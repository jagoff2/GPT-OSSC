from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from threading import Lock
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import WorkspaceConfig, load_config
from .logging_utils import init_logger
from .model_wrapper import GPTOSSHookedModel
from .types import GenerationRequestContext, HookToggles


class ChatMessage(BaseModel):
  role: str
  content: str


class ChatCompletionRequest(BaseModel):
  model: str
  messages: List[ChatMessage]
  max_tokens: int = Field(default=256, alias="max_tokens")
  temperature: float = 0.0
  top_p: float = 1.0
  stream: bool = False
  extra: Dict[str, Any] = Field(default_factory=dict)


class ChatCompletionChoice(BaseModel):
  index: int
  message: Dict[str, Any]
  finish_reason: str


class ChatCompletionResponse(BaseModel):
  id: str
  object: str
  created: int
  model: str
  choices: List[ChatCompletionChoice]
  usage: Dict[str, int]
  extra: Optional[Dict[str, Any]] = None


class CompletionRequest(BaseModel):
  model: str
  prompt: Union[str, List[str]]
  max_tokens: int = Field(default=256, alias="max_tokens")
  temperature: float = 0.0
  top_p: float = 1.0
  stream: bool = False
  extra: Dict[str, Any] = Field(default_factory=dict)


class CompletionChoice(BaseModel):
  index: int
  text: str
  logprobs: Optional[Any] = None
  finish_reason: str


class CompletionResponse(BaseModel):
  id: str
  object: str
  created: int
  model: str
  choices: List[CompletionChoice]
  usage: Dict[str, int]
  extra: Optional[Dict[str, Any]] = None


def _set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


class ServerState:
  def __init__(
    self,
    config: WorkspaceConfig,
    base_seed: int,
    default_temperature: float,
    default_top_p: float,
  ) -> None:
    self.config = config
    self.logger = init_logger("gpt_oss_ws.api", config.log_level)
    self.model = GPTOSSHookedModel(config)
    self.base_seed = base_seed
    self.default_temperature = default_temperature
    self.default_top_p = default_top_p
    self._seed_lock = Lock()
    self._request_counter = 0
    _set_seed(self.base_seed)
    try:
      torch.use_deterministic_algorithms(True)
    except Exception as exc:  # pragma: no cover
      self.logger.warning("Unable to enable deterministic algorithms: %s", exc)
    if hasattr(torch.backends, "cudnn"):
      try:
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
      except Exception as exc:  # pragma: no cover
        self.logger.warning("Unable to configure cuDNN determinism: %s", exc)

  def shutdown(self) -> None:
    self.logger.info("Shutting down workspace model")
    self.model.close()

  def next_seed(self) -> int:
    with self._seed_lock:
      seed = self.base_seed + self._request_counter
      self._request_counter += 1
    return seed

  def apply_seed(self, seed: int) -> None:
    _set_seed(seed)

  def snapshot_workspace(self) -> Dict[str, Any]:
    return self.model.kv_projector.store.state_dict()

  def restore_workspace(self, state_dict: Dict[str, Any]) -> None:
    self.model.kv_projector.store.load_state_dict(state_dict)

  def reset_runtime(self) -> None:
    self.model._current_plan_energy = None  # type: ignore[attr-defined]
    self.model._current_slots = None  # type: ignore[attr-defined]
    self.model._layer_residuals.clear()  # type: ignore[attr-defined]


def _build_prompt(messages: List[ChatMessage]) -> str:
  lines = [f"{message.role}: {message.content}" for message in messages]
  lines.append("assistant:")
  return "\n".join(lines)


def _normalize_prompt(prompt: Union[str, List[str]]) -> str:
  if isinstance(prompt, str):
    return prompt
  return "\n".join(prompt)


def _toggles_from_extra(extra: Dict[str, Any]) -> HookToggles:
  toggles_cfg = extra.get("toggles", {}) if extra else {}
  return HookToggles(
    kv_append=toggles_cfg.get("kv_append", True),
    residual_delta=toggles_cfg.get("residual_delta", True),
    read_probes=toggles_cfg.get("read_probes", True),
    broadcast=toggles_cfg.get("broadcast", True),
  )


def _decode_increment(tokenizer, token_id: int) -> str:
  text = tokenizer.decode([token_id], skip_special_tokens=True)
  if text == "":
    text = tokenizer.decode([token_id], skip_special_tokens=False)
  return text


def _messages_to_dicts(messages: Sequence[ChatMessage]) -> List[Dict[str, str]]:
  return [message.model_dump() for message in messages]


async def _stream_chat_events(
  state: ServerState,
  payload: ChatCompletionRequest,
  request_ctx: GenerationRequestContext,
  input_ids: torch.Tensor,
  attention_mask: torch.Tensor,
  prompt_tokens: int,
  seed: int,
  temperature: float,
  top_p: float,
) -> AsyncGenerator[str, None]:
  queue: asyncio.Queue[Tuple[str, Any]] = asyncio.Queue()
  loop = asyncio.get_event_loop()
  generated_ids: List[int] = []
  last_sent_len = 0
  role_sent = False
  prefill_ids = input_ids[0].tolist()

  def callback(token: torch.Tensor, logits: torch.Tensor) -> None:
    token_id = int(token.squeeze().item())
    generated_ids.append(token_id)
    loop.call_soon_threadsafe(queue.put_nowait, ("token", token_id))

  def run_generation() -> None:
    try:
      snapshot = state.snapshot_workspace()
      state.reset_runtime()
      state.apply_seed(seed)
      tokens = state.model.generate(
        request_ctx,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=payload.max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream_callback=callback,
      )
      loop.call_soon_threadsafe(queue.put_nowait, ("final", tokens))
    except Exception as exc:  # pragma: no cover
      loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
    finally:
      state.restore_workspace(snapshot)
      state.reset_runtime()

  loop.run_in_executor(None, run_generation)

  tokenizer = state.model.tokenizer

  while True:
    item_type, payload_obj = await queue.get()
    if item_type == "token":
      full_sequence = prefill_ids + generated_ids
      parse = state.model._parse_harmony_tokens(full_sequence)
      final_text = parse.final
      if not final_text:
        continue
      delta_text = final_text[last_sent_len:]
      if not delta_text:
        continue
      chunk_delta: Dict[str, Any] = {"content": delta_text}
      if not role_sent:
        chunk_delta["role"] = "assistant"
        role_sent = True
      chunk = {
        "id": request_ctx.request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": payload.model,
        "choices": [
          {
            "index": 0,
            "delta": chunk_delta,
            "finish_reason": None,
          }
        ],
      }
      last_sent_len = len(final_text)
      yield f"data: {json.dumps(chunk)}\n\n"
    elif item_type == "final":
      tokens = payload_obj
      completion_tokens = tokens.shape[-1] - prompt_tokens
      parse = state.model.decode_generated(tokens[0], prompt_tokens)
      output_text = parse.final.strip()
      if not output_text:
        output_text = state.model.tokenizer_decode(tokens[0, prompt_tokens:])
      final_chunk = {
        "id": request_ctx.request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.model,
        "choices": [
          {
            "index": 0,
            "message": {"role": "assistant", "content": output_text},
            "finish_reason": "stop",
          }
        ],
        "usage": {
          "prompt_tokens": int(prompt_tokens),
          "completion_tokens": int(completion_tokens),
          "total_tokens": int(prompt_tokens + completion_tokens),
        },
        "extra": payload.extra or None,
      }
      yield f"data: {json.dumps(final_chunk)}\n\n"
      yield "data: [DONE]\n\n"
      break
    elif item_type == "error":
      error_chunk = {"error": {"message": payload_obj, "type": "server_error"}}
      yield f"data: {json.dumps(error_chunk)}\n\n"
      yield "data: [DONE]\n\n"
      break


async def _stream_completion_events(
  state: ServerState,
  payload: CompletionRequest,
  request_ctx: GenerationRequestContext,
  input_ids: torch.Tensor,
  attention_mask: torch.Tensor,
  prompt_tokens: int,
  seed: int,
  temperature: float,
  top_p: float,
) -> AsyncGenerator[str, None]:
  queue: asyncio.Queue[Tuple[str, Any]] = asyncio.Queue()
  loop = asyncio.get_event_loop()
  generated_ids: List[int] = []
  last_sent_len = 0
  prefill_ids = input_ids[0].tolist()

  def callback(token: torch.Tensor, logits: torch.Tensor) -> None:
    token_id = int(token.squeeze().item())
    generated_ids.append(token_id)
    loop.call_soon_threadsafe(queue.put_nowait, ("token", token_id))

  def run_generation() -> None:
    try:
      snapshot = state.snapshot_workspace()
      state.reset_runtime()
      state.apply_seed(seed)
      tokens = state.model.generate(
        request_ctx,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=payload.max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream_callback=callback,
      )
      loop.call_soon_threadsafe(queue.put_nowait, ("final", tokens))
    except Exception as exc:  # pragma: no cover
      loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
    finally:
      state.restore_workspace(snapshot)
      state.reset_runtime()

  loop.run_in_executor(None, run_generation)

  tokenizer = state.model.tokenizer

  while True:
    item_type, payload_obj = await queue.get()
    if item_type == "token":
      full_sequence = prefill_ids + generated_ids
      parse = state.model._parse_harmony_tokens(full_sequence)
      final_text = parse.final
      if not final_text:
        continue
      delta_text = final_text[last_sent_len:]
      if not delta_text:
        continue
      chunk = {
        "id": request_ctx.request_id,
        "object": "text_completion.chunk",
        "created": int(time.time()),
        "model": payload.model,
        "choices": [
          {
            "index": 0,
            "delta": {"text": delta_text},
            "finish_reason": None,
          }
        ],
      }
      last_sent_len = len(final_text)
      yield f"data: {json.dumps(chunk)}\n\n"
    elif item_type == "final":
      tokens = payload_obj
      completion_tokens = tokens.shape[-1] - prompt_tokens
      parse = state.model.decode_generated(tokens[0], prompt_tokens)
      output_text = parse.final.strip()
      if not output_text:
        output_text = state.model.tokenizer_decode(tokens[0, prompt_tokens:])
      final_chunk = {
        "id": request_ctx.request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": payload.model,
        "choices": [
          {
            "index": 0,
            "text": output_text,
            "logprobs": None,
            "finish_reason": "stop",
          }
        ],
        "usage": {
          "prompt_tokens": int(prompt_tokens),
          "completion_tokens": int(completion_tokens),
          "total_tokens": int(prompt_tokens + completion_tokens),
        },
        "extra": payload.extra or None,
      }
      yield f"data: {json.dumps(final_chunk)}\n\n"
      yield "data: [DONE]\n\n"
      break
    elif item_type == "error":
      error_chunk = {"error": {"message": payload_obj, "type": "server_error"}}
      yield f"data: {json.dumps(error_chunk)}\n\n"
      yield "data: [DONE]\n\n"
      break


def create_app(
  config_path: str,
  overrides: Optional[Dict[str, Any]] = None,
  *,
  base_seed: int = 99,
  default_temperature: float = 0.0,
  default_top_p: float = 1.0,
) -> FastAPI:
  config = load_config(config_path, overrides or {})
  state = ServerState(
    config,
    base_seed=base_seed,
    default_temperature=default_temperature,
    default_top_p=default_top_p,
  )
  app = FastAPI()

  @app.on_event("shutdown")
  async def _shutdown() -> None:  # pragma: no cover
    state.shutdown()

  @app.get("/health")
  def health() -> Dict[str, str]:
    return {"status": "ok"}

  @app.get("/v1/models")
  def list_models() -> Dict[str, Any]:
    return {
      "object": "list",
      "data": [
        {
          "id": config.model_name,
          "object": "model",
          "created": int(time.time()),
          "owned_by": "gpt-oss",
        }
      ],
    }

  @app.post("/v1/chat/completions")
  async def chat_completions(payload: ChatCompletionRequest):
    if payload.model != config.model_name:
      raise HTTPException(status_code=400, detail="model mismatch")
    message_dicts = _messages_to_dicts(payload.messages)
    input_ids, attention_mask = state.model.prepare_chat_inputs(message_dicts, add_generation_prompt=True)
    prompt_tokens = input_ids.shape[-1]
    toggles = _toggles_from_extra(payload.extra)
    request_ctx = GenerationRequestContext(
      request_id=str(uuid.uuid4()),
      toggles=toggles,
      retention_overrides=payload.extra.get("retention") if payload.extra else None,
    )
    effective_temperature = payload.temperature if payload.temperature is not None else state.default_temperature
    effective_top_p = payload.top_p if payload.top_p is not None else state.default_top_p
    seed_value = state.next_seed()

    if payload.stream:
      generator = _stream_chat_events(
        state,
        payload,
        request_ctx,
        input_ids,
        attention_mask,
        prompt_tokens,
        seed_value,
        effective_temperature,
        effective_top_p,
      )
      return StreamingResponse(generator, media_type="text/event-stream")

    snapshot = state.snapshot_workspace()
    try:
      state.reset_runtime()
      state.apply_seed(seed_value)
      tokens = state.model.generate(
        request_ctx,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=payload.max_tokens,
        temperature=effective_temperature,
        top_p=effective_top_p,
      )
    finally:
      state.restore_workspace(snapshot)
      state.reset_runtime()
    completion_tokens = tokens.shape[-1] - prompt_tokens
    parse = state.model.decode_generated(tokens[0], prompt_tokens)
    text = parse.final.strip()
    if not text:
      text = state.model.tokenizer_decode(tokens[0, prompt_tokens:])
    response = ChatCompletionResponse(
      id=request_ctx.request_id,
      object="chat.completion",
      created=int(time.time()),
      model=payload.model,
      choices=[
        ChatCompletionChoice(
          index=0,
          message={"role": "assistant", "content": text},
          finish_reason="stop",
        )
      ],
      usage={
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens + completion_tokens),
      },
      extra=payload.extra or None,
    )
    return JSONResponse(content=json.loads(response.json()))

  @app.post("/v1/completions")
  async def completions(payload: CompletionRequest):
    if payload.model != config.model_name:
      raise HTTPException(status_code=400, detail="model mismatch")
    prompt_text = _normalize_prompt(payload.prompt)
    messages = [{"role": "user", "content": prompt_text}]
    input_ids, attention_mask = state.model.prepare_chat_inputs(messages, add_generation_prompt=True)
    prompt_tokens = input_ids.shape[-1]
    toggles = _toggles_from_extra(payload.extra)
    request_ctx = GenerationRequestContext(
      request_id=str(uuid.uuid4()),
      toggles=toggles,
      retention_overrides=payload.extra.get("retention") if payload.extra else None,
    )
    effective_temperature = payload.temperature if payload.temperature is not None else state.default_temperature
    effective_top_p = payload.top_p if payload.top_p is not None else state.default_top_p
    seed_value = state.next_seed()

    if payload.stream:
      generator = _stream_completion_events(
        state,
        payload,
        request_ctx,
        input_ids,
        attention_mask,
        prompt_tokens,
        seed_value,
        effective_temperature,
        effective_top_p,
      )
      return StreamingResponse(generator, media_type="text/event-stream")

    snapshot = state.snapshot_workspace()
    try:
      state.reset_runtime()
      state.apply_seed(seed_value)
      tokens = state.model.generate(
        request_ctx,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=payload.max_tokens,
        temperature=effective_temperature,
        top_p=effective_top_p,
      )
    finally:
      state.restore_workspace(snapshot)
      state.reset_runtime()
    completion_tokens = tokens.shape[-1] - prompt_tokens
    parse = state.model.decode_generated(tokens[0], prompt_tokens)
    text = parse.final.strip()
    if not text:
      text = state.model.tokenizer_decode(tokens[0, prompt_tokens:])
    response = CompletionResponse(
      id=request_ctx.request_id,
      object="text_completion",
      created=int(time.time()),
      model=payload.model,
      choices=[
        CompletionChoice(index=0, text=text, finish_reason="stop")
      ],
      usage={
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens + completion_tokens),
      },
      extra=payload.extra or None,
    )
    return JSONResponse(content=json.loads(response.json()))

  return app
