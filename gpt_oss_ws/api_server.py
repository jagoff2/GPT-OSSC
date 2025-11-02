from __future__ import annotations

import json
import random
import time
import uuid
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import WorkspaceConfig, load_config
from .logging_utils import init_logger
from .model_wrapper import GPTOSSHookedModel
from .runtime import effective_max_new_tokens
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
  prompt: Sequence[str] | str
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
    self._history_lock = Lock()
    self._text_lock = Lock()
    self._request_counter = 0
    self.chat_history: List[Dict[str, str]] = []
    self.completion_history: str = ""
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

  def reset_runtime(self) -> None:
    self.model._current_plan_energy = None  # type: ignore[attr-defined]
    self.model._current_slots = None  # type: ignore[attr-defined]
    self.model._layer_residuals.clear()  # type: ignore[attr-defined]

  def clear_virtual_kv(self) -> None:
    store_cls = self.model.kv_projector.store.__class__
    layer_cnt = len(self.model.kv_projector.layer_ids)
    self.model.kv_projector.store = store_cls(layer_cnt, self.config.retention)

  def reset_chat_history(self) -> None:
    with self._history_lock:
      self.chat_history.clear()
    self.clear_virtual_kv()

  def reset_completion_history(self) -> None:
    with self._text_lock:
      self.completion_history = ""
    self.clear_virtual_kv()


def _normalize_prompt(prompt: Sequence[str] | str) -> str:
  if isinstance(prompt, str):
    return prompt
  return "\n".join(prompt)


def _decode_tokens(state: ServerState, tokens: torch.Tensor, prompt_tokens: int) -> str:
  parse = state.model.decode_generated(tokens[0], prompt_tokens)
  text = parse.final.strip()
  if text:
    return text
  return state.model.tokenizer_decode(tokens[0, prompt_tokens:]).strip()



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

    extra = dict(payload.extra or {})
    if payload.stream:
      extra["warning"] = "Streaming is not supported in this build; returning full response."
    reset = bool(extra.pop("reset_history", False))

    default_system = {
      "role": "system",
      "content": (
        "You are ChatGPT, a helpful assistant. Answer every user question directly. "
        "When a user asks for a calculation, compute it exactly and return the numeric result."
      ),
    }

    if reset:
      state.reset_chat_history()
    with state._history_lock:
      payload_messages = [msg.model_dump() for msg in payload.messages]
      if payload_messages:
        if state.chat_history:
          overlap = 0
          max_overlap = min(len(state.chat_history), len(payload_messages))
          for k in range(max_overlap, 0, -1):
            if state.chat_history[-k:] == payload_messages[:k]:
              overlap = k
              break
          if overlap == len(payload_messages) and len(state.chat_history) >= len(payload_messages):
            state.chat_history = list(payload_messages)
          else:
            state.chat_history.extend(payload_messages[overlap:])
        else:
          state.chat_history = list(payload_messages)
      conversation = list(state.chat_history)
      if not conversation or conversation[0]["role"] != "system":
        conversation.insert(0, default_system)

    if not conversation:
      raise HTTPException(status_code=400, detail="Chat history is empty; provide at least one user message.")

    input_ids, attention_mask = state.model.prepare_chat_inputs(conversation, add_generation_prompt=True)
    prompt_tokens = input_ids.shape[-1]
    max_new_tokens = effective_max_new_tokens(state.model, payload.max_tokens)
    toggles = HookToggles(kv_append=True, residual_delta=True, read_probes=True, broadcast=True)
    request_ctx = GenerationRequestContext(
      request_id=str(uuid.uuid4()),
      toggles=toggles,
    )
    temperature = payload.temperature if payload.temperature is not None else state.default_temperature
    top_p = payload.top_p if payload.top_p is not None else state.default_top_p
    seed_value = state.next_seed()

    state.reset_runtime()
    state.apply_seed(seed_value)
    tokens = state.model.generate(
      request_ctx,
      input_ids=input_ids,
      attention_mask=attention_mask,
      max_new_tokens=max_new_tokens,
      temperature=temperature,
      top_p=top_p,
    )
    state.reset_runtime()

    assistant_text = _decode_tokens(state, tokens, prompt_tokens)
    last_assistant = None
    with state._history_lock:
      for entry in reversed(state.chat_history):
        if entry["role"] == "assistant":
          last_assistant = entry["content"]
          break
    if last_assistant and assistant_text.strip() == last_assistant.strip():
      fallback_temperature = max(0.7, temperature if temperature > 0 else 0.7)
      fallback_top_p = top_p if top_p < 0.95 else 0.95
      if fallback_temperature != temperature or fallback_top_p != top_p:
        state.reset_runtime()
        alt_seed = state.next_seed()
        state.apply_seed(alt_seed)
        tokens_alt = state.model.generate(
          request_ctx,
          input_ids=input_ids,
          attention_mask=attention_mask,
          max_new_tokens=max_new_tokens,
          temperature=fallback_temperature,
          top_p=fallback_top_p,
        )
        state.reset_runtime()
        alt_text = _decode_tokens(state, tokens_alt, prompt_tokens)
        if alt_text.strip() != last_assistant.strip():
          assistant_text = alt_text
          tokens = tokens_alt

    with state._history_lock:
      state.chat_history.append({"role": "assistant", "content": assistant_text})

    completion_tokens = tokens.shape[-1] - prompt_tokens
    response = ChatCompletionResponse(
      id=request_ctx.request_id,
      object="chat.completion",
      created=int(time.time()),
      model=payload.model,
      choices=[
        ChatCompletionChoice(
          index=0,
          message={"role": "assistant", "content": assistant_text},
          finish_reason="stop",
        )
      ],
      usage={
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens + completion_tokens),
      },
      extra=extra or None,
    )
    return JSONResponse(content=json.loads(response.json()))

  @app.post("/v1/completions")
  async def completions(payload: CompletionRequest):
    if payload.model != config.model_name:
      raise HTTPException(status_code=400, detail="model mismatch")

    extra = dict(payload.extra or {})
    if payload.stream:
      extra["warning"] = "Streaming is not supported in this build; returning full response."
    reset = bool(extra.pop("reset_history", False))

    if reset:
      state.reset_completion_history()
    with state._text_lock:
      prompt_text = _normalize_prompt(payload.prompt)
      combined_prompt = state.completion_history + prompt_text
      state.completion_history = combined_prompt

    messages = [{"role": "user", "content": combined_prompt}]
    input_ids, attention_mask = state.model.prepare_chat_inputs(messages, add_generation_prompt=True)
    prompt_tokens = input_ids.shape[-1]
    max_new_tokens = effective_max_new_tokens(state.model, payload.max_tokens)
    toggles = HookToggles(kv_append=True, residual_delta=True, read_probes=True, broadcast=True)
    request_ctx = GenerationRequestContext(
      request_id=str(uuid.uuid4()),
      toggles=toggles,
    )
    temperature = payload.temperature if payload.temperature is not None else state.default_temperature
    top_p = payload.top_p if payload.top_p is not None else state.default_top_p
    seed_value = state.next_seed()

    state.reset_runtime()
    state.apply_seed(seed_value)
    tokens = state.model.generate(
      request_ctx,
      input_ids=input_ids,
      attention_mask=attention_mask,
      max_new_tokens=max_new_tokens,
      temperature=temperature,
      top_p=top_p,
    )
    state.reset_runtime()

    completion_text = _decode_tokens(state, tokens, prompt_tokens)

    with state._text_lock:
      state.completion_history += completion_text

    completion_tokens = tokens.shape[-1] - prompt_tokens
    response = CompletionResponse(
      id=request_ctx.request_id,
      object="text_completion",
      created=int(time.time()),
      model=payload.model,
      choices=[
        CompletionChoice(index=0, text=completion_text, finish_reason="stop")
      ],
      usage={
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens + completion_tokens),
      },
      extra=extra or None,
    )
    return JSONResponse(content=json.loads(response.json()))

  return app
