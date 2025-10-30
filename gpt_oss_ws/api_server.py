from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

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
  temperature: float = 0.8
  top_p: float = 0.95
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
  temperature: float = 0.8
  top_p: float = 0.95
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


class ServerState:
  def __init__(self, config: WorkspaceConfig) -> None:
    self.config = config
    self.logger = init_logger("gpt_oss_ws.api", config.log_level)
    self.model = GPTOSSHookedModel(config)

  def shutdown(self) -> None:
    self.logger.info("Shutting down workspace model")
    self.model.close()


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


async def _stream_chat_events(
  state: ServerState,
  payload: ChatCompletionRequest,
  request_ctx: GenerationRequestContext,
  input_ids: torch.Tensor,
  prompt_tokens: int,
) -> AsyncGenerator[str, None]:
  queue: asyncio.Queue[Tuple[str, Any]] = asyncio.Queue()
  loop = asyncio.get_event_loop()

  def callback(token: torch.Tensor, logits: torch.Tensor) -> None:
    loop.call_soon_threadsafe(queue.put_nowait, ("token", token.squeeze().item()))

  def run_generation() -> None:
    try:
      tokens = state.model.generate(
        request_ctx,
        input_ids=input_ids,
        max_new_tokens=payload.max_tokens,
        temperature=payload.temperature,
        top_p=payload.top_p,
        stream_callback=callback,
      )
      loop.call_soon_threadsafe(queue.put_nowait, ("final", tokens))
    except Exception as exc:  # pragma: no cover
      loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

  loop.run_in_executor(None, run_generation)

  tokenizer = state.model.tokenizer

  while True:
    item_type, payload_obj = await queue.get()
    if item_type == "token":
      token_id = int(payload_obj)
      chunk = {
        "id": request_ctx.request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": payload.model,
        "choices": [
          {
            "index": 0,
            "delta": {"role": "assistant", "content": _decode_increment(tokenizer, token_id)},
            "finish_reason": None,
          }
        ],
      }
      yield f"data: {json.dumps(chunk)}\n\n"
    elif item_type == "final":
      tokens = payload_obj
      completion_tokens = tokens.shape[-1] - prompt_tokens
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
  prompt_tokens: int,
) -> AsyncGenerator[str, None]:
  queue: asyncio.Queue[Tuple[str, Any]] = asyncio.Queue()
  loop = asyncio.get_event_loop()

  def callback(token: torch.Tensor, logits: torch.Tensor) -> None:
    loop.call_soon_threadsafe(queue.put_nowait, ("token", token.squeeze().item()))

  def run_generation() -> None:
    try:
      tokens = state.model.generate(
        request_ctx,
        input_ids=input_ids,
        max_new_tokens=payload.max_tokens,
        temperature=payload.temperature,
        top_p=payload.top_p,
        stream_callback=callback,
      )
      loop.call_soon_threadsafe(queue.put_nowait, ("final", tokens))
    except Exception as exc:  # pragma: no cover
      loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

  loop.run_in_executor(None, run_generation)

  tokenizer = state.model.tokenizer

  while True:
    item_type, payload_obj = await queue.get()
    if item_type == "token":
      token_id = int(payload_obj)
      chunk = {
        "id": request_ctx.request_id,
        "object": "text_completion.chunk",
        "created": int(time.time()),
        "model": payload.model,
        "choices": [
          {
            "index": 0,
            "delta": {"text": _decode_increment(tokenizer, token_id)},
            "finish_reason": None,
          }
        ],
      }
      yield f"data: {json.dumps(chunk)}\n\n"
    elif item_type == "final":
      tokens = payload_obj
      completion_tokens = tokens.shape[-1] - prompt_tokens
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


def create_app(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> FastAPI:
  config = load_config(config_path, overrides)
  state = ServerState(config)
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
    prompt = _build_prompt(payload.messages)
    input_ids = state.model.tokenizer_encode(prompt)
    prompt_tokens = input_ids.shape[-1]
    toggles = _toggles_from_extra(payload.extra)
    request_ctx = GenerationRequestContext(
      request_id=str(uuid.uuid4()),
      toggles=toggles,
      retention_overrides=payload.extra.get("retention") if payload.extra else None,
    )

    if payload.stream:
      generator = _stream_chat_events(state, payload, request_ctx, input_ids, prompt_tokens)
      return StreamingResponse(generator, media_type="text/event-stream")

    tokens = state.model.generate(
      request_ctx,
      input_ids=input_ids,
      max_new_tokens=payload.max_tokens,
      temperature=payload.temperature,
      top_p=payload.top_p,
    )
    completion_tokens = tokens.shape[-1] - prompt_tokens
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
    input_ids = state.model.tokenizer_encode(prompt_text)
    prompt_tokens = input_ids.shape[-1]
    toggles = _toggles_from_extra(payload.extra)
    request_ctx = GenerationRequestContext(
      request_id=str(uuid.uuid4()),
      toggles=toggles,
      retention_overrides=payload.extra.get("retention") if payload.extra else None,
    )

    if payload.stream:
      generator = _stream_completion_events(state, payload, request_ctx, input_ids, prompt_tokens)
      return StreamingResponse(generator, media_type="text/event-stream")

    tokens = state.model.generate(
      request_ctx,
      input_ids=input_ids,
      max_new_tokens=payload.max_tokens,
      temperature=payload.temperature,
      top_p=payload.top_p,
    )
    completion_tokens = tokens.shape[-1] - prompt_tokens
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
