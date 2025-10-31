from __future__ import annotations

import asyncio
import json
import time
from typing import Dict

import torch

from gpt_oss_ws.config import WorkspaceConfig
from gpt_oss_ws.model_wrapper import GPTOSSHookedModel
from gpt_oss_ws.types import GenerationRequestContext, HookToggles

from .tasks_long_horizon import TASKS


async def run(task: str, config: WorkspaceConfig) -> None:
  if task not in ("fluency", "long_horizon"):
    raise ValueError(f"Unsupported task: {task}")
  model = GPTOSSHookedModel(config)
  prompts = TASKS.get("tool_plan" if task == "long_horizon" else "long_context", [])
  if not prompts:
    prompts = ["Describe how the workspace controller decides to broadcast."]
  request_ctx = GenerationRequestContext(request_id=str(time.time()), toggles=HookToggles(True, True, True, True))
  for prompt in prompts:
    messages = [{"role": "user", "content": prompt}]
    input_ids, attention_mask = model.prepare_chat_inputs(messages, add_generation_prompt=True)
    prompt_tokens = input_ids.shape[-1]
    outputs = model.generate(request_ctx, input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=128)
    parsed = model.decode_generated(outputs[0], prompt_tokens)
    text = (parsed.final or "").strip()
    if not text:
      text = model.tokenizer_decode(outputs[0, prompt_tokens:])
    print(json.dumps({"prompt": prompt, "completion": text}))
  model.close()
