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
    input_ids = model.tokenizer_encode(prompt)
    outputs = model.generate(request_ctx, input_ids=input_ids, max_new_tokens=128)
    text = model.tokenizer_decode(outputs[0])
    print(json.dumps({"prompt": prompt, "completion": text}))
  model.close()
