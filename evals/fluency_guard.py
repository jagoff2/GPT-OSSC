from __future__ import annotations

import asyncio
import statistics
from typing import Iterable, List, Tuple

import torch

from gpt_oss_ws.config import WorkspaceConfig
from gpt_oss_ws.model_wrapper import GPTOSSHookedModel
from gpt_oss_ws.types import GenerationRequestContext, HookToggles


async def compare(baseline_cfg: WorkspaceConfig, workspace_cfg: WorkspaceConfig, samples: int = 32) -> None:
  loop = asyncio.get_event_loop()
  prompts = [
    "Explain how the workspace augments attention."
    " Summarize the last 24 hours of decisions."[:128],
    "Outline steps for multi-tool reasoning."
  ]
  prompts = (prompts * ((samples + len(prompts) - 1) // len(prompts)))[:samples]

  baseline_model = await loop.run_in_executor(None, GPTOSSHookedModel, baseline_cfg)
  workspace_model = await loop.run_in_executor(None, GPTOSSHookedModel, workspace_cfg)

  baseline_scores: List[float] = []
  workspace_scores: List[float] = []

  for prompt in prompts:
    messages = [{"role": "user", "content": prompt}]
    input_ids, attention_mask = baseline_model.prepare_chat_inputs(messages, add_generation_prompt=False)
    ctx = GenerationRequestContext(request_id="baseline", toggles=HookToggles(False, False, False, False))

    input_ids_device = input_ids.to(baseline_model.primary_device())
    attention_mask_device = attention_mask.to(baseline_model.primary_device())
    with torch.no_grad():
      outputs = baseline_model.model(input_ids=input_ids_device, attention_mask=attention_mask_device, use_cache=False)
    log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
    target = input_ids_device[:, 1:]
    neg_log_likelihood = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean().item()
    baseline_scores.append(neg_log_likelihood)

    workspace_ids, workspace_mask = workspace_model.prepare_chat_inputs(messages, add_generation_prompt=False)
    ws_ctx = GenerationRequestContext(request_id="workspace", toggles=HookToggles(True, True, True, True))
    workspace_ids_device = workspace_ids.to(workspace_model.primary_device())
    workspace_mask_device = workspace_mask.to(workspace_model.primary_device())
    with torch.no_grad():
      ws_outputs = workspace_model.model(input_ids=workspace_ids_device, attention_mask=workspace_mask_device, use_cache=False)
    ws_log_probs = torch.log_softmax(ws_outputs.logits[:, :-1, :], dim=-1)
    ws_target = workspace_ids_device[:, 1:]
    ws_nll = -ws_log_probs.gather(-1, ws_target.unsqueeze(-1)).squeeze(-1).mean().item()
    workspace_scores.append(ws_nll)

  baseline_ppl = statistics.mean(baseline_scores)
  workspace_ppl = statistics.mean(workspace_scores)
  print(f"Baseline NLL: {baseline_ppl:.3f}, Workspace NLL: {workspace_ppl:.3f}")

  await loop.run_in_executor(None, baseline_model.close)
  await loop.run_in_executor(None, workspace_model.close)
