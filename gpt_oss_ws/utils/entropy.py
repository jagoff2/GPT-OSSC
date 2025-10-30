from __future__ import annotations

import torch


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
  orig_dtype = logits.dtype
  if logits.dtype not in (torch.float32, torch.float64):
    logits = logits.to(torch.float32)
  probs = torch.nn.functional.softmax(logits, dim=-1)
  log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
  entropy = -(probs * log_probs).sum(dim=-1)
  if entropy.dtype != orig_dtype and orig_dtype in (torch.float16, torch.bfloat16):
    entropy = entropy.to(orig_dtype)
  return entropy


def batch_entropy_floor(logits: torch.Tensor) -> float:
  return float(token_entropy(logits)[..., -1].mean().item())
