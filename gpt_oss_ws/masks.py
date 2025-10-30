from __future__ import annotations

from typing import Optional, Tuple

import torch


def extend_causal_mask(
  base_mask: torch.Tensor,
  virtual_kv: Optional[torch.Tensor],
  device: Optional[str] = None,
) -> torch.Tensor:
  if virtual_kv is None:
    return base_mask
  virtual_len = virtual_kv.shape[-2]
  bsz, num_heads, seq_len, _ = base_mask.shape
  device = device or base_mask.device
  virtual_mask = torch.zeros((bsz, num_heads, seq_len, virtual_len), device=device)
  extended = torch.cat([virtual_mask, base_mask], dim=-1)
  return extended


def build_attention_bias(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
  mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
  mask = torch.triu(mask, diagonal=1)
  return mask
