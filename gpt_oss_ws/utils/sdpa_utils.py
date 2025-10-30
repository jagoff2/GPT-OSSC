from __future__ import annotations

from typing import Optional

import torch


def safe_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
  try:
    return torch.nn.functional.scaled_dot_product_attention(
      q,
      k,
      v,
      attn_mask=attn_mask,
      dropout_p=0.0,
      is_causal=False,
    )
  except RuntimeError:
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
    if attn_mask is not None:
      scores += attn_mask
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)
