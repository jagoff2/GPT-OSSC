from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Iterable, Tuple

import torch


def concat_kv(real: torch.Tensor, virtual: torch.Tensor) -> torch.Tensor:
  if virtual is None or virtual.numel() == 0:
    return real
  if real.shape[:-2] != virtual.shape[:-2]:
    raise ValueError(f"KV shape mismatch: {real.shape} vs {virtual.shape}")
  return torch.cat([real, virtual], dim=-2)


def masked_softmax(attn_scores: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
  attn_scores = attn_scores + attn_mask
  return torch.softmax(attn_scores, dim=-1)


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
  bsz, seq_len, hidden = x.shape
  head_dim = hidden // num_heads
  return x.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
  bsz, num_heads, seq_len, head_dim = x.shape
  return x.transpose(1, 2).contiguous().view(bsz, seq_len, num_heads * head_dim)


@contextmanager
def inference_mode() -> Generator[None, None, None]:
  with torch.inference_mode():
    yield


def batched_index_select(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
  gather_shape = (values.shape[0], indices.shape[1], values.shape[-1])
  expanded_indices = indices.unsqueeze(-1).expand(-1, -1, values.shape[-1])
  return torch.gather(values, 1, expanded_indices).view(*gather_shape)
