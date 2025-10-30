from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .config import WorkspaceConfig
from .scheduling import VirtualKVSegment, VirtualKVStore


class VirtualKVProjector(nn.Module):
  def __init__(self, config: WorkspaceConfig, hidden_size: int, target_dtype: torch.dtype) -> None:
    super().__init__()
    self.config = config
    self.hidden_size = hidden_size
    self.kv_heads = 8
    self.head_dim = 64
    self.nvirt = config.nvirt
    proj_dim = self.kv_heads * self.nvirt * self.head_dim * 2
    self.linear = nn.Linear(config.slot_dim, proj_dim)
    nn.init.zeros_(self.linear.weight)
    nn.init.zeros_(self.linear.bias)
    self.layer_ids = list(config.hooked_layers)
    self.layer_to_slot: Dict[int, int] = {layer: idx for idx, layer in enumerate(self.layer_ids)}
    self.store = VirtualKVStore(len(self.layer_ids), config.retention)
    self.output_dtype = target_dtype

  def forward(
    self,
    slots: torch.Tensor,
    layer_idx: int,
    device: str,
    target_dtype: Optional[torch.dtype] = None,
  ) -> VirtualKVSegment:
    slot_idx = self.layer_to_slot[layer_idx]
    bsz = slots.size(0)
    pooled = slots.mean(dim=1)
    projected = self.linear(pooled)
    total = self.kv_heads * self.nvirt * self.head_dim
    key_flat = projected[:, :total]
    value_flat = projected[:, total:]
    key = key_flat.view(bsz, self.kv_heads, self.nvirt, self.head_dim)
    value = value_flat.view(bsz, self.kv_heads, self.nvirt, self.head_dim)
    dtype = target_dtype or self.output_dtype
    key = key.to(dtype=dtype)
    value = value.to(dtype=dtype)
    segment = VirtualKVSegment(
      key=key.to(device=device, dtype=dtype, non_blocking=True),
      value=value.to(device=device, dtype=dtype, non_blocking=True),
      created_step=self.store.step,
      ttl_steps=self.config.retention.virt_kv_ttl_steps,
      device=device,
    )
    self.store.append(slot_idx, segment)
    return segment

  def fetch(self, layer_idx: int, device: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if layer_idx not in self.layer_to_slot:
      return None
    slot_idx = self.layer_to_slot[layer_idx]
    segment = self.store.fetch(slot_idx, device)
    if segment is None:
      return None
    return segment.key, segment.value

  def advance_step(self) -> None:
    self.store.advance()
