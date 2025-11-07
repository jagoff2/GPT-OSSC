from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .config import WorkspaceConfig
from .scheduling import VirtualKVSegment, VirtualKVStore


class VirtualKVProjector(nn.Module):
  def __init__(
    self,
    config: WorkspaceConfig,
    hidden_size: int,
    target_dtype: Optional[torch.dtype] = None,
    workspace_dtype: Optional[torch.dtype] = None,
  ) -> None:
    super().__init__()
    self.config = config
    self.hidden_size = hidden_size
    self.kv_heads = 8
    self.head_dim = 64
    self.nvirt = config.nvirt
    proj_dim = self.kv_heads * self.nvirt * self.head_dim * 2
    self.linear = nn.Linear(config.slot_dim, proj_dim)
    nn.init.orthogonal_(self.linear.weight)
    self.linear.weight.data *= 0.02
    nn.init.zeros_(self.linear.bias)
    self.layer_ids = list(config.hooked_layers)
    self.layer_to_slot: Dict[int, int] = {layer: idx for idx, layer in enumerate(self.layer_ids)}
    self.store = VirtualKVStore(len(self.layer_ids), config.retention)
    base_dtype = target_dtype or torch.float32
    self.output_dtype = workspace_dtype or base_dtype
    if workspace_dtype is not None and self.linear.weight.dtype != workspace_dtype:
      self.linear = self.linear.to(dtype=workspace_dtype)

    total = self.kv_heads * self.nvirt * self.head_dim
    slot_dim = config.slot_dim
    freq = torch.arange(slot_dim, dtype=torch.float32).unsqueeze(1)
    span = torch.arange(total, dtype=torch.float32).unsqueeze(0) + 1.0
    plan_key_matrix = torch.sin(freq * span * torch.pi / max(slot_dim, 1))
    plan_val_matrix = torch.cos((freq + 0.5) * span * torch.pi / max(slot_dim, 1))
    plan_key_matrix = plan_key_matrix / plan_key_matrix.norm(dim=0, keepdim=True).clamp_min(1e-6)
    plan_val_matrix = plan_val_matrix / plan_val_matrix.norm(dim=0, keepdim=True).clamp_min(1e-6)
    self.register_buffer("plan_key_matrix", plan_key_matrix.to(dtype=self.output_dtype))
    self.register_buffer("plan_value_matrix", plan_val_matrix.to(dtype=self.output_dtype))
    self.register_buffer("plan_scale", torch.tensor(config.kv_plan_scale, dtype=torch.float32))
    self.register_buffer("plan_bias", torch.tensor(config.kv_plan_bias, dtype=torch.float32))
    self.projection_scale = float(config.kv_projection_scale)

  def forward(
    self,
    slots: torch.Tensor,
    layer_idx: int,
    device: str,
    target_dtype: Optional[torch.dtype] = None,
    plan_energy: Optional[torch.Tensor] = None,
  ) -> VirtualKVSegment:
    slot_idx = self.layer_to_slot[layer_idx]
    bsz = slots.size(0)
    linear_dtype = self.linear.weight.dtype
    if slots.dtype != linear_dtype:
      slots = slots.to(dtype=linear_dtype)
    pooled = slots.mean(dim=1)
    projected = self.linear(pooled)
    if self.projection_scale != 1.0:
      projected = projected * self.projection_scale
    total = self.kv_heads * self.nvirt * self.head_dim
    key_flat = projected[:, :total]
    value_flat = projected[:, total:]
    if plan_energy is not None:
      base_dtype = pooled.dtype
      tensor_device = pooled.device
      pe = plan_energy.to(device=tensor_device, dtype=base_dtype)
      plan_drive = torch.matmul(pooled, self.plan_key_matrix).to(device=tensor_device, dtype=base_dtype)
      scale = self.plan_scale.to(device=tensor_device, dtype=base_dtype)
      bias = self.plan_bias.to(device=tensor_device, dtype=base_dtype)
      scaled_energy = scale * pe.unsqueeze(-1) + bias
      plan_weight = torch.sigmoid(scaled_energy)
      key_flat = key_flat + plan_weight * plan_drive
      plan_value_drive = torch.matmul(pooled, self.plan_value_matrix).to(device=tensor_device, dtype=base_dtype)
      value_flat = value_flat + plan_weight * plan_value_drive
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
