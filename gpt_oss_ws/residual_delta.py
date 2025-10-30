from __future__ import annotations

from typing import List

import torch
from torch import nn

from .config import WorkspaceConfig


class ResidualDeltaHook(nn.Module):
  def __init__(self, config: WorkspaceConfig, hidden_size: int) -> None:
    super().__init__()
    self.rank = config.residual_rank
    self.hidden_size = hidden_size
    self.slot_dim = config.slot_dim
    self.hooked_layers: List[int] = list(config.hooked_layers)
    self.u = nn.Parameter(torch.zeros(len(self.hooked_layers), self.rank, self.hidden_size))
    self.v = nn.Parameter(torch.zeros(len(self.hooked_layers), self.slot_dim, self.rank))
    self.gate = nn.Parameter(torch.zeros(len(self.hooked_layers)))
    nn.init.normal_(self.u, mean=0.0, std=0.02)
    nn.init.normal_(self.v, mean=0.0, std=0.02)

  def apply(self, layer_idx: int, residual: torch.Tensor, slots: torch.Tensor, entropy: float, entropy_floor: float) -> torch.Tensor:
    if slots is None or residual.size(1) == 0:
      return residual
    try:
      hooked_idx = self.hooked_layers.index(layer_idx)
    except ValueError:
      return residual
    gate = torch.sigmoid(self.gate[hooked_idx])
    if entropy < entropy_floor:
      gate = gate * 0.1
    # Work in the same dtype as the residual/hidden states to avoid promoting to float32.
    target_dtype = residual.dtype
    slot_avg = slots.mean(dim=1)
    if slot_avg.dtype != target_dtype:
      slot_avg = slot_avg.to(dtype=target_dtype)
    # Retrieve parameters for this layer in target dtype
    v = self.v[hooked_idx]
    u = self.u[hooked_idx]
    if v.dtype != target_dtype:
      v = v.to(dtype=target_dtype)
    if u.dtype != target_dtype:
      u = u.to(dtype=target_dtype)
    coeff = torch.matmul(slot_avg, v)  # [B, rank]
    delta = torch.matmul(coeff, u)  # [B, hidden_size]
    # Ensure dtype compatibility with residual (which may be bfloat16)
    if delta.dtype != target_dtype:
      delta = delta.to(dtype=target_dtype)
    if gate.dtype != target_dtype:
      gate = gate.to(dtype=target_dtype)
    if not getattr(self, "_logged_dtype_once", False):
      print(
        f"[residual-delta] layer={layer_idx} residual={residual.dtype} slots={slots.dtype} delta={delta.dtype} gate={gate.dtype}",
        flush=True,
      )
      self._logged_dtype_once = True
    residual[:, -1, :] = residual[:, -1, :] + gate * delta
    return residual

  def forward(self, layer_idx: int, residual: torch.Tensor, slots: torch.Tensor, entropy: float, entropy_floor: float) -> torch.Tensor:
    return self.apply(layer_idx, residual, slots, entropy, entropy_floor)
