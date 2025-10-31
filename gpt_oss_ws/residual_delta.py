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

    freq = torch.arange(self.slot_dim, dtype=torch.float32).unsqueeze(1)
    span = torch.arange(self.hidden_size, dtype=torch.float32).unsqueeze(0) + 1.0
    action_basis = torch.sin(freq * span * torch.pi / max(self.hidden_size, 1))
    action_basis = action_basis / action_basis.norm(dim=0, keepdim=True).clamp_min(1e-6)
    self.register_buffer("action_basis", action_basis)

  def apply(
    self,
    layer_idx: int,
    residual: torch.Tensor,
    slots: torch.Tensor,
    entropy: float,
    entropy_floor: float,
    plan_energy: torch.Tensor,
  ) -> torch.Tensor:
    if slots is None or residual.size(1) == 0:
      return residual
    try:
      hooked_idx = self.hooked_layers.index(layer_idx)
    except ValueError:
      return residual
    target_dtype = residual.dtype
    slot_avg = slots.mean(dim=1)
    if slot_avg.dtype != target_dtype:
      slot_avg = slot_avg.to(dtype=target_dtype)
    plan_energy = plan_energy.to(target_dtype)
    energy_term = torch.tanh(plan_energy).unsqueeze(-1).to(target_dtype)

    gate_scalar = torch.sigmoid(self.gate[hooked_idx]).to(target_dtype)
    gate = gate_scalar * (1.0 + energy_term.squeeze(-1))
    if entropy < entropy_floor:
      gate = gate * 0.1

    v = self.v[hooked_idx]
    u = self.u[hooked_idx]
    if v.dtype != target_dtype:
      v = v.to(dtype=target_dtype)
    if u.dtype != target_dtype:
      u = u.to(dtype=target_dtype)
    coeff = torch.matmul(slot_avg, v)
    delta = torch.matmul(coeff, u)
    if delta.dtype != target_dtype:
      delta = delta.to(dtype=target_dtype)

    action_drive = torch.matmul(slot_avg, self.action_basis)
    action_drive = torch.tanh(action_drive).to(target_dtype)
    delta = delta + energy_term * action_drive

    gate = gate.unsqueeze(-1)
    residual[:, -1, :] = residual[:, -1, :] + gate * delta
    return residual

  def forward(
    self,
    layer_idx: int,
    residual: torch.Tensor,
    slots: torch.Tensor,
    entropy: float,
    entropy_floor: float,
    plan_energy: torch.Tensor,
  ) -> torch.Tensor:
    return self.apply(layer_idx, residual, slots, entropy, entropy_floor, plan_energy)
