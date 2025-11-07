from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .config import WorkspaceConfig


class SlotAttentionWorkspace(nn.Module):
  def __init__(self, config: WorkspaceConfig, input_dim: int = 128) -> None:
    super().__init__()
    self.slot_dim = config.slot_dim
    self.slot_count = config.slot_count
    self.iters = max(config.slot_iterations, 2)
    self.scale = (self.slot_dim) ** -0.5

    self.slot_mu = nn.Parameter(torch.zeros(1, 1, self.slot_dim))
    self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, self.slot_dim))

    self.project_q = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
    self.project_k = nn.Linear(input_dim, self.slot_dim, bias=False)
    self.project_v = nn.Linear(input_dim, self.slot_dim, bias=False)
    self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
    self.mlp = nn.Sequential(
      nn.LayerNorm(self.slot_dim),
      nn.Linear(self.slot_dim, self.slot_dim * 2),
      nn.GELU(),
      nn.Linear(self.slot_dim * 2, self.slot_dim)
    )

    freq = torch.arange(self.slot_dim, dtype=torch.float32).unsqueeze(1)
    phase = (torch.arange(self.slot_dim, dtype=torch.float32).unsqueeze(0) + 0.5)
    basis = torch.cos(freq * phase * torch.pi / max(1, self.slot_dim))
    basis = basis / basis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    self.register_buffer("plan_basis", basis)
    self.register_buffer("plan_coupling", torch.tensor(config.slot_dim ** -0.5, dtype=torch.float32))

  def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = inputs.size(0)
    mu = self.slot_mu.expand(bsz, self.slot_count, -1)
    sigma = torch.exp(self.slot_log_sigma).expand_as(mu)
    slots = mu + sigma * torch.randn_like(mu)

    k = self.project_k(inputs)
    v = self.project_v(inputs)

    plan_energy = torch.zeros(bsz, device=inputs.device, dtype=inputs.dtype)
    for _ in range(self.iters):
      slots_prev = slots
      q = self.project_q(slots)
      attn_logits = torch.einsum("bnd,bmd->bnm", q, k) * self.scale
      attn = torch.softmax(attn_logits, dim=-1)
      updates = torch.einsum("bnm,bmd->bnd", attn, v)
      slots = self.gru(
        updates.reshape(-1, self.slot_dim),
        slots_prev.reshape(-1, self.slot_dim)
      )
      slots = slots.reshape(bsz, self.slot_count, self.slot_dim)
      plan_drive = torch.matmul(slots, self.plan_basis)
      plan_energy = plan_energy + plan_drive.pow(2).mean(dim=(1, 2))
      slots = slots + self.plan_coupling * torch.tanh(plan_drive)
      slots = slots + self.mlp(slots)

    plan_energy = plan_energy / float(self.iters)
    return slots, plan_energy

  def device(self) -> torch.device:
    return next(self.parameters()).device
