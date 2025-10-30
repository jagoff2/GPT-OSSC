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
    self.iters = config.slot_iterations
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

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    bsz = inputs.size(0)
    mu = self.slot_mu.expand(bsz, self.slot_count, -1)
    sigma = torch.exp(self.slot_log_sigma).expand_as(mu)
    slots = mu + sigma * torch.randn_like(mu)

    k = self.project_k(inputs)
    v = self.project_v(inputs)

    for _ in range(max(1, self.iters)):
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
      slots = slots + self.mlp(slots)
    return slots

  def device(self) -> torch.device:
    return next(self.parameters()).device
