from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from .config import WorkspaceConfig
from .utils.entropy import batch_entropy_floor


@dataclass
class ControllerOutput:
  broadcast: bool
  retrieve: bool
  write_memory: bool
  halt: bool


class WorkspaceController(nn.Module):
  def __init__(self, config: WorkspaceConfig) -> None:
    super().__init__()
    self.config = config
    self.entropy_floor = config.controller_entropy_floor
    self.norm_cap = config.controller_norm_cap
    self.mlp = nn.Sequential(
      nn.LayerNorm(config.slot_dim),
      nn.Linear(config.slot_dim, config.slot_dim // 2),
      nn.GELU(),
      nn.Linear(config.slot_dim // 2, 4)
    )

  def forward(self, slots: torch.Tensor, logits: torch.Tensor) -> ControllerOutput:
    entropy = batch_entropy_floor(logits)
    slot_norm = slots.norm(dim=-1).mean().item()
    heur_broadcast = entropy < self.entropy_floor and slot_norm < self.norm_cap
    mlp_out = torch.sigmoid(self.mlp(slots.mean(dim=1)))
    decisions = mlp_out.mean(dim=0)
    broadcast = heur_broadcast or decisions[0].item() > 0.5
    retrieve = decisions[1].item() > 0.5
    write_memory = decisions[2].item() > 0.5
    halt = decisions[3].item() > 0.8
    return ControllerOutput(broadcast, retrieve, write_memory, halt)
