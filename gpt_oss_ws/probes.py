from __future__ import annotations

from typing import Dict, Iterable, List

import torch
from torch import nn

from .config import WorkspaceConfig


class ResidualProbe(nn.Module):
  def __init__(self, hidden_size: int, feature_dim: int) -> None:
    super().__init__()
    self.linear = nn.Linear(hidden_size, feature_dim)

  def forward(self, residual: torch.Tensor) -> torch.Tensor:
    """Project residual tensor [B, T, H] -> [B, T, F]."""
    # Ensure dtype compatibility with linear layer weights
    if residual.dtype != self.linear.weight.dtype:
      residual = residual.to(dtype=self.linear.weight.dtype)
    return self.linear(residual)


class LayerProbeBank(nn.Module):
  def __init__(self, config: WorkspaceConfig, hidden_size: int, feature_dim: int = 128) -> None:
    super().__init__()
    self.hooked_layers = config.hooked_layers
    self.probes = nn.ModuleDict({
      str(layer): ResidualProbe(hidden_size, feature_dim) for layer in self.hooked_layers
    })

  def forward(self, layer_residuals: Dict[int, torch.Tensor]) -> torch.Tensor:
    features: List[torch.Tensor] = []
    for layer, residual in layer_residuals.items():
      proj = self.probes[str(layer)](residual[:, -1:, :])
      features.append(proj.squeeze(1))
    return torch.stack(features, dim=1)
