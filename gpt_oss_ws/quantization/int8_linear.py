from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class QuantLinear(nn.Module):
  def __init__(self, linear: nn.Linear, target_dtype: torch.dtype = torch.bfloat16) -> None:
    super().__init__()
    weight = linear.weight.detach().cpu().to(torch.float32)
    bias = linear.bias.detach().cpu().to(torch.float32) if linear.bias is not None else None

    max_vals = weight.abs().amax(dim=1)
    max_vals.clamp_(min=1e-8)
    scale = max_vals / 127.0
    qweight = torch.round(weight / scale.unsqueeze(1)).clamp_(-127, 127).to(torch.int8)

    self.in_features = linear.in_features
    self.out_features = linear.out_features
    self.register_buffer("qweight", qweight, persistent=False)
    self.register_buffer("scale", scale, persistent=False)
    if bias is not None:
      self.register_buffer("bias", bias, persistent=False)
    else:
      self.bias = None
    self.weight_dtype = target_dtype

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    compute_dtype = self.weight_dtype
    if input.dtype != compute_dtype:
      input = input.to(dtype=compute_dtype)
    dequant = (self.qweight.float() * self.scale.unsqueeze(1)).to(compute_dtype)
    bias = self.bias.to(compute_dtype) if isinstance(self.bias, torch.Tensor) else None
    return F.linear(input, dequant, bias)

  @property
  def weight(self) -> torch.Tensor:
    return (self.qweight.float() * self.scale.unsqueeze(1)).to(self.weight_dtype)

  def to(self, *args, **kwargs):  # type: ignore[override]
    super().to(*args, **kwargs)
    return self


def quantize_linear_module(linear: nn.Linear, target_dtype: torch.dtype = torch.bfloat16) -> QuantLinear:
  return QuantLinear(linear, target_dtype)
