from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class CaptureRecord:
  input_ids: torch.Tensor
  attention_mask: torch.Tensor
  residuals: Dict[int, torch.Tensor]
  logits: torch.Tensor
  plan_energy: Optional[torch.Tensor]
  slots: Optional[torch.Tensor]
  kv_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)
  metadata: Dict[str, Any] = field(default_factory=dict)


class CaptureBuffer:
  def __init__(self, metadata: Optional[Dict[str, Any]] = None) -> None:
    self.metadata = metadata or {}
    self._residuals: Dict[int, torch.Tensor] = {}
    self._logits: Optional[torch.Tensor] = None
    self._plan_energy: Optional[torch.Tensor] = None
    self._slots: Optional[torch.Tensor] = None
    self._input_ids: Optional[torch.Tensor] = None
    self._attention_mask: Optional[torch.Tensor] = None
    self._kv_metrics: Dict[int, Dict[str, float]] = {}

  def record_residual(self, layer_idx: int, tensor: torch.Tensor) -> None:
    self._residuals[layer_idx] = tensor.detach().cpu()

  def record_logits(self, logits: torch.Tensor) -> None:
    self._logits = logits.detach().cpu()

  def record_workspace(self, slots: Optional[torch.Tensor], plan_energy: Optional[torch.Tensor]) -> None:
    self._slots = None if slots is None else slots.detach().cpu()
    self._plan_energy = None if plan_energy is None else plan_energy.detach().cpu()
  def record_kv_metrics(self, metrics: Dict[int, Dict[str, float]]) -> None:
    self._kv_metrics = {layer: dict(values) for layer, values in metrics.items()}

  def record_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
    self._input_ids = input_ids.detach().cpu()
    self._attention_mask = attention_mask.detach().cpu()

  def finalize(self) -> CaptureRecord:
    if self._input_ids is None or self._attention_mask is None:
      raise RuntimeError("CaptureBuffer missing inputs; call record_inputs before finalize.")
    if self._logits is None:
      raise RuntimeError("CaptureBuffer missing logits; ensure record_logits was called.")
    return CaptureRecord(
      input_ids=self._input_ids,
      attention_mask=self._attention_mask,
      residuals=dict(self._residuals),
      logits=self._logits,
      plan_energy=self._plan_energy,
      slots=self._slots,
      metadata=self.metadata,
      kv_metrics=self._kv_metrics,
    )


class CaptureContext:
  def __init__(self, model, buffer: CaptureBuffer) -> None:
    self._model = model
    self._buffer = buffer

  def __enter__(self) -> CaptureBuffer:
    self._model._begin_capture(self._buffer)  # type: ignore[attr-defined]
    return self._buffer

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self._model._end_capture()  # type: ignore[attr-defined]


def capture_workspace(model, metadata: Optional[Dict[str, Any]] = None) -> CaptureContext:
  """
  Context manager that collects residual/logit/slot data during a single generation call.
  """
  return CaptureContext(model, CaptureBuffer(metadata))