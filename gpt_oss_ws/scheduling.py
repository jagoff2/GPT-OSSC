from __future__ import annotations

import dataclasses
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, Optional

import torch

from .config import RetentionConfig


@dataclass
class VirtualKVSegment:
  key: torch.Tensor
  value: torch.Tensor
  created_step: int
  ttl_steps: int
  device: str

  @property
  def length(self) -> int:
    return self.key.shape[-2]

  def to(self, device: str) -> "VirtualKVSegment":
    if self.device == device:
      return self
    self.key = self.key.to(device=device, non_blocking=True)
    self.value = self.value.to(device=device, non_blocking=True)
    self.device = device
    return self


class VirtualKVStore:
  def __init__(self, num_layers: int, cfg: RetentionConfig) -> None:
    self.cfg = cfg
    self.layers: Dict[int, Deque[VirtualKVSegment]] = {i: deque() for i in range(num_layers)}
    self.step: int = 0
    ttl = max(cfg.virt_kv_ttl_steps, 1)
    self.decay = float(torch.exp(torch.tensor(-1.0 / ttl)))

  def advance(self) -> None:
    self.step += 1
    decay = self.decay
    for queue in self.layers.values():
      for segment in queue:
        segment.key.mul_(decay)
        segment.value.mul_(decay)

  def append(self, layer: int, segment: VirtualKVSegment) -> None:
    queue = self.layers[layer]
    queue.append(segment)
    self._enforce_limits(layer)

  def spill_if_needed(self, layer: int) -> None:
    if not self.cfg.spill_to_cpu:
      return
    for segment in self.layers[layer]:
      if segment.device.startswith("cuda"):
        segment.to("cpu")

  def fetch(self, layer: int, device: str) -> Optional[VirtualKVSegment]:
    self._enforce_limits(layer)
    segments = self.layers[layer]
    if not segments:
      return None
    concat_k = torch.cat([seg.key.to(device=device, non_blocking=True) for seg in segments], dim=-2)
    concat_v = torch.cat([seg.value.to(device=device, non_blocking=True) for seg in segments], dim=-2)
    return VirtualKVSegment(concat_k, concat_v, self.step, self.cfg.virt_kv_ttl_steps, device)

  def _enforce_limits(self, layer: int) -> None:
    queue = self.layers[layer]
    max_tokens = self.cfg.virt_kv_max_tokens_per_layer
    ttl = self.cfg.virt_kv_ttl_steps
    total = sum(segment.length for segment in queue)
    while queue and (total > max_tokens or self.step - queue[0].created_step > ttl):
      queue.popleft()
      total = sum(segment.length for segment in queue)

  def _clone_segment(self, segment: VirtualKVSegment) -> VirtualKVSegment:
    return VirtualKVSegment(
      key=segment.key.detach().clone(),
      value=segment.value.detach().clone(),
      created_step=segment.created_step,
      ttl_steps=segment.ttl_steps,
      device=segment.device,
    )

  def state_dict(self) -> Dict[str, Any]:
    payload_layers: Dict[int, Iterable[Dict[str, Any]]] = {}
    for layer, segments in self.layers.items():
      cloned_segments = [
        {
          "key": segment.key.detach().clone(),
          "value": segment.value.detach().clone(),
          "created_step": segment.created_step,
          "ttl_steps": segment.ttl_steps,
          "device": segment.device,
        }
        for segment in segments
      ]
      payload_layers[layer] = cloned_segments
    return {"step": self.step, "layers": payload_layers}

  def load_state_dict(self, state_dict) -> None:
    if not isinstance(state_dict, dict):
      raise TypeError("VirtualKVStore.load_state_dict expects a dict payload")

    raw_layers = state_dict.get("layers")
    if raw_layers is None:
      # backward compatibility with older format {str(layer): [segments], "step": step}
      raw_layers = {
        int(layer): value for layer, value in state_dict.items() if layer != "step"
      }
    else:
      raw_layers = {int(layer): value for layer, value in raw_layers.items()}

    self.step = int(state_dict.get("step", self.step))
    for layer in self.layers:
      self.layers[layer].clear()
      layer_segments = raw_layers.get(layer, [])
      new_queue: Deque[VirtualKVSegment] = deque()
      for segment in layer_segments:
        if isinstance(segment, VirtualKVSegment):
          cloned = self._clone_segment(segment)
        else:
          key_tensor = segment["key"].detach().clone()
          value_tensor = segment["value"].detach().clone()
          device = segment.get("device", str(key_tensor.device))
          cloned = VirtualKVSegment(
            key=key_tensor,
            value=value_tensor,
            created_step=segment["created_step"],
            ttl_steps=segment["ttl_steps"],
            device=device,
          )
        new_queue.append(cloned)
      self.layers[layer] = new_queue
