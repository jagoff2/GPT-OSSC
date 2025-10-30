from __future__ import annotations

import dataclasses
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Optional

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

  def advance(self) -> None:
    self.step += 1

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

  def state_dict(self) -> Dict[int, Iterable[VirtualKVSegment]]:
    return {layer: list(segments) for layer, segments in self.layers.items()}

  def load_state_dict(self, state_dict) -> None:
    for layer, items in state_dict.items():
      self.layers[layer] = deque(items)
