from __future__ import annotations

from typing import Dict

import torch


def build_balanced_device_map(num_layers: int, num_gpus: int) -> Dict[str, int]:
  if num_gpus < 1:
    raise ValueError("At least one GPU required")
  layers_per_gpu = (num_layers + num_gpus - 1) // num_gpus
  device_map: Dict[str, int] = {}
  for idx in range(num_layers):
    device_map[f"model.layers.{idx}"] = min(idx // layers_per_gpu, num_gpus - 1)
  device_map["model.embed_tokens"] = 0
  device_map["model.norm"] = num_gpus - 1
  device_map["lm_head"] = num_gpus - 1
  return device_map


def auto_device_map(num_layers: int) -> Dict[str, int]:
  if torch.cuda.device_count() >= 2:
    return build_balanced_device_map(num_layers, 2)
  return {"": 0}
