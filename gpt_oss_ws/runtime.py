from __future__ import annotations

import torch

from .model_wrapper import GPTOSSHookedModel


from typing import Any


def effective_max_new_tokens(model: Any, requested: int) -> int:
  """
  Mirror the CLI behaviour: on pure CPU runtimes clamp workspace generations
  to a smaller token budget to avoid multi-minute stalls.
  """
  device = None
  if hasattr(model, "primary_device"):
    try:
      device = model.primary_device()
    except Exception:
      device = None
  if device is None:
    return requested
  if getattr(device, "type", None) == "cpu":
    return min(requested, 512)
  return requested
