from __future__ import annotations

import torch

from .model_wrapper import GPTOSSHookedModel


def effective_max_new_tokens(model: GPTOSSHookedModel, requested: int) -> int:
  """
  Mirror the CLI behaviour: on pure CPU runtimes clamp workspace generations
  to a smaller token budget to avoid multi-minute stalls.
  """
  device = model.primary_device()
  if device.type == "cpu":
    return min(requested, 4096)
  return requested
