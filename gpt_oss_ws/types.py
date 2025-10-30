from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import torch

TokenLogits = torch.Tensor
PastKeyValues = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
HookName = Literal["kv_append", "residual_delta", "read_probe"]


@dataclass
class HookToggles:
  kv_append: bool = True
  residual_delta: bool = True
  read_probes: bool = True
  broadcast: bool = True


@dataclass
class GenerationRequestContext:
  request_id: str
  toggles: HookToggles
  retention_overrides: Optional[Dict[str, int]] = None
