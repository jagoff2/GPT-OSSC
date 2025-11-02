from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import yaml

HookLayer = Literal[1, 5, 9, 13, 17, 21]


@dataclass
class RetentionConfig:
  virt_kv_max_tokens_per_layer: int = 512
  virt_kv_ttl_steps: int = 1024
  spill_to_cpu: bool = True
  prefetch_margin: int = 16


@dataclass
class WorkspaceConfig:
  model_name: str = "openai/gpt-oss-20b"
  quantization: Literal["bnb-4bit", "bf16", "Mxfp4"] = "Mxfp4"
  device_map: Literal["auto", "balanced", "sequential"] = "auto"
  hooked_layers: List[HookLayer] = field(default_factory=lambda: [1, 5, 9, 13, 17, 21])
  nvirt: int = 2
  residual_rank: int = 8
  slot_count: int = 4
  slot_dim: int = 128
  slot_iterations: int = 1
  enable_kv_append: bool = True
  enable_residual_delta: bool = True
  enable_read_probes: bool = True
  enable_broadcast: bool = True
  workspace_device: Literal["cpu", "cuda:0", "cuda:1"] = "cpu"
  retention: RetentionConfig = field(default_factory=RetentionConfig)
  log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
  api_host: str = "0.0.0.0"
  api_port: int = 8000
  controller_entropy_floor: float = 2.5
  controller_norm_cap: float = 4.5
  sqlite_path: str = "workspace_memory.sqlite"
  faiss_index_path: str = "workspace_memory.faiss"
  memory_embedding_dim: int = 384
  max_context_tokens: int = 8192
  bf16_fallback: bool = True


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> WorkspaceConfig:
  """Load YAML config and merge overrides."""
  data: dict = {}
  if path:
    cfg_path = Path(path)
    if not cfg_path.exists():
      raise FileNotFoundError(f"Config file not found: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text()) or {}
  if overrides:
    data.update(overrides)
  retention_kwargs = data.pop("retention", {})
  config = WorkspaceConfig(**data)
  config.retention = RetentionConfig(**{**RetentionConfig().__dict__, **retention_kwargs})
  return config
