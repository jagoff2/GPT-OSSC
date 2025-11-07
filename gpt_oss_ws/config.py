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
class RetroStrategyConfig:
  enabled: bool = False
  margin: float = 1.0
  window: Optional[int] = 64
  max_retracts: int = 3
  retro_iters: int = 12
  damping: float = 0.5
  chunk_size: Optional[int] = None
  edit_budget: Optional[int] = None
  max_tokens: Optional[int] = None
  diffusion_blend: float = 0.5
  diffusion_temperature: float = 0.0


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
  workspace_state_path: Optional[str] = None
  log_virtual_kv_stats: bool = False
  kv_plan_scale: float = 1.0
  kv_plan_bias: float = 0.0
  kv_projection_scale: float = 1.0
  enable_torch_compile: bool = False
  torch_compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead"
  inference_threads: Optional[int] = None
  chunk_size: Optional[int] = None
  structured_output_enabled: bool = False
  structured_plan_threshold: float = 0.25
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
  bf16_fallback: bool = False
  retro: RetroStrategyConfig = field(default_factory=RetroStrategyConfig)


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
  retro_kwargs = data.pop("retro", {})
  config = WorkspaceConfig(**data)
  config.retention = RetentionConfig(**{**RetentionConfig().__dict__, **retention_kwargs})
  config.retro = RetroStrategyConfig(**{**RetroStrategyConfig().__dict__, **retro_kwargs})
  return config
