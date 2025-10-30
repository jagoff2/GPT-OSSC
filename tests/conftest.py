import pytest

from gpt_oss_ws.config import WorkspaceConfig, RetentionConfig


@pytest.fixture()
def small_config() -> WorkspaceConfig:
  cfg = WorkspaceConfig()
  cfg.hooked_layers = [1]
  cfg.nvirt = 2
  cfg.slot_count = 2
  cfg.slot_dim = 64
  cfg.retention = RetentionConfig(virt_kv_max_tokens_per_layer=16, virt_kv_ttl_steps=8, spill_to_cpu=False, prefetch_margin=4)
  return cfg
