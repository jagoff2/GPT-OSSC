import torch

from gpt_oss_ws.config import WorkspaceConfig
from gpt_oss_ws.kv_projector import VirtualKVProjector


def test_virtual_kv_projector_shapes():
  cfg = WorkspaceConfig()
  projector = VirtualKVProjector(cfg, hidden_size=2880)
  slots = torch.randn(2, cfg.slot_count, cfg.slot_dim)
  layer = cfg.hooked_layers[0]
  segment = projector(slots, layer, "cpu")
  assert segment.key.shape == (2, 8, cfg.nvirt, 64)
  projector.store.advance()
  fetched = projector.fetch(layer, "cpu")
  assert fetched is not None
  key, value = fetched
  assert key.shape[-2] == cfg.nvirt
  assert value.shape == key.shape
