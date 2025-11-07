import torch

from gpt_oss_ws.config import RetentionConfig
from gpt_oss_ws.scheduling import VirtualKVSegment, VirtualKVStore


def test_virtual_kv_retention_limits_tokens():
  cfg = RetentionConfig(virt_kv_max_tokens_per_layer=4, virt_kv_ttl_steps=10, spill_to_cpu=False)
  store = VirtualKVStore(num_layers=1, cfg=cfg)
  for step in range(3):
    store.step = step
    key = torch.randn(1, 8, 2, 64)
    value = torch.randn(1, 8, 2, 64)
    seg = VirtualKVSegment(key=key, value=value, created_step=step, ttl_steps=cfg.virt_kv_ttl_steps, device="cpu")
    store.append(0, seg)
  fetched = store.fetch(0, "cpu")
  assert fetched.key.shape[-2] <= cfg.virt_kv_max_tokens_per_layer


def test_virtual_kv_retention_ttl():
  cfg = RetentionConfig(virt_kv_max_tokens_per_layer=8, virt_kv_ttl_steps=1, spill_to_cpu=False)
  store = VirtualKVStore(num_layers=1, cfg=cfg)
  key = torch.randn(1, 8, 2, 64)
  value = torch.randn(1, 8, 2, 64)
  store.append(0, VirtualKVSegment(key=key, value=value, created_step=0, ttl_steps=cfg.virt_kv_ttl_steps, device="cpu"))
  store.step = 2
  fetched = store.fetch(0, "cpu")
  assert fetched is None or fetched.key.shape[-2] == 0


def test_virtual_kv_spill_to_cpu():
  cfg = RetentionConfig(virt_kv_max_tokens_per_layer=16, virt_kv_ttl_steps=8, spill_to_cpu=True)
  store = VirtualKVStore(num_layers=1, cfg=cfg)
  key = torch.randn(1, 8, 2, 64)
  value = torch.randn(1, 8, 2, 64)
  segment = VirtualKVSegment(key=key, value=value, created_step=0, ttl_steps=cfg.virt_kv_ttl_steps, device="cuda:0")
  store.append(0, segment)
  store.spill_if_needed(0)
  assert all(seg.device == "cpu" for seg in store.layers[0])
