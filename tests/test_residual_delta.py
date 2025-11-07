import torch

from gpt_oss_ws.config import WorkspaceConfig
from gpt_oss_ws.residual_delta import ResidualDeltaHook


def test_residual_delta_updates_last_token():
  cfg = WorkspaceConfig()
  hook = ResidualDeltaHook(cfg, hidden_size=2880)
  residual = torch.zeros(1, 2, 2880)
  slots = torch.randn(1, cfg.slot_count, cfg.slot_dim)
  updated = hook.apply(cfg.hooked_layers[0], residual.clone(), slots, entropy=1.0, entropy_floor=cfg.controller_entropy_floor)
  diff = updated[:, -1, :] - residual[:, -1, :]
  assert torch.any(diff != 0)


def test_residual_delta_gates_low_entropy():
  torch.manual_seed(0)
  cfg = WorkspaceConfig()
  hook = ResidualDeltaHook(cfg, hidden_size=2880)
  residual = torch.zeros(1, 2, 2880)
  slots = torch.randn(1, cfg.slot_count, cfg.slot_dim)
  high_entropy = cfg.controller_entropy_floor + 1.0
  low_entropy = cfg.controller_entropy_floor - 1.0

  high_update = hook.apply(cfg.hooked_layers[0], residual.clone(), slots, entropy=high_entropy, entropy_floor=cfg.controller_entropy_floor)
  low_update = hook.apply(cfg.hooked_layers[0], residual.clone(), slots, entropy=low_entropy, entropy_floor=cfg.controller_entropy_floor)

  high_norm = torch.norm(high_update[:, -1, :])
  low_norm = torch.norm(low_update[:, -1, :])
  assert low_norm < high_norm * 0.25
  assert torch.allclose(low_update[:, :-1, :], residual[:, :-1, :])
