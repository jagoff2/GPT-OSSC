import torch

from gpt_oss_ws.config import WorkspaceConfig
from gpt_oss_ws.controller import WorkspaceController


def test_controller_outputs_flags():
  cfg = WorkspaceConfig()
  controller = WorkspaceController(cfg)
  slots = torch.zeros(1, cfg.slot_count, cfg.slot_dim)
  logits = torch.randn(1, 1, 16)
  decision = controller(slots, logits)
  assert hasattr(decision, "broadcast")
  assert hasattr(decision, "retrieve")
  assert hasattr(decision, "write_memory")
  assert hasattr(decision, "halt")


def test_controller_entropy_heuristic_triggers_broadcast():
  cfg = WorkspaceConfig()
  controller = WorkspaceController(cfg)
  # Make MLP outputs sub-threshold so heuristic drives broadcast.
  for module in controller.mlp:
    if hasattr(module, "reset_parameters"):
      torch.nn.init.constant_(module.weight, 0.0)  # type: ignore[arg-type]
      if hasattr(module, "bias"):
        torch.nn.init.constant_(module.bias, -10.0)  # type: ignore[arg-type]
  slots = torch.zeros(1, cfg.slot_count, cfg.slot_dim)
  logits = torch.full((1, 1, 16), -10.0)
  logits[..., -1] = 10.0  # sharply peaked distribution -> low entropy
  decision = controller(slots, logits)
  assert decision.broadcast is True
  assert decision.halt is False
