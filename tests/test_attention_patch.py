import torch
from torch import nn

from gpt_oss_ws.attention_patch import AttentionPatcher, WorkspaceRuntimeState, workspace_runtime
from gpt_oss_ws.types import HookToggles


class DummyAttention(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.last_past = None

  def forward(self, hidden_states, past_key_value=None, attention_mask=None, use_cache=True):
    self.last_past = past_key_value
    new_hidden = hidden_states + 1.0
    new_key = torch.zeros(1, 8, 1, 64)
    new_value = torch.zeros(1, 8, 1, 64)
    return new_hidden, (new_key, new_value)


def test_attention_patcher_concatenates_virtual_kv():
  attn = DummyAttention()
  patcher = AttentionPatcher(layer_idx=0)
  patcher.patch(attn)
  real_k = torch.ones(1, 8, 2, 64)
  real_v = torch.ones(1, 8, 2, 64)
  virt_k = torch.zeros(1, 8, 1, 64)
  virt_v = torch.zeros(1, 8, 1, 64)
  toggles = HookToggles(True, True, True, True)

  state = WorkspaceRuntimeState(
    toggles=toggles,
    kv_fetch=lambda layer, device: (virt_k, virt_v),
    residual_delta=lambda layer, tensor: tensor + 1.0,
    record_residual=lambda layer, tensor: None,
    post_attention_hook=None,
    device="cpu",
    slots=None,
    entropy=0.0,
  )

  hidden = torch.zeros(1, 2, 64)
  attention_mask = torch.zeros(1, 8, 2, 2)
  with workspace_runtime(state):
    output, present = attn(hidden, past_key_value=(real_k, real_v), attention_mask=attention_mask)
  assert attn.last_past[0].shape[-2] == real_k.shape[-2] + virt_k.shape[-2]
  assert torch.allclose(output, torch.ones_like(output) * 2)
  assert present[0].shape == (1, 8, 1, 64)
