import torch

from gpt_oss_ws.masks import extend_causal_mask


def test_extend_causal_mask_preserves_shape():
  base = torch.zeros(1, 8, 4, 4)
  virtual = torch.zeros(1, 8, 2, 64)
  extended = extend_causal_mask(base, virtual)
  assert extended.shape[-1] == base.shape[-1] + virtual.shape[-2]
  assert torch.allclose(extended[..., :virtual.shape[-2]], torch.zeros_like(extended[..., :virtual.shape[-2]]))
