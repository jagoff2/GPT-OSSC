from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from contextlib import nullcontext

from .model_wrapper import GPTOSSHookedModel
from .types import GenerationRequestContext


def _sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
  device = logits.device
  float_logits = logits.to(torch.float32)
  float_logits = float_logits / max(temperature, 1e-5)
  probs = F.softmax(float_logits, dim=-1)
  if top_p < 1.0:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative > top_p
    mask[..., 0] = False
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    next_tokens = torch.multinomial(sorted_probs, num_samples=1)
    next_token = torch.gather(sorted_indices, -1, next_tokens)
  else:
    next_token = torch.multinomial(probs, num_samples=1)
  next_token = next_token.to(device=device)
  return next_token.squeeze(-1)


def generate_with_workspace(
  model: GPTOSSHookedModel,
  request: GenerationRequestContext,
  input_ids: torch.Tensor,
  attention_mask: Optional[torch.Tensor] = None,
  max_new_tokens: int = 4096,
  temperature: float = 0.8,
  top_p: float = 0.95,
  eos_token_id: Optional[int] = None,
  stream_callback: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
) -> torch.Tensor:
  device = model.primary_device()
  input_ids = input_ids.to(device)
  if attention_mask is None:
    attention_mask = torch.ones_like(input_ids, device=device)
  else:
    attention_mask = attention_mask.to(device)
  eos_token_id = eos_token_id or model.tokenizer.eos_token_id
  cache: Optional[Cache] = None
  generated_tokens = []
  full_input = input_ids
  prompt_length = input_ids.shape[-1]
  for step in range(max_new_tokens):
    step_input = full_input if cache is None else full_input[:, -1:]
    total_length = full_input.shape[-1]
    step_length = step_input.shape[-1]
    position_start = total_length - step_length
    cache_position = torch.arange(position_start, total_length, device=device)
    if cache is not None:
        if isinstance(cache, tuple):
            target_cache_dtype = getattr(model, "model_dtype", None)
            # Ensure legacy tuple cache matches model compute dtype (e.g. bfloat16)
            cache = tuple(
                tuple(
                    t.to(dtype=target_cache_dtype)
                    if isinstance(t, torch.Tensor) and target_cache_dtype is not None and t.dtype != target_cache_dtype
                    else t
                    for t in layer
                )
                for layer in cache
            )
        # Convert to DynamicCache if not already
        if not isinstance(cache, Cache):
            cache = DynamicCache.from_legacy_cache(cache)
    
    autocast_context = nullcontext()
    if device.type in {"cpu", "cuda"}:
      try:
        autocast_context = torch.autocast(device.type, dtype=model.model_dtype)
      except Exception:
        autocast_context = nullcontext()
    with model.runtime_context(request.toggles):
      with autocast_context:
        outputs = model.model(
          input_ids=step_input,
          attention_mask=attention_mask,
          past_key_values=cache,
          cache_position=cache_position,
          use_cache=True,
        )
    logits = outputs.logits[:, -1, :]
    new_cache = getattr(outputs, "past_key_values", outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None)
    if isinstance(new_cache, tuple):
      new_cache = DynamicCache.from_legacy_cache(new_cache)
    cache = new_cache
    decision, pending_entry = model.workspace_step(request.toggles, outputs.logits)
    next_token = _sample_next_token(logits, temperature, top_p)
    generated_tokens.append(next_token)
    next_token_unsqueezed = next_token.unsqueeze(-1)
    full_input = torch.cat([full_input, next_token_unsqueezed], dim=-1)
    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_unsqueezed)], dim=-1)
    if pending_entry is not None:
      generated_slice = full_input[:, prompt_length:]
      if generated_slice.numel() > 0:
        token_view = generated_slice[0].detach().cpu()
        decoded = model.tokenizer.decode(token_view.tolist(), skip_special_tokens=True)
        if not decoded:
          decoded = model.tokenizer_decode(token_view)
        pending_entry.text = decoded
        model.memory.add(pending_entry)
    if stream_callback:
      stream_callback(next_token_unsqueezed, logits)
    if eos_token_id is not None and (next_token == eos_token_id).all():
      break
    if decision.halt:
      break
  if generated_tokens:
    generated = torch.stack(generated_tokens, dim=1)
    return torch.cat([input_ids.cpu(), generated.cpu()], dim=1)
  return input_ids.cpu()
