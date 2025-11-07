from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import random

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


@dataclass
class RetroGenerationSettings:
  margin: float = 1.0
  window: Optional[int] = 64
  max_retracts: int = 3
  retro_iters: int = 12
  damping: float = 0.5
  chunk_size: int = 128
  edit_budget: Optional[int] = None
  max_tokens: Optional[int] = None
  diffusion_blend: float = 0.5
  diffusion_temperature: float = 0.0


@dataclass
class _ChunkRecord:
  token: torch.Tensor
  logits: torch.Tensor


def _ensure_attention_mask(base_mask: Optional[torch.Tensor], length: int) -> Optional[torch.Tensor]:
  if base_mask is None:
    return torch.ones(1, length, dtype=torch.long)
  base_len = base_mask.shape[-1]
  if length <= base_len:
    return base_mask[:, :length].clone()
  pad = torch.ones(1, length - base_len, dtype=base_mask.dtype)
  return torch.cat([base_mask.clone(), pad], dim=-1)


def _log_causal_chunk(model: GPTOSSHookedModel, chunk_start: int, records: List[_ChunkRecord]) -> None:
  if not records:
    return
  for offset, record in enumerate(records):
    logits = record.logits
    if logits.dim() == 2 and logits.size(0) == 1:
      logits = logits[0]
    top_id = int(torch.argmax(logits).item())
    token_id = int(record.token.view(-1)[0].item())
    top_logit = float(logits[top_id].item())
    model.logger.debug(
      "retro chunk causal step",
      extra={
        "position": chunk_start + offset,
        "token": token_id,
        "argmax": top_id,
      "logit": top_logit,
    },
  )


def _collect_mismatches(
  retro_logits: torch.Tensor,
  tokens: torch.Tensor,
  chunk_start: int,
  editable_floor: int,
  special_ids: Set[int],
  settings: RetroGenerationSettings,
  edit_counts: Dict[int, int],
) -> Tuple[List[Tuple[int, float, int, int]], torch.Tensor]:
  retro_argmax = retro_logits.argmax(dim=-1)
  seq_len = tokens.shape[-1]
  scan_start = max(chunk_start, editable_floor)
  mismatches: List[Tuple[int, float, int, int]] = []
  for absolute_idx in range(scan_start, seq_len):
    emitted_id = int(tokens[0, absolute_idx].item())
    candidate_id = int(retro_argmax[0, absolute_idx].item())
    if emitted_id in special_ids or candidate_id in special_ids:
      continue
    if candidate_id == emitted_id:
      continue
    diff = float(retro_logits[0, absolute_idx, candidate_id].item() - retro_logits[0, absolute_idx, emitted_id].item())
    if diff < settings.margin:
      continue
    if settings.edit_budget is not None and edit_counts.get(absolute_idx, 0) >= settings.edit_budget:
      continue
    mismatches.append((absolute_idx, diff, emitted_id, candidate_id))
  return mismatches, retro_argmax


def _snapshot_rng() -> Dict[str, Any]:
  state: Dict[str, Any] = {
    "python": random.getstate(),
    "numpy": np.random.get_state(),
    "torch": torch.get_rng_state(),
  }
  if torch.cuda.is_available():
    state["cuda"] = torch.cuda.get_rng_state_all()
  return state


def _restore_rng(state: Dict[str, Any]) -> None:
  random.setstate(state["python"])
  np.random.set_state(state["numpy"])
  torch.set_rng_state(state["torch"])
  if "cuda" in state and torch.cuda.is_available():
    torch.cuda.set_rng_state_all(state["cuda"])


def _records_from_logits(tokens: torch.Tensor, logits: torch.Tensor, start: int, end: int) -> List[_ChunkRecord]:
  length = max(0, end - start)
  if length == 0:
    return []
  logits_len = logits.shape[1]
  if logits_len >= end:
    slice_logits = logits[:, start:end, :]
  else:
    relative_len = min(length, logits_len)
    slice_logits = logits[:, :relative_len, :]
    if relative_len < length:
      pad = torch.zeros(
        slice_logits.shape[0],
        length - relative_len,
        slice_logits.shape[2],
        device=slice_logits.device,
        dtype=slice_logits.dtype,
      )
      slice_logits = torch.cat([slice_logits, pad], dim=1)
  slice_logits = slice_logits.detach().cpu()
  records: List[_ChunkRecord] = []
  for offset, absolute_idx in enumerate(range(start, end)):
    token_tensor = tokens[0, absolute_idx].view(1, 1).detach().cpu()
    logit_tensor = slice_logits[:, offset, :]
    records.append(_ChunkRecord(token_tensor, logit_tensor))
  return records


def _scaffold_intact(model: GPTOSSHookedModel, tokens: torch.Tensor, prompt_length: int) -> bool:
  try:
    text = model.tokenizer.decode(tokens[0, prompt_length:].tolist(), skip_special_tokens=False)
  except Exception:
    return False
  return "<|start|>assistant" in text


def _editable_floor(
  model: GPTOSSHookedModel,
  tokens: torch.Tensor,
  chunk_origin: int,
  chunk_len: int,
  base_floor: int,
  special_ids: Set[int],
) -> Tuple[int, List[Tuple[int, int, Optional[str]]]]:
  floor = max(base_floor, chunk_origin)
  barrier = chunk_origin
  limit = chunk_origin + chunk_len
  inspected: List[Tuple[int, int, Optional[str]]] = []
  for pos in range(chunk_origin, limit):
    token_id = int(tokens[0, pos].item())
    try:
      token_str = model.tokenizer.convert_ids_to_tokens(token_id)
    except Exception:
      token_str = None
    inspected.append((pos, token_id, token_str))
    if token_id in special_ids:
      barrier = pos + 1
      continue
    if token_str and token_str.startswith("<|"):
      barrier = pos + 1
      continue
    if token_str and token_str.strip() == "":
      barrier = pos + 1
      continue
    if token_str and token_str.strip().lower() in {"analysis", "final", "assistant", "user", "system", "profile"}:
      barrier = pos + 1
      continue
    break
  return max(floor, barrier), inspected


def _forward_teacher_logits(
  model: GPTOSSHookedModel,
  tokens: torch.Tensor,
  attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
  device = model.primary_device()
  tokens_device = tokens.to(device)
  mask_device = attention_mask.to(device) if attention_mask is not None else None
  with torch.no_grad():
    with model.baseline_mode():
      outputs = model.model(
        input_ids=tokens_device,
        attention_mask=mask_device,
        use_cache=False,
      )
  return outputs.logits.detach().cpu()


def _retro_bidirectional_logits(
  model: GPTOSSHookedModel,
  tokens: torch.Tensor,
  attention_mask: Optional[torch.Tensor],
  settings: RetroGenerationSettings,
  chunk_start: int,
) -> torch.Tensor:
  device = model.primary_device()
  editable_floor = chunk_start if settings.window is None else max(chunk_start - settings.window, 0)
  mask_device = attention_mask.to(device=device) if attention_mask is not None else None
  current_tokens = tokens.to(device=device)
  retro_logits: Optional[torch.Tensor] = None
  passes = max(settings.retro_iters, 1)
  for iteration in range(passes):
    with torch.no_grad():
      with model.baseline_mode():
        forward_out = model.model(
          input_ids=current_tokens,
          attention_mask=mask_device,
          use_cache=False,
        )
        forward_logits = forward_out.logits
        reversed_tokens = torch.flip(current_tokens, dims=(1,))
        reversed_mask = torch.flip(mask_device, dims=(1,)) if mask_device is not None else None
        reverse_out = model.model(
          input_ids=reversed_tokens,
          attention_mask=reversed_mask,
          use_cache=False,
        )
        reverse_logits = torch.flip(reverse_out.logits, dims=(1,))
    blended = (forward_logits + reverse_logits) * 0.5
    if retro_logits is None:
      retro_logits = blended
    else:
      retro_logits = settings.damping * blended + (1.0 - settings.damping) * retro_logits
    if iteration < passes - 1:
      argmax_tokens = retro_logits.argmax(dim=-1)
      if editable_floor > 0:
        left = current_tokens[:, :editable_floor]
        right = argmax_tokens[:, editable_floor:]
        current_tokens = torch.cat([left, right], dim=-1)
      else:
        current_tokens = argmax_tokens
  if retro_logits is None:
    raise RuntimeError("retro logits computation failed")
  return retro_logits.detach().cpu()


def _retro_refine_chunk(
  model: GPTOSSHookedModel,
  base_tokens: torch.Tensor,
  base_mask: Optional[torch.Tensor],
  chunk_start: int,
  generation_start: int,
  settings: RetroGenerationSettings,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool, Optional[torch.Tensor], Set[int]]:
  original_tokens = base_tokens.clone()
  original_mask = base_mask.clone() if base_mask is not None else None
  working_tokens = base_tokens.clone()
  working_mask = base_mask.clone() if base_mask is not None else None
  edit_counts: Dict[int, int] = {}
  changed_positions: Set[int] = set()
  last_logits: Optional[torch.Tensor] = None
  max_passes = max(settings.max_retracts, 1)
  for attempt in range(max_passes):
    retro_logits = _retro_bidirectional_logits(model, working_tokens, working_mask, settings, chunk_start)
    last_logits = retro_logits
    retro_argmax = retro_logits.argmax(dim=-1)
    seq_len = working_tokens.shape[-1]
    scan_start = chunk_start
    if settings.window is not None:
      scan_start = max(generation_start, chunk_start - settings.window)
    mismatches: List[Tuple[int, float, int, int]] = []
    for absolute_idx in range(scan_start, seq_len):
      if absolute_idx < generation_start:
        continue
      emitted_id = int(working_tokens[0, absolute_idx].item())
      candidate_id = int(retro_argmax[0, absolute_idx].item())
      if candidate_id == emitted_id:
        continue
      diff = float(retro_logits[0, absolute_idx, candidate_id].item() - retro_logits[0, absolute_idx, emitted_id].item())
      if diff < settings.margin:
        continue
      if settings.edit_budget is not None and edit_counts.get(absolute_idx, 0) >= settings.edit_budget:
        continue
      mismatches.append((absolute_idx, diff, emitted_id, candidate_id))
    if not mismatches:
      return working_tokens, working_mask, True, last_logits, changed_positions
    model.logger.info(
      "retro mismatch batch",
      extra={
        "attempt": attempt,
        "chunk_start": int(chunk_start),
        "prompt_length": int(generation_start),
        "count": len(mismatches),
        "first": int(mismatches[0][0]),
        "last": int(mismatches[-1][0]),
      },
    )
    updated_tokens = working_tokens.clone()
    for absolute_idx, diff, emitted_id, candidate_id in mismatches:
      updated_tokens[0, absolute_idx] = candidate_id
      edit_counts[absolute_idx] = edit_counts.get(absolute_idx, 0) + 1
      changed_positions.add(absolute_idx)
      model.logger.debug(
        "retro apply token",
        extra={
          "attempt": attempt,
          "position": int(absolute_idx),
          "emitted_id": emitted_id,
          "retro_id": candidate_id,
          "margin": diff,
        },
      )
    working_tokens = updated_tokens
  model.logger.warning(
    "retro chunk unresolved after retries",
    extra={"chunk_start": int(chunk_start), "attempts": max_passes, "changed": len(changed_positions)},
  )
  return original_tokens, original_mask, False, last_logits, set()


def generate_with_baseline_model(
  model: GPTOSSHookedModel,
  request: GenerationRequestContext,
  input_ids: torch.Tensor,
  attention_mask: Optional[torch.Tensor] = None,
  max_new_tokens: int = 4096,
  temperature: float = 0.8,
  top_p: float = 0.95,
  eos_token_id: Optional[int] = None,
  stream_callback: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
  chunk_size: Optional[int] = None,
) -> torch.Tensor:
  device = model.primary_device()
  full_input = input_ids.to(device)
  if attention_mask is None:
    attention_mask = torch.ones_like(full_input, device=device)
  else:
    attention_mask = attention_mask.to(device)
  eos_token_id = eos_token_id or model.tokenizer.eos_token_id
  cache: Optional[Cache] = None
  ones_cache: Optional[torch.Tensor] = None
  prompt_length = full_input.shape[-1]
  with model.baseline_mode():
    for step in range(max_new_tokens):
      step_input = full_input if cache is None else full_input[:, -1:]
      total_length = full_input.shape[-1]
      step_length = step_input.shape[-1]
      position_start = total_length - step_length
      cache_position = torch.arange(position_start, total_length, device=device)
      autocast_context = nullcontext()
      if device.type in {"cpu", "cuda"}:
        try:
          autocast_context = torch.autocast(device.type, dtype=model.model_dtype)
        except Exception:
          autocast_context = nullcontext()
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
      next_token = _sample_next_token(logits, temperature, top_p)
      next_token_unsqueezed = next_token.unsqueeze(-1)
      full_input = torch.cat([full_input, next_token_unsqueezed], dim=-1)
      if ones_cache is None or ones_cache.dtype != attention_mask.dtype or ones_cache.device != attention_mask.device:
        ones_cache = torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)
      broadcast_ones = ones_cache.expand_as(next_token_unsqueezed)
      attention_mask = torch.cat([attention_mask, broadcast_ones], dim=-1)
      if stream_callback is not None:
        stream_callback(next_token_unsqueezed.detach().cpu(), logits.detach().cpu())
      if eos_token_id is not None and (next_token == eos_token_id).all():
        break
      if chunk_size and chunk_size > 0 and ((step + 1) % chunk_size == 0):
        cache = None
  return full_input.detach().cpu()


def retro_generate_baseline(
  model: GPTOSSHookedModel,
  request: GenerationRequestContext,
  input_ids: torch.Tensor,
  attention_mask: Optional[torch.Tensor] = None,
  max_new_tokens: int = 4096,
  temperature: float = 0.8,
  top_p: float = 0.95,
  eos_token_id: Optional[int] = None,
  stream_callback: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
  chunk_size: Optional[int] = None,
  retro_settings: Optional[RetroGenerationSettings] = None,
) -> torch.Tensor:
  settings = retro_settings or RetroGenerationSettings()
  if settings.max_retracts is None or settings.max_retracts <= 0:
    settings.max_retracts = 3
  if settings.max_tokens is not None:
    max_new_tokens = min(max_new_tokens, settings.max_tokens)
  chunk_limit = settings.chunk_size or chunk_size or 128
  if chunk_limit <= 0:
    raise ValueError("Retro chunk size must be positive")
  special_ids: Set[int] = set(getattr(model.tokenizer, "all_special_ids", []) or [])
  base_mask = attention_mask.clone() if attention_mask is not None else None
  current_tokens = input_ids.clone()
  prompt_length = input_ids.shape[-1]
  eos_token_id = eos_token_id or model.tokenizer.eos_token_id
  total_generated = max(0, current_tokens.shape[-1] - prompt_length)
  retro_enabled = True
  while total_generated < max_new_tokens:
    remaining = max_new_tokens - total_generated
    chunk_budget = min(chunk_limit, remaining)
    if chunk_budget <= 0:
      break
    chunk_origin = current_tokens.shape[-1]
    allow_retro = retro_enabled and (chunk_origin - prompt_length) >= 16
    current_mask = _ensure_attention_mask(base_mask, chunk_origin)
    baseline_records: List[_ChunkRecord] = []

    def _capture(token: torch.Tensor, logits: torch.Tensor) -> None:
      baseline_records.append(_ChunkRecord(token.detach().cpu(), logits.detach().cpu()))

    baseline_chunk = generate_with_baseline_model(
      model,
      request,
      current_tokens,
      attention_mask=current_mask,
      max_new_tokens=chunk_budget,
      temperature=temperature,
      top_p=top_p,
      eos_token_id=eos_token_id,
      stream_callback=_capture,
      chunk_size=chunk_size,
    )
    chunk_len = baseline_chunk.shape[-1] - chunk_origin
    _log_causal_chunk(model, chunk_origin, baseline_records)
    if chunk_len <= 0:
      current_tokens = baseline_chunk
      total_generated = current_tokens.shape[-1] - prompt_length
      break
    mask_after_chunk = _ensure_attention_mask(base_mask, baseline_chunk.shape[-1])

    final_tokens = baseline_chunk
    final_mask = mask_after_chunk
    final_records = baseline_records
    retro_success = False

    if allow_retro:
      editable_floor = chunk_origin
      if settings.window is not None:
        editable_floor = max(prompt_length, chunk_origin - settings.window)
      editable_floor, inspected_tokens = _editable_floor(
        model,
        baseline_chunk,
        chunk_origin,
        chunk_len,
        editable_floor,
        special_ids,
      )
      preview_tokens = inspected_tokens[: min(8, len(inspected_tokens))]
      model.logger.debug(
        "retro editable floor chunk_start=%d chunk_len=%d floor=%d inspected=%s",
        int(chunk_origin),
        int(chunk_len),
        int(editable_floor),
        preview_tokens,
      )
      candidate_tokens = baseline_chunk.clone()
      candidate_mask = mask_after_chunk.clone() if mask_after_chunk is not None else None
      chunk_rng_state = _snapshot_rng()
      passes = 0
      blend = max(0.0, min(1.0, settings.diffusion_blend))
      while passes < settings.max_retracts:
        _restore_rng(chunk_rng_state)
        forward_logits = _forward_teacher_logits(model, candidate_tokens, candidate_mask)
        retro_logits = _retro_bidirectional_logits(model, candidate_tokens, candidate_mask, settings, chunk_origin)
        updated_tokens = candidate_tokens.clone()
        changed = False
        modified: List[Tuple[int, int, int, float]] = []
        for position in range(editable_floor, candidate_tokens.shape[-1]):
          current_id = int(candidate_tokens[0, position].item())
          if current_id in special_ids:
            continue
          forward_index = position - 1
          if forward_index < 0:
            continue
          forward_vec = forward_logits[:, forward_index, :]
          retro_vec = retro_logits[:, position, :]
          mixed = (1.0 - blend) * forward_vec + blend * retro_vec
          temp = settings.diffusion_temperature
          if temp is not None and temp > 0.0:
            probs = torch.softmax(mixed / temp, dim=-1)
            new_id = int(torch.multinomial(probs, num_samples=1).item())
          else:
            new_id = int(torch.argmax(mixed, dim=-1).item())
          if new_id in special_ids:
            continue
          improvement = float(mixed[0, new_id].item() - mixed[0, current_id].item())
          if improvement < settings.margin:
            continue
          if new_id != current_id:
            updated_tokens[0, position] = new_id
            changed = True
            modified.append((position, current_id, new_id, float(mixed.max().item())))
        if changed:
          model.logger.debug(
            "retro diffusion pass=%d modified=%s",
            passes,
            modified[:10],
          )
        if not changed:
          final_tokens = updated_tokens
          candidate_mask = _ensure_attention_mask(base_mask, final_tokens.shape[-1])
          final_mask = candidate_mask
          final_forward_logits = _forward_teacher_logits(model, final_tokens, final_mask)
          if final_forward_logits.shape[1] < chunk_origin + chunk_len:
            model.logger.warning(
              "retro forward logits truncated; reverting to baseline",
              extra={
                "chunk_start": int(chunk_origin),
                "available": int(final_forward_logits.shape[1]),
                "required": int(chunk_origin + chunk_len),
              },
            )
            retro_success = False
            break
          combined_logits = final_forward_logits[:, -chunk_len:, :]
          if combined_logits.shape[1] == 0:
            model.logger.warning(
              "retro combined logits empty after teacher forcing; reverting to baseline",
              extra={
                "chunk_start": int(chunk_origin),
                "chunk_len": int(chunk_len),
              },
            )
            retro_success = False
            break
          final_records = _records_from_logits(final_tokens, combined_logits, chunk_origin, chunk_origin + chunk_len)
          retro_success = True
          break
        candidate_tokens = updated_tokens
        candidate_mask = _ensure_attention_mask(base_mask, candidate_tokens.shape[-1])
        passes += 1
      if not retro_success:
        model.logger.warning(
          "retro chunk unresolved after diffusion passes; adopting last candidate",
          extra={"chunk_start": int(chunk_origin), "attempts": passes},
        )
        final_tokens = candidate_tokens
        candidate_mask = _ensure_attention_mask(base_mask, final_tokens.shape[-1])
        final_mask = candidate_mask
        final_forward_logits = _forward_teacher_logits(model, final_tokens, final_mask)
        combined_logits = final_forward_logits[:, -chunk_len:, :]
        final_records = _records_from_logits(final_tokens, combined_logits, final_tokens.shape[-1] - chunk_len, final_tokens.shape[-1])
    if allow_retro and retro_success:
      try:
        parsed = model.decode_generated(final_tokens[0], prompt_length)
        text_valid = bool((parsed.final and parsed.final.strip()) or (parsed.analysis and parsed.analysis.strip()))
      except Exception:
        text_valid = False
      if not text_valid:
        model.logger.warning("retro parse check failed; reverting to baseline and disabling retro")
        final_tokens = baseline_chunk
        final_mask = mask_after_chunk
        final_records = baseline_records
        retro_success = False
        retro_enabled = False

    if stream_callback is not None:
      for record in final_records:
        stream_callback(record.token, record.logits)

    current_tokens = final_tokens
    base_mask = final_mask
    total_generated = current_tokens.shape[-1] - prompt_length

    if eos_token_id is not None and current_tokens[0, -1].item() == eos_token_id:
      break
    if chunk_len < chunk_budget:
      break
    if total_generated >= max_new_tokens:
      break
  return current_tokens


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
  chunk_size: Optional[int] = None,
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
  ones_cache: Optional[torch.Tensor] = None
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
    if ones_cache is None or ones_cache.dtype != attention_mask.dtype or ones_cache.device != attention_mask.device:
      ones_cache = torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)
    broadcast_ones = ones_cache.expand_as(next_token_unsqueezed)
    attention_mask = torch.cat([attention_mask, broadcast_ones], dim=-1)
    if pending_entry is not None:
      generated_slice = full_input[:, prompt_length:]
      if generated_slice.numel() > 0:
        token_view = generated_slice[0].detach().cpu()
        decoded = model.tokenizer.decode(token_view.tolist(), skip_special_tokens=True)
        if not decoded:
          decoded = model.tokenizer_decode(token_view)
        pending_entry.text = decoded
        model.memory.add(pending_entry)
    if chunk_size and chunk_size > 0 and (len(generated_tokens) % chunk_size == 0):
      cache = None
      model.reset_workspace_state()
      model.reset_virtual_kv()
    if stream_callback:
      stream_callback(next_token_unsqueezed, logits)
    if eos_token_id is not None and (next_token == eos_token_id).all():
      break
    if decision.halt:
      break
  if getattr(model, "_capture_buffer", None) is not None:
    model._finalize_capture(full_input, attention_mask)
  if generated_tokens:
    generated = torch.stack(generated_tokens, dim=1)
    return torch.cat([input_ids.cpu(), generated.cpu()], dim=1)
  return input_ids.cpu()
