from __future__ import annotations

import contextlib
import contextvars
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import torch

try:
  from transformers.utils import ModelOutput as HFModelOutput  # type: ignore[attr-defined]
except Exception:
  HFModelOutput = None  # type: ignore[assignment]

from transformers.cache_utils import Cache

from .masks import extend_causal_mask
from .types import HookToggles

RuntimeVar = contextvars.ContextVar("workspace_runtime_state", default=None)


@dataclass
class LayerRuntime:
  residual: Optional[torch.Tensor] = None
  attn_output: Optional[torch.Tensor] = None


@dataclass
class WorkspaceRuntimeState:
  toggles: HookToggles
  layer_map: Dict[int, LayerRuntime] = field(default_factory=dict)
  kv_fetch: Callable[[int, str], Optional[Tuple[torch.Tensor, torch.Tensor]]] = lambda *_: None
  residual_delta: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None
  record_residual: Optional[Callable[[int, torch.Tensor], None]] = None
  post_attention_hook: Optional[Callable[[int, torch.Tensor], None]] = None
  device: str = "cpu"
  slots: Optional[torch.Tensor] = None
  entropy: float = 0.0
  model_dtype: Optional[torch.dtype] = None


def _append_virtual_to_past(
  past: Optional[Tuple[torch.Tensor, torch.Tensor]], virtual_k: torch.Tensor, virtual_v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
  if past is None:
    return virtual_k, virtual_v
  real_k, real_v = past
  if real_k is None or real_v is None:
    return virtual_k, virtual_v
  if real_k.shape[0] != virtual_k.shape[0]:
    raise ValueError(f"Batch mismatch for virtual KV: {real_k.shape} vs {virtual_k.shape}")
  target_key_dtype = real_k.dtype
  target_value_dtype = real_v.dtype
  # No conversion to float32
  if not hasattr(_append_virtual_to_past, "_debug_printed"):
    print("workspace virtual KV dtype alignment (before cast): "
      f"real_k={real_k.dtype}, real_v={real_v.dtype}, "
      f"virtual_k={virtual_k.dtype}, virtual_v={virtual_v.dtype}, "
      f"target_key={target_key_dtype}, target_value={target_value_dtype}", flush=True)
    setattr(_append_virtual_to_past, "_debug_printed", True)
  if virtual_k.dtype != target_key_dtype:
    virtual_k = virtual_k.to(dtype=target_key_dtype)
  if real_k.dtype != target_key_dtype:
    real_k = real_k.to(dtype=target_key_dtype)
  if virtual_v.dtype != target_value_dtype:
    virtual_v = virtual_v.to(dtype=target_value_dtype)
  if real_v.dtype != target_value_dtype:
    real_v = real_v.to(dtype=target_value_dtype)
  key = torch.cat([virtual_k, real_k], dim=-2)
  value = torch.cat([virtual_v, real_v], dim=-2)
  if not hasattr(_append_virtual_to_past, "_debug_printed_post"):
    print("workspace virtual KV dtype alignment (after cast): "
      f"key={key.dtype}, value={value.dtype}", flush=True)
    setattr(_append_virtual_to_past, "_debug_printed_post", True)
  return key, value


def _extend_mask(attention_mask: Optional[torch.Tensor], virtual_k: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
  if attention_mask is None or virtual_k is None:
    return attention_mask
  if attention_mask.dim() == 4:
    return extend_causal_mask(attention_mask, virtual_k, device=attention_mask.device)
  return attention_mask


class _VirtualCacheProxy:
  """
  Delegates to the underlying Hugging Face cache while injecting virtual KV
  slices for the hooked layer before returning to the attention kernel.
  """

  def __init__(self, base: Optional[Cache], layer_idx: int, virtual_k: Optional[torch.Tensor], virtual_v: Optional[torch.Tensor]) -> None:
    self._base = base
    self._layer_idx = layer_idx
    self._virtual_k = virtual_k
    self._virtual_v = virtual_v

  def update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: Optional[Dict[str, Any]] = None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    if self._base is not None:
      key_states, value_states = self._base.update(key_states, value_states, layer_idx, cache_kwargs)
    if layer_idx == self._layer_idx and self._virtual_k is not None and self._virtual_v is not None:
      virtual_k = self._virtual_k
      virtual_v = self._virtual_v
      target_dtype = key_states.dtype
      if not getattr(_VirtualCacheProxy, "_logged_dtype_once", False):
        print(
          f"[kv-proxy] key_states dtype={key_states.dtype} value_states dtype={value_states.dtype} virtual_k dtype={virtual_k.dtype} virtual_v dtype={virtual_v.dtype}",
          flush=True,
        )
        _VirtualCacheProxy._logged_dtype_once = True
      if virtual_k.dtype != target_dtype:
        virtual_k = virtual_k.to(dtype=target_dtype)
      if virtual_v.dtype != target_dtype:
        virtual_v = virtual_v.to(dtype=target_dtype)
      key_states, value_states = _append_virtual_to_past((key_states, value_states), virtual_k, virtual_v)
    return key_states, value_states

  def __getattr__(self, name: str):
    if name in {"_base", "_layer_idx", "_virtual_k", "_virtual_v"}:
      return super().__getattribute__(name)
    if self._base is None:
      raise AttributeError(name)
    return getattr(self._base, name)


class AttentionPatcher:
  def __init__(self, layer_idx: int) -> None:
    self.layer_idx = layer_idx

  def patch(self, module: torch.nn.Module) -> None:
    if hasattr(module, "_workspace_original_forward"):
      return

    if not hasattr(module, "_workspace_dtype_guard_handle"):
      def _dtype_guard(mod: torch.nn.Module, args, kwargs):
        hidden_states = None
        use_kwargs = False
        if kwargs and "hidden_states" in kwargs:
          hidden_states = kwargs["hidden_states"]
          use_kwargs = True
        elif args:
          hidden_states = args[0]
        if not isinstance(hidden_states, torch.Tensor) or not hidden_states.is_floating_point():
          return args, kwargs
        target = getattr(mod, "q_proj", None)
        target_dtype = getattr(target, "weight", None).dtype if target is not None and hasattr(target, "weight") else None
        if target_dtype is None or hidden_states.dtype == target_dtype:
          return args, kwargs
        cast_hidden = hidden_states.to(dtype=target_dtype)
        if use_kwargs:
          kwargs = dict(kwargs)
          kwargs["hidden_states"] = cast_hidden
        else:
          args = (cast_hidden,) + args[1:]
        return args, kwargs

      guard_handle = module.register_forward_pre_hook(_dtype_guard, with_kwargs=True)
      module._workspace_dtype_guard_handle = guard_handle

    original_forward = module.forward
    signature = inspect.signature(original_forward)
    accepts_attention_mask = "attention_mask" in signature.parameters
    accepts_past_values = "past_key_values" in signature.parameters
    accepts_past_value = "past_key_value" in signature.parameters

    def _cast_to_dtype(obj, dtype):
      if dtype is None:
        return obj
      if torch.is_tensor(obj) and obj.is_floating_point() and obj.dtype != dtype:
        return obj.to(dtype=dtype)
      if HFModelOutput is not None and isinstance(obj, HFModelOutput):
        return obj.to(dtype=dtype)
      if isinstance(obj, tuple) and obj:
        head = _cast_to_dtype(obj[0], dtype)
        return (head,) + obj[1:]
      return obj

    def patched_forward(*args, **kwargs):
      runtime: WorkspaceRuntimeState = RuntimeVar.get()
      runtime_dtype = getattr(runtime, "model_dtype", None) if runtime is not None else None
      target_attn_dtype = None
      if hasattr(module, "q_proj") and hasattr(module.q_proj, "weight"):
        target_attn_dtype = module.q_proj.weight.dtype
      if target_attn_dtype is not None:
        if args and isinstance(args[0], torch.Tensor) and args[0].is_floating_point() and args[0].dtype != target_attn_dtype:
          first = args[0].to(dtype=target_attn_dtype)
          args = (first, *args[1:])
        if "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
          hidden_states_kw = kwargs["hidden_states"]
          if hidden_states_kw.is_floating_point() and hidden_states_kw.dtype != target_attn_dtype:
            kwargs["hidden_states"] = hidden_states_kw.to(dtype=target_attn_dtype)
      if not getattr(patched_forward, "_logged_input_once", False):
        first_arg = args[0] if args else kwargs.get("hidden_states")
        dtype = first_arg.dtype if isinstance(first_arg, torch.Tensor) else None
        print(f"[attention-patch] layer={self.layer_idx} input_dtype={dtype}", flush=True)
        patched_forward._logged_input_once = True

      if runtime is None:
        return original_forward(*args, **kwargs)

      bound = signature.bind_partial(*args, **kwargs)
      toggles = runtime.toggles
      past_arg_name: Optional[str] = None
      past: Optional[Any] = None
      if accepts_past_values and "past_key_values" in bound.arguments:
        past_arg_name = "past_key_values"
        past = bound.arguments["past_key_values"]
      if past is None and accepts_past_value and "past_key_value" in bound.arguments:
        past_arg_name = "past_key_value"
        past = bound.arguments["past_key_value"]
      if past_arg_name is None:
        if accepts_past_values:
          past_arg_name = "past_key_values"
        elif accepts_past_value:
          past_arg_name = "past_key_value"

      virtual_k = None
      virtual_v = None
      if toggles.kv_append and runtime.kv_fetch:
        device = args[0].device if args else runtime.device
        fetched = runtime.kv_fetch(self.layer_idx, str(device))
        if fetched is not None:
          virtual_k, virtual_v = fetched
          if not getattr(AttentionPatcher, "_logged_dtype_once", False):
            hidden_states_arg = args[0] if args else bound.arguments.get("hidden_states")
            hs_dtype = hidden_states_arg.dtype if isinstance(hidden_states_arg, torch.Tensor) else None
            past_dtype = None
            if isinstance(past, tuple) and past and isinstance(past[0], torch.Tensor):
              past_dtype = past[0].dtype
            print(f"[workspace-debug] layer={self.layer_idx} hidden_states={hs_dtype} past={past_dtype} virtual={virtual_k.dtype}", flush=True)
            AttentionPatcher._logged_dtype_once = True
          expected_dtype = runtime_dtype
          if expected_dtype is None and args and isinstance(args[0], torch.Tensor):
            expected_dtype = args[0].dtype
          if expected_dtype is None:
            hidden_states_arg = bound.arguments.get("hidden_states")
            if isinstance(hidden_states_arg, torch.Tensor):
              expected_dtype = hidden_states_arg.dtype
          if expected_dtype is not None:
            if virtual_k.dtype != expected_dtype:
              virtual_k = virtual_k.to(dtype=expected_dtype)
            if virtual_v.dtype != expected_dtype:
              virtual_v = virtual_v.to(dtype=expected_dtype)
          target_key = past_arg_name or ("past_key_values" if accepts_past_values else "past_key_value")
          if target_key not in signature.parameters and accepts_past_value:
            target_key = "past_key_value"
          if target_key in signature.parameters:
            if isinstance(past, Cache) or past is None:
              base_cache = past if isinstance(past, Cache) else None
              bound.arguments[target_key] = _VirtualCacheProxy(base_cache, self.layer_idx, virtual_k, virtual_v)
            else:
              combined = _append_virtual_to_past(past, virtual_k, virtual_v)
              bound.arguments[target_key] = combined
              past = combined
            if target_key == "past_key_values":
              bound.arguments.pop("past_key_value", None)
            elif target_key == "past_key_value":
              bound.arguments.pop("past_key_values", None)

      if toggles.kv_append and virtual_k is not None and accepts_attention_mask:
        attention_mask = bound.arguments.get("attention_mask")
        bound.arguments["attention_mask"] = _extend_mask(attention_mask, virtual_k)

      outputs = original_forward(*bound.args, **bound.kwargs)
      model_output = HFModelOutput is not None and isinstance(outputs, HFModelOutput)
      if not model_output and not isinstance(outputs, tuple):
        if target_attn_dtype is not None and isinstance(outputs, torch.Tensor) and outputs.is_floating_point() and outputs.dtype != target_attn_dtype:
          return outputs.to(dtype=target_attn_dtype)
        return outputs
      hidden_states = outputs[0]
      if target_attn_dtype is not None and isinstance(hidden_states, torch.Tensor) and hidden_states.is_floating_point() and hidden_states.dtype != target_attn_dtype:
        hidden_states = hidden_states.to(dtype=target_attn_dtype)
      if runtime.record_residual:
        runtime.record_residual(self.layer_idx, hidden_states.detach())
      final_hidden = hidden_states
      if target_attn_dtype is not None and isinstance(final_hidden, torch.Tensor) and final_hidden.is_floating_point() and final_hidden.dtype != target_attn_dtype:
        final_hidden = final_hidden.to(dtype=target_attn_dtype)
      if toggles.residual_delta and runtime.residual_delta:
        final_hidden = runtime.residual_delta(self.layer_idx, hidden_states)
      if isinstance(final_hidden, torch.Tensor) and isinstance(hidden_states, torch.Tensor) and final_hidden.dtype != hidden_states.dtype:
        final_hidden = final_hidden.to(dtype=hidden_states.dtype)
      if model_output:
        if final_hidden is not hidden_states:
          data = dict(outputs)
          if hasattr(outputs, "last_hidden_state"):
            data["last_hidden_state"] = final_hidden
          else:
            first_key = next(iter(outputs.keys()), None)
            if first_key is not None:
              data[first_key] = final_hidden
          outputs = outputs.__class__(**data)
      else:
        if final_hidden is not hidden_states:
          outputs = (final_hidden,) + outputs[1:]
        elif target_attn_dtype is not None and isinstance(outputs[0], torch.Tensor) and outputs[0].is_floating_point() and outputs[0].dtype != target_attn_dtype:
          outputs = (outputs[0].to(dtype=target_attn_dtype),) + outputs[1:]
      if runtime.post_attention_hook:
        runtime.post_attention_hook(self.layer_idx, final_hidden.detach())
      if not getattr(patched_forward, "_logged_dtype_once", False):
        print(
          f"[attention-patch] layer={self.layer_idx} hidden_out={final_hidden.dtype} orig_hidden={hidden_states.dtype}",
          flush=True,
        )
        patched_forward._logged_dtype_once = True
      return outputs

    module._workspace_original_forward = original_forward
    module.forward = patched_forward  # type: ignore[assignment]


@contextlib.contextmanager
def workspace_runtime(state: WorkspaceRuntimeState):
  token = RuntimeVar.set(state)
  try:
    yield
  finally:
    RuntimeVar.reset(token)


def restore_attention(module: torch.nn.Module) -> None:
  if hasattr(module, "_workspace_original_forward"):
    module.forward = module._workspace_original_forward
    delattr(module, "_workspace_original_forward")
