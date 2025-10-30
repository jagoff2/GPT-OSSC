from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
from transformers import AutoTokenizer
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

from .attention_patch import AttentionPatcher, WorkspaceRuntimeState, workspace_runtime
from .config import WorkspaceConfig
from .controller import ControllerOutput, WorkspaceController
from .kv_projector import VirtualKVProjector
from .logging_utils import init_logger
from .memory import MemoryEntry, WorkspaceMemory
from .probes import LayerProbeBank
from .residual_delta import ResidualDeltaHook
from .types import GenerationRequestContext, HookToggles
from .utils.entropy import batch_entropy_floor
from .utils.quant_utils import load_model_config, load_quantized_model
from .workspace import SlotAttentionWorkspace


@dataclass
class WorkspaceSnapshot:
  slots: torch.Tensor
  controller: ControllerOutput


class GPTOSSHookedModel:
  def __init__(self, config: WorkspaceConfig) -> None:
    self.config = config
    self.logger = init_logger("gpt_oss_ws", config.log_level)
    self.model = load_quantized_model(config)
    self.model_config = load_model_config(config.model_name)
    self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    self.hidden_size = getattr(self.model_config, "hidden_size", None)
    if self.hidden_size is None:
      raise ValueError("Model config missing hidden_size")
    self.num_layers = getattr(self.model_config, "num_hidden_layers", None)
    if self.num_layers is None:
      raise ValueError("Model config missing num_hidden_layers")
    try:
      self.model_dtype = next(self.model.parameters()).dtype
    except StopIteration:
      self.model_dtype = torch.float32
    self._validate_layers()
    self.probes = LayerProbeBank(config, self.hidden_size)
    self.workspace = SlotAttentionWorkspace(config)
    self.kv_projector = VirtualKVProjector(config, self.hidden_size, self.model_dtype)
    self.residual_delta = ResidualDeltaHook(config, self.hidden_size)
    self.controller = WorkspaceController(config)
    self.memory = WorkspaceMemory(config)
    self.layer_patchers: Dict[int, AttentionPatcher] = {}
    self._layer_residuals: Dict[int, torch.Tensor] = {}
    self._current_slots: Optional[torch.Tensor] = None
    self._current_entropy: float = 0.0
    self._apply_patches()
    self._attach_moe_dtype_guards()
    self.logger.info("GPT-OSS workspace model initialized")

  def primary_device(self) -> torch.device:
    try:
      return next(self.model.parameters()).device
    except StopIteration:
      return torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def _validate_layers(self) -> None:
    missing = [layer for layer in self.config.hooked_layers if layer >= self.num_layers]
    if missing:
      raise ValueError(f"Hooked layer indices out of range: {missing}")

  def _apply_patches(self) -> None:
    decoder_layers = self._decoder_layers()
    for layer_idx in self.config.hooked_layers:
      patcher = AttentionPatcher(layer_idx)
      patcher.patch(decoder_layers[layer_idx].self_attn)
      self.layer_patchers[layer_idx] = patcher

  def _decoder_layers(self) -> List[torch.nn.Module]:
    if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
      return list(self.model.model.layers)
    raise ValueError("Unsupported model architecture for GPT-OSS workspace hooks")

  def _attach_moe_dtype_guards(self) -> None:
    for module in self.model.modules():
      if isinstance(module, GptOssExperts):
        if hasattr(module, "_workspace_original_experts_forward"):
          continue

        param_device = module.gate_up_proj.device
        if param_device.type == "cpu" and module.gate_up_proj.dtype != torch.float32:
          module.gate_up_proj = torch.nn.Parameter(module.gate_up_proj.to(dtype=torch.float32))
          module.gate_up_proj_bias = torch.nn.Parameter(module.gate_up_proj_bias.to(dtype=torch.float32))
          module.down_proj = torch.nn.Parameter(module.down_proj.to(dtype=torch.float32))
          module.down_proj_bias = torch.nn.Parameter(module.down_proj_bias.to(dtype=torch.float32))

        original_forward = module.forward

        def patched_forward(
          hidden_states: torch.Tensor,
          *args: torch.Tensor,
          _module: GptOssExperts = module,
          _original_forward: Callable[..., torch.Tensor] = original_forward,
          **kwargs: torch.Tensor,
        ) -> torch.Tensor:
          target_dtype = _module.gate_up_proj.dtype
          if isinstance(hidden_states, torch.Tensor) and hidden_states.is_floating_point() and hidden_states.dtype != target_dtype:
            hidden_states = hidden_states.to(dtype=target_dtype)

          converted_args = []
          for arg in args:
            if isinstance(arg, torch.Tensor) and arg.is_floating_point() and arg.dtype != target_dtype:
              converted_args.append(arg.to(dtype=target_dtype))
            else:
              converted_args.append(arg)

          converted_kwargs = {}
          for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != target_dtype:
              converted_kwargs[key] = value.to(dtype=target_dtype)
            else:
              converted_kwargs[key] = value

          return _original_forward(hidden_states, *converted_args, **converted_kwargs)

        module._workspace_original_experts_forward = original_forward
        module.forward = patched_forward  # type: ignore[assignment]

  def _record_residual(self, layer_idx: int, tensor: torch.Tensor) -> None:
    self._layer_residuals[layer_idx] = tensor

  def _residual_delta(self, layer_idx: int, residual: torch.Tensor) -> torch.Tensor:
    return self.residual_delta.apply(layer_idx, residual, self._current_slots, self._current_entropy, self.config.controller_entropy_floor)

  def _apply_feature_flags(self, toggles: HookToggles) -> HookToggles:
    return HookToggles(
      kv_append=toggles.kv_append and self.config.enable_kv_append,
      residual_delta=toggles.residual_delta and self.config.enable_residual_delta,
      read_probes=toggles.read_probes and self.config.enable_read_probes,
      broadcast=toggles.broadcast and self.config.enable_broadcast,
    )

  def _runtime_state(self, toggles: HookToggles) -> WorkspaceRuntimeState:
    effective = self._apply_feature_flags(toggles)
    return WorkspaceRuntimeState(
      toggles=effective,
      kv_fetch=self._kv_fetch,
      residual_delta=self._residual_delta if effective.residual_delta else None,
      record_residual=self._record_residual if effective.read_probes else None,
      post_attention_hook=None,
      device=str(self.primary_device()),
      slots=self._current_slots,
      entropy=self._current_entropy,
    )

  def _kv_fetch(self, layer_idx: int, device: str) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    return self.kv_projector.fetch(layer_idx, device)

  def _prepare_slots(self, toggles: HookToggles) -> Optional[torch.Tensor]:
    if not toggles.read_probes or not self._layer_residuals:
      return None
    layer_residuals = {layer: tensor for layer, tensor in self._layer_residuals.items() if layer in self.config.hooked_layers}
    if not layer_residuals:
      return None
    features = self.probes(layer_residuals)
    device = self.workspace.slot_mu.device
    features = features.to(device=device, dtype=self.workspace.slot_mu.dtype)
    slots = self.workspace(features)
    self._current_slots = slots
    return slots

  def _controller_step(self, slots: Optional[torch.Tensor], logits: torch.Tensor) -> ControllerOutput:
    controller_dtype = next(self.controller.parameters()).dtype
    if slots is None:
      slots = torch.zeros(
        logits.size(0),
        self.config.slot_count,
        self.config.slot_dim,
        device=logits.device,
        dtype=controller_dtype,
      )
    elif slots.dtype != controller_dtype:
      slots = slots.to(dtype=controller_dtype)
    return self.controller(slots, logits)

  def generate(self, request: GenerationRequestContext, **kwargs) -> torch.Tensor:
    from .generation import generate_with_workspace

    return generate_with_workspace(self, request, **kwargs)

  def tokenizer_encode(self, text: str, **kwargs) -> torch.Tensor:
    return self.tokenizer(text, return_tensors="pt", **kwargs)["input_ids"]

  def tokenizer_decode(self, tokens: torch.Tensor) -> str:
    return self.tokenizer.decode(tokens, skip_special_tokens=True)

  def workspace_step(self, toggles: HookToggles, logits: torch.Tensor) -> ControllerOutput:
    effective = self._apply_feature_flags(toggles)
    self._current_entropy = batch_entropy_floor(logits)
    slots = self._prepare_slots(effective)
    decision = self._controller_step(slots, logits)
    if effective.kv_append and decision.broadcast and slots is not None:
      device = str(self.primary_device())
      for layer_idx in self.config.hooked_layers:
        residual = self._layer_residuals.get(layer_idx)
        target_dtype = residual.dtype if residual is not None else self.model_dtype
        self.kv_projector(slots, layer_idx, device, target_dtype)
    if decision.write_memory and slots is not None:
      snapshot = slots.detach().cpu().reshape(-1).tolist()
      entry = MemoryEntry(
        time=time.time(),
        goal="generation",
        decision="broadcast" if decision.broadcast else "observe",
        outcome="pending",
        ws_snapshot=snapshot,
        tags=["generation"],
        text="",
      )
      self.memory.add(entry)
    self.kv_projector.advance_step()
    self._layer_residuals.clear()
    return decision

  def runtime_context(self, toggles: HookToggles):
    return workspace_runtime(self._runtime_state(toggles))

  def close(self) -> None:
    self.memory.close()
