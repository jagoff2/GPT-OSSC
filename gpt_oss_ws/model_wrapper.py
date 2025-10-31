from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoTokenizer
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

from .attention_patch import AttentionPatcher, WorkspaceRuntimeState, workspace_runtime, restore_attention
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


@dataclass
class HarmonyParseResult:
  analysis: str
  analysis_complete: bool
  final: str
  final_complete: bool


class GPTOSSHookedModel:
  def __init__(self, config: WorkspaceConfig) -> None:
    self.config = config
    self.logger = init_logger("gpt_oss_ws", config.log_level)
    self.model = load_quantized_model(config)
    self.model_config = load_model_config(config.model_name)
    self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
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
    self.workspace_dtype = self._select_workspace_dtype()
    self._validate_layers()
    self.probes = LayerProbeBank(config, self.hidden_size)
    self.workspace = SlotAttentionWorkspace(config)
    self.kv_projector = VirtualKVProjector(
      config,
      self.hidden_size,
      self.model_dtype,
      workspace_dtype=self.workspace_dtype,
    )
    self.residual_delta = ResidualDeltaHook(config, self.hidden_size)
    self.controller = WorkspaceController(config)
    self.memory = WorkspaceMemory(config)
    self.layer_patchers: Dict[int, AttentionPatcher] = {}
    self._layer_residuals: Dict[int, torch.Tensor] = {}
    self._current_slots: Optional[torch.Tensor] = None
    self._current_plan_energy: Optional[torch.Tensor] = None
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

  def _select_workspace_dtype(self) -> torch.dtype:
    preferred = torch.bfloat16
    device = None
    try:
      device = self.primary_device()
      torch.zeros(1, device=device, dtype=preferred)
    except Exception:
      self.logger.warning(
        "Workspace dtype bfloat16 unsupported on device %s; falling back to %s.",
        device if device is not None else "unknown",
        self.model_dtype,
      )
      return self.model_dtype
    return preferred

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
    plan_energy = self._current_plan_energy
    if plan_energy is None or not isinstance(plan_energy, torch.Tensor):
      plan_energy = torch.zeros(residual.size(0), device=residual.device, dtype=residual.dtype)
    else:
      plan_energy = plan_energy.to(device=residual.device, dtype=residual.dtype)
    return self.residual_delta.apply(
      layer_idx,
      residual,
      self._current_slots,
      self._current_entropy,
      self.config.controller_entropy_floor,
      plan_energy,
    )

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
      model_dtype=self.workspace_dtype,
    )

  def _kv_fetch(self, layer_idx: int, device: str) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    return self.kv_projector.fetch(layer_idx, device)

  def _prepare_slots(self, toggles: HookToggles) -> Optional[torch.Tensor]:
    if not toggles.read_probes or not self._layer_residuals:
      self._current_plan_energy = None
      return None
    layer_residuals = {layer: tensor for layer, tensor in self._layer_residuals.items() if layer in self.config.hooked_layers}
    if not layer_residuals:
      self._current_plan_energy = None
      return None
    features = self.probes(layer_residuals)
    device = self.workspace.slot_mu.device
    features = features.to(device=device, dtype=self.workspace.slot_mu.dtype)
    slots, plan_energy = self.workspace(features)
    self._current_slots = slots
    self._current_plan_energy = plan_energy.to(device=slots.device, dtype=slots.dtype)
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
    if tokens.numel() == 0:
      return ""
    parsed = self._parse_harmony_tokens(tokens.tolist())
    if parsed.final:
      return parsed.final.strip()
    return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()

  def workspace_step(self, toggles: HookToggles, logits: torch.Tensor) -> ControllerOutput:
    effective = self._apply_feature_flags(toggles)
    self._current_entropy = batch_entropy_floor(logits)
    slots = self._prepare_slots(effective)
    decision = self._controller_step(slots, logits)
    if effective.kv_append and decision.broadcast and slots is not None:
      device = str(self.primary_device())
      for layer_idx in self.config.hooked_layers:
        residual = self._layer_residuals.get(layer_idx)
        target_dtype = self.workspace_dtype
        plan_energy = self._current_plan_energy
        if plan_energy is None:
          plan_energy = torch.zeros(slots.size(0), device=slots.device, dtype=slots.dtype)
        self.kv_projector(slots, layer_idx, device, target_dtype, plan_energy=plan_energy)
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
    if not (toggles.kv_append or toggles.residual_delta or toggles.read_probes or toggles.broadcast):
      return contextlib.nullcontext()
    return workspace_runtime(self._runtime_state(toggles))

  @contextlib.contextmanager
  def baseline_mode(self):
    decoder_layers = self._decoder_layers()
    restored = []
    for layer_idx in self.config.hooked_layers:
      module = decoder_layers[layer_idx].self_attn
      if hasattr(module, "_workspace_original_forward"):
        restore_attention(module)
        restored.append((layer_idx, module))
    try:
      yield
    finally:
      for layer_idx, module in restored:
        patcher = self.layer_patchers[layer_idx]
        patcher.patch(module)

  def close(self) -> None:
    self.memory.close()

  def prepare_chat_inputs(
    self,
    messages: Sequence[Dict[str, str]],
    tools: Optional[Sequence[Dict[str, object]]] = None,
    add_generation_prompt: bool = True,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    chat = self.tokenizer.apply_chat_template(
      messages,
      tools=tools,
      tokenize=True,
      add_generation_prompt=add_generation_prompt,
      return_tensors="pt",
    )
    if hasattr(chat, "keys"):
      input_ids = chat["input_ids"]
      attention_mask = chat.get("attention_mask")
      if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
      return input_ids, attention_mask
    if isinstance(chat, torch.Tensor):
      token_tensor = chat
    else:
      token_tensor = torch.tensor(chat, dtype=torch.long)
    if token_tensor.dim() == 1:
      token_tensor = token_tensor.unsqueeze(0)
    attention_mask = torch.ones_like(token_tensor, dtype=torch.long)
    return token_tensor, attention_mask

  def decode_generated(self, full_tokens: torch.Tensor, prompt_tokens: int) -> HarmonyParseResult:
    parse_full = self._parse_harmony_tokens(full_tokens.tolist())
    if parse_full.final:
      return parse_full
    generated = full_tokens[prompt_tokens:]
    return self._parse_harmony_tokens(generated.tolist())

  def _parse_harmony_tokens(self, token_ids: Sequence[int]) -> HarmonyParseResult:
    if not token_ids:
      return HarmonyParseResult("", False, "", False)
    text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
    analysis, analysis_complete = self._extract_channel_text(text, "analysis")
    final, final_complete = self._extract_channel_text(text, "final")
    if analysis:
      analysis = self._clean_channel_text(analysis)
    if final:
      final = self._clean_channel_text(final)
    return HarmonyParseResult(analysis, analysis_complete, final, final_complete)

  @staticmethod
  def _extract_channel_text(text: str, channel: str) -> Tuple[str, bool]:
    marker = f"<|start|>assistant<|channel|>{channel}<|message|>"
    end_token = "<|end|>"
    start = text.rfind(marker)
    if start == -1:
      return "", False
    start += len(marker)
    end = text.find(end_token, start)
    if end == -1:
      return text[start:], False
    return text[start:end], True

  @staticmethod
  def _clean_channel_text(text: str) -> str:
    cleaned = text.replace("<|return|>", "")
    return cleaned.strip()
