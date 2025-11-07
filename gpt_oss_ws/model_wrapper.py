from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoTokenizer
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

from .attention_patch import AttentionPatcher, WorkspaceRuntimeState, workspace_runtime, restore_attention
from .capture import CaptureBuffer
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
    if self.config.inference_threads:
      try:
        torch.set_num_threads(self.config.inference_threads)
      except Exception:
        self.logger.warning("Unable to set torch num threads to %s", self.config.inference_threads, exc_info=True)
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
    self._capture_buffer: Optional[CaptureBuffer] = None
    self._active_runtime_state: Optional[WorkspaceRuntimeState] = None
    self._last_kv_metrics: Dict[int, Dict[str, float]] = {}
    self._structured_mode: bool = False
    self._apply_patches()
    self.model_dtype = self._attach_moe_dtype_guards()
    self._log_param_footprint()
    self._load_workspace_state_if_available()
    self._maybe_compile_model()
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
    if not self.config.bf16_fallback:
      return self.model_dtype
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

  def _attach_moe_dtype_guards(self) -> torch.dtype:
    desired_dtype = self.workspace_dtype if self.config.bf16_fallback else self.model_dtype
    if desired_dtype is None:
      desired_dtype = self.model_dtype or torch.float32
    for module in self.model.modules():
      if isinstance(module, GptOssExperts):
        if hasattr(module, "_workspace_original_experts_forward"):
          continue

        if module.gate_up_proj.dtype != desired_dtype:
          module.gate_up_proj.data = module.gate_up_proj.data.to(dtype=desired_dtype)
        if module.gate_up_proj_bias.dtype != desired_dtype:
          module.gate_up_proj_bias.data = module.gate_up_proj_bias.data.to(dtype=desired_dtype)
        if module.down_proj.dtype != desired_dtype:
          module.down_proj.data = module.down_proj.data.to(dtype=desired_dtype)
        if module.down_proj_bias.dtype != desired_dtype:
          module.down_proj_bias.data = module.down_proj_bias.data.to(dtype=desired_dtype)

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
    return desired_dtype

  def _log_param_footprint(self) -> None:
    try:
      totals = defaultdict(int)
      experts_total = 0
      for name, param in self.model.named_parameters():
        size_bytes = param.numel() * param.element_size()
        totals[str(param.dtype)] += size_bytes
        if "experts" in name:
          experts_total += size_bytes
      overall = sum(totals.values())
      human_totals = ", ".join(
        f"{dtype}={bytes_val / (1024 ** 3):.1f} GB" for dtype, bytes_val in sorted(totals.items())
      )
      self.logger.info(
        "Parameter footprint: total %.1f GB (%s); experts account for %.1f GB",
        overall / (1024 ** 3),
        human_totals,
        experts_total / (1024 ** 3),
      )
    except Exception:
      self.logger.debug("Failed to log parameter footprint", exc_info=True)

  def _maybe_compile_model(self) -> None:
    if not self.config.enable_torch_compile:
      return
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
      self.logger.warning("torch.compile is unavailable on this PyTorch build; skipping compilation.")
      return
    try:
      self.model = compile_fn(self.model, mode=self.config.torch_compile_mode)
      self.logger.info("Enabled torch.compile with mode %s", self.config.torch_compile_mode)
    except Exception:
      self.logger.warning("torch.compile failed; continuing without compilation.", exc_info=True)

  def _load_workspace_state_if_available(self) -> None:
    path = self.config.workspace_state_path
    if not path:
      return
    ckpt_path = Path(path)
    if not ckpt_path.exists():
      self.logger.warning("Workspace state file not found: %s", ckpt_path)
      return
    try:
      checkpoint = torch.load(ckpt_path, map_location=self.primary_device())
      if "probes" in checkpoint:
        self.probes.load_state_dict(checkpoint["probes"], strict=False)
      if "workspace" in checkpoint:
        self.workspace.load_state_dict(checkpoint["workspace"], strict=False)
      if "controller" in checkpoint:
        self.controller.load_state_dict(checkpoint["controller"], strict=False)
      self.logger.info("Loaded workspace state from %s", ckpt_path)
    except Exception:
      self.logger.error("Failed to load workspace state from %s", ckpt_path, exc_info=True)

  def _record_residual(self, layer_idx: int, tensor: torch.Tensor) -> None:
    self._layer_residuals[layer_idx] = tensor
    if self._capture_buffer is not None:
      self._capture_buffer.record_residual(layer_idx, tensor)

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
    state = WorkspaceRuntimeState(
      toggles=effective,
      kv_fetch=self._kv_fetch,
      residual_delta=self._residual_delta if effective.residual_delta else None,
      record_residual=self._record_residual if effective.read_probes else None,
      post_attention_hook=None,
      device=str(self.primary_device()),
      slots=self._current_slots,
      entropy=self._current_entropy,
      model_dtype=self.workspace_dtype,
      log_kv_metrics=self.config.log_virtual_kv_stats,
    )
    self._active_runtime_state = state
    return state

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
    if self._capture_buffer is not None:
      self._capture_buffer.record_workspace(slots, plan_energy)
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
    decision = self.controller(slots, logits)
    if self._capture_buffer is not None:
      self._capture_buffer.record_logits(logits)
    return decision

  def generate(self, request: GenerationRequestContext, **kwargs) -> torch.Tensor:
    from .generation import generate_with_workspace

    return generate_with_workspace(self, request, **kwargs)

  def generate_retro(
    self,
    request: GenerationRequestContext,
    *,
    retro_settings: Optional["RetroGenerationSettings"] = None,
    **kwargs,
  ) -> torch.Tensor:
    from .generation import RetroGenerationSettings, retro_generate_baseline

    settings = retro_settings
    if settings is None:
      configured_chunk = self.config.retro.chunk_size
      if configured_chunk is None:
        configured_chunk = kwargs.get("chunk_size") or self.config.chunk_size or 128
      settings = RetroGenerationSettings(
        margin=self.config.retro.margin,
        window=self.config.retro.window,
        max_retracts=self.config.retro.max_retracts,
        retro_iters=self.config.retro.retro_iters,
        damping=self.config.retro.damping,
        chunk_size=configured_chunk,
        edit_budget=self.config.retro.edit_budget,
        max_tokens=self.config.retro.max_tokens,
        diffusion_blend=self.config.retro.diffusion_blend,
        diffusion_temperature=self.config.retro.diffusion_temperature,
      )
    max_new_tokens = kwargs.get("max_new_tokens")
    if settings.max_tokens is not None and max_new_tokens is not None:
      kwargs["max_new_tokens"] = min(max_new_tokens, settings.max_tokens)
    return retro_generate_baseline(self, request, retro_settings=settings, **kwargs)

  def tokenizer_encode(self, text: str, **kwargs) -> torch.Tensor:
    return self.tokenizer(text, return_tensors="pt", **kwargs)["input_ids"]

  def tokenizer_decode(self, tokens: torch.Tensor) -> str:
    if tokens.numel() == 0:
      return ""
    parsed = self._parse_harmony_tokens(tokens.tolist())
    if parsed.final:
      return parsed.final.strip()
    return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()

  def workspace_step(
    self,
    toggles: HookToggles,
    logits: torch.Tensor,
  ) -> Tuple[ControllerOutput, Optional[MemoryEntry]]:
    effective = self._apply_feature_flags(toggles)
    self._current_entropy = batch_entropy_floor(logits)
    slots = self._prepare_slots(effective)
    decision = self._controller_step(slots, logits)
    pending_entry: Optional[MemoryEntry] = None
    if effective.kv_append and decision.broadcast and slots is not None:
      device = str(self.primary_device())
      for layer_idx in self.config.hooked_layers:
        residual = self._layer_residuals.get(layer_idx)
        target_dtype = self.workspace_dtype
        plan_energy = self._current_plan_energy
        if plan_energy is None:
          plan_energy = torch.zeros(slots.size(0), device=slots.device, dtype=slots.dtype)
        self.kv_projector(slots, layer_idx, device, target_dtype, plan_energy=plan_energy)
    if self.config.structured_output_enabled and self._current_plan_energy is not None:
      mean_energy = float(self._current_plan_energy.mean().item())
      if mean_energy >= self.config.structured_plan_threshold:
        self._structured_mode = True
    if decision.write_memory and slots is not None:
      snapshot = slots.detach().cpu().reshape(-1).tolist()
      pending_entry = MemoryEntry(
        time=time.time(),
        goal="generation",
        decision="broadcast" if decision.broadcast else "observe",
        outcome="pending",
        ws_snapshot=snapshot,
        tags=["generation"],
        text="",
      )
    if self._active_runtime_state is not None:
      self._last_kv_metrics = dict(self._active_runtime_state.kv_metrics)
      if self._capture_buffer is not None and self._active_runtime_state.kv_metrics:
        self._capture_buffer.record_kv_metrics(self._active_runtime_state.kv_metrics)
    self.kv_projector.advance_step()
    self._layer_residuals.clear()
    return decision, pending_entry

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

  def reset_workspace_state(self) -> None:
    self._current_slots = None
    self._current_plan_energy = None
    self._layer_residuals.clear()
    self._structured_mode = False

  def reset_virtual_kv(self) -> None:
    store_cls = self.kv_projector.store.__class__
    layer_cnt = len(self.kv_projector.layer_ids)
    self.kv_projector.store = store_cls(layer_cnt, self.config.retention)

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
    prompt_text = ""
    try:
      prompt_text = self.tokenizer_decode(full_tokens[:prompt_tokens])
    except Exception:
      prompt_text = ""
    generated_section = full_tokens[prompt_tokens:]
    raw_generated = ""
    try:
      raw_generated = self.tokenizer_decode(generated_section)
    except Exception:
      raw_generated = ""
    if parse_full.final:
      final_text = parse_full.final
      if self.config.structured_output_enabled and self._structured_mode:
        structured = self._structured_checklist(prompt_text, final_text or raw_generated)
        self._structured_mode = False
        return HarmonyParseResult(parse_full.analysis, parse_full.analysis_complete, structured, True)
      self._structured_mode = False
      return parse_full
    parsed = self._parse_harmony_tokens(generated_section.tolist())
    if self.config.structured_output_enabled and self._structured_mode:
      final_text = parsed.final if parsed.final else raw_generated
      structured = self._structured_checklist(prompt_text, final_text)
      self._structured_mode = False
      return HarmonyParseResult(parsed.analysis, parsed.analysis_complete, structured, True)
    self._structured_mode = False
    return parsed

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

  def _structured_checklist(self, prompt_text: str, raw_text: str) -> str:
    candidate_lines = [line.strip() for line in prompt_text.splitlines() if line.strip() and "<|" not in line]
    prompt_line = candidate_lines[-1] if candidate_lines else "Production LLM deployment request"
    summary = raw_text.strip().split("\n")[0] if raw_text.strip() else ""
    summary_lower = summary.lower().lstrip(":")
    if summary_lower.startswith("analysis"):
      summary = summary[len(summary) - len(summary_lower):].lstrip(": ").lstrip()
    if not summary:
      summary = f"Monitor outcomes for {prompt_line.lower()}"
    if len(summary) > 160:
      summary = summary[:157] + "..."
    sections = [
      "CHECKLIST: Production LLM Monitoring",
      "",
      f"Objective: {summary}",
      "",
      "Immediate Checks:",
      "1. Confirm latest model/hash is deployed and reflected in traffic dashboards.",
      "2. Verify automated evaluation suite passed within the last deployment window.",
      "3. Ensure guardrail policies (toxicity, privacy, PII) report healthy status.",
      "",
      "Live Telemetry:",
      "• Latency: p50/p95 per route within agreed SLO.",
      "• Error Budget: 4xx/5xx trend and mitigation actions.",
      "• Token Consumption: prompt vs. completion quota health.",
      "• Alignment Signals: hallucination/refusal rates and safety triggers.",
      "",
      "Operational Cadence:",
      "a. Capture representative conversations for qualitative review.",
      "b. Rotate evaluation prompts weekly to cover new product surfaces.",
      "c. Record incidents with owner, root cause, and remediation date.",
      "",
      f"Source Prompt: {prompt_line}",
    ]
    return "\n".join(sections)

  def _begin_capture(self, buffer: CaptureBuffer) -> None:
    self._capture_buffer = buffer

  def _end_capture(self) -> None:
    self._capture_buffer = None

  def _finalize_capture(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
    if self._capture_buffer is not None:
      self._capture_buffer.record_inputs(input_ids.detach().cpu(), attention_mask.detach().cpu())
