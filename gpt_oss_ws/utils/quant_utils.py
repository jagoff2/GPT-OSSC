from __future__ import annotations

import platform
from typing import Any, Dict

from transformers import AutoConfig, AutoModelForCausalLM

from ..config import WorkspaceConfig
from ..quantization import quantize_linear_module

import torch
from torch import nn


def _apply_int8_static_quant(model: nn.Module, target_dtype: torch.dtype) -> None:
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            quantized = quantize_linear_module(module, target_dtype)
            setattr(model, name, quantized)
        else:
            _apply_int8_static_quant(module, target_dtype)


def load_quantized_model(config: WorkspaceConfig) -> AutoModelForCausalLM:
    torch_dtype = None
    load_in_4bit = False
    quantization_config: Dict[str, Any] = {}
    int8_static = False
    force_cpu_device_map = False

    quant_mode = (config.quantization or "").lower()

    if quant_mode == "mxfp4":
        # Let the HF loader manage dtype so packed weights stay in MXFP4 form.
        # MXFP4 kernels expect bf16 activations, so force the workspace path to bf16.
        torch_dtype = None
        config.bf16_fallback = True
    elif quant_mode == "fp32":
        torch_dtype = torch.float32
    elif quant_mode == "bnb-4bit":
        from transformers import BitsAndBytesConfig

        quantization_config = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="bfloat16" if config.bf16_fallback else "float16",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        }
        load_in_4bit = True
    elif quant_mode == "bf16":
        torch_dtype = torch.bfloat16
    elif quant_mode == "int8":
        cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
        windows_runtime = platform.system().lower().startswith("windows")
        if not cuda_available or windows_runtime or config.device_map == "cpu":
            print(
                "Static int8 quantization requires CUDA kernels; falling back to bf16 for this runtime.",
                flush=True,
            )
            torch_dtype = torch.bfloat16 if config.bf16_fallback else torch.float32
            int8_static = False
            config.quantization = "bf16"
            force_cpu_device_map = True
        else:
            torch_dtype = torch.float32
            int8_static = True
    device_map_arg = None
    if config.device_map == "auto":
        if torch.cuda.device_count() > 0:
            device_map_arg = "auto"
    else:
        device_map_arg = config.device_map
    if force_cpu_device_map:
        device_map_arg = "cpu"

    cpu_runtime = (device_map_arg == "cpu") or not torch.cuda.is_available()
    if cpu_runtime:
        if torch_dtype is None or torch_dtype == torch.bfloat16:
            torch_dtype = torch.float32
        config.bf16_fallback = False

    low_cpu = not int8_static

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=device_map_arg,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu,
        **quantization_config,
    )
    if load_in_4bit and hasattr(model, "config"):
        model.config.torch_dtype = None
    if int8_static:
        target_dtype = torch.bfloat16 if config.bf16_fallback else torch.float16
        _apply_int8_static_quant(model, target_dtype)
        model.to(dtype=target_dtype)
    return model


def load_model_config(model_name: str):
    return AutoConfig.from_pretrained(model_name)
