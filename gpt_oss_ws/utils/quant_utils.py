from __future__ import annotations

from typing import Any, Dict

from transformers import AutoConfig, AutoModelForCausalLM

from ..config import WorkspaceConfig

import torch


def load_quantized_model(config: WorkspaceConfig) -> AutoModelForCausalLM:
    torch_dtype = None
    load_in_4bit = False
    quantization_config: Dict[str, Any] = {}
    if config.quantization == "Mxfp4":
        torch_dtype = torch.bfloat16
        # MXFP4 quantization is not supported directly; we fallback to bfloat16
    elif config.quantization == "fp32":
        torch_dtype = torch.float32
    elif config.quantization == "bnb-4bit":
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
    elif config.quantization == "bf16":
        torch_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=config.device_map if config.device_map != "auto" else None,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        **quantization_config,
    )
    if load_in_4bit and hasattr(model, "config"):
        model.config.torch_dtype = None
    return model


def load_model_config(model_name: str):
    return AutoConfig.from_pretrained(model_name)
