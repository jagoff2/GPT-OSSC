from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

# Ensure Windows consoles handle UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")

from gpt_oss_ws.config import WorkspaceConfig, load_config
from gpt_oss_ws.model_wrapper import GPTOSSHookedModel
from gpt_oss_ws.types import GenerationRequestContext, HookToggles


DEFAULT_PROMPTS = [
  "Hello",
  "Who are you?",
  "What is 2 + 2?",
  "Explain the workspace controller in one sentence.",
]


def _load_config(path: Path | None) -> WorkspaceConfig:
  if path is None:
    path = Path("configs/server.yaml")
  return load_config(str(path))


def run_chat(prompts: List[str], config_path: Path | None) -> None:
  cfg = _load_config(config_path)
  model = GPTOSSHookedModel(cfg)
  history: List[dict[str, str]] = []
  try:
    for prompt in prompts:
      history.append({"role": "user", "content": prompt})
      input_ids, attention_mask = model.prepare_chat_inputs(history, add_generation_prompt=True)
      decoded_prefill = model.tokenizer.decode(input_ids[0], skip_special_tokens=False)
      print("=== Prefill ===")
      print(decoded_prefill)
      ctx = GenerationRequestContext(
        request_id=str(time.time()),
        toggles=HookToggles(True, True, True, True),
      )
      output = model.generate(
        ctx,
        input_ids=input_ids.to(model.primary_device()),
        attention_mask=attention_mask.to(model.primary_device()),
        max_new_tokens=128,
      )
      prompt_tokens = input_ids.shape[-1]
      parsed = model.decode_generated(output[0], prompt_tokens)
      reply = (parsed.final or "").strip()
      if not reply:
        reply = model.tokenizer_decode(output[0, prompt_tokens:])
      print("=== Raw decode ===")
      print(model.tokenizer.decode(output[0], skip_special_tokens=False))
      energy = getattr(model, "_current_plan_energy", None)
      if isinstance(energy, torch.Tensor):
        print("=== Plan energy ===")
        print(energy.detach().cpu().tolist())
      history.append({"role": "assistant", "content": reply})
      print(json.dumps({"user": prompt, "assistant": reply}, ensure_ascii=False))
  finally:
    model.close()


def main() -> None:
  parser = argparse.ArgumentParser(description="Quick harmony chat sanity check.")
  parser.add_argument("--config", type=Path, default=None, help="Config YAML path")
  parser.add_argument("prompts", nargs="*", help="Prompts to send in order")
  args = parser.parse_args()
  prompts = args.prompts or DEFAULT_PROMPTS
  run_chat(prompts, args.config)


if __name__ == "__main__":
  main()
