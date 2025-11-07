from __future__ import annotations

import argparse
import json
import random
import uuid
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from gpt_oss_ws.capture import capture_workspace
from gpt_oss_ws.config import WorkspaceConfig, load_config
from gpt_oss_ws.model_wrapper import GPTOSSHookedModel
from gpt_oss_ws.runtime import effective_max_new_tokens
from gpt_oss_ws.types import GenerationRequestContext, HookToggles


def _load_prompts(path: Path) -> List[str]:
  if not path.exists():
    raise FileNotFoundError(f"Prompt file not found: {path}")
  return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def _build_history(prompt: str) -> List[dict[str, str]]:
  system_msg = {
    "role": "system",
    "content": (
      "You are ChatGPT, a helpful assistant. Answer every user question directly. "
      "When a user asks for a calculation, compute it exactly and return the numeric result."
    ),
  }
  return [system_msg, {"role": "user", "content": prompt}]


def capture_dataset(
  prompts: Iterable[str],
  output_dir: Path,
  *,
  config_path: Optional[Path],
  seed: int,
  temperature: float,
  top_p: float,
  max_new_tokens: int,
) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  cfg: WorkspaceConfig = load_config(str(config_path)) if config_path else load_config()
  model = GPTOSSHookedModel(cfg)
  manifest_path = output_dir / "manifest.jsonl"
  with manifest_path.open("w", encoding="utf-8") as manifest:
    for idx, prompt in enumerate(prompts):
      sample_id = f"{idx:05d}"
      turn_seed = seed + idx
      _set_seed(turn_seed)
      history = _build_history(prompt)
      input_ids, attention_mask = model.prepare_chat_inputs(history, add_generation_prompt=True)
      toggles = HookToggles(kv_append=True, residual_delta=True, read_probes=True, broadcast=True)
      request_ctx = GenerationRequestContext(
        request_id=str(uuid.uuid4()),
        toggles=toggles,
      )
      token_cap = effective_max_new_tokens(model, max_new_tokens)
      metadata = {
        "prompt": prompt,
        "seed": turn_seed,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": token_cap,
      }
      with capture_workspace(model, metadata) as buffer:
        model.generate(
          request_ctx,
          input_ids=input_ids,
          attention_mask=attention_mask,
          max_new_tokens=token_cap,
          temperature=temperature,
          top_p=top_p,
        )
        record = buffer.finalize()
      torch.save(record, output_dir / f"{sample_id}.pt")
      manifest.write(
        json.dumps(
          {
            "id": sample_id,
            "file": f"{sample_id}.pt",
            "prompt": prompt,
            "seed": turn_seed,
          }
        )
        + "\n"
      )
  model.close()


def main() -> None:
  parser = argparse.ArgumentParser(description="Capture workspace training samples.")
  parser.add_argument("--config", type=Path, default=None, help="Path to workspace config (optional).")
  parser.add_argument("--prompts", type=Path, required=True, help="Text file with one prompt per line.")
  parser.add_argument("--output", type=Path, required=True, help="Directory to write captured samples.")
  parser.add_argument("--seed", type=int, default=2025, help="Base seed for deterministic captures.")
  parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
  parser.add_argument("--top-p", type=float, default=1.0, dest="top_p", help="Top-p nucleus sampling cut-off.")
  parser.add_argument("--max-new-tokens", type=int, default=512, dest="max_new_tokens", help="Maximum new tokens.")
  args = parser.parse_args()

  prompts = _load_prompts(args.prompts)
  capture_dataset(
    prompts,
    args.output,
    config_path=args.config,
    seed=args.seed,
    temperature=args.temperature,
    top_p=args.top_p,
    max_new_tokens=args.max_new_tokens,
  )


if __name__ == "__main__":
  main()
