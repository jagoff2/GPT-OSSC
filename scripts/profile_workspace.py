from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from gpt_oss_ws.config import WorkspaceConfig, load_config
from gpt_oss_ws.model_wrapper import GPTOSSHookedModel
from gpt_oss_ws.runtime import effective_max_new_tokens
from gpt_oss_ws.types import GenerationRequestContext, HookToggles

try:
  import psutil  # type: ignore
except Exception:  # pragma: no cover
  psutil = None  # type: ignore


def _memory_usage_mb() -> float:
  if psutil is None:
    return 0.0
  process = psutil.Process(os.getpid())
  return process.memory_info().rss / (1024 ** 2)


def profile_prompt(
  model: GPTOSSHookedModel,
  prompt: str,
  max_new_tokens: int,
  temperature: float,
  top_p: float,
) -> None:
  history = [
    {
      "role": "system",
      "content": (
        "You are ChatGPT, a helpful assistant. Answer every user question directly. "
        "When a user asks for a calculation, compute it exactly and return the numeric result."
      ),
    },
    {"role": "user", "content": prompt},
  ]
  input_ids, attention_mask = model.prepare_chat_inputs(history, add_generation_prompt=True)
  toggles = HookToggles(kv_append=True, residual_delta=True, read_probes=True, broadcast=True)
  request_ctx = GenerationRequestContext(request_id="profile", toggles=toggles)
  token_cap = effective_max_new_tokens(model, max_new_tokens)

  start_mem = _memory_usage_mb()
  start_time = time.perf_counter()
  tokens = model.generate(
    request_ctx,
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=token_cap,
    temperature=temperature,
    top_p=top_p,
    chunk_size=model.config.chunk_size,
  )
  duration = time.perf_counter() - start_time
  end_mem = _memory_usage_mb()
  generated = tokens.shape[-1] - input_ids.shape[-1]
  tok_per_sec = generated / duration if duration > 0 else 0.0
  print(f"Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
  print(f"  Time: {duration:.2f}s | tokens: {generated} | tokens/s: {tok_per_sec:.2f}")
  if psutil is not None:
    print(f"  RSS delta: {end_mem - start_mem:+.1f} MB | current RSS: {end_mem:.1f} MB")
  if model._last_kv_metrics:  # type: ignore[attr-defined]
    for layer, metrics in sorted(model._last_kv_metrics.items()):
      ratio = metrics.get("ratio") or 0.0
      print(
        f"  Layer {layer}: virtual_norm={metrics.get('virtual_norm', 0.0):.4f} "
        f"real_norm={metrics.get('real_norm', 0.0):.4f} ratio={ratio:.4f}"
      )
  print("")


def main() -> None:
  parser = argparse.ArgumentParser(description="Profile workspace generation for CPU usage.")
  parser.add_argument("--config", type=Path, default=None, help="Workspace config (defaults to configs/server.yaml).")
  parser.add_argument("--prompt", type=str, default="Summarize the benefits of workspace-augmented inference.")
  parser.add_argument("--max-new-tokens", type=int, default=512, dest="max_new_tokens")
  parser.add_argument("--temperature", type=float, default=0.0)
  parser.add_argument("--top-p", type=float, default=1.0, dest="top_p")
  args = parser.parse_args()

  cfg: WorkspaceConfig = load_config(str(args.config)) if args.config else load_config()
  model = GPTOSSHookedModel(cfg)
  profile_prompt(
    model,
    args.prompt,
    max_new_tokens=args.max_new_tokens,
    temperature=args.temperature,
    top_p=args.top_p,
  )
  model.close()


if __name__ == "__main__":
  main()
