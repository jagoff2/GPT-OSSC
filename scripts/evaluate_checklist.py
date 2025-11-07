from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Iterable, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from gpt_oss_ws.config import WorkspaceConfig, load_config
from gpt_oss_ws.model_wrapper import GPTOSSHookedModel
from gpt_oss_ws.runtime import effective_max_new_tokens
from gpt_oss_ws.types import GenerationRequestContext, HookToggles

REQUIRED_SECTIONS = [
  "CHECKLIST: Production LLM Monitoring",
  "Immediate Checks:",
  "Live Telemetry:",
  "Operational Cadence:",
]


def _prepare_history(prompt: str):
  return [
    {
      "role": "system",
      "content": (
        "You are ChatGPT, a helpful assistant. Answer every user question directly. "
        "When a user asks for a calculation, compute it exactly and return the numeric result."
      ),
    },
    {"role": "user", "content": prompt},
  ]


def _generate(model: GPTOSSHookedModel, prompt: str, toggles: HookToggles, max_new_tokens: int) -> str:
  history = _prepare_history(prompt)
  input_ids, attention_mask = model.prepare_chat_inputs(history, add_generation_prompt=True)
  request_ctx = GenerationRequestContext(request_id=str(uuid.uuid4()), toggles=toggles)
  out = model.generate(
    request_ctx,
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=max_new_tokens,
    temperature=0.0,
    top_p=1.0,
  )
  prompt_tokens = input_ids.shape[-1]
  return model.decode_generated(out[0], prompt_tokens).final


def _score(text: str) -> bool:
  if not text:
    return False
  return all(section in text for section in REQUIRED_SECTIONS)


def _run_suite(config_path: Path, prompts: Iterable[str], structured: bool) -> Tuple[int, int, list[str]]:
  cfg = load_config(str(config_path))
  model = GPTOSSHookedModel(cfg)
  results = []
  successes = 0
  total = 0
  max_tokens = effective_max_new_tokens(model, 256)
  toggles = HookToggles(kv_append=True, residual_delta=True, read_probes=True, broadcast=True)
  for prompt in prompts:
    if structured:
      text = _generate(model, prompt, toggles, max_tokens)
    else:
      with model.baseline_mode():
        text = _generate(model, prompt, toggles, max_tokens)
    results.append(text)
    total += 1
    if _score(text):
      successes += 1
  model.close()
  return successes, total, results


def main() -> None:
  os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
  prompts = [line.strip() for line in Path("data/eval_prompts.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
  baseline_hits, baseline_total, baseline_outputs = _run_suite(Path("configs/cpu_small.yaml"), prompts, structured=False)
  workspace_hits, workspace_total, workspace_outputs = _run_suite(Path("configs/cpu_small_trained.yaml"), prompts, structured=True)

  print("PROMPTS TESTED:\n")
  for prompt, base, ws in zip(prompts, baseline_outputs, workspace_outputs):
    print(f"Prompt: {prompt}")
    print(f"Baseline Score: {'PASS' if _score(base) else 'FAIL'}")
    print(f"Workspace Score: {'PASS' if _score(ws) else 'FAIL'}")
    baseline_preview = base[:200].strip().replace("\n", " ")
    print("Baseline Output Snippet:\n", baseline_preview.encode('ascii', errors='ignore').decode('ascii'))
    workspace_preview = ws[:200].strip().replace("\n", " ")
    print("Workspace Output Snippet:\n", workspace_preview.encode('ascii', errors='ignore').decode('ascii'))
    print("-" * 80)

  print("\nSUMMARY:")
  print(f"Baseline checklist pass rate: {baseline_hits}/{baseline_total}")
  print(f"Workspace checklist pass rate: {workspace_hits}/{workspace_total}")


if __name__ == "__main__":
  main()
