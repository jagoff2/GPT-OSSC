from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")

from gpt_oss_ws.config import WorkspaceConfig, load_config
from gpt_oss_ws.model_wrapper import GPTOSSHookedModel
from gpt_oss_ws.types import GenerationRequestContext, HookToggles


def _load_model(config_path: Path | None) -> Tuple[WorkspaceConfig, GPTOSSHookedModel]:
  cfg = load_config(str(config_path)) if config_path else load_config()
  model = GPTOSSHookedModel(cfg)
  return cfg, model


def _set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def _generate_turn(
  model: GPTOSSHookedModel,
  history: List[dict[str, str]],
  toggles: HookToggles,
  max_new_tokens: int = 4096,
  temperature: float = 0.0,
  top_p: float = 1.0,
) -> Tuple[str, torch.Tensor | None]:
  device = model.primary_device()
  input_ids, attention_mask = model.prepare_chat_inputs(history, add_generation_prompt=True)
  ctx = GenerationRequestContext(
    request_id=str(time.time()),
    toggles=toggles,
  )
  output = model.generate(
    ctx,
    input_ids=input_ids.to(device),
    attention_mask=attention_mask.to(device),
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_p=top_p,
  )
  prompt_tokens = input_ids.shape[-1]
  parsed = model.decode_generated(output[0], prompt_tokens)
  text = (parsed.final or "").strip()
  if not text:
    text = model.tokenizer_decode(output[0, prompt_tokens:])
  history.append({"role": "assistant", "content": text})
  plan_energy = getattr(model, "_current_plan_energy", None)
  if isinstance(plan_energy, torch.Tensor):
    plan_energy = plan_energy.detach().cpu()
  return text, plan_energy


def interactive_compare(
  config_path: Path | None,
  prompt_list: List[str] | None = None,
  *,
  seed: int = 1234,
  temperature: float = 0.0,
  top_p: float = 1.0,
) -> None:
  cfg, model = _load_model(config_path)
  _set_seed(seed)
  try:
    torch.use_deterministic_algorithms(True)
  except Exception:
    pass
  if hasattr(torch.backends, "cudnn"):
    try:
      torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
      torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    except Exception:
      pass
  baseline_history: List[dict[str, str]] = []
  workspace_history: List[dict[str, str]] = []
  base_seed = seed
  turn_counter = 0

  def run_turn(user: str) -> None:
    nonlocal turn_counter
    baseline_history.append({"role": "user", "content": user})
    workspace_history.append({"role": "user", "content": user})

    store_state = model.kv_projector.store.state_dict()
    turn_seed = base_seed + turn_counter
    # Baseline run with workspace fully disabled
    _set_seed(turn_seed)
    with model.baseline_mode():
      baseline_response, _ = _generate_turn(
        model,
        baseline_history,
        HookToggles(kv_append=False, residual_delta=False, read_probes=False, broadcast=False),
        temperature=temperature,
        top_p=top_p,
      )
    model.kv_projector.store.load_state_dict(store_state)
    model._current_plan_energy = None  # type: ignore[attr-defined]

    # Workspace-enabled run
    _set_seed(turn_seed)
    workspace_response, plan_energy = _generate_turn(
      model,
      workspace_history,
      HookToggles(kv_append=True, residual_delta=True, read_probes=True, broadcast=True),
      temperature=temperature,
      top_p=top_p,
    )

    print(f"\nBaseline> {baseline_response}")
    print(f"Workspace> {workspace_response}")
    if plan_energy is not None:
      print(f"Plan energy: {plan_energy.tolist()}")
    print("")
    turn_counter += 1

  try:
    if prompt_list:
      for prompt in prompt_list:
        run_turn(prompt)
      return

    print("Interactive comparison started. Type 'exit' to quit, 'reset' to clear history.")
    while True:
      try:
        user = input("You> ").strip()
      except EOFError:
        print("\nInput stream closed; exiting.")
        break
      if not user:
        continue
      lower = user.lower()
      if lower in {"exit", "quit"}:
        break
      if lower == "reset":
        baseline_history.clear()
        workspace_history.clear()
        model.kv_projector.store = model.kv_projector.store.__class__(len(model.kv_projector.layer_ids), cfg.retention)  # type: ignore[arg-type]
        print("History cleared.\n")
        continue

      run_turn(user)
  finally:
    model.close()


def main() -> None:
  parser = argparse.ArgumentParser(description="Interactive comparison of baseline vs workspace-enhanced responses.")
  parser.add_argument("--config", type=Path, default=None, help="Path to config YAML (defaults to configs/server.yaml if present)")
  parser.add_argument("--seed", type=int, default=1234, help="Random seed used for both baseline and workspace runs")
  parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation (0 for greedy)")
  parser.add_argument("--top-p", type=float, default=1.0, dest="top_p", help="Nucleus sampling top-p cutoff (1.0 disables)")
  parser.add_argument("prompts", nargs="*", help="Optional prompts to compare without entering interactive mode")
  args = parser.parse_args()
  prompt_list = args.prompts or None
  interactive_compare(
    args.config,
    prompt_list,
    seed=args.seed,
    temperature=args.temperature,
    top_p=args.top_p,
  )


if __name__ == "__main__":
  main()
