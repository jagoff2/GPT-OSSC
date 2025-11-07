from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from gpt_oss_ws.config import WorkspaceConfig, load_config
from gpt_oss_ws.model_wrapper import GPTOSSHookedModel
from gpt_oss_ws.types import GenerationRequestContext, HookToggles


class WorkspaceCaptureDataset(Dataset):
  def __init__(self, manifest_path: Path):
    if not manifest_path.exists():
      raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    self.root = manifest_path.parent
    self.entries: List[Dict[str, str]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
      if line.strip():
        self.entries.append(json.loads(line))

  def __len__(self) -> int:
    return len(self.entries)

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    entry = self.entries[idx]
    record = torch.load(self.root / entry["file"])
    input_ids = record.input_ids.long()
    attention_mask = record.attention_mask.long()
    return input_ids, attention_mask


def freeze_backbone(model: GPTOSSHookedModel) -> None:
  for param in model.model.parameters():
    param.requires_grad_(False)


def move_workspace_modules(model: GPTOSSHookedModel, device: torch.device) -> None:
  model.model.to(device)
  model.probes.to(device)
  model.workspace.to(device)
  model.controller.to(device)
  model.residual_delta.to(device)


def train_workspace(
  manifest_path: Path,
  *,
  config_path: Path | None,
  epochs: int,
  batch_size: int,
  lr: float,
  device: str,
) -> None:
  cfg: WorkspaceConfig = load_config(str(config_path)) if config_path else load_config()
  model = GPTOSSHookedModel(cfg)
  device_obj = torch.device(device)
  move_workspace_modules(model, device_obj)
  freeze_backbone(model)
  # Avoid populating the persistent memory database during training.
  model.memory.add = lambda *_args, **_kwargs: None  # type: ignore[assignment]
  dataset = WorkspaceCaptureDataset(manifest_path)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  params = list(model.probes.parameters()) + list(model.workspace.parameters()) + list(model.controller.parameters())
  optimizer = torch.optim.Adam(params, lr=lr)
  toggles = HookToggles(kv_append=True, residual_delta=True, read_probes=True, broadcast=True)

  for epoch in range(epochs):
    total_loss = 0.0
    total_tokens = 0
    for input_ids, attention_mask in loader:
      input_ids = input_ids.to(device_obj)
      attention_mask = attention_mask.to(device_obj)
      # Teacher forcing loop; process one sample at a time to simplify caching.
      if input_ids.size(0) != 1:
        raise ValueError("Batching >1 is not supported yet due to sequential teacher forcing requirements.")
      optimizer.zero_grad(set_to_none=True)
      model.reset_virtual_kv()
      model.reset_workspace_state()

      full_input = input_ids[:, :1]
      full_mask = attention_mask[:, :1]
      cache = None
      losses: List[torch.Tensor] = []
      target_tokens = input_ids[:, 1:]
      total_steps = target_tokens.shape[1]
      request_ctx = GenerationRequestContext(request_id="training", toggles=toggles)

      if total_steps == 0:
        continue

      for step in range(total_steps):
        step_input = full_input if cache is None else full_input[:, -1:]
        step_mask = full_mask if cache is None else full_mask[:, -1:]
        cache_position = torch.arange(
          full_input.shape[-1] - step_input.shape[-1],
          full_input.shape[-1],
          device=device_obj,
        )
        with model.runtime_context(request_ctx.toggles):
          outputs = model.model(
            input_ids=step_input,
            attention_mask=full_mask,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
          )
        logits = outputs.logits[:, -1, :]
        cache = getattr(outputs, "past_key_values", outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None)
        target = target_tokens[:, step]
        loss = F.cross_entropy(logits, target)
        losses.append(loss)
        # Feed the ground truth token back in (teacher forcing).
        next_token = target.unsqueeze(-1)
        full_input = torch.cat([full_input, next_token], dim=-1)
        full_mask = torch.cat([full_mask, torch.ones_like(next_token)], dim=-1)
        model.workspace_step(request_ctx.toggles, outputs.logits)

      if not losses:
        continue
      batch_loss = torch.stack(losses).mean()
      batch_loss.backward()
      optimizer.step()
      total_loss += batch_loss.item() * total_steps
      total_tokens += total_steps

    avg_loss = total_loss / max(total_tokens, 1)
    print(f"Epoch {epoch+1}/{epochs}: avg token loss {avg_loss:.4f}")

  torch.save(
    {
      "probes": model.probes.state_dict(),
      "workspace": model.workspace.state_dict(),
      "controller": model.controller.state_dict(),
      "config": asdict(cfg),
    },
    manifest_path.parent / "workspace_trained.pt",
  )
  model.close()


def main() -> None:
  parser = argparse.ArgumentParser(description="Train workspace probes/controller on captured data.")
  parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest.jsonl produced by capture script.")
  parser.add_argument("--config", type=Path, default=None, help="Workspace config (optional).")
  parser.add_argument("--epochs", type=int, default=1)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--device", type=str, default="cpu")
  args = parser.parse_args()
  train_workspace(
    args.manifest,
    config_path=args.config,
    epochs=args.epochs,
    batch_size=args.batch_size,
    lr=args.lr,
    device=args.device,
  )


if __name__ == "__main__":
  main()
