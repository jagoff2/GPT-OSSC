from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

import typer
import uvicorn

from gpt_oss_ws.api_server import create_app
from gpt_oss_ws.config import WorkspaceConfig, load_config
from gpt_oss_ws.logging_utils import init_logger

app = typer.Typer(help="CLI entrypoints for GPT-OSS latent workspace package")


def _resolve_config(path: Optional[Path]) -> Path:
  if path is not None:
    return path
  default = Path("configs/server.yaml")
  if default.exists():
    return default
  raise typer.BadParameter("Config path must be provided if configs/server.yaml does not exist")


@app.command()
def serve(
  config: Optional[Path] = typer.Option(None, "--config", help="Path to server YAML config"),
  host: Optional[str] = typer.Option(None, "--host"),
  port: Optional[int] = typer.Option(None, "--port"),
  workspace_state: Optional[Path] = typer.Option(None, "--workspace-state", help="Path to trained workspace state checkpoint"),
  seed: int = typer.Option(99, "--seed", help="Base seed for deterministic generation"),
  temperature: float = typer.Option(0.0, "--temperature", help="Default sampling temperature"),
  top_p: float = typer.Option(1.0, "--top-p", help="Default nucleus sampling top-p (1.0 disables filtering)"),
  retro_generation: bool = typer.Option(False, "--retro-generation", "--retro", help="Enable retro refinement generation pipeline"),
  retro_margin: Optional[float] = typer.Option(None, "--retro-margin", help="Retro logit margin needed to roll back a chunk"),
  retro_window: Optional[int] = typer.Option(None, "--retro-window", help="Retro lookback window in tokens"),
  retro_max_retracts: Optional[int] = typer.Option(None, "--retro-max-retracts", help="Maximum retro retract attempts per chunk"),
  retro_iters: Optional[int] = typer.Option(None, "--retro-iters", help="Retro bidirectional smoothing passes"),
  retro_damping: Optional[float] = typer.Option(None, "--retro-damping", help="Retro damping factor (0-1)"),
  retro_chunk_size: Optional[int] = typer.Option(None, "--retro-chunk-size", help="Chunk size override for retro refinement"),
  retro_edit_budget: Optional[int] = typer.Option(None, "--retro-edit-budget", help="Maximum retro edits per position"),
  retro_max_tokens: Optional[int] = typer.Option(None, "--retro-max-tokens", help="Cap total new tokens for retro mode"),
  retro_blend: Optional[float] = typer.Option(None, "--retro-blend", help="Blend weight for retro logits vs forward logits (0-1)"),
  retro_diffusion_temperature: Optional[float] = typer.Option(None, "--retro-diffusion-temperature", help="Sampling temperature for diffusion-style retro updates"),
) -> None:
  cfg_path = _resolve_config(config)
  cfg = load_config(str(cfg_path))
  if host:
    cfg.api_host = host
  if port:
    cfg.api_port = port
  if workspace_state:
    cfg.workspace_state_path = str(workspace_state)
  logger = init_logger("cli.serve", cfg.log_level)
  logger.info("Starting API server", extra={"host": cfg.api_host, "port": cfg.api_port})
  overrides = {}
  if workspace_state:
    overrides["workspace_state_path"] = str(workspace_state)
  retro_overrides: Dict[str, Any] = {}
  if retro_generation:
    retro_overrides["enabled"] = True
  if retro_margin is not None:
    retro_overrides["margin"] = retro_margin
  if retro_window is not None:
    retro_overrides["window"] = retro_window
  if retro_max_retracts is not None:
    retro_overrides["max_retracts"] = retro_max_retracts
  if retro_iters is not None:
    retro_overrides["retro_iters"] = retro_iters
  if retro_damping is not None:
    retro_overrides["damping"] = retro_damping
  if retro_chunk_size is not None:
    retro_overrides["chunk_size"] = retro_chunk_size
  if retro_edit_budget is not None:
    retro_overrides["edit_budget"] = retro_edit_budget
  if retro_max_tokens is not None:
    retro_overrides["max_tokens"] = retro_max_tokens
  if retro_blend is not None:
    retro_overrides["diffusion_blend"] = retro_blend
  if retro_diffusion_temperature is not None:
    retro_overrides["diffusion_temperature"] = retro_diffusion_temperature
  if retro_overrides:
    existing_retro = overrides.get("retro", {})
    existing_retro.update(retro_overrides)
    overrides["retro"] = existing_retro
  app_instance = create_app(
    str(cfg_path),
    overrides=overrides or None,
    base_seed=seed,
    default_temperature=temperature,
    default_top_p=top_p,
  )
  uvicorn.run(
    app_instance,
    host=cfg.api_host,
    port=cfg.api_port,
    log_level=cfg.log_level.lower(),
  )


@app.command()
def eval(
  config: Optional[Path] = typer.Option(None, "--config", help="Path to eval config"),
  task: str = typer.Option("fluency", "--task", help="Which eval task to run"),
) -> None:
  cfg_path = _resolve_config(config)
  cfg = load_config(str(cfg_path))
  from evals import report

  try:
    loop = asyncio.get_running_loop()
  except RuntimeError:
    loop = None
  if loop is None:
    asyncio.run(report.run(task, cfg))
  else:
    loop.create_task(report.run(task, cfg))


@app.command(name="fluency-guard")
def fluency_guard(
  baseline_config: Optional[Path] = typer.Option(None, "--baseline", help="Baseline config path"),
  workspace_config: Optional[Path] = typer.Option(None, "--workspace", help="Workspace config path"),
  samples: int = typer.Option(64, "--samples", help="Number of samples to compare"),
) -> None:
  baseline = load_config(str(_resolve_config(baseline_config)))
  workspace = load_config(str(_resolve_config(workspace_config)))
  from evals import fluency_guard as fg

  asyncio.run(fg.compare(baseline, workspace, samples))


if __name__ == "__main__":
  app()
