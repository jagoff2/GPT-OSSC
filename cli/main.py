from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

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
def serve(config: Optional[Path] = typer.Option(None, "--config", help="Path to server YAML config"), host: Optional[str] = typer.Option(None, "--host"), port: Optional[int] = typer.Option(None, "--port")) -> None:
  cfg_path = _resolve_config(config)
  cfg = load_config(str(cfg_path))
  if host:
    cfg.api_host = host
  if port:
    cfg.api_port = port
  logger = init_logger("cli.serve", cfg.log_level)
  logger.info("Starting API server", extra={"host": cfg.api_host, "port": cfg.api_port})
  uvicorn.run(
    create_app(str(cfg_path)),
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
