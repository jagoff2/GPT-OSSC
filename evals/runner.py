from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

DEFAULT_TIMEOUT_SECONDS = 300


def _quote_for_powershell(value: str) -> str:
  escaped = value.replace("'", "''")
  return "'" + escaped + "'"


def build_eval_command(repo_root: Path, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> List[str]:
  repo = repo_root.resolve()
  quoted_root = _quote_for_powershell(str(repo))
  script = (
    "$job = Start-Job -ScriptBlock { "
    f"Set-Location -Path {quoted_root}; "
    "py -m cli.main eval "
    "}; "
    "try { "
    f"if (-not (Wait-Job -Job $job -Timeout {timeout_seconds})) {{ "
    "Stop-Job -Job $job -Force; "
    "throw 'eval timed out' "
    "}}; "
    "Receive-Job -Job $job "
    "} finally { "
    "Remove-Job -Job $job -Force -ErrorAction SilentlyContinue "
    "}"
  )
  return ["powershell.exe", "-NoLogo", "-Command", script]


def run_eval(timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS, repo_root: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess[str]:
  root = repo_root.resolve() if repo_root else Path(__file__).resolve().parents[1]
  command = build_eval_command(root, timeout_seconds)
  completed = subprocess.run(
    command,
    cwd=str(root),
    capture_output=True,
    text=True,
    check=False,
  )
  if check and completed.returncode != 0:
    raise subprocess.CalledProcessError(
      completed.returncode,
      command,
      output=completed.stdout,
      stderr=completed.stderr,
    )
  return completed


if __name__ == "__main__":
  try:
    result = run_eval()
  except subprocess.CalledProcessError as exc:
    if exc.output:
      sys.stdout.write(exc.output)
    if exc.stderr:
      sys.stderr.write(exc.stderr)
    raise
  else:
    if result.stdout:
      sys.stdout.write(result.stdout)
    if result.stderr:
      sys.stderr.write(result.stderr)


__all__ = ["build_eval_command", "run_eval", "DEFAULT_TIMEOUT_SECONDS"]
