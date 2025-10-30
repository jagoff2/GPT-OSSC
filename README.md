# GPT-OSS Workspace Hooks

This repository provides a Python 3.11 package that augments `openai/gpt-oss-20b` with a latent global workspace, persistent virtual KV append, and residual-delta hooks. It delivers:

- Hooked model wrapper with selective layer patching for GPT-OSS-20B full-attention blocks.
- Slot-Attention workspace and controller integrating SQLite+FAISS memory.
- OpenAI-compatible FastAPI server with streaming support and request-level hook toggles.
- Typer-based CLI for serving, evaluations, and fluency guard checks.
- Configs and evaluation utilities aimed at dual 16 GB GPUs and 192 GB system RAM.

Refer to `docs/architecture.md` for a high-level overview and `configs/*.yaml` for operational presets.
