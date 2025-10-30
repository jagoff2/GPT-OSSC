# GPT-OSS Workspace Architecture

This document summarizes the latent global workspace integration designed for `openai/gpt-oss-20b`.

## Components

- **Attention Patcher**: Wraps full-attention layers `{1,5,9,13,17,21}` and concatenates virtual KV tensors before the attention score computation. Residual deltas are injected after attention to adjust the last token representation without altering the base weights.
- **Virtual KV Store**: Maintains persistent virtual key/value segments per hooked layer with configurable retention limits and TTLs. Segments can spill to CPU memory when GPU pressure rises.
- **Workspace Module**: Slot-Attention workspace aggregates probe features from post-attention residuals to produce latent slots that inform virtual KV synthesis and residual deltas.
- **Controller**: Combines entropy heuristics with a lightweight MLP to decide when to broadcast workspace state, retrieve from memory, or halt generation.
- **Memory System**: SQLite metadata store paired with FAISS vector index enabling episodic recall. Retrieval updates workspace slots while enforcing a strict token budget for prompt injection.
- **FastAPI Server**: Exposes OpenAI-compatible endpoints with SSE streaming. Request-level toggles control hook activation to support A/B tests and regression comparisons.

## Data Flow

1. A request enters the server, the tokenizer encodes the chat transcript, and the generation loop begins.
2. During each hooked layer forward pass, virtual KV segments are concatenated to real cache entries. Residual probes capture last-token representations and feed the workspace.
3. Slot-Attention produces workspace slots which drive the virtual KV projector and residual delta hook. Controller decisions determine whether to persist the synthesized segments.
4. After the forward pass, retention logic advances the virtual KV store, optionally spilling segments to CPU. Controller outputs may trigger memory writes or retrieval.
5. The generation loop samples the next token, streams it to the client if requested, and repeats until completion or controller halt.

## Deployment Considerations

- The package targets 4-bit quantization across two 30 GB GPUs with CPU spillover for KV caches.
- Use `python -m cli.main serve --config configs/server.yaml` to launch the OpenAI-compatible server.
- Evaluation utilities (`cli.main eval` and `cli.main fluency-guard`) provide throughput, fluency, and tool-usage regressions tailored to long-horizon tasks.
