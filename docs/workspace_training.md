## Workspace Training Roadmap

### 1. Capture Targets
- Residual tensors for each `hooked_layer` at the final token position.
- Decoder logits for the same step, before workspace modifications.
- Controller entropy heuristics, plan energy, and workspace slots.
- Generated tokens and attention metadata required for loss computation.

### 2. Storage Schema
- Persist batches as torch `.pt` blobs with keys:
  - `input_ids`, `attention_mask`
  - `residuals` (dict[layer] -> tensor)
  - `logits`
  - `plan_energy`, `slots`
  - `metadata` (prompt text, seed, toggles)
- Index with lightweight JSON manifest for streaming.

### 3. Training Objectives
- Cross-entropy loss on next-token logits with gradients flowing through probes, workspace, controller.
- Auxiliary broadcast loss measuring perplexity delta when virtual KV is appended.
- Optional KL term to regularise controller outputs toward heuristic baseline.

### 4. Runtime Integration
- Load trained weights via new `WorkspaceConfig` fields.
- Preserve backward-compatible defaults (fallback to heuristic behaviour when weights absent).

### 5. Performance Considerations (CPU Only)
- Prefer bf16 for probes/workspace to minimise RAM.
- Reuse preallocated buffers during capture/training loops.
- Gate optional features behind CLI flags for iterative experimentation.


### 6. Offline Capture & Training Workflow
- Capture prompts into `.pt` blobs: `python scripts/capture_workspace_data.py --prompts prompts.txt --output data/capture`
- Fine-tune probes/controller: `python scripts/train_workspace.py --manifest data/capture/manifest.jsonl --epochs 3 --device cpu`
- Load trained weights via config or `--workspace-state` flag when launching the CLI server.

### 7. Runtime Options
- `kv_plan_scale`, `kv_plan_bias`, `kv_projection_scale` let you regulate virtual KV strength.
- `log_virtual_kv_stats` enables per-layer norm telemetry retrievable from capture buffers and the profiling script.
- `chunk_size` caps active cache length; the server forwards this automatically when set in config.
- `enable_torch_compile`, `torch_compile_mode`, and `inference_threads` expose CPU-friendly optimisations.

### 8. Tooling
- `scripts/profile_workspace.py` reports latency and RSS deltas (uses `psutil` when available).
- `configs/cpu_small.yaml` provides a reduced-footprint preset for constrained CPU hosts.
`scripts/capture_workspace_data.py` and `scripts/train_workspace.py` share the same config loader, so they honour workspace tweaks automatically.
