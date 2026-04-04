---
name: primus-pretrain
description: Primus pretraining via OmniHub -- job generation, config overrides, MASTER_PORT, HF_HOME, GPU arch (FP8), and debugging multinode hangs. Use when the user works with applications/primus-pretrain/, pretrain_wrapper.py, Primus config, primus_turbo, or multinode Primus training.
---

# Primus pretrain via OmniHub

Primus pretraining is launched through `applications/primus-pretrain/pretrain_wrapper.py`. This skill covers Primus-specific job generation, configuration, and debugging. For generic OmniHub job generation see the [generate-job-and-run](.cursor/skills/generate-job-and-run/) skill.

## Job generation

Always use `--tasks-per-node 1` when generating jobs for Primus. The pretrain wrapper expects one Slurm task per node (`SLURM_NTASKS == SLURM_NNODES`). With `--runner manual`, the default is one task per GPU, which breaks multi-node Primus unless overridden.

```bash
# Multi-node (always --runner manual --tasks-per-node 1)
./omnihub-generate-job --omnihub-dir $PWD \
  --app-config applications/primus-pretrain/config-example.yaml \
  --num-nodes 2 --runner manual --tasks-per-node 1 --partition mi2508x \
  --output job.slurm
sbatch job.slurm
```

- **Multi-node:** use `--runner manual --tasks-per-node 1` together.
- **Single-node:** include `--tasks-per-node 1` for consistency; without `--runner`, OmniHub already defaults to one task, but the explicit flag avoids mistakes if `--runner manual` is added later.
- **`--runner torchrun`**: OmniHub already uses one Slurm task per node, so `--tasks-per-node 1` is redundant. This is an alternative to `--runner manual --tasks-per-node 1` when you want an **outer** torchrun; match `run_mode` and `use_primus_cli` in the config to the flow you intend.

## MASTER_PORT

`pretrain_wrapper.py` sets the **inner** `MASTER_PORT` (used by Primus's torchrun) to `outer_port + 1` by default (e.g. 29401 when OmniHub uses 29400). This avoids `EADDRINUSE` when both OmniHub's torchrun and Primus's inner torchrun bind on the same node. Override via the `master_port` key in `config-example.yaml`.

## HuggingFace model assets and HF_HOME

`pretrain_wrapper.py` auto-resolves HuggingFace repo IDs (e.g. `meta-llama/Llama-3.1-8B`) to local snapshots when `$HF_HOME` is set and the model was previously downloaded with `huggingface-cli download`. No `hf_assets_path` override is needed in `config-example.yaml` -- just set `HF_HOME` in your environment. This avoids the need for `HF_TOKEN` at runtime.

If `HF_HOME` is not set or no matching snapshot is found, Primus falls back to its own download path (which requires `HF_TOKEN`). You can also set `hf_assets_path` explicitly in `config-example.yaml` under `modules.pre_trainer.overrides.model.hf_assets_path` to a full local path.

## GPU architecture (FP8 / primus_turbo)

The default `config-example.yaml` disables all `primus_turbo` and FP8 features for compatibility with MI210 (gfx90a), which lacks native FP8. On MI300X+ GPUs, remove or enable these overrides as appropriate.

Key overrides (under `modules.pre_trainer.overrides.primus_turbo`):

- `enable_primus_turbo: false`
- `enable_attention_float8: false`
- `use_turbo_float8_linear: false`
- `use_moe_fp8: false`
- `use_turbo_mx_linear: false`
- `use_turbo_async_tp: false`
- `use_turbo_attention: false`

**Symptom**: `ImportError: ScaleDtype from primus_turbo` or SIGSEGV during training means FP8 code paths are being hit on a GPU without FP8 support. Check that the overrides above are at the correct YAML nesting depth.

Also clear `model.converters: []` if the model config lists `primus_turbo` as a converter -- this avoids the converter import path entirely.

## Memory tuning

If you hit `HIP out of memory` on MI210 (64 GB HBM2e) with Llama 3.1 8B:

- Reduce `training.local_batch_size` (e.g. 1)
- Reduce `training.seq_len` (e.g. 2048)
- Enable `activation_checkpoint.mode: "full"` (full activation checkpointing)
- Disable `compile.enable: false` (avoids torch.compile memory overhead)

FSDP is typically enabled by default in TorchTitan configs. Tensor parallelism is another option for very large models.

## Config file structure

`config-example.yaml` overrides the Primus experiment config at runtime:

```yaml
entrypoint: applications/primus-pretrain/pretrain_wrapper.py
primus_path: /app/Primus
exp_config: /app/Primus/examples/torchtitan/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
primus_log_level: DEBUG
use_primus_cli: true
modules:
 pre_trainer:
   overrides:
     primus_turbo:
       enable_primus_turbo: false
       # ... (FP8 overrides)
     training:
       mock_data: true
```

The `modules.pre_trainer.overrides` dict is deep-merged into the experiment config YAML before Primus is launched. The wrapper writes a temporary merged config to a tempfile and passes it as `EXP`.

## Debugging multinode hangs

### Built-in sanity checks pass but Primus hangs

Use the **dist-init smoke test** to isolate whether the hang is in distributed init or in Primus training:

```bash
./omnihub-generate-job --omnihub-dir $PWD \
  --app-config applications/primus-pretrain/config-dist-smoke.yaml \
  --num-nodes 2 --runner manual --tasks-per-node 1 --partition mi2104x \
  --output dist-smoke-job.sh
sbatch dist-smoke-job.sh
```

This reproduces `pretrain_wrapper.py`'s full environment setup (same env vars, same MASTER_PORT+1) but stops after `torch.distributed.init_process_group`, running a broadcast + all_reduce test. If the smoke test passes but Primus hangs, the issue is in Primus hooks or training, not distributed init.

### Common Primus hang causes

1. **Hook failures on non-rank-0 nodes**: Primus hooks (`prepare_experiment.sh/py`) only log on `NODE_RANK == 0`. Other nodes exit silently, causing node 0's torchrun to hang waiting. Check **all** rank logs.
2. **`pip install -e torchtitan`** race conditions on shared filesystem overlays.
3. **HuggingFace tokenizer download waits**: Node 0 downloads, other nodes poll for a file that may be inside an isolated Apptainer overlay.
4. **`EADDRINUSE`**: Inner and outer torchrun competing for the same port. The wrapper's port+1 logic should prevent this, but verify with `MASTER_PORT` in logs.

### Log filtering

Primus uses `--local-ranks-filter`: only **local rank 0 on node 0** and the **last local rank on the last node** print training logs. Other ranks' stdout is empty by design -- not an error.

## Files

| File | Purpose |
|------|---------|
| `applications/primus-pretrain/config-example.yaml` | Main OmniHub config for Primus pretrain |
| `applications/primus-pretrain/pretrain_wrapper.py` | OmniHub entrypoint; env setup, config merge, launch |
| `applications/primus-pretrain/config-dist-smoke.yaml` | Config for the dist-init smoke test |
| `applications/primus-pretrain/dist_init_smoke_entrypoint.py` | Smoke test entrypoint (mimics wrapper, stops at init) |
| `applications/primus-pretrain/primus_child_dist_smoke.py` | Child process for smoke test (NCCL collectives) |
