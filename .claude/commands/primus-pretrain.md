# Primus pretrain via OmniHub

Primus pretraining is launched through `applications/primus-pretrain/pretrain_wrapper.py`. This covers Primus-specific job generation, configuration, and debugging. For generic OmniHub job generation see `/project:generate-job`.

## Job generation

Always use `--runner manual` for Primus. Do **not** use `--runner torchrun` -- OmniHub's torchrun runner passes `nproc_per_node=num_gpus` which conflicts with Primus's own process management, and the single Apptainer overlay per node causes `pip install` race conditions when multiple workers share it.

There are two supported launch modes. The choice depends on `run_mode` in the app config.

### Mode A: Primus owns torchrun (`use_primus_cli: true`, `run_mode` unset)

Primus's `run_pretrain_cli.sh` launches its own inner torchrun per node. OmniHub must give it **one Slurm task per node** so Primus controls all GPUs on the node.

```bash
./omnihub-generate-job --omnihub-dir $PWD \
  --app-config applications/primus-pretrain/config-example.yaml \
  --num-nodes 2 --runner manual --tasks-per-node 1 --partition mi2508x \
  --output job.slurm
sbatch job.slurm
```

- Use `--runner manual --tasks-per-node 1` together.
- Single-node: include `--tasks-per-node 1` for consistency.

### Mode B: OmniHub owns ranks (`use_primus_cli: true`, `run_mode: single`)

Setting `run_mode: single` tells Primus to run `python3` directly (no inner torchrun). OmniHub must provide **one Slurm task per GPU** so each task is one rank.

```bash
./omnihub-generate-job --omnihub-dir $PWD \
  --app-config applications/primus-pretrain/config-example.yaml \
  --num-nodes 2 --runner manual --partition mi2508x \
  --output job.slurm
sbatch job.slurm
```

- Use `--runner manual` **without** `--tasks-per-node 1` (the default is one task per GPU, which is correct here).
- Each srun task gets its own Apptainer container and overlay, so Primus's `pip install -e torchtitan` hook runs once per overlay with no contention.

### Why not `--runner torchrun`?

OmniHub's torchrun runner creates one Apptainer container per node with a single overlay, then launches torchrun inside with `--nproc_per_node=<num_gpus>`. This causes two problems for Primus:

1. **Overlay contention**: All GPU workers share one writable overlay. Primus's hooks run `pip install -e torchtitan` concurrently in each worker, racing on stale editable-install metadata.
2. **Double torchrun**: In Mode A, Primus launches its own inner torchrun. Having an outer torchrun that already spawned N workers means N independent inner torchruns each trying to claim all GPUs.

The manual runner avoids both issues: each Slurm task is its own isolated Apptainer container.

## MASTER_PORT

`pretrain_wrapper.py` sets the **inner** `MASTER_PORT` (used by Primus's torchrun) to `outer_port + 1` by default (e.g. 29401 when OmniHub uses 29400). This avoids `EADDRINUSE` when both OmniHub's torchrun and Primus's inner torchrun bind on the same node. Override via the `master_port` key in `config-example.yaml`.

## HuggingFace model assets and HF_HOME

`pretrain_wrapper.py` auto-resolves HuggingFace repo IDs (e.g. `meta-llama/Llama-3.1-8B`) to local snapshots when `$HF_HOME` is set and the model was previously downloaded with `huggingface-cli download`. No `hf_assets_path` override is needed -- just set `HF_HOME` in your environment.

If `HF_HOME` is not set or no matching snapshot is found, Primus falls back to its own download path (which requires `HF_TOKEN`). You can also set `hf_assets_path` explicitly in `config-example.yaml` under `modules.pre_trainer.overrides.model.hf_assets_path`.

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

Also clear `model.converters: []` if the model config lists `primus_turbo` as a converter.

## Memory tuning

If you hit `HIP out of memory` on MI210 (64 GB HBM2e) with Llama 3.1 8B:

- Reduce `training.local_batch_size` (e.g. 1)
- Reduce `training.seq_len` (e.g. 2048)
- Enable `activation_checkpoint.mode: "full"`
- Disable `compile.enable: false`

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

### Common Primus hang/crash causes

1. **Hook failures on non-rank-0 nodes**: Primus hooks only log on `NODE_RANK == 0`. Other nodes exit silently, causing node 0's torchrun to hang. Check **all** rank logs.
2. **`pip install -e torchtitan` overlay contention**: With the torchrun runner, multiple workers share one overlay and race on `pip install`. Use `--runner manual` instead.
3. **HuggingFace tokenizer download waits**: Node 0 downloads, other nodes poll for a file inside an isolated overlay.
4. **`EADDRINUSE`**: Inner and outer torchrun competing for the same port. Verify `MASTER_PORT` in logs.
5. **No DNS inside Apptainer**: Primus hooks calling pip may fail with `Temporary failure in name resolution`. Use `--no-deps` or pre-install packages.
6. **"Failed to initialize any NET plugin" on single-node Frontier**: `config/frontier.yaml` lists `NCCL_NET` / `NCCL_NET_PLUGIN` under `multinode-only-env` so they are omitted for `--num-nodes 1`. Verify the generated job script does **not** contain `export NCCL_NET=`.

## Files

| File | Purpose |
|------|---------|
| `applications/primus-pretrain/config-example.yaml` | Main OmniHub config for Primus pretrain |
| `applications/primus-pretrain/pretrain_wrapper.py` | OmniHub entrypoint; env setup, config merge, launch |
| `applications/primus-pretrain/config-dist-smoke.yaml` | Config for the dist-init smoke test |
| `applications/primus-pretrain/dist_init_smoke_entrypoint.py` | Smoke test entrypoint (mimics wrapper, stops at init) |
| `applications/primus-pretrain/primus_child_dist_smoke.py` | Child process for smoke test (NCCL collectives) |
