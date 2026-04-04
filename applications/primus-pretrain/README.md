# Primus Pretrain Application for OmniHub

This application provides a wrapper to run Primus pretraining through the OmniHub framework.

## Overview

The Primus pretrain wrapper allows you to execute the `run_pretrain.sh` shell script from Primus using OmniHub's unified interface and profiling capabilities.

## Files

- `pretrain_wrapper.py` - Python wrapper script that executes Primus run_pretrain.sh
- `config-example.yaml` - Example configuration file
- `template-llama3.1_8B-BF16-pretrain.yaml` - Template Primus experiment config

## Usage

### Generating and Running SLURM Jobs

Use `omnihub-generate-job` to create a SLURM job script for Primus pretraining:

```bash
# Navigate to the omnihub repository
cd $HOME/omnihub

# Generate a SLURM job script
./omnihub-generate-job \
  --omnihub-dir $HOME/omnihub \
  --app-config applications/primus-pretrain/config-example.yaml \
  --cluster hpcfund \
  --partition mi3008x \
  --num-nodes 1 \
  --output primus-pretrain.slurm

# Submit the job
sbatch primus-pretrain.slurm
```

The generated SLURM script will automatically:
- Set up the appropriate container environment
- Configure distributed training parameters
- Set OMNIHUB environment variables (OMNIHUB_SRC_DIR, OMNIHUB_DATA_DIR, etc.)
- Execute the Primus pretrain wrapper with your configuration

### Multi-node training (Primus CLI / node-level launch)

For multinode pretraining, use **one task per node** so Primus (e.g. `run_pretrain_cli.sh` / `primus-cli direct`) can launch torchrun per node. Use `--runner manual --tasks-per-node 1`:

```bash
./omnihub-generate-job \
  --omnihub-dir $HOME/omnihub \
  --app-config applications/primus-pretrain/config-example.yaml \
  --cluster hpcfund \
  --partition mi3008x \
  --num-nodes 2 \
  --runner manual \
  --tasks-per-node 1 \
  --output primus-pretrain-multinode.slurm
```

The wrapper requires one task per node when running under Slurm (it checks `SLURM_NTASKS == SLURM_NNODES`) and will exit with an error otherwise. Distributed env is set only inside the Primus example; the entrypoint launches training via `run_pretrain_cli.sh` or `run_pretrain.sh`, and Primus runs torchrun itself.

### Running Directly with omnihub-run

For testing without SLURM:

```bash
omnihub-run \
  --output-dir=/path/to/results \
  --app-config=applications/primus-pretrain/config-example.yaml
```

### Configuration

The `config-example.yaml` file supports the following parameters:

#### Required
- `entrypoint`: Path to the wrapper script (e.g., `applications/primus-pretrain/pretrain_wrapper.py`)
- `exp_config`: Path to Primus experiment config file (YAML file defining model, training params, etc.)

#### Optional
- `primus_path`: Path to Primus installation (default: `${OMNIHUB_SRC_DIR}/../Primus`)
- `data_path`: Path to datasets directory (used if `OMNIHUB_DATA_DIR` is not set)
- `nnodes`: Number of nodes (default: from environment or 1)
- `node_rank`: Current node rank (default: from environment or 0)
- `gpus_per_node`: Number of GPUs per node (default: from environment or 8)
- `master_addr`: Master node address (default: from environment or localhost)
- `master_port`: Master node port (default: from environment or 29400)
- `hipblaslt_tuning_stage`: HipBLASLt tuning stage 0/1/2/3 (default: 0)
- `backend_path`: Path to backend (e.g., Megatron-LM)
- `work_group`: Work group name (default: from `PRIMUS_TEAM` env var or 'amd')
- `user_name`: User name (default: from `PRIMUS_USER` env var or 'root')
- `exp_name`: Experiment name (default: from `PRIMUS_EXP_NAME` env var or 'llama3.1_8B-pretrain')
- `workspace`: Workspace directory for outputs (used if `OMNIHUB_RESULTS_DIR` is not set)
- `models_path`: Path to models directory (used if `OMNIHUB_MODELS_DIR` is not set)
- `use_primus_cli`: If true, prefer `run_pretrain_cli.sh` (Primus CLI direct); default is to use it when multinode (`nnodes` > 1)
- `run_mode`: Default is **single**, set explicitly in the example config. Set to `"single"` for multi-node runs so that Primus simply calls python and not torchrun. Multi-node with `run_mode` unset is an error.

You can also include `modules` section with overrides for Primus experiment config (see `config-example.yaml` for details).

### Distributed environment (set only in this example)

Distributed env vars for Primus are set **only inside this Primus example** (the wrapper). OmniHub job scripts do not set Primus-specific vars, so the same launchers (manual, torchrun) work for non-Primus applications. **Both `--runner torchrun` and `--runner manual`** behave the same for the Primus example: the wrapper either sets distributed vars when `run_mode` is set (e.g. single) or unsets them when `run_mode` is omitted (single-node only) so Primus's torchrun can set them.

When `run_mode` is set, the wrapper derives `NNODES`, `NODE_RANK`, `GPUS_PER_NODE`, `MASTER_ADDR`, and `MASTER_PORT` from:

- **OmniHub runner env** when present: e.g. `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`
- **Slurm** when available: `SLURM_NNODES`, `SLURM_NODEID`, `SLURM_NODELIST` (first host for `MASTER_ADDR`)
- **Config** as explicit overrides

Primus does not always launch torchrun (it supports `--single` for one process). For distributed runs, Primus scripts typically launch torchrun themselves, which is why one task per node is required when using this wrapper (`--runner manual --tasks-per-node 1`).

### Environment Variables

The wrapper automatically uses and sets the following environment variables:

**OmniHub Variables (automatically set by omnihub-generate-job):**
- `OMNIHUB_SRC_DIR`: Path to OmniHub repository
- `OMNIHUB_DATA_DIR`: Data directory path
- `OMNIHUB_RESULTS_DIR`: Results output directory
- `OMNIHUB_MODELS_DIR`: Models directory path

**Primus Variables (set or derived by wrapper):**
- `PRIMUS_PATH`: Primus installation path
- `EXP`: Experiment config path
- `NNODES`: Number of nodes (from config, env, or derived from `WORLD_SIZE`/`GPUS_PER_NODE` or `SLURM_NNODES`)
- `NODE_RANK`: Current node rank (from config, env, or derived from `RANK`/`GPUS_PER_NODE` or `SLURM_NODEID`)
- `GPUS_PER_NODE`: GPUs per node (from config, env, or `torch.cuda.device_count()`)
- `MASTER_ADDR`: Master address (from config, env, or first host from `SLURM_NODELIST`)
- `MASTER_PORT`: Master port (from config or env, default 29400)
- `DATA_PATH`: Data directory (if configured)
- `PRIMUS_WORKSPACE`: Workspace directory (if configured)
- `PRIMUS_HIPBLASLT_TUNING_STAGE`: HipBLASLt tuning stage (if set)
- `BACKEND_PATH`: Backend path (if configured)
- `HF_HUB_CACHE`: Hugging Face hub cache (if models_path configured)


## HipBLASLt Tuning Stages

- **0**: No tuning (default)
- **1**: Dump GEMM shapes
- **2**: Offline tuning
- **3**: Use tuned config

## Troubleshooting

### Single-node job fails with SIGSEGV or "Patch 'megatron.turbo.te_spec_provider' failed"

If the container's **Primus Turbo** is enabled in the experiment config but the installed `primus_turbo` package is out of sync with Primus, the patch `megatron.turbo.te_spec_provider` can fail with:

```text
ImportError: cannot import name 'ScaleDtype' from 'primus_turbo.pytorch.core.low_precision'
```

Training then continues with 6/7 patches; later, Turbo-related code runs and can cause **SIGSEGV** (signal 11) on one or more ranks.

**Fix:** Disable Primus Turbo via the app config so the `te_spec_provider` patch is not required. The example `config-example.yaml` sets `modules.pre_trainer.overrides.enable_primus_turbo: false` for this reason. Re-enable Turbo only after the container's `primus_turbo` package matches the Primus version (so `ScaleDtype` is available).

## Examples

### Single Node Training on HPC Fund

```bash
# Generate SLURM job for single-node training on MI300X
./omnihub-generate-job \
  --omnihub-dir $HOME/omnihub \
  --app-config applications/primus-pretrain/config-example.yaml \
  --cluster hpcfund \
  --partition mi3008x \
  --num-nodes 1 \
  --output primus-single-node.slurm

sbatch primus-single-node.slurm
```

### Multi-Node Training on HPC Fund

```bash
# Generate SLURM job for 4-node training on MI300X (one task per node for Primus)
./omnihub-generate-job \
  --omnihub-dir $HOME/omnihub \
  --app-config applications/primus-pretrain/config-example.yaml \
  --cluster hpcfund \
  --partition mi3008x \
  --num-nodes 4 \
  --runner manual \
  --tasks-per-node 1 \
  --output primus-multi-node.slurm

sbatch primus-multi-node.slurm
```

### With Profiling Tools

```bash
# Generate SLURM job with omnistat profiling
./omnihub-generate-job \
  --omnihub-dir $HOME/omnihub \
  --app-config applications/primus-pretrain/config-example.yaml \
  --cluster hpcfund \
  --partition mi3008x \
  --tools omnistat \
  --output primus-with-profiling.slurm

sbatch primus-with-profiling.slurm
```

### Configuration File Example

Create a custom config file based on `config-example.yaml`:

```yaml
entrypoint: applications/primus-pretrain/pretrain_wrapper.py
# You can use absolute paths or environment variables in your shell before submitting
primus_path: /work/username/Primus
# Point exp_config to a Primus experiment config file (e.g., from Primus repo)
exp_config: /work/username/Primus/examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
work_group: amd
user_name: myusername
exp_name: my-experiment
workspace: /work/username/results/primus

# Optional: Override Primus experiment config parameters
modules:
  pre_trainer:
    overrides:
      train_iters: 100
      micro_batch_size: 4
      global_batch_size: 256
```

**Note:** OmniHub currently loads `config-example.yaml` with `yaml.safe_load` and `pretrain_wrapper.py` does not perform any special placeholder expansion. Syntax like `${PRIMUS_TEAM:amd}` is **not** interpreted automatically, so your config should contain concrete values (as shown in the example above).

If you want to derive values from environment variables, do so before generating the config (e.g., via your own templating step or shell scripting) and write the resolved values into the YAML file.
## Integration with OmniHub Tools

When running through OmniHub, you can enable various profiling and monitoring tools:

```bash
# With PyTorch profiler
./omnihub-generate-job \
  --omnihub-dir $HOME/omnihub \
  --app-config applications/primus-pretrain/config-example.yaml \
  --tools pytorch-stats \
  --output job.slurm

# With multiple tools
./omnihub-generate-job \
  --omnihub-dir $HOME/omnihub \
  --app-config applications/primus-pretrain/config-example.yaml \
  --tools omnistat pytorch-stats \
  --output job.slurm
```

See the main [OmniHub README](../../README.md) for the full list of supported tools and their capabilities.
