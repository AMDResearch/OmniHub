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
- `master_port`: Master node port (default: from environment or 1234)
- `hipblaslt_tuning_stage`: HipBLASLt tuning stage 0/1/2/3 (default: 0)
- `backend_path`: Path to backend (e.g., Megatron-LM)
- `work_group`: Work group name (default: from `PRIMUS_TEAM` env var or 'amd')
- `user_name`: User name (default: from `PRIMUS_USER` env var or 'root')
- `exp_name`: Experiment name (default: from `PRIMUS_EXP_NAME` env var or 'llama3.1_8B-pretrain')
- `workspace`: Workspace directory for outputs (used if `OMNIHUB_RESULTS_DIR` is not set)
- `models_path`: Path to models directory (used if `OMNIHUB_MODELS_DIR` is not set)

You can also include `modules` section with overrides for Primus experiment config (see `config-example.yaml` for details).

### Environment Variables

The wrapper automatically uses and sets the following environment variables:

**OmniHub Variables (automatically set by omnihub-generate-job):**
- `OMNIHUB_SRC_DIR`: Path to OmniHub repository
- `OMNIHUB_DATA_DIR`: Data directory path
- `OMNIHUB_RESULTS_DIR`: Results output directory
- `OMNIHUB_MODELS_DIR`: Models directory path

**Primus Variables (set by wrapper):**
- `PRIMUS_PATH`: Primus installation path
- `EXP`: Experiment config path
- `NNODES`: Number of nodes
- `NODE_RANK`: Current node rank
- `GPUS_PER_NODE`: GPUs per node
- `MASTER_ADDR`: Master address
- `MASTER_PORT`: Master port
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
# Generate SLURM job for 4-node training on MI300X
./omnihub-generate-job \
  --omnihub-dir $HOME/omnihub \
  --app-config applications/primus-pretrain/config-example.yaml \
  --cluster hpcfund \
  --partition mi3008x \
  --num-nodes 4 \
  --runner manual \
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
