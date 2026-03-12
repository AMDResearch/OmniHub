---
name: generate-job-and-run
description: Generates SLURM job scripts with omnihub-generate-job and runs them with sbatch. Use when the user wants to generate a job, run a job, use omnihub-generate-job, sbatch, set app-config, partition, or run a single OmniHub job.
---

# Generate job and run

## Workflow

1. Generate a SLURM job script with **omnihub-generate-job**.
2. Submit with **sbatch** (e.g. `sbatch job.slurm`).

## Required arguments

- **--omnihub-dir** — Path to the OmniHub repo (e.g. `$PWD` or `$HOME/omnihub`).
- **--app-config** — Path to the application config file **relative to the OmniHub directory** (e.g. `applications/hf-infer/config-example.yaml`).

## Optional arguments

| Argument | Description | Default |
|----------|-------------|---------|
| --app-args | Application-specific CLI args (e.g. `--VllmArguments.model=meta-llama/Meta-Llama-3.1-8B`) | "" |
| --cluster | Cluster name: hpcfund, radha | hpcfund |
| --partition | Partition / queue name (see config/<cluster>.yaml) | cluster default |
| --num-nodes | Number of nodes | 1 |
| --runner | Distributed runner: manual, torchrun. **Required for multi-node.** | — |
| --tools | One or more profiling tools (see config/tools/*.yaml) | [] |
| --time-limit | Job time limit (e.g. 30m, 2h) | 1h |
| --output | Output script path (default: stdout) | — |
| --platform | apptainer, docker | apptainer |
| --rocm-version | ROCm version | 7.1.0 |
| --include-nodelist | Nodes to include (space-separated) | [] |
| --exclude-nodelist | Nodes to exclude (space-separated) | [] |
| --tasks-per-node | Override tasks per node (e.g. 1 for Primus); requires --runner manual | — |

## Runner choice

- **Multi-node:** Always set `--runner`. Use **manual** when the application starts its own distributed launcher (e.g. Primus pretrain); use **torchrun** when OmniHub should launch torchrun.
- **Single-node:** Runner is optional unless the app expects it.

## Example

```bash
./omnihub-generate-job --omnihub-dir $PWD \
  --app-config applications/vllm-latency/config-example.yaml \
  --cluster hpcfund --partition mi3008x \
  --tools omnistat --output job.slurm
sbatch job.slurm
```

Multi-node with manual runner (e.g. Primus):

```bash
./omnihub-generate-job --omnihub-dir $PWD \
  --app-config applications/primus-pretrain/config-example.yaml \
  --num-nodes 2 --runner manual --partition mi2508x \
  --output job.slurm
sbatch job.slurm
```

## Where to look

- **App configs:** `applications/*/config-example.yaml`
- **Cluster/partition definitions:** `config/<cluster>.yaml` (e.g. `config/hpcfund.yaml`)
- **Tool definitions:** `config/tools/*.yaml` (tools are auto-discovered; run `omnihub-generate-job --help` for the current tool list)
- **More examples:** [docs/generate-job-examples.md](docs/generate-job-examples.md)
