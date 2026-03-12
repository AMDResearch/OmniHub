---
name: sweeps
description: Creates and runs parameter sweeps with omnihub-sweep using templates, sweep-dir, and optional resume. Use when the user wants to run a sweep, use omnihub-sweep, create a parameter sweep, use a sweep template, set sweep-dir, or do a dry-run sweep.
---

# OmniHub sweeps

## How it works

1. **Template** — A YAML file (like an app config) where some values are **lists**. Each list is a sweep dimension.
2. **Configurations** — `omnihub-sweep` expands the template into all combinations and writes configs to `sweep-dir/configurations/` (e.g. `config-00000.yaml`, ...).
3. **Jobs** — For each combination of (partition × num_nodes × tool set × config), it generates a job script in `sweep-dir/jobs/` and submits it (unless `--dry-run`).
4. **Resume** — Submitted job IDs are stored in `sweep-dir/submitted.yaml`. Re-run the **same** sweep command to skip already-submitted combinations and continue.

## Required arguments

- **--omnihub-dir** — Path to the OmniHub repo.
- **--sweep-dir** — Directory for this sweep (configurations, jobs, submitted.yaml).
- **--template** — Path to the template YAML (list values define sweep dimensions).

## Optional arguments

| Argument | Description | Default |
|----------|-------------|---------|
| --cluster | hpcfund, radha | hpcfund |
| --partitions | Partitions to sweep (space-separated) | cluster default |
| --num-nodes | List of node counts (e.g. 1 2 4) | [1] |
| --tools | Tool sets; **repeat** to add multiple sets (e.g. `--tools omnistat` `--tools omnistat rocprofv3-stats`) | [[]] |
| --runner | manual, torchrun (for multi-node) | — |
| --time-limit | Job time limit | 1h |
| --delay | Seconds between submissions | 60 |
| --dry-run | Generate jobs only, do not sbatch | false |
| --max-active | Max concurrent (pending+running) jobs; wait until below before submitting more | 10 |
| --platform | apptainer, docker | apptainer |
| --rocm-version | ROCm version | 7.1.0 |
| --include-nodelist, --exclude-nodelist | Node filters | — |

## Template example

YAML with list values; each combination becomes one config:

```yaml
entrypoint: /app/vllm/benchmarks/benchmark_latency.py
model: meta-llama/Llama-3.1-8B-Instruct
tensor_parallel_size: 1
input_len: [32, 64]
output_len: [128, 256]
batch_size: 8
```

This yields 4 configs (2×2). Sweep then multiplies by partitions × num_nodes × tool sets.

## Directory layout

```
sweep-dir/
  configurations/   # config-00000.yaml, config-00001.yaml, ...
  jobs/             # job-<partition>-<nodes>-<toolset>-<config>.sh
  submitted.yaml    # list of { partition, num_nodes, tools, config, job_id } for resume
```

## Example

```bash
mkdir -p sweep-vllm
./omnihub-sweep --omnihub-dir $PWD --sweep-dir ./sweep-vllm \
  --template vllm-latency-template.yaml \
  --partitions mi2104x mi2508x \
  --tools omnistat --tools omnistat rocprofv3-stats
```

Dry-run (generate only):

```bash
./omnihub-sweep --omnihub-dir $PWD --sweep-dir ./sweep-vllm \
  --template vllm-latency-template.yaml --dry-run
```

## Resuming

Re-run the same command (same sweep-dir and parameters). Already-submitted (partition, num_nodes, tools, config) entries are read from `submitted.yaml` and skipped.
