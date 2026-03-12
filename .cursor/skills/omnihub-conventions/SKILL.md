---
name: omnihub-conventions
description: Documents OmniHub repo layout, app configs, cluster and tools config, results layout, and when to use runner manual vs torchrun. Use when the user asks about conventions, app config, cluster config, tools config, results layout, or runner manual vs torchrun.
---

# OmniHub conventions

## Repo layout

| Path | Purpose |
|------|--------|
| config/ | Cluster YAMLs (e.g. hpcfund.yaml, radha.yaml), job.template, config/tools/*.yaml |
| applications/ | One dir per app: config-example.yaml, entrypoint script, optional parse.py |
| omnihub/generate/ | job.py (omnihub-generate-job), sweep.py (omnihub-sweep), app_config.py |
| omnihub/process/ | processor.py (omnihub-process), parsers.py |
| omnihub/index/ | indexer.py (omnihub-index) |
| omnihub/run/ | runner.py (omnihub-run, used inside the container by the job) |

## App config (YAML)

Paths are **relative to the OmniHub repo root** (e.g. `applications/hf-infer/config-example.yaml`).

- **entrypoint** — Required. Script to run (e.g. `applications/hf-infer/infer.py` or `/app/vllm/benchmarks/benchmark_latency.py`).
- **App-specific keys** — ModelArguments, VllmArguments, tensor_parallel_size, input_len, output_len, etc. Depend on the application; see each app’s config-example.yaml.

Templates for sweeps use the same structure but with **list values** for dimensions to sweep (see the sweeps skill in `.cursor/skills/sweeps/`).

## Cluster config (config/<cluster>.yaml)

Under `cluster:`:

- **name**, **models-dir**, **shared-dir**, **data-dir**, **results-dir** (template, e.g. `$WORK/results/omnihub/$SLURM_JOB_ID`)
- **subsets** — List of partition definitions: `partition`, `gpu`, `num-gpus`, `num-nodes`

Example: `config/hpcfund.yaml` defines partitions such as mi2104x, mi2508x, mi3008x, mi3258x.

## Tools config (config/tools/*.yaml)

Each YAML defines one or more tools (name, conflicts, etc.). The job generator loads all `config/tools/*.yaml` and builds the tool list. Use `omnihub-generate-job --help` to see the current tool list. No hardcoded tool list in skills; tools are discovered at runtime.

## Results layout and env vars

Jobs write to **results_dir** (from the cluster config, typically `$WORK/results/omnihub/$SLURM_JOB_ID`). The job script sets:

- **OMNIHUB_RESULTS_DIR** — Same as results_dir; used by omnihub-run and apps (e.g. Primus workspace).
- **tools/** — omnihub-monitor, omnistat, rocprofv3-stats, etc., per job.
- **logs/** — srun stdout/stderr per rank.
- **job.sh**, **job.yaml**, **app.yaml**, **job-status.yaml** — For post-processing.

Processed output goes in **processed-data/** under each job dir (created by omnihub-process).

## Runner: manual vs torchrun

- **--runner torchrun** — OmniHub’s job script launches torchrun; one process per GPU/task. Use when the app is a standard single-process script that expects to be launched by torchrun.
- **--runner manual** — No torchrun from the job script; one task per node (or override with --tasks-per-node). Use when:
  - The app is **multi-node** and starts its own distributed launcher (e.g. Primus CLI that runs torchrun internally), or
  - You need one process per node and the app uses NNODES / NODE_RANK / torchrun itself.

For **single-node** jobs, runner is often omitted. For **multi-node**, always set --runner; choose manual for frameworks that manage their own launcher (e.g. Primus pretrain).
