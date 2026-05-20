# OmniHub — Guidance for AI assistants

Use this when helping users with OmniHub: job generation, sweeps, result processing, or the agent service.

## When the user wants to run a job

1. **Generate** a SLURM script: `./omnihub-generate-job --omnihub-dir $PWD --app-config <path> [options] --output job.slurm`
2. **Submit**: `sbatch job.slurm`

- **Paths:** App configs live under `applications/` (e.g. `applications/hf-infer/config-example.yaml`). Cluster definitions are in `config/` (e.g. `config/hpcfund.yaml`).
- **Multi-node:** They must pass `--runner manual` or `--runner torchrun`; use `manual` when the app launches its own distributed run (e.g. Primus).
- **Overrides:** `--app-args` passes through to the application; `--partition` selects the queue.

## When the user wants to run a sweep

Use **`omnihub-sweep`**: `--omnihub-dir`, `--sweep-dir`, `--template` (YAML with list values for dimensions). Optionally `--partitions`, `--num-nodes`, repeated `--tools`, `--dry-run`, `--max-active`. Sweep dir gets `configurations/`, `jobs/`, and `submitted.yaml`; re-running resumes from `submitted.yaml`.

## When the user has results to analyze

1. **Process** first: `./omnihub-process --results-dir /path/to/results/omnihub -j 4`
2. **Index** next: `./omnihub-index --results-dir /path/to/results/omnihub --output index`

Do not run index before process; index expects `processed-data/` in each job dir. Output is `index.csv` with two-level column headers (Pandas: `header=[0,1], index_col=0`).

## Key paths

| Path | Purpose |
|------|--------|
| `config/` | Cluster and job template; `config/tools/` for tool definitions |
| `applications/` | App configs and entrypoints; e.g. `applications/vllm-latency/config-example.yaml` |
| `omnihub/generate/` | Job and sweep generation (`job.py`, `sweep.py`, `app_config.py`) |
| `omnihub/process/` | Result processing and parsers |
| `omnihub/index/` | Indexer for processed results |

## Common pitfalls

- **Multi-node:** Forgetting `--runner` leads to single-process behavior; use `--runner manual` for apps that start torchrun themselves.
- **Index fails:** Run `omnihub-process` first so each job dir has `processed-data/`.

For more detail, see [AGENTS.md](AGENTS.md).
