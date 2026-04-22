# OmniHub — Agent orientation

OmniHub provides tools to generate and run SLURM jobs for AI/ML workloads, run parameter sweeps, and post-process results. Optionally, the **OmniHub Agent** (on branch `aaji/agent`) can run as a service for monitoring and recommendations.

## Generate and run a job

1. Use **`omnihub-generate-job`** to create a SLURM job script.
2. Submit with **`sbatch <output-file>`**.

**Required:** `--omnihub-dir`, `--app-config` (path relative to repo root).

**Common options:** `--cluster` (hpcfund, radha, frontier), `--partition`, `--num-nodes`, `--runner` (manual | torchrun; required for multi-node), `--tools`, `--time-limit`, `--output`, `--image`, `--app-args`.

- App configs: `applications/*/config-example.yaml`
- Cluster configs: `config/*.yaml` (e.g. `config/hpcfund.yaml`)
- More examples: [docs/generate-job-examples.md](docs/generate-job-examples.md)

**Skill:** [.cursor/skills/generate-job-and-run/](.cursor/skills/generate-job-and-run/)

## Primus pretrain

Primus pretraining (`applications/primus-pretrain/`) has its own wrapper, config overrides, and debugging tools. Two launch modes: Mode A (Primus owns torchrun, `--tasks-per-node 1`) and Mode B (`run_mode: single`, one task per GPU -- default in example config). Inner MASTER_PORT is outer+1, `$HF_HOME` auto-resolves model assets, FP8/primus_turbo must be disabled on MI210.

**Skill:** [.cursor/skills/primus-pretrain/](.cursor/skills/primus-pretrain/)

## Run sweeps

Use **`omnihub-sweep`** to generate many job configs from a **template** (YAML with list values for sweep dimensions) and submit jobs.

**Required:** `--omnihub-dir`, `--sweep-dir`, `--template`.

**Options:** `--partitions`, `--num-nodes` (list), `--tools` (repeatable; each use adds a tool set), `--dry-run`, `--max-active`, `--delay`. Sweep state is stored in `sweep-dir/submitted.yaml`; re-run the same command to resume.

**Skill:** [.cursor/skills/sweeps/](.cursor/skills/sweeps/)

## Post-process results

1. **`omnihub-process --results-dir <dir> -j N`** — processes each job dir (writes `processed-data/`). Run first.
2. **`omnihub-index --results-dir <dir> --output index`** — builds `index.csv` from processed data. Requires step 1.

Index CSV has two header rows; load in Pandas: `pandas.read_csv("index.csv", header=[0,1], index_col=0)`.

**Skill:** [.cursor/skills/post-process-results/](.cursor/skills/post-process-results/)

## OmniHub Agent (aaji/agent branch)

Only available on branch **`aaji/agent`**. CLI: `omnihub-agent start | stop | status | run-once`. Config: `~/.omnihub_agent/config.yaml` or env vars `OMNIHUB_AGENT_*`. See `docs/omnihub-agent.md` on that branch.

**Skill:** [.cursor/skills/omnihub-agent/](.cursor/skills/omnihub-agent/)

## Conventions

- **Results:** Jobs write to `$WORK/results/omnihub/$SLURM_JOB_ID/` (job.sh, job.yaml, app.yaml, tools/, logs/).
- **App config paths** are relative to the OmniHub repo root.
- **Tools** are defined in `config/tools/*.yaml`.

**Skill:** [.cursor/skills/omnihub-conventions/](.cursor/skills/omnihub-conventions/)

---

See also: [README.md](README.md).
