# Generate job and run

## Workflow

1. Generate a SLURM job script with **omnihub-generate-job**.
2. Submit with **sbatch**. If the cluster requires a charge account (common on OLCF-style systems such as Frontier), pass **`-A <account>`** on the `sbatch` command (e.g. `sbatch -A ven114 job.slurm`). Do **not** add account lines to the generated OmniHub script; keep account selection in the dispatch command. If the first `sbatch` fails for account reasons, obtain the correct project name (ask the user, derive from stdout, or try site-appropriate discovery such as `sshare` / `sacctmgr`) and resubmit with **`-A`**.

## Required arguments

- **--omnihub-dir** -- Path to the OmniHub repo (e.g. `$PWD` or `$HOME/omnihub`).
- **--app-config** -- Path to the application config file **relative to the OmniHub directory** (e.g. `applications/hf-infer/config-example.yaml`).

## Optional arguments

| Argument | Description | Default |
|----------|-------------|---------|
| --app-args | Application-specific CLI args (e.g. `--VllmArguments.model=meta-llama/Meta-Llama-3.1-8B`) | "" |
| --cluster | Cluster name: hpcfund, frontier | hpcfund |
| --partition | Partition / queue name (see config/<cluster>.yaml) | cluster default |
| --num-nodes | Number of nodes | 1 |
| --runner | Distributed runner: manual, torchrun. **Required for multi-node.** | -- |
| --tools | One or more profiling tools (see config/tools/*.yaml) | [] |
| --time-limit | Job time limit (e.g. 30m, 2h) | 1h |
| --output | Output script path (default: stdout) | -- |
| --platform | apptainer, docker | apptainer |
| --rocm-version | ROCm version | 7.1.0 |
| --include-nodelist | Nodes to include (space-separated) | [] |
| --exclude-nodelist | Nodes to exclude (space-separated) | [] |
| --tasks-per-node | Override tasks per node (e.g. **1 for Primus**) | -- |

## Built-in sanity checks

Generated jobs automatically run **`scripts/sanity-check.sh`** before the main workload. It performs three checks: ROCm availability (`rocminfo`), GPU compute-mode (warns on exclusive partitions), and PyTorch NCCL `broadcast` + `all_reduce` via `torchrun` (uses `scripts/sanity_torch_dist.py`).

If the sanity step fails (e.g. NCCL connection refused, traffic to link-local `169.254.x.x` addresses), fix cluster networking first (`NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `MASTER_ADDR` reachability) before debugging application code.

## Runner choice

- **Multi-node:** Always set `--runner`. Use **manual** when the application starts its own distributed launcher (e.g. Primus pretrain); use **torchrun** when OmniHub should launch torchrun.
- **Single-node:** Runner is optional unless the app expects it.

For Primus-specific runner guidance (tasks-per-node, MASTER_PORT, HF_HOME, GPU arch), see `/project:primus-pretrain`.

## Example

```bash
./omnihub-generate-job --omnihub-dir $PWD \
  --app-config applications/vllm-latency/config-example.yaml \
  --cluster hpcfund --partition mi3008x \
  --tools omnistat --output job.slurm
sbatch job.slurm
```

Multi-node with manual runner (Primus -- **always** `--tasks-per-node 1`):

```bash
./omnihub-generate-job --omnihub-dir $PWD \
  --app-config applications/primus-pretrain/config-example.yaml \
  --num-nodes 2 --runner manual --tasks-per-node 1 --partition mi2508x \
  --output job.slurm
sbatch job.slurm
```

## SLURM account when `sbatch` fails

Many systems reject jobs unless a valid **charge account** (project) is set. Typical errors mention **Invalid account**, **association**, **no valid account**, or **bank**.

**If submission fails on the first try:**

1. **Infer or obtain the account** — Ask the user for their project / allocation ID when unsure; optionally try `sshare` or `sacctmgr show assoc user=$USER format=Account%20` (site-dependent); `$SLURM_JOB_ACCOUNT` from a recent successful job may be the right value.

2. **Resubmit with `-A` on `sbatch`** — Do not modify the generated job script for account. Use:

```bash
sbatch -A YOUR_PROJECT_HERE job.slurm
```

## Where to look

- **App configs:** `applications/*/config-example.yaml`
- **Cluster/partition definitions:** `config/<cluster>.yaml` (e.g. `config/hpcfund.yaml`)
- **Tool definitions:** `config/tools/*.yaml` (tools are auto-discovered; run `omnihub-generate-job --help` for the current tool list)
- **More examples:** [docs/generate-job-examples.md](docs/generate-job-examples.md)
