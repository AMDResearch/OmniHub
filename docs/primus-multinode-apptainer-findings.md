# Primus Multinode Training with Apptainer -- Findings and Fixes

This document summarizes issues encountered and resolved while running
Primus multinode pretraining (Llama 3.1 8B, MI210/gfx90a) inside
**Apptainer containers on Slurm HPC clusters**. The goal is to share
lessons learned with the Primus team so these patterns can be addressed
upstream or documented for other Apptainer users.

## How Apptainer differs from bare-metal / Docker

Apptainer (formerly Singularity) is used on shared HPC clusters where
users cannot run Docker. The key difference that drives most of the issues
below:

- **The container image (SIF) is read-only.** A `.sif` file is a squashfs
  filesystem -- immutable at runtime. Any writes (pip installs, file
  downloads, config generation) go to a per-task **overlay filesystem**, a
  writable layer stacked on top of the read-only image.
- **Each Slurm task gets its own overlay.** Files written inside one
  task's overlay are invisible to other tasks. There is no equivalent of
  Docker's shared writable layer or bind-mounted volumes between tasks.

---

## Issues Found

### 1. Read-only SIF + stale `pip install -e` metadata

Primus's `prepare_experiment.py` hook runs
`pip install -e /app/Primus/third_party/torchtitan` at startup. Inside
Apptainer this fails because the SIF image contains **stale editable-install
metadata**: `torchtitan-0.0.2.dist-info/RECORD` references
`__editable__.torchtitan-0.0.2.pth`, but that `.pth` file doesn't exist
in the squashfs (lost during the Docker-to-SIF conversion). When pip tries
to update the existing installation, it fails with:

```
OSError: [Errno 2] No such file or directory:
  '/opt/venv/lib/python3.10/site-packages/__editable__.torchtitan-0.0.2.pth'
```

**Why Docker doesn't have this problem**: In Docker, the container's
writable layer preserves all files from the image build, including the
`.pth` file. The squashfs conversion that produces a `.sif` can lose files
that were created and referenced across different image layers, leaving
behind orphaned metadata.

### 2. Shared overlay + concurrent pip installs

When torchrun forks N GPU workers inside a **single** Apptainer container,
all N workers share one writable overlay. Each worker's
`prepare_experiment` hook runs `pip install -e torchtitan` concurrently on
the same overlay -- a race condition on the stale metadata from issue #1.
Workers corrupt each other's partial writes and the job crashes.

The fix is to run **one Apptainer container per Slurm task** so each task
gets its own isolated overlay. The pip install runs once per overlay with
no contention. This means either:

- **One task per node** (Pattern A below): Primus owns torchrun internally
  and manages all GPUs on the node. Each node is one Slurm task, one
  container, one overlay.
- **One task per GPU** (Pattern B below): Each GPU rank is its own Slurm
  task, own container, own overlay. Primus runs in `run_mode: single`
  (direct `python3`, no inner torchrun).

In both cases, the pip install runs exactly once per overlay with no
contention.

**Recommendation for Primus**: Consider making `prepare_experiment.py`
skip the `pip install -e torchtitan` if torchtitan is already importable.
For container images, prefer a non-editable install (`pip install .`
instead of `pip install -e .`) so the package files live directly in
site-packages rather than relying on `.pth` redirects that may not survive
image format conversions.

### 3. HuggingFace model assets not visible across overlays

Primus model YAML files set `model.hf_assets_path` to a HuggingFace repo
ID (e.g. `meta-llama/Llama-3.1-8B`). At runtime, Primus's
`prepare_experiment` hook resolves this by downloading via the HuggingFace
Hub API. Inside Apptainer this breaks for two reasons:

1. **Models may not be downloadable at runtime.** Some HPC clusters
   restrict outbound network access from compute nodes, and even where
   network is available, concurrent downloads from multiple tasks are
   wasteful and race-prone. `HF_TOKEN` may not propagate into the
   container namespace.

2. **Site-specific pre-downloaded models should be reusable.** Many HPC
   sites pre-stage popular models on shared filesystems (e.g. under
   `$HF_HOME`). Primus should be able to discover and use these without
   re-downloading.

Even if downloads succeed, each task's overlay is isolated -- files
downloaded inside node 0's overlay are invisible to node 1.

**Fix**: Pre-download models to a shared filesystem before the job runs:

```bash
HF_HOME=/shared/models huggingface-cli download meta-llama/Llama-3.1-8B
```

Then resolve the repo ID to the local `$HF_HOME` snapshot path and pass
it as `hf_assets_path` so Primus's prepare hook sees a local directory and
skips the download. The snapshot lives at:

```
$HF_HOME/hub/models--meta-llama--Llama-3.1-8B/snapshots/<commit_hash>/
```

**Recommendation for Primus**: If `$HF_HOME` is set, `prepare_experiment`
could check for a local snapshot before attempting a download. This would
make Primus work out of the box on clusters where models are pre-staged.

### 4. Distributed env var ownership and MASTER_PORT collision

These two issues are related -- they both stem from ambiguity about which
layer controls the distributed setup.

**Env var ownership**: Primus scripts, the outer job launcher, and Slurm
all set overlapping env vars (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`,
`MASTER_PORT`, `LOCAL_RANK`, etc.) with different semantics. There was no
clear contract on which layer "owns" each variable, leading to conflicts.

**MASTER_PORT collision**: When Primus runs inside a framework that already
uses torchrun (or any process that binds `MASTER_PORT` for distributed
rendezvous), Primus's `run_pretrain_cli.sh` starts its own **inner
torchrun** that tries to bind the same port on the same node, causing
`EADDRINUSE`. This was the most common hang/crash in multinode runs.

**Fix**: We established a clear resolution chain for env vars: explicit
config -> outer-launcher env vars -> Slurm env vars (`SLURM_NNODES`,
`SLURM_NODEID`, `SLURM_NODELIST`) -> defaults. `MASTER_ADDR` is derived
from `scontrol show hostnames $SLURM_NODELIST` when not set. For
`MASTER_PORT`, the inner (Primus) port is set to `outer_port + 1`
(e.g. 29401 when the outer process uses 29400).

**Recommendation for Primus**: Be aware that in nested-torchrun scenarios,
`MASTER_PORT` needs to differ from whatever the parent process already
bound. Primus could detect an occupied port and auto-increment.

### 5. FQDN hostname breaks Primus node matching

Some HPC clusters return fully qualified domain names (FQDNs) from
`socket.gethostname()` (e.g. `node001.cluster.example.com`), while
Slurm's `SLURM_NODELIST` and `scontrol show hostnames` return short
names (e.g. `node001`). Primus compares hostnames to determine
`MASTER_ADDR` and local rank assignment; when the formats don't match,
nodes fail to find themselves in the node list and distributed init hangs
or crashes.

**Fix**: Add a `sitecustomize.py` to the container's site-packages that
strips the domain suffix:

```python
import socket

original_gethostname = socket.gethostname

def patched_gethostname():
    hostname = original_gethostname()
    if '.' in hostname:
        return hostname.split('.')[0]
    return hostname

socket.gethostname = patched_gethostname
```

This ensures all Python code inside the container sees the short hostname,
matching what Slurm reports.

**Recommendation for Primus**: When comparing hostnames for distributed
setup, normalize to short names (strip everything after the first `.`)
rather than relying on exact `gethostname()` output.

### 6. Primus Turbo / FP8 on MI210 (gfx90a)

MI210 has no native FP8 hardware. The default Primus experiment config
enables `primus_turbo`, which patches in `te_spec_provider`. On gfx90a
the import fails (`ImportError: ScaleDtype`), the patch silently degrades,
and later code hits SIGSEGV.

**Fix**: Disable all `primus_turbo` and FP8 features via config overrides
for MI210/gfx90a:

```yaml
primus_turbo:
  enable_primus_turbo: false
  enable_attention_float8: false
  use_turbo_float8_linear: false
  use_moe_fp8: false
  use_turbo_mx_linear: false
  use_turbo_async_tp: false
  use_turbo_attention: false
```

Also set `model.converters: []` to prevent the turbo converter import.

**Recommendation for Primus**: The `ScaleDtype` import failure is handled
silently (6/7 patches succeed, training continues) but later causes
SIGSEGV. A louder failure or automatic disablement on GPUs without FP8
would help users diagnose this faster.

---

## Primus Assumptions Rethought for Apptainer

| Primus Assumption | Reality in Apptainer on Slurm | Workaround |
|---|---|---|
| Container filesystem is writable | SIF is read-only squashfs; `.pth` files from editable installs lost in conversion | Pre-install non-editable in image; or skip install if already importable |
| `pip install -e` can run concurrently | Shared overlay between workers causes race conditions | One container per Slurm task (isolated overlays); avoid forking workers inside one container |
| `hf_assets_path` repo IDs are downloadable at runtime | Downloads may fail (no network) or land in isolated overlays invisible to other tasks; site-specific models should be reusable | Pre-download models to shared FS; resolve to local `$HF_HOME` snapshot |
| Primus owns torchrun (and MASTER_PORT) | An outer launcher may already bind the port; env vars set by multiple layers | Inner port must differ (e.g. outer + 1); clear env var ownership chain |
| `gethostname()` returns a usable short name | HPC nodes may return FQDNs; Slurm uses short names | `sitecustomize.py` to strip domain suffix; or normalize in Primus |
| FP8/Turbo are safe defaults | MI210 (gfx90a) lacks FP8; silent patch failure causes SIGSEGV | Disable turbo on non-FP8 GPUs; fail loudly on import error |

---

## Two Working Launch Patterns

### Pattern A: Primus owns torchrun (one Slurm task per node)

Run one Slurm task per node. Each task runs inside its own Apptainer
container. Primus's `run_pretrain_cli.sh` launches torchrun internally to
manage all GPUs on the node.

- Set `use_primus_cli: true`, do not set `run_mode`
- Slurm: `--ntasks-per-node=1`
- The inner `MASTER_PORT` must differ from any port the outer launcher uses

### Pattern B: External rank management (one Slurm task per GPU)

Run one Slurm task per GPU. Each task is its own Apptainer container with
its own overlay. Set `run_mode: single` so Primus runs `python3` directly
(no inner torchrun).

- Set `use_primus_cli: true`, `run_mode: single`
- Slurm: `--ntasks-per-node=<num_gpus>`
- Each task is a single rank; distributed env vars (`RANK`, `WORLD_SIZE`,
  etc.) are set by the outer launcher

**Important**: Do not launch multiple workers inside a single Apptainer
container (e.g. via torchrun inside one srun task). The shared overlay
causes `pip install` race conditions and multiple Primus instances compete
for the same GPUs.

---

## Verified Configuration

Successfully trained Llama 3.1 8B on 2 nodes (8x MI210) using Pattern B:

- **Container**: `omnihub.training.gfx90a.710.sif`
- **Config**: `use_primus_cli: true`, `run_mode: single`, `mock_data: true`
- **Turbo**: All `primus_turbo` and FP8 features disabled
- **HF assets**: Pre-downloaded to shared filesystem, resolved via `$HF_HOME`
- **Performance**: ~1,755 tokens/sec, ~90 tflops, ~29% MFU per rank
- **Loss**: Converging (mock data), step 34 loss 4.62

### Known issue: single-node NCCL plugin failure

The aws-ofi-rccl plugin (built from commit `339b39d` at
`/ccs/sw/crusher/amdsw/aws-ofi-nccl/`) fails to initialize on
single-node jobs. Setting `NCCL_NET="AWS Libfabric"` forces NCCL to
load the plugin, but the CXI provider is not active for intra-node
communication (XGMI handles it), so initialization fails with "Failed
to initialize any NET plugin". `config/frontier.yaml` lists these
vars under `multinode-only-env` so `omnihub-generate-job` omits them
for single-node jobs.
