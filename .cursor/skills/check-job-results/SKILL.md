---
name: check-job-results
description: Periodically checks the results of the current or specified OmniHub job: inspects results_dir and logs, alerts on errors, and reports success. Use when the user wants to monitor a job, check for errors in results or logs, see job status, or get a success/failure report for an OmniHub job.
---

# Check OmniHub job results

Use this skill when the user wants to **periodically check** or **inspect** the results of an executed OmniHub job--look in the results directory and logs, **alert on errors**, and **report success** when the job completed successfully.

## Identifying the job

- **Explicit**: User provides `--results-dir` (root, e.g. `$WORK/results/omnihub`) and optionally a job ID, or a full path to one job dir (e.g. `$WORK/results/omnihub/279009`).
- **Current / latest (default)**: If no path or job ID is given, use **SLURM commands** to find the job(s) to check:
  - **Running jobs**: `squeue -u $USER -h -o "%i"` (or equivalent) to list the user's job IDs. Use the **latest** (e.g. highest job ID, or most recently submitted) as the default job to check.
  - **Completed / recent**: If no running jobs, use `sacct -u $USER -j <recent_range> --format=JobID,State,End -n` (or similar) to get the most recent completed job ID and treat that as the latest.
  - Do **not** search the filesystem for all job directories to pick the latest; use SLURM as the source of truth for "latest" job.
- **Multiple jobs running**: If the user has **many jobs running** at once:
  - Either **collate results for all** (check each job's results dir and report status/log errors for every one), or
  - **Ask the user** which job ID (or which job) to get results for.
  - Prefer collating when the count is small (e.g. 2-3); prefer asking when there are many (e.g. 5+), unless the user said "check all" or similar.

Results root is typically `$WORK/results/omnihub` (from cluster config or env); if `$WORK` is unset, ask or use a sensible default. A **single job directory** is `$WORK/results/omnihub/<job_id>/` and contains `job.sh`, `job.yaml`, and usually `app.yaml`, `logs/`, and `tools/`.

## What to check

1. **Job status (success vs failure)**
   - If **`job-status.yaml`** exists: read it (YAML). **`exit_code: 0`** -> success; **non-zero or missing** -> failure or abnormal exit.
   - If **`job-status.yaml`** does not exist: job may still be **running**, or may have **crashed** before writing status (e.g. pre-execute failure). Report "no status file yet" and still scan logs for errors.

2. **Logs for errors**
   - **Primary (srun)**: Under **`<job_dir>/logs/`**, look at **`srun-*.err`** first (stderr per rank), then **`srun-*.out`** if needed.
   - **Torchrun / elastic runner**: Logs may also live in a nested layout written by the runner (e.g. torchrun, Primus). Check **`<job_dir>/logs/*/attempt_<N>/<rank>/stdout.log`** and **`<job_dir>/logs/*/attempt_<N>/<rank>/stderr.log`**:
     - Top-level `*` is a run/group ID (e.g. `none_abc123`, one per node or process group).
     - **`attempt_0`**, **`attempt_1`**, ... are attempt indices (retries).
     - **`<rank>`** is the local rank (numeric dirs: `0`, `1`, ...).
     - Prefer the **latest attempt** (highest `attempt_<N>`) for final status; scan **stderr.log** first, then stdout.log for context.
   - **Alert** on lines that indicate errors: e.g. `Error`, `ERROR`, `Exception`, `Traceback`, `Fatal`, `failed`, `FAILED`, `exit code`, `Exit code`, `killed`, `OOM`, `CUDA out of memory`, `HIP out of memory`, `ROCm`, `Segmentation fault`, `SIGSEGV`, `signal 9`, `signal 11`, `ImportError`, `EADDRINUSE`, etc.
   - Include **file name and line** (or last few lines) when reporting so the user can open the log.
   - If multiple ranks have similar errors, summarize (e.g. "same error on ranks 0,1,2") and show one representative snippet.

3. **Primus-specific log behavior**
   - **Rank filtering**: Primus uses `--local-ranks-filter` so only **local rank 0 on node 0** and the **last local rank on the last node** print training logs. Other ranks' stdout is empty by design -- this is **not** an error. Focus on these two ranks for training output.
   - **Silent failures on non-rank-0 nodes**: Primus hooks (`prepare_experiment.sh/py`) only log errors on `NODE_RANK == 0`; other nodes call `sys.exit(1)` silently. If node 0's torchrun hangs waiting for other nodes to join, check **all** rank stderr/err logs -- the root cause is often a hook failure on a non-zero node that produced no log output. Common hook failures:
     - `pip install -e torchtitan` race conditions on shared filesystem overlays
     - HuggingFace tokenizer download waits (node 0 downloads, other nodes poll for a file that may be inside an isolated Apptainer overlay)
   - **FP8 / primus_turbo crashes**: `ImportError: ScaleDtype from primus_turbo` or SIGSEGV during training indicate FP8 code paths on GPUs without native FP8 (e.g. MI210/gfx90a). Check that `primus_turbo` overrides are correctly nested in the app config (see generate-job-and-run skill).

4. **Sanity check failures**
   - Generated jobs run `scripts/sanity-check.sh` (ROCm, GPU compute-mode, NCCL smoke test) **before** the main app. If the job fails during this step, it is a **cluster networking issue**, not an application bug. Look for NCCL connection refused, link-local addresses (`169.254.x.x`), or `MASTER_ADDR` unreachable. Fix `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` and verify node connectivity before re-running.

5. **Benign warnings to ignore**
   - `fuse-overlayfs exited: fuse: reading device: Software caused connection abort` at job exit -- benign Apptainer/fuse-overlayfs cleanup race condition. Does not affect results.
   - `WARNING: AMD GPU compute partition may restrict concurrent access` -- informational from the GPU compute-mode check in `sanity-check.sh`; only an error if training actually fails with GPU access issues.

6. **Success report**
   - If **`job-status.yaml`** has **`exit_code: 0`** and no concerning lines in the logs (or only benign warnings): report **success** clearly.
   - Optionally: mention key artifacts (e.g. `processed-data/`, `tools/`) or a one-line summary from stdout (e.g. "Benchmark completed") if useful.

7. **Timeout / external termination: training metrics summary**
   - When the job **failed** with an **external termination** (e.g. **exit_code 143** / SIGTERM, time limit, **CANCELLED** step in sacct) and **no training/application errors** were found in the logs, add a **summarized training result** from the logs.
   - **Where to look**: Scan **stdout** (srun-*.out and `logs/<run_id>/attempt_<N>/<rank>/stdout.log`) for telemetry lines. Common patterns:
     - **Throughput**: `throughput`, `samples/s`, `tokens/s`, `samples per second`, `tokens per second`, `iter/s`, `iters/s`, `step/s`.
     - **FLOPs / compute**: `FLOPs`, `TFLOPS`, `PFLOPS`, `flops`, `FLOPS`.
     - **Model Flops Utilization**: `MFU`, `model flops utilization`, `Model Flops Utilization`.
     - **Other**: `loss`, `lr`/`learning rate`, `step`, `iteration`, `global_step`, `epoch`, latency, `ms/iter`, `time per step`, GPU utilization, memory.
   - **Extraction**: Use regex or line-by-line scan to capture numeric values associated with these names (e.g. `MFU: 0.42`, `throughput: 1234.5 samples/s`, `1234.5 TFLOPS`). Prefer the **latest attempt** and all ranks (or a representative set if many ranks).
   - **Consolidation**: For each metric that appears multiple times (e.g. per step or per rank):
     - Report **count** (number of values), **mean (average)**, **std dev**, **min**, **max**.
     - If only one or a few values exist, report them as-is (e.g. "MFU: 0.41, 0.42 (2 steps)").
   - **Output**: Include a short "Training telemetry (before exit)" subsection in the report: list each metric with its statistics (e.g. "Throughput (samples/s): n=100, mean=1200, std=50, min=1100, max=1300") so the user gets a concise summary of what was achieved before timeout.

8. **Network verification (running jobs)**
   - To check whether a running job uses InfiniBand or TCP sockets, examine IB traffic counters: `cat /sys/class/infiniband/mlx5_*/ports/1/counters/port_rcv_data` and `port_xmit_data` (sample before and after a short interval; increasing values confirm IB traffic). Also check `ibstat` for link state (Active = good, Down = not cabled/configured).
   - Enable `NCCL_DEBUG=INFO` in the job environment for explicit transport logs (e.g. "NET/IB" vs "NET/Socket").
   - If only 1 of N HCAs is active (e.g. only `mlx5_0` at 200 Gb/s, others down), flag the physical cabling issue to sysadmin.

## Workflow (periodic or on-demand)

- **On-demand**: Run the check once when the user asks ("how's my job?", "any errors?", "check the results", "did it succeed?").
- **Periodically**: When the user asks to **monitor** or **periodically check**:
  1. Run the check (status file + logs).
  2. If job still running (no `job-status.yaml`): report "Job still running; no status file yet" and any errors found in logs so far. Suggest re-running the check later (e.g. "Re-run this check in a few minutes" or "I can check again when you ask").
  3. If job finished: report success or alert on errors as above, and stop suggesting further checks unless the user asks again.

## Output format

- **Success**: e.g. "Job **succeeded** (exit_code: 0). No errors found in logs."
- **Failure**: e.g. "Job **failed** (exit_code: 143). Errors in logs: ..." with file and snippet. If failure is due to **timeout or external termination** (e.g. exit_code 143, CANCELLED step) and **no training/application errors** were found, also include a **Training telemetry (before exit)** subsection with consolidated metrics (mean, std, min, max, count) as in section 7.
- **Still running**: e.g. "Job **still running** (no job-status.yaml). So far in logs: ..." and note any errors already present.
- **No status, errors in logs**: e.g. "No job-status.yaml yet; possible crash. Errors in logs: ..."

Keep the report concise but actionable: user should know at a glance whether the job succeeded and where to look if it didn't. When reporting timeout-without-app-error, the telemetry summary gives a quick view of throughput, MFU, FLOPs, and other metrics achieved before the job was stopped.

## Conventions (reference)

- Results root: typically `$WORK/results/omnihub/`; each job is a subdir (usually `$SLURM_JOB_ID`).
- Job dir contains: `job.sh`, `job.yaml`, `app.yaml`, `job-status.yaml` (when finished), `logs/`, `tools/`.
- **Logs (two layouts)**:
  - **Srun**: `logs/srun-<rank>.out`, `logs/srun-<rank>.err` per rank.
  - **Torchrun / elastic**: `logs/<run_id>/attempt_<N>/<rank>/stdout.log`, `logs/<run_id>/attempt_<N>/<rank>/stderr.log` (run_id varies; attempt and rank are numeric). Always scan both layouts when checking for errors.

For full results layout and post-processing, see [omnihub-conventions](.cursor/skills/omnihub-conventions/) and [post-process-results](.cursor/skills/post-process-results/).
