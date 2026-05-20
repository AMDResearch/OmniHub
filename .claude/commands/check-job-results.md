# Check OmniHub job results

Use this command to **periodically check** or **inspect** the results of an executed OmniHub job -- look in the results directory and logs, **alert on errors**, and **report success** when the job completed successfully.

## Identifying the job

- **Explicit**: User provides `--results-dir` (root, e.g. `$WORK/results/omnihub`) and optionally a job ID, or a full path to one job dir (e.g. `$WORK/results/omnihub/279009`).
- **Current / latest (default)**: If no path or job ID is given, use **SLURM commands** to find the job(s) to check:
  - **Running jobs**: `squeue -u $USER -h -o "%i"` to list the user's job IDs. Use the **latest** (highest job ID or most recently submitted) as the default.
  - **Completed / recent**: If no running jobs, use `sacct -u $USER -j <recent_range> --format=JobID,State,End -n` to get the most recent completed job ID.
  - Do **not** search the filesystem for all job directories; use SLURM as the source of truth for "latest" job.
- **Multiple jobs running**: Collate results for all when count is small (2-3); ask the user which job when there are many (5+), unless the user said "check all".

Results root is typically `$WORK/results/omnihub` (from cluster config or env).

## What to check

1. **Job status (success vs failure)**
   - If **`job-status.yaml`** exists: read it. **`exit_code: 0`** -> success; **non-zero or missing** -> failure or abnormal exit.
   - If **`job-status.yaml`** does not exist: job may still be **running**, or may have **crashed** before writing status. Report "no status file yet" and scan logs for errors.

2. **Logs for errors**
   - **Primary (srun)**: Under **`<job_dir>/logs/`**, look at **`srun-*.err`** first (stderr per rank), then **`srun-*.out`** if needed.
   - **Torchrun / elastic runner**: Check **`<job_dir>/logs/*/attempt_<N>/<rank>/stdout.log`** and **`stderr.log`**. Prefer the **latest attempt** (highest `attempt_<N>`); scan **stderr.log** first.
   - **Alert** on: `Error`, `ERROR`, `Exception`, `Traceback`, `Fatal`, `failed`, `FAILED`, `exit code`, `killed`, `OOM`, `CUDA out of memory`, `HIP out of memory`, `Segmentation fault`, `SIGSEGV`, `signal 9`, `signal 11`, `ImportError`, `EADDRINUSE`.
   - Include **file name and line** when reporting. If multiple ranks have similar errors, summarize and show one representative snippet.

3. **Primus-specific log behavior**
   - **Rank filtering**: Primus uses `--local-ranks-filter` so only **local rank 0 on node 0** and the **last local rank on the last node** print training logs. Other ranks' stdout is empty by design -- **not** an error.
   - **Silent failures on non-rank-0 nodes**: Primus hooks only log on `NODE_RANK == 0`; other nodes call `sys.exit(1)` silently. If node 0's torchrun hangs, check **all** rank stderr/err logs.

4. **Sanity check failures**
   - If the job fails during `scripts/sanity-check.sh`, it is a **cluster networking issue**, not an application bug. Look for NCCL connection refused, link-local addresses (`169.254.x.x`), or `MASTER_ADDR` unreachable.

5. **Benign warnings to ignore**
   - `fuse-overlayfs exited: fuse: reading device: Software caused connection abort` -- benign Apptainer cleanup race.
   - `WARNING: AMD GPU compute partition may restrict concurrent access` -- informational.

6. **Timeout / external termination: training metrics summary**
   - When the job failed with external termination (exit_code 143 / SIGTERM, time limit, CANCELLED) and no training errors, include a **Training telemetry** summary.
   - Scan stdout for: `throughput`, `samples/s`, `tokens/s`, `FLOPs`, `TFLOPS`, `MFU`, `loss`, `lr`, `step`, `epoch`.
   - Report **count**, **mean**, **std dev**, **min**, **max** for each metric.

## Output format

- **Success**: "Job **succeeded** (exit_code: 0). No errors found in logs."
- **Failure**: "Job **failed** (exit_code: N). Errors in logs: ..." with file and snippet. For timeout-without-app-error, include training telemetry subsection.
- **Still running**: "Job **still running** (no job-status.yaml). So far in logs: ..."
