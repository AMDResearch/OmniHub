---
name: check-job-results
description: Periodically checks the results of the current or specified OmniHub job: inspects results_dir and logs, alerts on errors, and reports success. Use when the user wants to monitor a job, check for errors in results or logs, see job status, or get a success/failure report for an OmniHub job.
---

# Check OmniHub job results

Use this skill when the user wants to **periodically check** or **inspect** the results of an executed OmniHub job—look in the results directory and logs, **alert on errors**, and **report success** when the job completed successfully.

## Identifying the job

- **Explicit**: User provides `--results-dir` (root, e.g. `$WORK/results/omnihub`) and optionally a job ID, or a full path to one job dir (e.g. `$WORK/results/omnihub/279009`).
- **Current / latest (default)**: If no path or job ID is given, use **SLURM commands** to find the job(s) to check:
  - **Running jobs**: `squeue -u $USER -h -o "%i"` (or equivalent) to list the user’s job IDs. Use the **latest** (e.g. highest job ID, or most recently submitted) as the default job to check.
  - **Completed / recent**: If no running jobs, use `sacct -u $USER -j <recent_range> --format=JobID,State,End -n` (or similar) to get the most recent completed job ID and treat that as the latest.
  - Do **not** search the filesystem for all job directories to pick the latest; use SLURM as the source of truth for “latest” job.
- **Multiple jobs running**: If the user has **many jobs running** at once:
  - Either **collate results for all** (check each job’s results dir and report status/log errors for every one), or
  - **Ask the user** which job ID (or which job) to get results for.
  - Prefer collating when the count is small (e.g. 2–3); prefer asking when there are many (e.g. 5+), unless the user said “check all” or similar.

Results root is typically `$WORK/results/omnihub` (from cluster config or env); if `$WORK` is unset, ask or use a sensible default. A **single job directory** is `$WORK/results/omnihub/<job_id>/` and contains `job.sh`, `job.yaml`, and usually `app.yaml`, `logs/`, and `tools/`.

## What to check

1. **Job status (success vs failure)**
   - If **`job-status.yaml`** exists: read it (YAML). **`exit_code: 0`** → success; **non-zero or missing** → failure or abnormal exit.
   - If **`job-status.yaml`** does not exist: job may still be **running**, or may have **crashed** before writing status (e.g. pre-execute failure). Report “no status file yet” and still scan logs for errors.

2. **Logs for errors**
   - Under **`<job_dir>/logs/`**: look at **`srun-*.err`** first (stderr per rank), then **`srun-*.out`** if needed.
   - **Alert** on lines that indicate errors: e.g. `Error`, `ERROR`, `Exception`, `Traceback`, `Fatal`, `failed`, `FAILED`, `exit code`, `Exit code`, `killed`, `OOM`, `CUDA out of memory`, `ROCm`, `Segmentation fault`, `signal 9`, `signal 11`, etc.
   - Include **file name and line** (or last few lines) when reporting so the user can open the log.
   - If multiple ranks have similar errors, summarize (e.g. “same error on ranks 0,1,2”) and show one representative snippet.

3. **Success report**
   - If **`job-status.yaml`** has **`exit_code: 0`** and no concerning lines in the logs (or only benign warnings): report **success** clearly.
   - Optionally: mention key artifacts (e.g. `processed-data/`, `tools/`) or a one-line summary from stdout (e.g. “Benchmark completed”) if that’s useful.

## Workflow (periodic or on-demand)

- **On-demand**: Run the check once when the user asks (“how’s my job?”, “any errors?”, “check the results”, “did it succeed?”).
- **Periodically**: When the user asks to **monitor** or **periodically check**:
  1. Run the check (status file + logs).
  2. If job still running (no `job-status.yaml`): report “Job still running; no status file yet” and any errors found in logs so far. Suggest re-running the check later (e.g. “Re-run this check in a few minutes” or “I can check again when you ask”).
  3. If job finished: report success or alert on errors as above, and stop suggesting further checks unless the user asks again.

## Output format

- **Success**: e.g. “Job **succeeded** (exit_code: 0). No errors found in logs.”
- **Failure**: e.g. “Job **failed** (exit_code: 143). Errors in logs: …” with file and snippet.
- **Still running**: e.g. “Job **still running** (no job-status.yaml). So far in logs: …” and note any errors already present.
- **No status, errors in logs**: e.g. “No job-status.yaml yet; possible crash. Errors in logs: …”

Keep the report concise but actionable: user should know at a glance whether the job succeeded and where to look if it didn’t.

## Conventions (reference)

- Results root: typically `$WORK/results/omnihub/`; each job is a subdir (usually `$SLURM_JOB_ID`).
- Job dir contains: `job.sh`, `job.yaml`, `app.yaml`, `job-status.yaml` (when finished), `logs/`, `tools/`.
- Logs: `logs/srun-<rank>.out`, `logs/srun-<rank>.err` per rank.

For full results layout and post-processing, see [omnihub-conventions](.cursor/skills/omnihub-conventions/) and [post-process-results](.cursor/skills/post-process-results/).
