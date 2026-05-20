# Debug stdout from OmniHub jobs and containers

Analyze log output from OmniHub jobs and containers, identify root causes, and propose fixes in one of two forms:

1. **Image fixes** — Changes to incorporate into the OmniHub container image (Dockerfile, entrypoint scripts, or image build/contents) so the fix is permanent for all runs.
2. **Pre-launch fixes** — Commands or steps to run on the **host or login node** (or in the job script) **before** launching the container (e.g., env vars, bind mounts, or one-off setup).

## When invoked

1. **Capture context**: Get the full stdout/stderr snippet (from SLURM logs, `logs/` under `$WORK/results/omnihub/$SLURM_JOB_ID/`, or container output). Note cluster, partition, platform (docker vs apptainer), and how the job was launched.
2. **Identify the failure**: Pinpoint the exact message(s) that indicate the problem (error, exception, missing file, wrong version, permission issue, etc.).
3. **Classify the fix**:
   - **Image**: Fix belongs in the image if it's about missing/correct packages, entrypoint behavior, default paths, or runtime behavior inside the container.
   - **Pre-launch**: Fix is pre-launch if it's about host-side env, mounts, secrets, or one-time setup that shouldn't be baked into the image.
4. **Output format**: For each finding, provide:
   - **Symptom**: Quote or summarize the relevant stdout/stderr.
   - **Root cause**: Short explanation.
   - **Solution type**: Either "Incorporate into OmniHub image" or "Run before launching image (CMD line / job script)".
   - **Concrete fix**: Exact steps, code snippets, or commands.

## OmniHub context

- Jobs run via SLURM; stdout/stderr often in job output files or under `$WORK/results/omnihub/$SLURM_JOB_ID/logs/`.
- Containers can be Docker or Apptainer; image locations and build process are documented in `docs/images.md`.
- Runner and entrypoint live in `omnihub/run/`; cluster and tool config in `config/*.yaml` and `config/tools/*.yaml`.

Always prefer the smallest, clearest fix. If both an image change and a pre-launch workaround are possible, suggest both and label which is preferred for permanence vs quick iteration.
