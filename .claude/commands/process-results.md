# Post-process OmniHub results

## Order of operations

1. **omnihub-process** — Run first. Writes `processed-data/` inside each job directory.
2. **omnihub-index** — Run after process. Reads all `processed-data/` and writes a single table (e.g. `index.csv`). Fails if no processed data exists.

## Results directory layout

OmniHub jobs write under a root results dir (e.g. `$WORK/results/omnihub/`). Each job has its own directory, typically named by `$SLURM_JOB_ID`:

```
results-dir/
  <job_id>/
    job.sh           # Copy of the submitted script
    job.yaml         # Job config (cluster, partition, app-config, etc.)
    app.yaml         # Application config used
    job-status.yaml  # Exit code, etc. (if job completed)
    logs/            # srun stdout/stderr per rank
    tools/           # omnihub-monitor, omnistat, rocprofv3-stats, etc.
    processed-data/  # Created by omnihub-process (flat YAML/JSON)
```

Process discovers job dirs by **rglob("job.sh")** under `--results-dir`; each such parent dir is one execution.

## omnihub-process

```bash
./omnihub-process --results-dir /path/to/results/omnihub -j 4
```

| Argument | Description | Default |
|----------|-------------|---------|
| --results-dir | Root results directory (contains job subdirs) | required |
| -j, --jobs | Parallel workers | 1 |
| --force | Re-process dirs that already have processed-data | false |
| --successful-executions | Skip jobs with non-zero exit code in job-status.yaml | false |

Process runs a registry of parsers per job: job config, app config, app logs, job status, omnihub-monitor, omnistat variants, pytorch-trace, rccl-info, rocprofv3-stats, report card, hash. Application-specific parsers are invoked when the app has an executable **applications/<name>/parse.py** and the entrypoint is known (e.g. hf-finetune, vllm-latency, vllm-throughput). Parsed outputs are written under that job's `processed-data/`.

## omnihub-index

```bash
./omnihub-index --results-dir /path/to/results/omnihub --output index
```

Produces **index.csv** (or `index.<format>`). Requires that `omnihub-process` has already run; otherwise exits with an error.

| Argument | Description |
|----------|-------------|
| --results-dir | Same root as for process |
| --output | Base name for output file (e.g. index -> index.csv) |
| --format | csv (default) |

## Loading the index in Pandas

The CSV uses **two header rows** (source and field name). Load with:

```python
import pandas as pd
df = pd.read_csv("index.csv", header=[0, 1], index_col=0)
```

## Application-specific parsers

If the job's `app.yaml` has an entrypoint that matches a known app (e.g. `applications/vllm-throughput/bench_throughput_wrapper.py`), the processor looks for **applications/<name>/parse.py** and, if executable, runs it with the job directory as the argument. Those parsers can write additional artifacts into the job dir or into `processed-data/` for inclusion in the index.
