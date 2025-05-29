# Examples for `omnihub-generate-job`

This document contains detailed examples of how to use the `omnihub-generate-job` tool to generate SLURM job scripts for
various use cases. Below are just example ways to mix and match models, tools, and clusters. There are some combinations
that work better than the others because of deficiencies with some of the tools (e.g., rocprof-compute has known issues
in multi-node configurations).

> **Note:** These examples assume you have already navigated to the top level of the omnihub repository. If you're in
> another directory, please update the $PWD variable to point to your working copy.

## Llama3.1 on HPC Fund

### Infer Llama3.1 (405B) with a single-node execution on MI300s (vLLM) with Omnistat

```console
./omnihub-generate-job --omnihub-dir $PWD --cluster hpcfund --partition mi3008x --app-args=--VllmArguments.model=meta-llama/Meta-Llama-3.1-405B-Instruct --app-config applications/vllm-infer/config-example.yaml --tools omnistat --output job.slurm
sbatch job.slurm
```

### Infer Llama3.1 (405B) with a single-node execution on MI250s (Hugging Face) with PyTorch Profiler traces

```console
./omnihub-generate-job --omnihub-dir $PWD --cluster hpcfund --partition mi2508x --app-args=--VllmArguments.model=meta-llama/Meta-Llama-3.1-405B-Instruct --app-config applications/hf-infer/config-example.yaml --tools pytorch-trace --output job.slurm
sbatch job.slurm
```

### Finetune Llama3.1 (8B) with a multi-node execution on MI250s (Hugging Face) with Rocprof stats

```console
./omnihub-generate-job --omnihub-dir $PWD --cluster hpcfund --partition mi2508x --num-nodes 2 --app-config applications/hf-finetune/config-example.yaml --runner manual --tools rocprofv3-stats --output job.slurm
sbatch job.slurm
```

### Finetune Llama3.1 (8B) with a single-node execution on MI250s (Hugging Face) with Rocprof performance counters

```console
./omnihub-generate-job --omnihub-dir $PWD --cluster hpcfund --partition mi2508x --app-config applications/hf-finetune/config-example.yaml --runner manual --tools rocprofv2-pmc --output job.slurm
sbatch job.slurm
```

### Infer Llama3.1 (8B) on MI210s (Hugging Face) with rocprof-compute

```console
./omnihub-generate-job --omnihub-dir $PWD --cluster hpcfund --partition mi2104x --app-config applications/hf-infer/config-example.yaml --tools rocprof-compute --output job.slurm
sbatch job.slurm
```

## Llama3.1 on Radha

### Infer on MI210s (Hugging Face) with rocprof-compute

```console
./omnihub-generate-job --omnihub-dir $PWD --cluster radha --app-config applications/hf-infer/config-example.yaml --tools rocprof-compute --output job.slurm
sbatch job.slurm
```

If the run was successful, you will find the rocprof-compute output stats at
`$HOME/results/omnihub/$SLURM_JOB_ID/rocprof-compute`, with which you may do further
analysis (e.g., roofline analysis).

### Infer on MI210s (Hugging Face) with Omnitrace

```console
./omnihub-generate-job --omnihub-dir $PWD --cluster radha --app-config applications/hf-infer/config-example.yaml --tools omnitrace --output job.slurm
sbatch job.slurm
```

If the run was successful, you will find the omnitrace output stats under
`$HOME/results/omnihub/$SLURM_JOB_ID/omnitrace`, with which
you may use [Perfetto](https://ui.perfetto.dev/) for interactive exploration.
