# Examples for `omnihub-generate-job`

This document contains detailed examples of how to use the `omnihub-generate-job` tool to generate SLURM job scripts for various use cases.

## Llama3.1 on HPC Fund

### Infer Llama3.1 (405B) with a single-node execution on MI300s (vLLM) with Omnistat

```console
./omnihub-generate-job --omnihub-dir $HOME/omnihub --cluster hpcfund --partition mi2508x --app-args=--model-dir=Meta-Llama-3.1-405B-Instruct-safetensors --app-config applications/vllm-infer/config.yaml --tools omnistat > job.slurm
sbatch job.slurm
```

### Infer Llama3.1 (405B) with a single-node execution on MI250s (Hugging Face) with PyTorch Profiler traces

```console
./omnihub-generate-job --omnihub-dir $HOME/omnihub --cluster hpcfund --partition mi2508x --app-args=--model-dir=Meta-Llama-3.1-405B-Instruct-safetensors --app-config applications/hf-infer/config.yaml --tools pytorch-trace > job.slurm
sbatch job.slurm
```

### Finetune Llama3.1 (8B) with a single-node execution on MI250s (Hugging Face) with Rocprof stats

```console
./omnihub-generate-job --omnihub-dir $HOME/omnihub --cluster hpcfund --partition mi2508x --app-config applications/hf-finetune/config.yaml --runner manual --tools rocprofv1-stats  > job.slurm
sbatch job.slurm
```

### Finetune Llama3.1 (8B) with a single-node execution on MI250s (Hugging Face) with Rocprof performance counters

```console
./omnihub-generate-job --omnihub-dir $HOME/omnihub --cluster hpcfund --partition mi2508x --app-config applications/hf-finetune/config.yaml --runner manual --tools rocprofv2-pmc  > job.slurm
sbatch job.slurm
```

### Finetune Llama3.1 (8B) with manual distributed execution on MI210s (Hugging Face) with rocprof-compute

```console
./omnihub-generate-job --omnihub-dir $HOME/omnihub --cluster hpcfund --partition mi2104x --num-nodes 2 --app-config applications/hf-finetune/config.yaml --runner manual --tools rocprof-compute > job.slurm
sbatch job.slurm
```

### Infer Llama3.1 (8B) via Torchrun on MI210s (Hugging Face) with rocprof-compute

```console
./omnihub-generate-job --omnihub-dir $HOME/omnihub --cluster hpcfund --partition mi2104x --num-nodes 2 --app-config applications/hf-infer/config.yaml --runner torchrun --tools rocprof-compute > job.slurm
sbatch job.slurm
```

### Finetune Llama3.1 (8B) with Torchrun on MI210s (Hugging Face) with Omnitrace and Omnistat

```console
./omnihub-generate-job --omnihub-dir $HOME/omnihub --cluster hpcfund --partition mi2104x --num-nodes 2 --app-config applications/hf-finetune/config.yaml --runner torchrun --tools omnitrace omnistat > job.slurm
sbatch job.slurm
```

## Llama3 on Radha

### Infer on MI210s (Hugging Face) with rocprof-compute

```console
./omnihub-generate-job --omnihub-dir $HOME/omnihub --cluster radha --app-config applications/hf-infer/config.yaml --tools rocprof-compute > job.slurm
sbatch job.slurm
```

If the run was successful, you will find the rocprof-compute output stats at
`$HOME/results/omnihub/$SLURM_JOB_ID/rocprof-compute`, with which you may do further
analysis (e.g., roofline analysis).

### Infer on MI210s (Hugging Face) with Omnitrace

```console
./omnihub-generate-job --omnihub-dir $HOME/omnihub --cluster radha --model Meta-Llama-3-8B-Instruct-safetensors --app-config applications/hf-infer/config.yaml --tools omnitrace > job.slurm
sbatch job.slurm
```

If the run was successful, you will find the omnitrace output stats under
`$HOME/results/omnihub/$SLURM_JOB_ID/omnitrace`, with which
you may use [Perfetto](https://ui.perfetto.dev/) for interactive exploration.