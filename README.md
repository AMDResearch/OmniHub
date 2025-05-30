# OmniHub: Tools for AI/ML Workload Analysis and Characterization
## Table of Contents

- [Introduction](#introduction)
- [Generating and Executing Jobs with OmniHub](#generating-and-executing-jobs-with-omnihub)
- [Sweeping Jobs and Application Arguments](#sweeping-jobs-and-application-arguments)
- [Processing Metrics and Results](#processing-metrics-and-results)
- [Known Issues](#known-issues)
- [Developer Corner](#developer-corner)
- [Contact](#contact)

## Introduction

This repository provides utilities (scripts, tools, and container images) to execute and analyze AI/ML workloads on AMD systems,
evaluating performance across system scales, model granularities, and AMD performance tools. Tested with local ML models
and pre-built Docker and Apptainer images, the tools enable flexible mix-and-match configurations via simple
command line arguments. We use the [Hugging Face](https://huggingface.co/) and [vLLM](https://github.com/vllm-project/vllm) APIs in our examples,
with support for additional frameworks available upon request (contact [us](#contact)).

**ML Models Sample List**

- Llama (v2-v4, 3B-405B)
- DeepSeek-R1
- AMD-OLMo (1B)
- Mistral (7B-24B)
- Qwen (v2.5-7B)
- NVLM (72B)
- more can be added on request (contact [us](#contact))

See [here](docs/models.md) for more details on the available ML models on Radha and the HPC Fund clusters.

**Frameworks**

- Hugging Face: inference and finetuning
- vLLM: inference
- PyTorch: training and inference
- SGLang: coming soon...

**Systems**

- HPC Fund: multi-node, multi-GPU; Apptainer
- Radha: multi-node, multi-GPU; Apptainer, Docker

**Tools Sample List**

- [rocprof](https://github.com/ROCm/rocprofiler)/[rocprof-compute](https://github.com/ROCm/rocprofiler-compute/)
- [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html)
- [omnitrace](https://github.com/ROCm/omnitrace)
- [omnistat](https://github.com/AMDResearch/omnistat/)
- more can be added on request (contact [us](#contact)).

Check the [list of supported tools](#list-of-supported-tools) for more details about tools and their execution modes.

## Generating and Executing Jobs with OmniHub

Our test clusters use SLURM for job management. OmniHub provides the tool [`omnihub-generate-job`](omnihub-generate-job)
to automatically create SLURM job scripts that fit your target environment. Using this tool, researchers can easily
generate and execute jobs with different combinations of ML models, container platforms, number of nodes, and
performance tools. For non-SLURM systems, the tool can serve as a reference to modify launch commands as needed.

In its most basic form, SLURM job script generation and job execution work as follows:

```console
git clone https://github.com/AARInternal/omnihub.git $HOME/omnihub
cd $HOME/omnihub

./omnihub-generate-job --omnihub-dir $HOME/omnihub \
  --app-config applications/hf-infer/config-example.yaml \
  --output hf-infer.slurm
sbatch hf-infer.slurm
```

Where `--omnihub-dir` points to your working copy of the OmniHub repository in
the cluster and `--app-config` points to the path to the application configuration file relative to the OmniHub directory.
Refer to [this document](docs/generate-job-examples.md) for more examples of using the `omnihub-generate-job` tool.

### List of Command Line Options for `omnihub-generate-job`

| Flag          | Options                                   | Description                          |
|:--------------|:------------------------------------------|:-------------------------------------|
| `omnihub-dir` |                                           | OmniHub working directory.           |
| `app-config`  |                                           | Relative path to the app config.     |
| `cluster`     | **`hpcfund`**, `radha`                    | Cluster name.                        |
| `partition`   |                                           | Partition or job queue name.   |
| `num-nodes`   |                                           | Number of nodes to allocate.         |
| `platform`    | **`apptainer`**, `docker`                 | Container platform to use.           |
| `runner`      | `manual`, `torchrun`                      | Distributed runner for multi-node.   |
| `tools`       | [List of tools](#list-of-supported-tools) | List of profiling tools.             |
| `time-limit`  |                                           | SLURM job time limit.                |

### List of Supported Tools

| Tool                        | Description                                                                            |
| :-------------------------- | :------------------------------------------------------------------------------------- |
| `rocprof-compute`           | Collect all performance counters.                                                      |
| `omnistat`                  | Low-overhead system metrics, sampled at 1s intervals.                                  |
| `omnistat-rocprofiler-pmc1` | Low-overhead performance counter collection, sampled at 1s intervals, 1st set of PMCs. |
| `omnistat-rocprofiler-pmc2` | Low-overhead performance counter collection, sampled at 1s intervals, 2nd set of PMCs. |
| `omnitrace`                 | Application tracing.                                                                   |
| `pytorch-stats`             | Collects detailed statistics of PyTorch operations.                                    |
| `pytorch-trace`             | PyTorch execution traces compatible with TensorBoard.                                  |
| `rocprofv1-stats`           | Kernel execution stats (to be deprecated soon).                                        |
| `rocprofv2-pmc`             | Profiling with performance counters ([configuration](config/rocprof.txt)).             |
| `rocprofv3-stats`           | Kernel execution stats.                                                                |
| `rccl-info`                 | Collects statistics of RCCL collective calls                                           |

### Example Applications

Explore [example applications](applications/) that demonstrate the usage of various ML models and configurations:

- **Hugging Face Inference:** LLM inference using the Hugging Face API.
- **Hugging Face Finetuning:** Finetune LLM models.
- **vLLM Inference:** Inference via the vLLM API.
- **vLLM Latency Benchmark:** Measure response times.
- **vLLM Throughput Benchmark:** Test concurrent processing.
- **PyTorch Training:** Train a simple CNN with PyTorch.
- **PyTorch Inference:** Run inference on a simple CNN.

All examples include YAML configuration files specifying the main entrypoint, tensor parallel size, and other settings.
A snippet of the Hugging Face inference configuration file is provided below.

```yaml
# Specifies the main script to run the application. The main function in the
# application may be decorated with `omnihub.entrypoint` and other functions may be
# decorated with `omnihub.tools.profile` to enable detailed profiling.
entrypoint: applications/hf-infer/infer.py

# Provides details for loading the model. The script will check if the model exists in
# `OMNIHUB_MODELS_DIR` before automatically downloading from Hugging Face.
ModelArguments:
  pretrained_model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
```

## Sweeping Jobs and Application Arguments

The OmniHub sweep tool ([`omnihub-sweep`](omnihub-sweep)) automates job generation and submission. It creates different
job setups based on CLI flags and generates application configurations from templates.

Application templates let you list multiple values for certain fields. The sweep tool then produces every possible
combination of those fields. For example, the following template that sweeps the input and output lengths for the VLLM
latency benchmark will generate 4 configuration files with different combinations of `input_len`/`output_len`: 32/128,
32/256, 64/128, and 64/256.
```yaml
entrypoint: /app/vllm/benchmarks/benchmark_latency.py
model: meta-llama/Llama-3.1-8B-Instruct
tensor_parallel_size: 1
input_len:
  - 32
  - 64
output_len:
  - 128
  - 256
batch_size: 8
# more configuration params go here...
```

The `omnihub-sweep` CLI also allows sweeping over job-related options, including:
- `partitions`: cluster partitions.
- `num-nodes`: number of nodes.
- `tools`: profiling tools to enable (`tools` can be set multiple times to enable different sets of tools).

For example, using the previously listed template stored in a file named
`vllm-latency-template.yaml`, the following `omnihub-sweep` will generate 4
configuration files, and then it will submit jobs for all of them in 2
partitions using 2 different sets of tools:
```console
mkdir sweep-vllm
./omnihub-sweep --omnihub-dir $PWD --sweep-dir ./sweep-vllm \
  --template vllm-latency-template.yaml \
  --partitions mi2104x mi2508x \
  --tools omnistat --tools omnistat rocprofv3-stats
```
```console
Starting a new sweep
.. Generated configurations: 4
Number of jobs in this sweep: 16
Submitting job: mi2104x/1/omnistat/config-00001.yaml
Submitting job: mi2104x/1/omnistat/config-00003.yaml
Submitting job: mi2104x/1/omnistat/config-00002.yaml
Submitting job: mi2104x/1/omnistat/config-00000.yaml
Submitting job: mi2104x/1/omnistat,rocprofv3-stats/config-00001.yaml
Submitting job: mi2104x/1/omnistat,rocprofv3-stats/config-00003.yaml
Submitting job: mi2104x/1/omnistat,rocprofv3-stats/config-00002.yaml
Submitting job: mi2104x/1/omnistat,rocprofv3-stats/config-00000.yaml
Submitting job: mi2508x/1/omnistat/config-00001.yaml
Submitting job: mi2508x/1/omnistat/config-00003.yaml
Submitting job: mi2508x/1/omnistat/config-00002.yaml
Submitting job: mi2508x/1/omnistat/config-00000.yaml
Submitting job: mi2508x/1/omnistat,rocprofv3-stats/config-00001.yaml
Submitting job: mi2508x/1/omnistat,rocprofv3-stats/config-00003.yaml
Submitting job: mi2508x/1/omnistat,rocprofv3-stats/config-00002.yaml
Submitting job: mi2508x/1/omnistat,rocprofv3-stats/config-00000.yaml
```

To test sweeps and job generation without submitting jobs to the cluster, use
the `--dry-run` flag.

## Processing Metrics and Results

After running OmniHub-generated jobs, you can process the results to create a summary in a standardized format
([`omnihub-process`](omnihub-process)). The summary includes details from various sources:

- Job configuration options
- Application configuration options
- Application metrics (if available)
- Default monitor metrics
- Omnistat report metrics

To process and index the results, use:

```console
./omnihub-process --results-dir /path/to/results/omnihub -j 4
./omnihub-index --results-dir /path/to/results/omnihub --output index
```

> **Note:** `omnihub-index` requires that `omnihub-process` has completed successfully and that processed data
is present in each job directory. If no processed data is found, `omnihub-index` will exit with an error message.
Please ensure all jobs have been processed before running the index step.

This will generate an `index.csv` file in the top directory of the repository. The CSV uses two header rows and can be
loaded in Pandas as follows:

```python
import pandas
df = pandas.read_csv("index.csv", header=[0,1], index_col=0)
```

## Known Issues

- `rocprofiler-compute` does not work with multi-node runs.
- Tools like `pytorch-trace`, `omnitrace`, and `rocprofv3-stats` tend to generate many GBs trace data per rank and you may run out of disk space very soon.

## Developer Corner

If you want to contribute to OmniHub, make sure you read [this document](docs/developer.md) for developer pre-requisites.

## Contact

- Email: [dl.RAD-omnihub@amd.com](mailto:dl.RAD-omnihub@amd.com)
- [GitHub Discussions](https://github.com/AARInternal/omnihub/discussions)
