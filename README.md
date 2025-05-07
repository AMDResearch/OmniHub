# OmniHub: Tools for AI/ML Workload Analysis and Characterization

This repository contains a set of utilities (scripts, tools, container images)
to analyze and characterize AI/ML workloads on AMD systems. The goal of this project is to characterize
performance of different stages of the ML lifecycle (inference, finetuning) at different system scales (single/multi-node, single/multi-GPU)
and different model granularities (kernel, op, layer, entire model) and model types (LLM, CNN, GNN).
The analysis of each workload may be done using available tools such as rocprof-compute, omnitrace, rocprof, and so on.

We have currently tested the scripts in this repository to work with the following ML models, stages, systems, and tools. We have
ML models copied locally and pre-built [Docker and Apptainer images](docs/images.md) pre-installed and customized with the necessary
tools on the below systems for your ready consumption.
We have used [Hugging Face](https://huggingface.co/) and [vLLM](https://github.com/vllm-project/vllm) API so far in our scripts, but support for other ML frameworks could be added (talk to [us](#contact)).
Our scripts are very user-friendly where you can mix-and-match the models, tasks, and tools of interest with simple command line arguments.

**ML Models**

- Llama (v2-v3.3, 3B-405B)
- DeepSeek-R1
- AMD-OLMo (1B)
- Mistral (7B-24B)
- Qwen (v2.5-7B)
- NVLM (72B)

See [here](docs/models.md) for more details on the available ML models on Radha and the HPC Fund clusters.

**Frameworks**

- Hugging Face: inference and finetuning
- vLLM: inference
- PyTorch: training and inference

**Systems**

- HPC Fund: multi-node, multi-GPU; Apptainer
- Radha: multi-node, multi-GPU; Apptainer, Docker

**Tools**

- [rocprof](https://github.com/ROCm/rocprofiler)/[rocprof-compute](https://github.com/ROCm/rocprofiler-compute/)
- [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html)
- [omnitrace](https://github.com/ROCm/omnitrace)
- [omnistat](https://github.com/AMDResearch/omnistat/)

Check the [list of supported tools](#list-of-supported-tools) for more details
about tools and their execution modes.

## Usage

Our test clusters use SLURM for job submission and management. Researchers may want to create several SLURM job scripts to experiment with
different combinations of ML model, stage, container platform, number of nodes, and profiling and analytical tools.
To this end, OmniHub provides a tool called [`omnihub-generate-job`](omnihub-generate-job) to easily generate job scripts that are tailored to be executed in a particular
environment. The `omnihub-generate-job` tool takes the target cluster characteristics
into consideration as well as user-defined options like the version
of the model, the scale of the execution, or the kind of performance data that
needs to be collected. If the target system does not use SLURM, one could still experiment with this tool
to generate reference job scripts and modify application launch commands as needed.

In its most basic form, SLURM job script generation and job execution work as follows:

```console
./omnihub-generate-job --omnihub-dir $HOME/omnihub --app-config applications/hf-infer/config-example.yaml > hf-infer.slurm
sbatch hf-infer.slurm
```

Where `--omnihub-dir` points to your working copy of the OmniHub repository in
the cluster and `--app-config` points to the path to the application configuration file relative to the OmniHub directory.

For more examples of using `omnihub-generate-job`, refer to [this document](docs/generate-job-examples.md).

### List of Command Line Options for `omnihub-generate-job`

| Flag            | Options                                   | Description                                                                              |
| :------------:  | :---------------------------------------: | :--------------------------------------------------------------------------------------- |
| `--omnihub-dir` |                                           | Path to OmniHub working copy. Should be accessible by all nodes.                         |
| `--app-config`  |                                           | Path to the application configuration file relative to OmniHub directory.                |
| `--cluster`     | **`hpcfund`**, `radha`                    | Name of the cluster.                                                                     |
| `--partition`   |                                           | Partition of the cluster; defaults to first partition in the cluster configuration file. |
| `--num-nodes`   |                                           | Number of nodes to allocate for the execution.                                           |
| `--platform`    | **`apptainer`**, `docker`                 | Container platform for the execution; supported platforms in cluster configuration file. |
| `--runner`      | `manual`, `torchrun`                      | Distributed runner. Required for multi-node executions.                                  |
| `--tools`       | [List of tools](#list-of-supported-tools) | Space-separated list of tools to use.                                                    |
| `--time-limit`  |                                           | Time limit for the SLURM job as an integer followed by a time unit.                      |

### List of Supported Tools

| Tool              | Description                                                                 |
| :---------------- | :-------------------------------------------------------------------------- |
| `rocprof-compute`        | Collect all performance counters.                                           |
| `omnistat`        | Low-overhead system metrics, sampled at 1s intervals.                       |
| `omnitrace`       | Application tracing.                                                        |
| `pytorch-stats`   | Collects detailed statistics of PyTorch operations.                         |
| `pytorch-trace`   | PyTorch execution traces compatible with TensorBoard.                       |
| `rocprofv1-stats` | Kernel execution stats.                                                     |
| `rocprofv2-pmc`   | Profiling with performance counters ([configuration](config/rocprof.txt)).  |

### Example Applications and Configuration Files

The repository includes several example applications to help you get started with different ML models and stages. Below is a list of some of the example applications provided:

- **Hugging Face Inference**: Example scripts for running inference using Hugging Face models.
- **Hugging Face Finetuning**: Example scripts for finetuning models using Hugging Face.
- **vLLM Inference**: Example scripts for running inference using vLLM models.
- **PyTorch Training**: Example scripts for training a simple CNN using PyTorch.
- **PyTorch Inference**: Example scripts for running inference using a simple CNN in PyTorch.

You can find these example applications in the `applications` directory of the repository. Each application comes with its own configuration file, which defines various settings and hyper-parameters required for the application and the chosen framework. This file typically includes entries that specify the path to the application's main entrypoint, tensor parallel size, and other configuration details.

#### Required Entry: `entrypoint`

- **Description**: The `entrypoint` field specifies the main script of the application. This script must decorate the main function, which should be first executed when the application is launched, with `omnihub.entrypoint`. Optionally, the script may decorate individual functions with `omnihub.tools.profile` to apply Omnitrace and other tools to specific code sections of the application.
- **Type**: String
- **Example**: `"entrypoint": "infer.py"`

#### Model Name or Path Entry: `ModelArguments`

- **Description**: The `ModelArguments` field specifies important fields that can be used when loading the model including the pretrained model name or path. If a model name is passed, scripts will check if it exists in `OMNIHUB_MODELS_DIR` before downloading from HuggingFace.
- **Type**: String
-  **Example**:
```
ModelArguments:
  pretrained_model_name_or_path: Meta-Llama-3.1-8B-Instruct-safetensors
```

## Sweeping Job and Application Arguments

The OmniHub sweep tool is designed to automate the generation of OmniHub jobs
and facilitate their submission to a cluster. It allows users to automatically
generate different job and application configurations: job configurations
change depending on the provided CLI flags, while application configurations
are generated based on templates.

Application templates are configuration files that allow several values
provided as lists. The OmniHub sweep tool will generate all possible
combinations of values for a given template. For example, the following
template to sweep the input and output lengths in the VLLM latency benchmark
will generate 4 configuration files with different combinations of
`input_len`/`output_len`: `32/128`, `32/256`, `64/128`, `64/256`.
```yaml
entrypoint: /app/vllm/benchmarks/benchmark_latency.py
model: Meta-Llama-3.1-8B-Instruct-safetensors
tensor_parallel_size: 1
load_format: dummy
num_iters_warmup: 5
num_iters: 15
input_len:
  - 32
  - 64
output_len:
  - 128
  - 256
batch_size: 8
dtype: auto
kv-cache-dtype: auto
output_json: latency_results.json
```

The `omnihub-sweep` CLI also allows sweeping over job-related options,
including:
- `--partitions`: cluster partitions.
- `--num-nodes`: number of nodes.
- `--tools`: profiling tools to enable (`--tools` can
  be set multiple times to enable different sets of tools).

For example, using the previously listed template stored in a file named
`vllm-latency-template.yaml`, the following `omnihub-sweep` will generate 4
configuration files, and then it will submit jobs for all of them in 2
partitions using 2 different sets of tools:
```
$ mkdir sweep-vllm
$ ./omnihub-sweep --omnihub-dir $PWD --sweep-dir ./sweep-vllm --template vllm-latency-template.yaml --partitions mi2104x mi2508x --tools omnistat --tools omnistat rocprofv3-stats
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

## Processing Results

After running jobs generated with OmniHub, the results can be processed to
extract summary information about each job in a standardized format.
Processed executions can then be combined to generate a tabular index exported
as a CSV file. Rows in the index represent different job executions, while
columns identify key indicators from different sources. Currently supported
sources for indexing are:
- Job configuration options
- Application configuration options
- Application metrics (if optional application parser is available)
- Default monitor metrics
- Omnistat report metrics

Use the following commands to process and index OmniHub executions:
```
./omnihub-process --results-dir /path/to/results/omnihub -j 4
./omnihub-index --results-dir /path/to/results/omnihub
```

After processing and indexing, a `index.csv` file should be present in the top
directory of the repository. The resulting CSV uses the first two rows as
headers, and can be loaded as a Pandas Dataframe as follows:
```
import pandas
df = pandas.read_csv("index.csv", header=[0,1], index_col=0)
```

## Developer Corner

If you want to contribute to OmniHub, make sure you read [this document](docs/developer.md) for developer pre-requisites.

## Contact

- Email: [dl.RAD-omnihub@amd.com](mailto:dl.RAD-omnihub@amd.com)
- [GitHub Discussions](https://github.com/AARInternal/omnihub/discussions)
