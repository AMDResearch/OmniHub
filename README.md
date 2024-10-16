# OmniHub: Tools for AI/ML Workload Analysis and Characterization

This repository contains a set of utilities (scripts, tools, container images)
to analyze and characterize AI/ML workloads on AMD systems. The goal of this project is to characterize
performance of different stages of the ML lifecycle (inference, finetuning) at different system scales (single/multi-node, single/multi-GPU)
and different model granularities (kernel, op, layer, entire model) and model types (LLM, CNN, GNN).
The analysis of each workload may be done using available tools such as omniperf, omnitrace, rocprof, and so on.

We have currently tested the scripts in this repository to work with the following ML models, stages, systems, and tools. We have
ML models copied locally and pre-built [Docker and Apptainer images](docs/images.md) pre-installed and customized with the necessary 
tools on the below systems for your ready consumption. 
We have used [Hugging Face](https://huggingface.co/) and [vLLM](https://github.com/vllm-project/vllm) API so far in our scripts, but support for other ML frameworks could be added (talk to [us](#contact)).
Our scripts are very user-friendly where you can mix-and-match the models, tasks, and tools of interest with simple command line arguments.

**ML Models**
- Llama2 (7B, 13B, 70B)
- Llama3 (8B, 70B)
- Llama3.1 (8B, 70B, 405B)

See [here](docs/models.md) for more details on the available ML models on Radha and the HPC Fund clusters.

**Stages**
- Infer
- Finetune

**Frameworks**
- Hugging Face: inference and finetuning
- vLLM: inference

**Systems**
- HPC Fund: multi-node, multi-GPU; Apptainer
- Radha: multi-node, multi-GPU; Apptainer, Docker

**Tools**
- [omniperf](https://github.com/ROCm/omniperf)
- [omnitrace](https://github.com/ROCm/omnitrace)
- [omnistat](https://github.com/AMDResearch/omnistat/)

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
./omnihub-generate-job --omnihub-dir $HOME/omnihub > infer-single.slurm
sbatch infer-single.slurm
```
Where `--omnihub-dir` points to your working copy of the OmniHub repository in
the cluster.

### List of Command Line Options for `omnihub-generate-job`

| Flag            | Options                             | Description                                                                              |
| :------------:  | :--------------------------------:  | :--------------------------------------------------------------------------------------  |
| `--omnihub-dir` |                                     | Path to OmniHub working copy. Should be accessible by all nodes.                         |
| `--cluster`     | **`hpcfund`**, `radha`              | Name of the cluster.                                                                     |
| `--partition`   |                                     | Partition of the cluster; defaults to first partition in the cluster configuration file. |
| `--num-nodes`   |                                     | Number of nodes to allocate for the execution.                                           |
| `--model`       |                                     | Model to evaluate; defaults to first model in the cluster configuration file.            |
| `--platform`    | **`apptainer`**, `docker`           | Container platform for the execution; supported platforms in cluster configuration file. |
| `--stage`       | **`infer`**, `finetune`             | ML lifecycle stage.                                                                      |
| `--framework`   | **`hf`**, `vllm`                    | ML framework of choice.                                                                  |
| `--runner`      | `manual`, `torchrun`                | Distributed runner. Required for multi-node executions.                                  |
| `--profile`     | `omniperf`, `omnitrace`, `omnistat` | Space-separated list of profilers to use.                                                |

## Use cases

### Llama3.1 on HPC Fund

Below are some example steps you can follow to generate scripts and run inference or finetuning of Llama3.1 model on the
HPC Fund cluster and use different tools to collect comprehensive performance metrics.
If the run was successful, you will find the execution logs at
`$WORK/results/omnihub/$SLURM_JOB_ID`, with which you may do further analysis (e.g., roofline
analysis, perfetto trace analysis, and GPU telemetry analysis).
More specifically, you will find the omnitrace output stats under
`$WORK/results/omnihub/$SLURM_JOB_ID/omnitrace-omnihub-hf-output`, with which you may use
[Perfetto](https://ui.perfetto.dev/) for interactive exploration. The omnistat GPU telemetry
data can be found at `$WORK/results/omnihub/$SLURM_JOB_ID/omnistat`.
Change `$HOME/omnihub` to the installed location of OmniHub in your environment.

#### Infer Llama3.1 (405B) with a single-node execution on MI250s with Omnitrace and Omnistat
```
./omnihub-generate-job --omnihub-dir $HOME/omnihub --cluster hpcfund --partition mi2508x --model Meta-Llama-3.1-405B-Instruct-safetensors --stage infer --profile omnitrace omnistat > job.slurm
sbatch job.slurm
```
#### Finetune Llama3.1 (8B) with manual distributed execution on MI210s with Omniperf
```
./omnihub-generate-job --omnihub-dir $HOME/omnihub --cluster hpcfund --partition mi2104x --num-nodes 2 --model Meta-Llama-3.1-8B-Instruct-safetensors --stage finetune --runner manual --profile omniperf > job.slurm
sbatch job.slurm
```
#### Infer Llama3.1 (8B) via Torchrun on MI210s with Omniperf
```
./omnihub-generate-job --omnihub-dir $HOME/omnihub --cluster hpcfund --partition mi2104x --num-nodes 2 --model Meta-Llama-3.1-8B-Instruct-safetensors --stage infer --runner torchrun --profile omniperf > job.slurm
sbatch job.slurm
```
#### Finetune Llama3.1 (8B) with Torchrun on MI210s with Omnitrace and Omnistat
```
./omnihub-generate-job --omnihub-dir $HOME/omnihub --cluster hpcfund --partition mi2104x --num-nodes 2 --model Meta-Llama-3.1-8B-Instruct-safetensors --stage finetune --runner torchrun --profile omnitrace omnistat > job.slurm
sbatch job.slurm
```

### Llama3 on Radha

Below are some example steps you can follow to run Llama3 (8B Instruct) model on radha and use different tools to collect comprehensive performance metrics.

#### Infer on MI210s with Omniperf
```
./omnihub-generate-job --omnihub-dir $HOME/omnihub --cluster radha --model Meta-Llama-3-8B-Instruct-safetensors --stage infer --profile omniperf > job.slurm
sbatch job.slurm
```
If the run was successful, you will find the omniperf output stats at
`$HOME/results/omnihub/$SLURM_JOB_ID/omniperf`, with which you may do further
analysis (e.g., roofline analysis).

#### Infer on MI210s with Omnitrace
```
./omnihub-generate-job --omnihub-dir $HOME/omnihub --cluster radha --model Meta-Llama-3-8B-Instruct-safetensors --stage infer --profile omnitrace > job.slurm
sbatch job.slurm
```
If the run was successful, you will find the omnitrace output stats under
`$HOME/results/omnihub/$SLURM_JOB_ID/omnitrace-omnihub-hf-output`, with which
you may use [Perfetto](https://ui.perfetto.dev/) for interactive exploration.

## Developer Corner
If you want to contribute to OmniHub, make sure you read [this document](docs/developer.md) for developer pre-requisites.

## Contact
- Email: dl.RAD-omnihub@amd.com
- [GitHub Discussions](https://github.com/AARInternal/omnihub/discussions)
