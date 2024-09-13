# OmniHub: Tools for AI/ML Workload Analysis and Characterization

This repository contains a set of utilities (scripts, tools, container images) 
to analyze and characterize AI/ML workloads on AMD systems. The goal of this project is to characterize 
performance of different ML tasks (infer, finetune) at different system scales (single/multi-node, single/multi-GPU)
and different model granularities (kernel, op, layer, entire model) and model types (LLM, CNN, GNN).
The analysis of each workload may be done using available tools such as omniperf, omnitrace, rocprof, and so on.

We have currently tested the scripts in this repository to work with the following models, tasks, systems, and tools. 
Our scripts are very user-friendly where you can mix-and-match the models, tasks, and tools of interest with simple command line arguments.

**Models**
- Llama2 (7B, 13B, 70B)
- Llama3 (8B, 70B)

See [here](docs/models.md) for more details on the available ML models on Radha and the HPC Fund clusters.

**Tasks**
- Infer
- Finetune

**Systems**
- HPC Fund: multi-node, multi-GPU; Apptainer
- Radha: single-node, multi-GPU; Docker

**Tools**
- omniperf
- omnitrace
- omnistat

## Usage

Omnihub operates using SLURM jobs, which can be generated with the
`generate-job` tool and are tailored to be executed in a particular
environment. `generate-job` takes into consideration characteristics of the
cluster where a job will run as well as user-defined options like the version
of the model, the scale of the execution, or the kind of performance data that
needs to be collected.

In its most basic form, SLURM job generation and execution work as follows:
```console
./slurm/generate-job --omnihub-dir $HOME/omnihub > infer-single.slurm
sbatch infer-single.slurm
```
Where `--omnihub-dir` points to your working copy of the Omnihub repository in
the cluster.

| Flag            | Options                             | Description                                                                              |
| --------------- | ----------------------------------- | ---------------------------------------------------------------------------------------- |
| `--omnihub-dir` |                                     | Path to Omnihub working copy. Should be accessible by all nodes.                         |
| `--cluster`     | **`hpcfund`**, `radha`              | Name of the cluster.                                                                     |
| `--partition`   |                                     | Partition of the cluster; defaults to first partition in the cluster configuration file. |
| `--num-nodes`   |                                     | Number of nodes to allocate for the execution.                                           |
| `--model`       |                                     | Model to evaluate; defaults to first model in the cluster configuration file.            |
| `--platform`    | **`apptainer`**, `docker`           | Container platform for the execution; supported platforms in cluster configuration file. |
| `--stage`       | **`infer`**, `finetune`             | ML lifecycle stage.                                                                      |
| `--runner`      | `manual`, `torchrun`                | Distributed runner. Required for multi-node executions.                                  |
| `--profile`     | `omniperf`, `omnitrace`, `omnistat` | Space-separated list of profilers to use.                                                |


## Developer
If you want to contribute to OmniHub, make sure you read [this document](docs/developer.md) for developer pre-requisites.

## Use cases

### Multi-node Llama2 on HPC Fund

Below are some example steps you can follow to run inference or finetuning of Llama2 (7B Chat) model on the
HPC Fund cluster and use different tools to collect comprehensive performance metrics.
If the run was successful, you will find the execution logs at
`$WORK/results/omnihub/$SLURM_JOB_ID`, with which you may do further analysis (e.g., roofline
analysis, perfetto trace analysis, and GPU telemetry analysis).
More specifically, you will find the omnitrace output stats under
`$WORK/results/omnihub/$SLURM_JOB_ID/omnitrace-omnihub-hf-output`, with which you may use
[Perfetto](https://ui.perfetto.dev/) for interactive exploration. The omnistat GPU telemetry
data can be found at `$WORK/results/omnihub/$SLURM_JOB_ID/omnistat`.

#### Finetune with manual distributed execution, Omniperf
```
./slurm/generate-job --omnihub-dir $HOME/omnihub --cluster hpcfund --partition mi2104x --partition mi2108x --num-nodes 2 --model Meta-Llama-2-7B-Chat-safetensors --stage finetune --runner manual --profile omniperf
```
#### Infer via Torchrun, Omniperf 
```
./slurm/generate-job --omnihub-dir $HOME/omnihub --cluster hpcfund --partition mi2104x --num-nodes 2 --model Meta-Llama-2-7B-Chat-safetensors --stage infer --runner torchrun --profile omniperf
```
#### Infer with manual distributed execution, Omnitrace and Omnistat
```
./slurm/generate-job --omnihub-dir $HOME/omnihub --cluster hpcfund --partition mi2104x --num-nodes 2 --model Meta-Llama-2-7B-Chat-safetensors --stage infer --runner manual --profile omnitrace omnistat
```
#### Finetune with Torchrun, Omnitrace and Omnistat
```
./slurm/generate-job --omnihub-dir $HOME/omnihub --cluster hpcfund --partition mi2104x --num-nodes 2 --model Meta-Llama-2-7B-Chat-safetensors --stage finetune --runner torchrun --profile omnitrace omnistat
```

### (WIP) Single-node Llama3 on Radha

Below are the steps you can follow to run Llama3 (8B Instruct) model on radha and use different tools to collect comprehensive performance metrics. 
We assume that this repo is available in the user's `HOME` directory hierarchy.

#### Slurm batch script for Omniperf
```
./slurm/generate-job --omnihub-dir $HOME/omnihub --cluster radha --model Meta-Llama-3-8B-Instruct-safetensors --stage infer --profile omniperf
```
If the run was successful, you will find the omniperf output stats at
`$HOME/results/omnihub/$SLURM_JOB_ID/omniperf`, with which you may do further
analysis (e.g., roofline analysis).

#### Slurm batch script for Omnitrace
```
./slurm/generate-job --omnihub-dir $HOME/omnihub --cluster radha --model Meta-Llama-3-8B-Instruct-safetensors --stage infer --profile omnitrace
```
If the run was successful, you will find the omnitrace output stats under
`$HOME/results/omnihub/$SLURM_JOB_ID/omnitrace-omnihub-hf-output`, with which
you may use [Perfetto](https://ui.perfetto.dev/) for interactive exploration.
