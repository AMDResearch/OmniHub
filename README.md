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

OmniHub operates using SLURM scripts, which you can submit to a compute environment using `sbatch`. Your OmniHub command may vary depending on the system you are using, how you would like to execute the model, and the analysis data that you wish to collect.

```console
sbatch slurm/<container>/<dist-execution-mode> -M <model-operation> [analysis options]
```

  - **Containerization:**
    - Docker: `sbatch slurm/docker/<dist-execution-mode>`
    - Apptainer/Singularity: `sbatch slurm/apptainer/<dist-execution-mode>`

  - **Distributed Execution Mode:**
    - Manual: `sbatch slurm/<container>/manual.slurm`
    - Torchrun: `sbatch slurm/<container>/torchrun.slurm`

  - **Model Task:**
    - Infer: `sbatch slurm/<container>/<dist-execution-mode> -M infer`
    - Finetune:  `sbatch slurm/<container>/<dist-execution-mode> -M finetune`

To collect analyitic data from your run, explore these supported analysis options:

  - **Analysis Options**
    - Use [Omniperf](https://github.com/ROCm/omniperf): `-p`
    - Use [Omnitrace](https://github.com/ROCm/omnitrace): `-t`
    - Use [Omnistat](https://github.com/AMDResearch/omnistat): `-s`


## Use cases

### Multi-node Llama2 on HPC Fund

Below are some example steps you can follow to run inference or finetuning of Llama2 (7B Chat) model on the
HPC Fund cluster and use different tools to collect comprehensive performance metrics.
If the run was successful, you will find the execution logs at
`$WORK/results/omnihub/$SLURM_JOB_ID`, with which you may do further analysis (e.g., roofline
analysis, perfetto trace analysis, and GPU telemetry analysis).
More specifically, you will find the omnitrace output stats under
`$WORK/results/omnihub/$SLURM_JOB_ID/omnitrace-omnihub-output`, with which you may use
[Perfetto](https://ui.perfetto.dev/) for interactive exploration. The omnistat GPU telemetry
data can be found at `$WORK/results/omnihub/$SLURM_JOB_ID/omnistat`.

#### Finetune with manual distributed execution, Omniperf
```
sbatch slurm/apptainer/manual.slurm -p -M finetune
```
#### Infer via Torchrun, Omniperf 
```
sbatch slurm/apptainer/torchrun.slurm -p -M infer
```
#### Infer with manual distributed execution, Omnitrace and Omnistat
```
sbatch slurm/apptainer/manual.slurm -t -s -M infer
```
#### Finetune with Torchrun, Omnitrace and Omnistat
```
sbatch slurm/apptainer/torchrun.slurm -t -s -M finetune
```

### (WIP) Single-node Llama3 on Radha

Below are the steps you can follow to run Llama3 (8B Instruct) model on radha and use different tools to collect comprehensive performance metrics. 
We assume that this repo is available in the user's `HOME` directory hierarchy.

#### Slurm batch script for Omniperf
```
sbatch slurm/omniperf_inference.sh
```
If the run was successful, you will find the omniperf output stats at
`~/workloads`, with which you may do further analysis (e.g., roofline
analysis).

#### Slurm batch script for Omnitrace
```
sbatch slurm/omnitrace_inference.sh
```
If the run was successful, you will find the omnitrace output stats under
`~/omnitrace-hf-inference-output`, with which you may use
[Perfetto](https://ui.perfetto.dev/) for interactive exploration.

#### Manual

##### Interactively login
```
salloc -N 1
```

##### Build docker image
```
bash docker/build-docker-ubuntu-dev.sh
```

##### Run docker container
```
docker run -itd --rm --name omnihub \
  -v $SHARED/projs/omnihub:/share -v $HOME:/host-home -w /host-home \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --shm-size 8G \
  omnihub:latest
```

##### Exec inference task with omniperf
```
rel_path=`realpath -s --relative-to=$HOME $PWD`
docker exec omnihub omniperf profile -n Meta_Llama_3_8b -- \
  ${rel_path}/scripts/hf-inference.py -p /share/ml-models/Meta-Llama-3-8B-Instruct-safetensors
```

##### Stop/remove container
```
docker container rm omnihub -f
```
