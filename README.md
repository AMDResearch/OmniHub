# OmniHub: Tools for AI/ML Workload Analysis and Characterization

This repository contains a set of utilities (scripts, tools, container images) 
to analyze and characterize AI/ML workloads on AMD systems. The goal of this project is to characterize 
performance at different system scales (single/multi-node, single/multi-GPU)
and different model granularities (kernel, op, layer, entire model) and model types (LLM, CNN, GNN).
The analysis of each workload may be done using available tools such as omniperf, omnitrace, rocprof, and so on.

## Current status

We have currently tested the scripts in this repository to work with the following models, systems, and tools.

**Models**
- Llama2 (7B, 13B, 70B)
- Llama3 (8B, 70B)

**Systems**
- Radha (single-node, multi-GPU)

**Tools**
- omniperf

## Use case: Omniperf and Llama3 on Radha

Below are the steps you can follow to run Llama3 (8B Instruct) model on radha and use omniperf to collect comprehensive performance metrics. 
If the run was successful, you will find the omniperf output stats at `~/workloads`, with which you may do further analysis (e.g., roofline analysis).
We assume that this repo is available in the user's `HOME` directory hierarchy.

### Slurm batch script
```
sbatch slurm/omniperf_inference.sh
```

### Manual

#### Interactively login
```
salloc -N 1
```

#### Build docker image
```
bash docker/build-docker-ubuntu-dev.sh
```

#### Run docker container
```
docker run -itd --rm --name omnihub \
  -v $SHARED/projs/omnihub:/share -v $HOME:/host-home -w /host-home \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --shm-size 8G \
  omnihub:latest
```

#### Exec inference task with omniperf
```
rel_path=`realpath -s --relative-to=$HOME $PWD`
docker exec omnihub omniperf profile -n Meta_Llama_3_8b -- \
  ${rel_path}/scripts/hf-inference.py -p /share/ml-models/Meta-Llama-3-8B-Instruct-safetensors
```

#### Stop/remove container
```
docker container rm omnihub -f
```

### Models
See [here](docs/models.md) for more details on the available ML models on radha.