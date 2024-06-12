#!/bin/bash

#SBATCH --job-name=omniperf_llama 
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH -o logs/job-%j.out
#SBATCH -e logs/job-%j.err

srun hostname
sstat -j $SLURM_JOB_ID

# build docker image
bash docker/build-docker-ubuntu-dev.sh

# run docker container
docker run -itd --rm --name omnihub \
  -v $SHARED/projs/omnihub:/share -v $HOME:/host-home -w /host-home \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --shm-size 8G \
  omnihub:latest

# exec inference workload with omniperf
rel_path=`realpath -s --relative-to=$HOME $PWD`
docker exec omnihub omniperf profile -n Meta_Llama_3_8b -- ${rel_path}/scripts/hf-inference.py -p /share/ml-models/Meta-Llama-3-8B-Instruct-safetensors

# fix permissions of omniperf stats dir
docker exec omnihub fix-host-owner workloads

# stop/remove container
docker container rm omnihub -f
