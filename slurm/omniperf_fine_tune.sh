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
  -v $SHARED/aaji:/share -v $HOME:/host-home -w /host-home \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --shm-size 8G \
  omnihub:latest

# exec inference workload with omniperf
rel_path=`realpath -s --relative-to=$HOME $PWD`
docker exec omnihub omniperf profile -n Meta_Llama_2_13b -- ${rel_path}/llama-hf/fine-tune.py -p /share/ml-models/meta-llama/Llama-2-13b-chat-hf/

# stop/remove container
docker container rm omnihub -f
