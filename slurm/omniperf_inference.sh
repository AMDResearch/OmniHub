#!/bin/bash

#SBATCH --job-name=profile_LLM_omniperf 
#SBATCH --gpus-per-node=2 
#SBATCH --nodes=1

srun hostname
sstat -j $SLURM_JOB_ID


docker build -f docker/Dockerfile-ubuntu-dev   -t pachyderm:latest   --build-arg UID=$(id -u)   --build-arg GID=$(id -g)   --build-arg RENDER_GID=$(getent group render | cut -d: -f3)   --build-arg VIDEO_GID=$(getent group video | cut -d: -f3)   .
docker run -itd --rm --name pachyderm -v $SHARED/$USER:/share -v $SHARED/aaji:/share/aaji -v $HOME:/host-home -w /host-home --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --shm-size 8G \
  pachyderm:latest

docker exec pachyderm omniperf profile -n Meta_Llama_2_70b -- ./pachyderm/llama-hf/inference.py -p /share/aaji/ml-models/meta-llama/Llama-2-70b-chat-hf/
docker container rm pachyderm -f




