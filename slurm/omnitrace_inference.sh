#!/bin/bash

#SBATCH --job-name=omnitrace_llama
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

# exec inference workload with omnitrace
rel_path=`realpath -s --relative-to=$HOME $PWD`
docker exec omnihub ${rel_path}/scripts/hf-inference.py --omnitrace -p /share/ml-models/Meta-Llama-3-8B-Instruct-safetensors

# copy perfetto trace, which is unexpectedly stored in the root directory
docker exec omnihub bash -c 'cp -r /omnitrace* /host-home/'
# fix permissions of omnitrace output dir
docker exec omnihub fix-host-owner omnitrace-hf-inference-output

# stop/remove container
docker container rm omnihub -f
