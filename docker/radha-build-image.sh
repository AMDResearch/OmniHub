#!/bin/bash
#SBATCH --job-name=docker-build-push
#SBATCH --output=docker-build-push-%j.out
#SBATCH --error=docker-build-push-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --partition=radhadefault

# Build Docker image
slurm_file=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
slurm_dir="$(dirname "$slurm_file")"
docker build -f ${slurm_dir}/Dockerfile-ubuntu-dev -t omnihub:latest ${slurm_dir}/..

# Get the tag of the newly created Docker image
image_tag=$(docker images -q omnihub:latest)

# Push the tag to the Docker registry
docker tag $image_tag docker-virtual.atlartifactory.amd.com/amd/omnihub/radha:latest
docker push docker-virtual.atlartifactory.amd.com/amd/omnihub/radha:latest
