#!/bin/bash

# Define the docker image to pull
docker_image="docker-virtual.atlartifactory.amd.com/amd/omnihub/radha:latest"

# Loop through each compute node and submit a SLURM job
for i in {0..6}; do
    compute_node="radha${i}"
    sbatch --nodelist="${compute_node}" -t 02:00:00 --wrap="docker pull ${docker_image}"
done
