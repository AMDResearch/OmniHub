#!/bin/bash
#SBATCH -J omnihub
#SBATCH -o %j-slurm.out
#SBATCH -N 2
#SBATCH --tasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH -p mi2104x

head_host=$(ip -f inet addr show eth0  | awk '/inet / {print $2}' | cut -d/ -f1)
head_port=29400
num_gpus=4

echo "Head node: $head_host"
echo "Head port: $head_port"
echo "Number of GPUs per node: $num_gpus"

export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export NCCL_ENABLE_DMABUF_SUPPORT=0
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0

srun apptainer run --rocm $WORK/omnihub.sif -c "cd; torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --master_addr=$head_host \
    --master_port=$head_port \
    --node-rank=\$SLURM_PROCID \
    $HOME/src/omnihub/scripts/distributed.py \
        --ddp -p /work1/amd/omnihub/ml-models/Meta-Llama-2-7B-Chat-safetensors/ \
        > $HOME/logs/$SLURM_JOB_ID-torchrun-\$SLURM_PROCID.out \
        2> $HOME/logs/$SLURM_JOB_ID-torchrun-\$SLURM_PROCID.err"
