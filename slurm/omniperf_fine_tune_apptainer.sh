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

module load rocm/6.0.2

LIB_PATH=/opt/rocm-6.0.2/lib:/opt/ohpc/pub/mpi/ucx-ohpc/1.14.0/lib:/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.5/lib:/opt/ohpc/pub/compiler/gcc/12.2.0/lib64:/opt/
srun apptainer run --rocm $WORK/omnihub.sif -c "cd $WORK/omnihub; LD_LIBRARY_PATH=$LIB_PATH slurm/run_omniperf_apptainer.bash /apptainer/conda/bin/torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --master_addr=$head_host \
    --master_port=$head_port \
    --node-rank=\$SLURM_PROCID \
    $WORK/omnihub/scripts/hf-fine-tune-ddp.py \
        --ddp -p /work1/amd/omnihub/Meta-Llama-3-8B-Instruct-safetensors/ "