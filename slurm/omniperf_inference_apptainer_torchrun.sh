#!/bin/bash
#SBATCH -J omnihub
#SBATCH -o %j-slurm.out
#SBATCH -N 2
#SBATCH --tasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH -p mi2104x


module load rocm/6.0.2

slurm_file=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
slurm_dir="$(dirname "$slurm_file")"

# Find omnihub directory based on the path of the slurm job we are launching.
omnihub_dir=$(builtin cd $slurm_dir/..; pwd)
shared_dir=/work1/amd/omnihub
results_dir=$WORK/results/omnihub/$SLURM_JOB_ID
model_dir=$shared_dir/ml-models/Meta-Llama-2-7B-Chat-safetensors

head_host=$(ip -f inet addr show eth0  | awk '/inet / {print $2}' | cut -d/ -f1)
head_port=29400
num_gpus=4

mkdir -p $results_dir

echo "Omnihub directory: $omnihub_dir"
echo "Shared directory: $shared_dir"
echo "Results directory: $results_dir"
echo "Model directory: $model_dir"
echo "Head node: $head_host"
echo "Head port: $head_port"
echo "Number of GPUs per node: $num_gpus"

export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export NCCL_ENABLE_DMABUF_SUPPORT=0
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# srun to launch one apptainer task per node; inside of the container, launch
# one process per GPU using torchrun.
srun \
    apptainer run --rocm \
        $WORK/omnihub.sif -c " \
        $omnihub_dir/slurm/run_omniperf_apptainer.bash \
        /apptainer/conda/bin/torchrun \
            --nnodes=$SLURM_JOB_NUM_NODES \
            --nproc_per_node=$num_gpus \
            --master_addr=$head_host \
            --master_port=$head_port \
            --node-rank=\$SLURM_PROCID \
            --log-dir $results_dir \
            --redirect 3 \
            $omnihub_dir/scripts/hf-fine-tune-ddp-torchrun.py \
            --ddp -p $model_dir --output=$results_dir"
