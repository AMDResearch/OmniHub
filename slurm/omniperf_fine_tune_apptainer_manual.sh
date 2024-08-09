#!/bin/bash
#SBATCH -J omnihub
#SBATCH -o %j-slurm.out
#SBATCH -N 2
#SBATCH --tasks-per-node=4
#SBATCH -t 08:00:00
#SBATCH -p mi2104x

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

module load rocm/6.0.2

LIB_PATH=/opt/rocm-6.0.0/lib:/opt/ohpc/pub/mpi/ucx-ohpc/1.14.0/lib:/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.5/lib:/opt/ohpc/pub/compiler/gcc/12.2.0/lib64:/opt/

# srun to launch one apptainer task per GPU in each node; each apptainer
# container then executes omniperf to profile each process separately.
srun \
    apptainer run --rocm \
        $shared_dir/apptainer/omnihub-mpi.sif -c "cd $results_dir; \
        LD_LIBRARY_PATH=$LIB_PATH $omnihub_dir/slurm/run_omniperf_apptainer.bash \
        /apptainer/conda/bin/python $omnihub_dir/scripts/hf-fine-tune-dist.py \
            --model-dir=$model_dir \
            --output-dir=$results_dir \
            --ddp \
            --manual-runner \
            --master_addr=$head_host \
            --master_port=$head_port \
            --rank=\$SLURM_PROCID \
            --world_size=$SLURM_NTASKS \
            > $results_dir/srun-\$SLURM_PROCID.out \
            2> $results_dir/srun-\$SLURM_PROCID.err"
