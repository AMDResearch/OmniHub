#!/bin/bash
#SBATCH -J omnihub
#SBATCH -o %j-slurm.out
#SBATCH -e %j-slurm.err
#SBATCH -N 2
#SBATCH --tasks-per-node=4
#SBATCH -t 01:00:00
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

# Set omnitrace env vars
export OMNITRACE_OUTPUT_PATH=$results_dir/omnitrace-%tag%-output

# srun to launch one apptainer task per GPU in each node; each apptainer
# container then executes a single python process.
srun \
    -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*4)) \
    apptainer run --rocm \
        $shared_dir/apptainer/omnihub.sif -c " \
        \$CONDA_DIR/bin/python $omnihub_dir/scripts/hf-fine-tune-dist.py \
            --model-dir=$model_dir \
            --output-dir=$results_dir \
            --ddp \
            --omnitrace \
            --manual-runner \
            --master-addr=$head_host \
            --master-port=$head_port \
            --rank=\$SLURM_PROCID \
            --world-size=\$SLURM_NTASKS \
            > $results_dir/srun-\$SLURM_PROCID.out \
            2> $results_dir/srun-\$SLURM_PROCID.err"
