#!/bin/bash
#SBATCH --job-name=apptainer-build-push
#SBATCH --output=apptainer-build-push-%j.out
#SBATCH --error=apptainer-build-push-%j.err
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --partition=mi2104x

set -e

# Define the paths
slurm_file=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
slurm_dir="$(dirname "$slurm_file")"
DEFINITION_FILE="${slurm_dir}/omnihub-ubuntu-dev.def"
OUTPUT_IMAGE="$WORK/omnihub_temp.sif"
FINAL_DESTINATION="/work1/amd/omnihub/apptainer/omnihub.sif"

# Build the Apptainer image
apptainer build --force $OUTPUT_IMAGE $DEFINITION_FILE

# Move the image to the final destination
mv $OUTPUT_IMAGE $FINAL_DESTINATION

# Verify the image build
if [ $? -eq 0 ]; then
    echo "Apptainer image built successfully!"
else
    echo "Failed to build the Apptainer image."
fi
