#!/usr/bin/bash
x=$1
export LD_LIBRARY_PATH=/opt/ohpc/pub/mpi/ucx-ohpc/1.14.0/lib:/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.5/lib:/opt/ohpc/pub/compiler/gcc/12.2.0/lib64:/opt/rocm-6.0.2/lib

ROCM_PATH=/opt/rocm-6.0.2 LD_LIBRARY_PATH=/opt/ohpc/pub/mpi/ucx-ohpc/1.14.0/lib:/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.5/lib:/opt/ohpc/pub/compiler/gcc/12.2.0/lib64:/opt/rocm-6.0.2/lib /apptainer/omniperf/bin/omniperf profile -n llama3_8_2n_${SLURM_PROCID} -- "$@"