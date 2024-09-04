#!/usr/bin/bash
x=$1
export LD_LIBRARY_PATH=/opt/ohpc/pub/mpi/ucx-ohpc/1.14.0/lib:/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.5/lib:/opt/ohpc/pub/compiler/gcc/12.2.0/lib64:/opt/rocm/lib
export ROCM_PATH=/opt/rocm
/opt/omniperf/bin/omniperf profile -n omnihub_perf_${SLURM_PROCID} -- "$@"