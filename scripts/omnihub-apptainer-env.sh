#!/usr/bin/env bash
# Sourced inside Apptainer before omnihub-run (see omnihub/generate/job.py).
# Sets ROCm paths and a per-task pip cache directory.

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export ROCM_LIB="${ROCM_LIB:-/opt/rocm/lib}"
export LD_LIBRARY_PATH="${ROCM_LIB}:${LD_LIBRARY_PATH:-}"

export OMNIHUB_PIP_LOCAL_DIR="${SLURM_TMPDIR:-/tmp}/omnihub-pip-${SLURM_JOB_ID}-${SLURM_PROCID}"
mkdir -p "${OMNIHUB_PIP_LOCAL_DIR}/cache"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$OMNIHUB_PIP_LOCAL_DIR/cache}"
