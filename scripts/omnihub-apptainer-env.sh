#!/usr/bin/env bash
# Sourced inside Apptainer before omnihub-run (see omnihub/generate/job.py).
# Sets ROCm paths, handles cluster-specific library overrides, and configures
# a per-task pip cache directory.

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export ROCM_LIB="${ROCM_LIB:-/opt/rocm/lib}"

# Disable conflicting container libraries listed in OMNIHUB_DISABLE_LIBS
# (colon-separated glob patterns, set via cluster YAML "container-disable-libs").
# Renaming happens in the writable overlay so the base image is untouched.
if [ -n "${OMNIHUB_DISABLE_LIBS:-}" ]; then
    IFS=':' read -ra _patterns <<< "${OMNIHUB_DISABLE_LIBS}"
    for _pat in "${_patterns[@]}"; do
        for _lib in $_pat; do
            [ -e "$_lib" ] && mv "$_lib" "${_lib}.disabled" 2>/dev/null || true
        done
    done
    unset _patterns _pat _lib
fi

# Strip /opt/ompi/lib from LD_LIBRARY_PATH. Some container images ship a
# broken OpenMPI 5.x build at /opt/ompi whose libmpi.so.40 is linked against the
# wrong libopen-pal SONAME (4.x instead of 5.x). Keeping it on the path causes
# "undefined symbol: opal_smsc_base_framework" when PyTorch loads MPI.
LD_LIBRARY_PATH="${LD_LIBRARY_PATH//:\/opt\/ompi\/lib/}"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH#/opt/ompi/lib:}"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH#/opt/ompi/lib}"

export LD_LIBRARY_PATH="${ROCM_LIB}:${LD_LIBRARY_PATH:-}"

# Prepend cluster-specific library paths (set via cluster YAML
# "container-ld-library-path", propagated as OMNIHUB_EXTRA_LD_LIBRARY_PATH).
if [ -n "${OMNIHUB_EXTRA_LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="${OMNIHUB_EXTRA_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
fi

# Set LD_PRELOAD for cluster-specific library overrides (set via cluster YAML
# "container-ld-preload", propagated as OMNIHUB_EXTRA_LD_PRELOAD).
if [ -n "${OMNIHUB_EXTRA_LD_PRELOAD:-}" ]; then
    export LD_PRELOAD="${OMNIHUB_EXTRA_LD_PRELOAD}"
fi

# Auto-detect a routable network interface for NCCL/Gloo OOB sockets.
# Some nodes expose a link-local BMC/IPMI NIC (169.254.x.x) that may appear
# first in `hostname -I`. Link-local addresses are identical across nodes and
# not routable, so multi-node NCCL bootstrap fails.
# Pick the first non-loopback, non-link-local IPv4 interface.
if [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
    _omnihub_ifname=""
    if command -v ip >/dev/null 2>&1; then
        _omnihub_ifname=$(ip -o -4 addr show \
            | awk '$2 != "lo" && $4 !~ /^169\.254\./ && $4 !~ /^127\./ {print $2; exit}')
    else
        # Fallback: prefer high-speed network interfaces, then any non-lo up interface.
        for _d in /sys/class/net/hsn* /sys/class/net/*; do
            [ -d "$_d" ] || continue
            _n=$(basename "$_d")
            [ "$_n" = "lo" ] && continue
            [ -f "$_d/operstate" ] && [ "$(cat "$_d/operstate")" = "up" ] || continue
            _omnihub_ifname="$_n"
            break
        done
        unset _d _n
    fi
    if [ -n "${_omnihub_ifname}" ]; then
        export NCCL_SOCKET_IFNAME="${_omnihub_ifname}"
        export GLOO_SOCKET_IFNAME="${_omnihub_ifname}"
    fi
    unset _omnihub_ifname
fi

# MIOpen kernel cache: write to node-local NVMe to avoid Lustre I/O issues.
export MIOPEN_DISABLE_CACHE="${MIOPEN_DISABLE_CACHE:-1}"
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-/tmp/${SLURM_JOB_ID:-miopen}}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR:-${MIOPEN_USER_DB_PATH}}"
mkdir -p "${MIOPEN_USER_DB_PATH}" 2>/dev/null || true

export OMNIHUB_PIP_LOCAL_DIR="${SLURM_TMPDIR:-/tmp}/omnihub-pip-${SLURM_JOB_ID}-${SLURM_PROCID}"
mkdir -p "${OMNIHUB_PIP_LOCAL_DIR}/cache"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$OMNIHUB_PIP_LOCAL_DIR/cache}"
