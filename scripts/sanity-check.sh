#!/usr/bin/env bash
# Consolidated sanity checks for OmniHub jobs.
# Invoked from job.template with env: head_host, head_port, apptainer_image,
# omnihub_dir, num_gpus_per_node, apptainer_bind_args.
#
# Checks (in order):
#   1. ROCm availability (rocminfo) on all nodes
#   2. GPU compute-mode (rocm-smi) — warns if exclusive partition detected
#   3. PyTorch NCCL broadcast + all_reduce (one srun task per GPU)
set -euo pipefail

: "${head_host:?}" "${head_port:?}" "${apptainer_image:?}" "${omnihub_dir:?}" "${num_gpus_per_node:?}"

NNODES="${SLURM_NNODES:-1}"
ENV_SCRIPT="${omnihub_dir}/scripts/omnihub-apptainer-env.sh"
PY_SCRIPT="${omnihub_dir}/scripts/sanity_torch_dist.py"
BIND_ARGS="${apptainer_bind_args:-}"

# --- 1. ROCm check ---
echo "OmniHub sanity: checking ROCm (rocminfo) on ${NNODES} node(s)"
srun -N"${NNODES}" -n"${NNODES}" apptainer exec "${apptainer_image}" \
  /bin/bash -c "rocminfo > /dev/null 2>&1"

# --- 2. GPU compute-mode check (non-fatal) ---
echo "OmniHub sanity: checking GPU compute mode on ${NNODES} node(s)"
srun -N"${NNODES}" -n"${NNODES}" apptainer exec "${apptainer_image}" /bin/bash -c '
set -euo pipefail
if ! command -v rocm-smi >/dev/null 2>&1; then exit 0; fi
output=$(rocm-smi --showcomputepartition 2>/dev/null) || exit 0
ok=true
while IFS= read -r line; do
  lower="${line,,}"
  case "${lower}" in *spx*|*cpx*|*tpx*|*dpx*) continue ;; esac
  if [[ "${lower}" == *exclusive* ]]; then
    echo "WARNING: AMD GPU compute partition may restrict concurrent access: ${line}" >&2
    ok=false
  fi
done <<< "${output}"
if $ok; then
  echo "GPU compute mode check: OK (no exclusive modes detected)"
fi
' || echo "Warning: GPU compute-mode check reported issues (see above)"

# --- 3. PyTorch NCCL check ---
# Uses one srun task per GPU — mirrors the main training job's process model.
# Each task gets its own Apptainer container (and overlay when bind mounts are
# needed), then runs sanity_torch_dist.py directly with env-var-based
# dist.init_process_group.  No torchrun involved.

NTASKS=$(( NNODES * num_gpus_per_node ))
echo "OmniHub sanity: PyTorch NCCL (master ${head_host}:${head_port}, nnodes=${NNODES}, gpus_per_node=${num_gpus_per_node})"

_nccl_launcher=$(mktemp -p "${omnihub_dir}" .sanity-nccl-XXXXXX.sh)
trap 'rm -f "${_nccl_launcher}"' EXIT

if [[ -n "${BIND_ARGS}" ]]; then
  cat > "${_nccl_launcher}" << NCCL_EOF
#!/bin/bash
set -euo pipefail
mkdir -p /tmp/omnihub-sanity-overlay.\${SLURM_PROCID}/{upper,work}
apptainer exec --overlay /tmp/omnihub-sanity-overlay.\${SLURM_PROCID} ${BIND_ARGS} ${apptainer_image} /bin/bash -c "source ${ENV_SCRIPT}; RANK=\${SLURM_PROCID} LOCAL_RANK=\$((\${SLURM_PROCID} % ${num_gpus_per_node})) WORLD_SIZE=${NTASKS} MASTER_ADDR='${head_host}' MASTER_PORT='${head_port}' exec python3 '${PY_SCRIPT}'"
NCCL_EOF
else
  cat > "${_nccl_launcher}" << NCCL_EOF
#!/bin/bash
set -euo pipefail
apptainer exec ${apptainer_image} /bin/bash -c "source ${ENV_SCRIPT}; RANK=\${SLURM_PROCID} LOCAL_RANK=\$((\${SLURM_PROCID} % ${num_gpus_per_node})) WORLD_SIZE=${NTASKS} MASTER_ADDR='${head_host}' MASTER_PORT='${head_port}' exec python3 '${PY_SCRIPT}'"
NCCL_EOF
fi

chmod +x "${_nccl_launcher}"
srun -N"${NNODES}" -n"${NTASKS}" "${_nccl_launcher}"

if [[ -n "${BIND_ARGS}" ]]; then
  rm -rf /tmp/omnihub-sanity-overlay.* 2>/dev/null || true
fi

echo "OmniHub sanity: done"
