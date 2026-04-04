#!/usr/bin/env bash
# Consolidated sanity checks for OmniHub jobs.
# Invoked from job.template with env: head_host, head_port, apptainer_image,
# omnihub_dir, num_gpus_per_node.
#
# Checks (in order):
#   1. ROCm availability (rocminfo) on all nodes
#   2. GPU compute-mode (rocm-smi) — warns if exclusive partition detected
#   3. PyTorch NCCL broadcast + all_reduce via torchrun
set -euo pipefail

: "${head_host:?}" "${head_port:?}" "${apptainer_image:?}" "${omnihub_dir:?}" "${num_gpus_per_node:?}"

NNODES="${SLURM_NNODES:-1}"
ENV_SCRIPT="${omnihub_dir}/scripts/omnihub-apptainer-env.sh"
PY_SCRIPT="${omnihub_dir}/scripts/sanity_torch_dist.py"

# --- 1. ROCm check ---
echo "OmniHub sanity: checking ROCm (rocminfo) on ${NNODES} node(s)"
srun -N"${NNODES}" -n"${NNODES}" apptainer exec "${apptainer_image}" \
  /bin/bash -c "rocminfo > /dev/null 2>&1"

# --- 2. GPU compute-mode check (non-fatal) ---
# Warn if any GPU reports an exclusive compute partition that would block
# concurrent parent + child device access (relevant to --runner manual).
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
echo "OmniHub sanity: PyTorch NCCL (master ${head_host}:${head_port}, nnodes=${NNODES}, gpus_per_node=${num_gpus_per_node})"

if [[ "${NNODES}" -gt 1 ]]; then
  srun -N"${NNODES}" -n"${NNODES}" apptainer exec "${apptainer_image}" /bin/bash -c "
set -euo pipefail
source ${ENV_SCRIPT}
exec torchrun --nnodes=\${SLURM_JOB_NUM_NODES} --nproc_per_node=${num_gpus_per_node} --node_rank=\${SLURM_PROCID} \
  --master_addr='${head_host}' --master_port='${head_port}' \
  '${PY_SCRIPT}'
"
else
  srun -N1 -n1 apptainer exec "${apptainer_image}" /bin/bash -c "
set -euo pipefail
source ${ENV_SCRIPT}
exec torchrun --nnodes=1 --nproc_per_node=${num_gpus_per_node} \
  --master_addr='${head_host}' --master_port='${head_port}' \
  '${PY_SCRIPT}'
"
fi

echo "OmniHub sanity: done"
