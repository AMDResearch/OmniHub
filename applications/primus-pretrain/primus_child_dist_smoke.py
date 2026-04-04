#!/usr/bin/env python3
"""Minimal distributed-init smoke test matching the Primus child environment.

Run inside a subprocess launched by the dist-init smoke entrypoint (or directly
via ``torchrun``).  Logs the distributed environment variables that matter for
debugging OmniHub + Primus multinode issues, then performs a single
``init_process_group`` + broadcast + all_reduce and exits.

This intentionally mirrors what Primus's inner torchrun workers do at startup
(through Megatron's ``initialize_megatron`` -> ``init_process_group(env://)``)
without importing any Primus or Megatron code.

Compare results with ``scripts/sanity_torch_dist.py`` (run by
``sanity-check.sh``) to isolate whether the problem is in the
wrapper-supplied env or deeper inside Primus.
"""

import os
import sys

_DIST_ENV_VARS = [
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "NNODES",
    "NODE_RANK",
    "GPUS_PER_NODE",
    "RUN_MODE",
    "DRY_RUN_MODE",
]

_SLURM_PREFIXES = (
    "SLURM_NNODES",
    "SLURM_NODEID",
    "SLURM_PROCID",
    "SLURM_JOB_NUM_NODES",
    "SLURM_NTASKS",
    "SLURM_NODELIST",
    "SLURM_JOB_ID",
)

_PRIMUS_PREFIXES = ("PRIMUS_PATH", "PRIMUS_LOG_LEVEL", "PRIMUS_WORKSPACE", "EXP")


def _dump_env() -> None:
    """Print all relevant env vars for debugging."""
    print("=" * 60)
    print("primus_child_dist_smoke — environment snapshot")
    print("=" * 60)
    for var in _DIST_ENV_VARS:
        val = os.environ.get(var)
        print(f"  {var} = {val!r}")
    for var in _SLURM_PREFIXES:
        val = os.environ.get(var)
        if val is not None:
            print(f"  {var} = {val!r}")
    for var in _PRIMUS_PREFIXES:
        val = os.environ.get(var)
        if val is not None:
            print(f"  {var} = {val!r}")
    print("=" * 60)
    sys.stdout.flush()


def main() -> None:
    _dump_env()

    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        print("primus_child_dist_smoke: CUDA not available", file=sys.stderr)
        sys.exit(1)

    torch.cuda.set_device(local_rank)

    print(
        f"primus_child_dist_smoke: calling init_process_group "
        f"(MASTER_ADDR={os.environ.get('MASTER_ADDR')}, "
        f"MASTER_PORT={os.environ.get('MASTER_PORT')}, "
        f"RANK={os.environ.get('RANK')}, "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}, "
        f"LOCAL_RANK={local_rank})",
        flush=True,
    )

    dist.init_process_group(backend="nccl", init_method="env://")
    try:
        t = torch.ones(4, device="cuda", dtype=torch.float32)
        dist.broadcast(t, src=0)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        ws = dist.get_world_size()
        expected = float(ws)
        if not torch.allclose(t, torch.full_like(t, expected)):
            print(
                f"primus_child_dist_smoke: tensor mismatch after all_reduce: "
                f"got {t} expected {expected}",
                file=sys.stderr,
            )
            sys.exit(1)
        if dist.get_rank() == 0:
            print(
                f"primus_child_dist_smoke: broadcast+all_reduce OK "
                f"(world_size={ws}, port={os.environ.get('MASTER_PORT')})"
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
