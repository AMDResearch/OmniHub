#!/usr/bin/env python3
"""Minimal NCCL smoke test for OmniHub jobs (run under torchrun)."""

import os
import sys

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        print("OmniHub network sanity: CUDA not available", file=sys.stderr)
        sys.exit(1)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    try:
        t = torch.ones(4, device="cuda", dtype=torch.float32)
        dist.broadcast(t, src=0)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        ws = dist.get_world_size()
        expected = float(ws)
        if not torch.allclose(t, torch.full_like(t, expected)):
            print(
                f"OmniHub network sanity: tensor mismatch after all_reduce: got {t} expected {expected}",
                file=sys.stderr,
            )
            sys.exit(1)
        if dist.get_rank() == 0:
            print(
                f"OmniHub network sanity: PyTorch NCCL broadcast+all_reduce OK (world_size={ws})"
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
