import os
import sys

import torch
import torch.distributed as dist


def is_manual_ddp(*manual_args):
    """Checks if all arguments needed for manual distributed execution are set (i.e., are not None)."""
    return all(arg is not None for arg in manual_args)


class Distributed:
    def __init__(self, args):
        self.manual_ddp = is_manual_ddp(
            args.master_addr, args.master_port, args.rank, args.world_size
        )
        if not self.manual_ddp:
            return

        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{args.master_addr}:{args.master_port}",
            rank=args.rank,
            world_size=args.world_size,
        )

        local_rank = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)

        print(f"Manual distributed runner")
        print(f"Number of tasks: {args.world_size}")
        print(f"Rank: {args.rank}")
        print(f"Local rank: {local_rank}")
        print(f"Using device: {torch.cuda.current_device()}")

        # Set LOCAL_RANK: this variable is also set by torchrun, and we do the
        # same in the manual runner to replicate the same behaviour and keep the
        # remaining code the same.
        os.environ["LOCAL_RANK"] = str(local_rank)

    def finalize(self):
        if self.manual_ddp:
            dist.destroy_process_group()
