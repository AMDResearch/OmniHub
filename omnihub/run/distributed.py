import os
import sys

import torch
from torch import distributed as dist

# Define _is_initialized as a global variable
_is_initialized = False


class Distributed:
    def __init__(self, args):
        global _is_initialized
        self.manual_launch_ddp = args.manual_launch_ddp
        if self.manual_launch_ddp:
            local_rank = args.rank % torch.cuda.device_count()
            torch.cuda.set_device(local_rank)
            # Set LOCAL_RANK *BEFORE* initializing the torch distributed process group. HF's PartialState logic needs the LOCAL_RANK env
            # var to ensure that the distribute type is MULTI_GPU. Moreover, this variable is also set by torchrun, and we explicitly set
            # if here to reuse this setup code for any type of distributed launcher (torchrun, accelerate, manual).
            os.environ["LOCAL_RANK"] = str(local_rank)

            # Set DDP-specific env variables *BEFORE* correctly initializing the torch distributed process group
            os.environ["RANK"] = str(args.rank)
            os.environ["WORLD_SIZE"] = str(args.world_size)
            os.environ["MASTER_ADDR"] = str(args.master_addr)
            os.environ["MASTER_PORT"] = str(args.master_port)

            dist.init_process_group(backend="nccl", init_method="env://")

            print(f"Manual DDP launcher")
            print(f"Number of tasks: {args.world_size}")
            print(f"Rank: {args.rank}")
            print(f"Local rank: {local_rank}")
            print(f"Using device: {torch.cuda.current_device()}")

        if all(
            var in os.environ
            for var in [
                "RANK",
                "WORLD_SIZE",
                "MASTER_ADDR",
                "MASTER_PORT",
                "LOCAL_RANK",
            ]
        ):
            _is_initialized = True

    def finalize(self):
        if self.manual_launch_ddp:
            dist.destroy_process_group()


def is_initialized():
    global _is_initialized
    return _is_initialized
