import os
import sys

import torch
from accelerate import PartialState


class Distributed:
    def __init__(self, args):
        self.manual_launch_ddp = args.manual_launch_ddp
        if self.manual_launch_ddp:
            local_rank = args.rank % torch.cuda.device_count()
            torch.cuda.set_device(local_rank)
            # Set LOCAL_RANK *BEFORE* calling the PartialState singleton object. The current PartialState logic needs the LOCAL_RANK env
            # var to ensure that the distribute type is MULTI_GPU. Moreover, this variable is also set by torchrun, and we explicitly set
            # if here to reuse this setup code for any type of distributed launcher (torchrun, accelerate, manual).
            os.environ["LOCAL_RANK"] = str(local_rank)

            # Set DDP-specific env variables and the first invocation of PartialState singleton object will correctly initialize the torch distributed process group
            os.environ["RANK"] = str(args.rank)
            os.environ["WORLD_SIZE"] = str(args.world_size)
            os.environ["MASTER_ADDR"] = str(args.master_addr)
            os.environ["MASTER_PORT"] = str(args.master_port)

            print(f"Manual DDP launcher")
            print(f"Number of tasks: {args.world_size}")
            print(f"Rank: {args.rank}")
            print(f"Local rank: {local_rank}")
            print(f"Using device: {torch.cuda.current_device()}")

        # First call to `PartialState()` internally initializes torch DDP with NCCL backend, so it is
        # equivalent to calling torch.distributed.init_process_group(backend="nccl", init_method='env://').
        # The singleton object is initialized JIT as needed by the specific task and does not have to be
        # initialized here.
        # self.state = PartialState()

    def finalize(self):
        if self.manual_launch_ddp:
            # no explicit way to destroy process group in this version of PartialState (but available in a later version)
            # self.state.destroy_process_group()
            pass
