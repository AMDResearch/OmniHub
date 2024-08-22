#!/usr/bin/env python3

import argparse
import os
import sys

import torch
import torch.distributed as dist

from omnihub import finetune
from omnihub import tracer

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model-dir", help="Path to model", type=str, required=True)
parser.add_argument("-o", "--output-dir", help="Path to store output", type=str, default=".")
parser.add_argument("--ddp", help="Run with DDP", action="store_true")
parser.add_argument("--manual-runner", help="Type of distributed execution runner", action="store_true")

# Optional arguments for manual distributed execution
parser.add_argument("--master-addr", type=str)
parser.add_argument("--master-port", type=str)
parser.add_argument("--rank", type=int)
parser.add_argument("--world-size", type=int)
parser.add_argument("--omnitrace", action="store_true", help="Enable omnitrace")

args = parser.parse_args()

if args.manual_runner:
    if None in [args.master_addr, args.master_port, args.rank, args.world_size]:
        print("Missing arguments for manual runner")
        sys.exit(1)

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
    os.environ['LOCAL_RANK'] = str(local_rank)

setup = finetune.setup_finetune(args)
model, train_dataset, peft_config, tokenizer, train_args = setup

with tracer.profile(use_omnitrace=args.omnitrace):
    finetune.run_finetune(model, train_dataset, peft_config, tokenizer, train_args)
finetune.run_inference(model, tokenizer)

if args.manual_runner:
    dist.destroy_process_group()
