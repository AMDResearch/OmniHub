from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from omnihub import distributed as dist
from contextlib import contextmanager
from enum import Enum
from omnihub import finetuner
from omnihub import inferencer
from omnihub import tracer

import os
import sys


class Mode(Enum):
    Infer = "infer"
    FineTune = "finetune"

    def __str__(self):
        return self.value


def setup_parser():
    parser = ArgumentParser(
        description="Analyze ML/AI workloads with Omnihub",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m", "--model-dir", help="Path to the model", type=str, required=True
    )
    parser.add_argument(
        "-o", "--output-dir", help="Path to store output", type=str, default="."
    )

    # Mode
    parser.add_argument(
        "-M",
        "--mode",
        help="Mode of execution",
        type=Mode,
        choices=list(Mode),
        default=Mode.Infer,
    )

    # Optional arguments for manual distributed execution
    parser.add_argument("--master-addr", type=str)
    parser.add_argument("--master-port", type=str)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int)

    # Tools
    parser.add_argument("--omnitrace", action="store_true", help="Enable omnitrace")
    return parser


class Omnihub:
    def __init__(self) -> None:
        parser = setup_parser()
        self.args = parser.parse_args()
        if not os.path.exists(self.args.model_dir) or not os.path.isdir(
            self.args.model_dir
        ):
            print("Model path does not exist")
            parser.print_help()
            sys.exit(1)

        if not os.path.exists(self.args.output_dir) or not os.path.isdir(
            self.args.output_dir
        ):
            print("Output path does not exist")
            parser.print_help()
            sys.exit(1)

        self.dist = dist.Distributed(args=self.args)

    def run(self):
        if self.args.mode == Mode.FineTune:
            ft = finetuner.FineTuner(args=self.args, manual_ddp=self.dist.manual_ddp)
            with tracer.profile(use_omnitrace=self.args.omnitrace):
                ft.run()
        elif self.args.mode == Mode.Infer:
            inf = inferencer.Inferencer(args=self.args)
            with tracer.profile(use_omnitrace=self.args.omnitrace):
                inf.run()

    def finalize(self):
        self.dist.finalize()


@contextmanager
def Init():
    o = Omnihub()
    try:
        yield o
    finally:
        o.finalize()
