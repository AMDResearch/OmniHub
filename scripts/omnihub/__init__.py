import functools
import importlib.util
import inspect
import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from contextlib import contextmanager
from enum import Enum

import yaml

from omnihub import distributed as dist
from omnihub import tools


def setup_parser():
    parser = ArgumentParser(
        description="Analyze ML/AI workloads with Omnihub",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=".",
        help="Output directory (default: current directory)",
    )

    # Optional arguments for manual distributed execution
    parser.add_argument("--master-addr", type=str)
    parser.add_argument("--master-port", type=str)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument(
        "--manual-launch-ddp",
        action="store_true",
        help="Manual launcher to run with DDP",
    )

    # Tools
    parser.add_argument("--omnitrace", action="store_true", help="Enable omnitrace")
    # PyTorch Profiler
    parser.add_argument(
        "--pytorch-profiler-stats",
        action="store_true",
        help="Enable PyTorch Profiler (Stats)",
    )
    parser.add_argument(
        "--pytorch-profiler-trace",
        action="store_true",
        help="Enable PyTorch Profiler (Tensorboard trace)",
    )

    # Configuration file
    parser.add_argument(
        "-c",
        "--app-config",
        help="Absolute path to the YAML configuration file describing application-specific args",
        type=str,
        required=True,
    )
    return parser


def entrypoint(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(
            f"Executing {func.__name__} with arguments {args} and keyword arguments {kwargs}"
        )
        result = func(*args, **kwargs)
        print(f"Finished executing {func.__name__}")
        return result

    wrapper.__wrapped__ = func
    wrapper.__qualname__ = "entrypoint"
    return wrapper


def find_decorated_function(module):
    decorated_functions = []
    for _, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and hasattr(obj, "__wrapped__"):
            if obj.__qualname__ == "entrypoint":
                decorated_functions.append(obj.__wrapped__)
    return decorated_functions[0] if decorated_functions else None


class Omnihub:
    def __init__(self) -> None:
        parser = setup_parser()
        # Parse known arguments, leaving the rest untouched
        self.args, self.extra_args = parser.parse_known_args()

        if not os.path.exists(self.args.output_dir) or not os.path.isdir(
            self.args.output_dir
        ):
            print(
                f"Output directory {self.args.output_dir} does not exist or is not a directory"
            )
            parser.print_help()
            sys.exit(1)

        if not os.path.exists(self.args.app_config) or not os.path.isfile(
            self.args.app_config
        ):
            print(f"Configuration file {self.args.app_config} does not exist")
            parser.print_help()
            sys.exit(1)

        with open(self.args.app_config, "r") as config_file:
            config = yaml.safe_load(config_file)

        entrypoint = config.get("entrypoint")
        if not entrypoint:
            print("Entrypoint not specified in the configuration file")
            sys.exit(1)

        # Add other key-value pairs from the config file to self.extra_args
        for key, value in config.items():
            if key != "entrypoint":
                self.extra_args.append(f"--{key}")
                self.extra_args.append(str(value))

        # Check if entrypoint is an absolute path. If not, make it relative to the config file.
        # Application writer is responsible to structuring the config file location and
        # entrypoint correctly.
        if not os.path.isabs(entrypoint):
            entrypoint = os.path.join(os.path.dirname(self.args.app_config), entrypoint)

        # Imports the module specified by the entrypoint.
        spec = importlib.util.spec_from_file_location("module.name", entrypoint)
        module = importlib.util.module_from_spec(spec)
        sys.modules["module.name"] = module
        spec.loader.exec_module(module)

        # Find the first function decorated with @entrypoint in the imported module
        self.func = find_decorated_function(module)

        self.dist = dist.Distributed(args=self.args)

        if self.args.omnitrace:
            tools.tracers.enable_omnitrace()
        if self.args.pytorch_profiler_stats:
            tools.tracers.enable_pytorch_profiler_stats()
        if self.args.pytorch_profiler_trace:
            tools.tracers.enable_pytorch_profiler_trace()

    def run(self):
        if self.func:
            self.func(self.extra_args)
        else:
            print("No function decorated with @omnihub.entrypoint found to execute.")
            sys.exit(1)

    def finalize(self):
        self.dist.finalize()


@contextmanager
def Init():
    o = Omnihub()
    try:
        yield o
    finally:
        o.finalize()
