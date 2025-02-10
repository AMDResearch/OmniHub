import importlib.util
import inspect
import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from contextlib import contextmanager
from enum import Enum

import yaml

import omnihub.tools
from omnihub.run import distributed as dist


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


def find_decorated_function(module):
    decorated_functions = []
    for _, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and hasattr(obj, "__wrapped__"):
            if obj.__qualname__ == "entrypoint":
                decorated_functions.append(obj.__wrapped__)
    return decorated_functions[0] if decorated_functions else None


class OmnihubRunner:
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

        # Remove the entrypoint from the config and store it separately
        entrypoint = config.pop("entrypoint", None)
        if not entrypoint:
            print("Entrypoint not specified in the configuration file")
            sys.exit(1)

        # Update self.config with CLI args if they match. CLI args take precedence.
        i = 0
        extra_args_indices = []
        while i < len(self.extra_args):
            if "=" in self.extra_args[i]:
                # Split the argument into key and value based on the first '=' and remove leading '--'
                key, value = self.extra_args[i].split("=", 1)
                key = key.lstrip("--")
                indices = [i]
                i += 1
            else:
                # Remove leading '--' and get the next argument as value
                key = self.extra_args[i].lstrip("--")
                value = self.extra_args[i + 1] if i + 1 < len(self.extra_args) else None
                indices = [i, i + 1]
                i += 2

            # Update the config with the CLI args and store the indices to remove them from extra_args
            if key in config:
                config[key] = value
                extra_args_indices.extend(indices)

        # Remove the CLI args from the extra_args based on the indices that matched the config keys
        self.extra_args = [
            arg for i, arg in enumerate(self.extra_args) if i not in extra_args_indices
        ]
        self.config = config

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
            omnihub.tools.tracers.enable_omnitrace()
        if self.args.pytorch_profiler_stats:
            omnihub.tools.tracers.enable_pytorch_profiler_stats()
        if self.args.pytorch_profiler_trace:
            omnihub.tools.tracers.enable_pytorch_profiler_trace()

    def run(self):
        if self.func:
            self.func(self.extra_args, self.config)
        else:
            print("No function decorated with @omnihub.entrypoint found to execute.")
            sys.exit(1)

    def finalize(self):
        self.dist.finalize()


@contextmanager
def Init():
    o = OmnihubRunner()
    try:
        yield o
    finally:
        o.finalize()


def main():
    with Init() as o:
        o.run()
