#!/usr/bin/env python3
"""
vLLM Latency Benchmark Wrapper for Omnihub
"""

import os
import subprocess
import sys
from typing import Any, Dict, List

try:
    import omnihub.run
    import omnihub.tools
except ImportError:

    class omnihub:
        class run:
            @staticmethod
            def entrypoint(func):
                return func

        class tools:
            @staticmethod
            def profile():
                def decorator(func):
                    return func

                return decorator


MODEL_KEYS = [
    "model",
    "model_name",
    "model_path",
    "model_dir",
    "pretrained_model_name_or_path",
]


def resolve_model_path(config, model_args):
    default_model_path = None
    model_key = None
    for key in MODEL_KEYS:
        default_model_path = config.get(key, model_args.get(key))
        if default_model_path is not None:
            model_key = key
            break
    if default_model_path is None:
        print("No model specified in the configuration file or CLI args")
        sys.exit(1)
    omnihub_model_path = os.path.join(
        os.getenv("OMNIHUB_MODELS_DIR", ""), default_model_path
    )
    model_path = default_model_path
    if not os.path.isdir(default_model_path) and os.path.isdir(omnihub_model_path):
        model_path = omnihub_model_path
    config[model_key] = model_path


def run_vllm_bench_latency(extra_args, config):
    resolve_model_path(config, {})
    cli_args = extra_args + [
        f"--{k.replace('_', '-')}={str(v)}" for k, v in config.items()
    ]
    argv = ["vllm", "bench", "latency"] + cli_args

    @omnihub.tools.profile()
    def run_subprocess():
        return subprocess.run(argv)

    result = run_subprocess()
    if result.returncode != 0:
        print(f"Subprocess exited with return code {result.returncode}")
        sys.exit(result.returncode)


@omnihub.run.entrypoint
def run(extra_args, config):
    run_vllm_bench_latency(extra_args, config)


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="vLLM bench latency wrapper")
    parser.add_argument("--app-config", required=True)
    args, extra = parser.parse_known_args()
    with open(args.app_config) as f:
        config = yaml.safe_load(f)
    config.pop("entrypoint", None)
    run_vllm_bench_latency(extra, config)
