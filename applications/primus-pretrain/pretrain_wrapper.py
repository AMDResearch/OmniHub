#!/usr/bin/env python3
"""
Primus Pretrain Wrapper for Omnihub

This wrapper allows the Primus run_pretrain.sh shell script to be executed
through the Omnihub framework.
"""

import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional

import yaml

try:
    import omnihub.run
    import omnihub.tools
except ImportError:
    # If omnihub is not available, create a dummy decorator
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


def get_default_primus_path():
    """
    Get the default Primus path based on OMNIHUB_SRC_DIR environment variable.

    Returns the path to Primus installation:
    - If OMNIHUB_SRC_DIR is set: ${OMNIHUB_SRC_DIR}/../Primus
    - Otherwise: ../Primus (relative to current directory)
    """
    return os.path.join(os.getenv("OMNIHUB_SRC_DIR", "."), "..", "Primus")


def deep_merge(base_dict, override_dict):
    """
    Recursively merge override_dict into base_dict.
    Values in override_dict take precedence over values in base_dict.

    Args:
        base_dict: The base dictionary
        override_dict: The dictionary with override values

    Returns:
        A new merged dictionary
    """
    result = base_dict.copy()

    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge(result[key], value)
        else:
            # Override the value
            result[key] = value

    return result


def get_path_with_fallback(
    env_var_name: str, config: Dict[str, Any], config_key: str
) -> Optional[str]:
    """
    Get a path from environment variable or config with proper fallback.

    Args:
        env_var_name (str): Name of the environment variable to check
        config (Dict[str, Any]): Configuration dictionary
        config_key (str): Key in config dictionary to use as fallback

    Returns:
        Optional[str]: The path value from environment or config, or None if neither
                       is set. Note that empty strings are treated as valid values.
    """
    value = os.getenv(env_var_name)
    if value is None:
        value = config.get(config_key, None)
    return value


def run_primus_pretrain(extra_args, config):
    """
    Execute the Primus run_pretrain.sh script with the provided configuration.

    Args:
        extra_args: Additional command-line arguments
        config: Configuration dictionary from YAML file

    Expected config keys:
        - primus_path: Path to Primus installation (default: ${OMNIHUB_SRC_DIR}/../Primus)
        - exp_config: Path to experiment config file (required)
        - nnodes: Number of nodes (optional, default: from env or 1)
        - node_rank: Current node rank (optional, default: from env or 0)
        - gpus_per_node: Number of GPUs per node (optional, default: from env or 8)
        - master_addr: Master node address (optional, default: from env or localhost)
        - master_port: Master node port (optional, default: from env or 1234)
        - hipblaslt_tuning_stage: HipBLASLt tuning stage 0/1/2/3 (optional, default: 0)

    Any additional keys in the config will be treated as overrides for the exp_config file.
    These override parameters will be recursively merged with the exp_config, with the
    config parameters taking precedence over those in the exp_config file.
    """

    # Get Primus path from config or use default based on OMNIHUB_SRC_DIR
    primus_path = config.get("primus_path", get_default_primus_path())
    run_pretrain_script = os.path.join(primus_path, "examples/run_pretrain.sh")

    # Check if the script exists
    if not os.path.exists(run_pretrain_script):
        print(
            f"Error: run_pretrain.sh not found at {run_pretrain_script}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Get experiment config (required)
    exp_config = config.get("exp_config") or config.get("EXP")
    if not exp_config:
        print(
            "Error: 'exp_config' or 'EXP' must be specified in the configuration",
            file=sys.stderr,
        )
        sys.exit(1)

    # Extract wrapper-specific parameters that should not be passed to exp_config
    wrapper_params = {
        "entrypoint",
        "primus_path",
        "exp_config",
        "EXP",
        "nnodes",
        "node_rank",
        "gpus_per_node",
        "master_addr",
        "master_port",
        "hipblaslt_tuning_stage",
        "backend_path",
    }

    # Extract additional parameters that should override exp_config
    override_params = {k: v for k, v in config.items() if k not in wrapper_params}

    # If there are override parameters, merge them with exp_config
    final_exp_config = exp_config
    temp_config_created = False
    if override_params:
        # Load the exp_config file
        if not os.path.exists(exp_config):
            print(
                f"Error: Experiment config file not found at {exp_config}",
                file=sys.stderr,
            )
            sys.exit(1)

        with open(exp_config, "r") as f:
            exp_config_data = yaml.safe_load(f)

        # Merge override parameters with exp_config (overrides take precedence)
        merged_config = deep_merge(exp_config_data, override_params)

        # Write merged config to a temporary file
        temp_config_fd, temp_config_path = tempfile.mkstemp(
            suffix=".yaml", prefix="primus_exp_"
        )
        try:
            with os.fdopen(temp_config_fd, "w") as f:
                yaml.safe_dump(
                    merged_config, f, default_flow_style=False, sort_keys=False
                )
            final_exp_config = temp_config_path
            temp_config_created = True
            print(f"Created merged experiment config at: {temp_config_path}")
        except Exception as e:
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
            print(f"Error creating merged config: {e}", file=sys.stderr)
            sys.exit(1)

    # Setup environment variables for the script
    env = os.environ.copy()

    # Required: Experiment config
    env["EXP"] = final_exp_config

    # CRITICAL: Set PRIMUS_PATH explicitly so run_pretrain.sh doesn't try to auto-detect it
    # The script normally does: PRIMUS_PATH=$(realpath "$(dirname "$0")/..")
    # But when run through omnihub, the working directory is the results dir
    env["PRIMUS_PATH"] = primus_path

    # Optional: Distributed training settings (fallback to existing env vars or defaults)
    env["NNODES"] = str(config.get("nnodes", os.getenv("NNODES", "1")))
    env["NODE_RANK"] = str(config.get("node_rank", os.getenv("NODE_RANK", "0")))
    env["GPUS_PER_NODE"] = str(
        config.get("gpus_per_node", os.getenv("GPUS_PER_NODE", "8"))
    )
    env["MASTER_ADDR"] = config.get(
        "master_addr", os.getenv("MASTER_ADDR", "localhost")
    )
    env["MASTER_PORT"] = str(
        config.get("master_port", os.getenv("MASTER_PORT", "1234"))
    )

    # Optional: HipBLASLt tuning stage
    if "hipblaslt_tuning_stage" in config:
        env["PRIMUS_HIPBLASLT_TUNING_STAGE"] = str(config["hipblaslt_tuning_stage"])

    # Optional: Set other Primus-related paths from config or environment
    # Use helper function to get paths with proper fallback logic
    data_dir = get_path_with_fallback("OMNIHUB_DATA_DIR", config, "data_path")
    # Prefer explicit Primus-specific key, but accept `workspace` as an alias to
    # match example configs and README.
    results_dir = get_path_with_fallback(
        "OMNIHUB_RESULTS_DIR", config, "primus_workspace"
    )
    if results_dir is None:
        results_dir = get_path_with_fallback("OMNIHUB_RESULTS_DIR", config, "workspace")
    models_dir = get_path_with_fallback("OMNIHUB_MODELS_DIR", config, "models_path")

    # Use explicit None checks to allow empty strings if explicitly set
    if data_dir is not None:
        env["DATA_PATH"] = data_dir
    if results_dir is not None:
        env["PRIMUS_WORKSPACE"] = results_dir
    if models_dir is not None:
        env["HF_HUB_CACHE"] = os.path.join(models_dir, "hub")

    if "backend_path" in config:
        env["BACKEND_PATH"] = config["backend_path"]

    # Log the configuration
    print("=" * 60)
    print("Primus Pretrain Wrapper - Configuration")
    print("=" * 60)
    print(f"Script: {run_pretrain_script}")
    print(f"PRIMUS_PATH: {env['PRIMUS_PATH']}")
    print(f"EXP: {env['EXP']}")
    print(f"NNODES: {env['NNODES']}")
    print(f"NODE_RANK: {env['NODE_RANK']}")
    print(f"GPUS_PER_NODE: {env['GPUS_PER_NODE']}")
    print(f"MASTER_ADDR: {env['MASTER_ADDR']}")
    print(f"MASTER_PORT: {env['MASTER_PORT']}")
    if "DATA_PATH" in env:
        print(f"DATA_PATH: {env['DATA_PATH']}")
    if "PRIMUS_WORKSPACE" in env:
        print(f"PRIMUS_WORKSPACE: {env['PRIMUS_WORKSPACE']}")
    if "BACKEND_PATH" in env:
        print(f"BACKEND_PATH: {env['BACKEND_PATH']}")
    if "PRIMUS_HIPBLASLT_TUNING_STAGE" in env:
        print(f"PRIMUS_HIPBLASLT_TUNING_STAGE: {env['PRIMUS_HIPBLASLT_TUNING_STAGE']}")
    print("=" * 60)
    sys.stdout.flush()

    # CRITICAL: Change to the Primus directory before running the script
    # The prepare_experiment.py script uses Path.cwd() to determine primus_path
    # So we must be in the Primus directory when running the script
    original_cwd = os.getcwd()
    try:
        os.chdir(primus_path)
        print(f"Changed working directory to: {os.getcwd()}")
        sys.stdout.flush()

        # Execute the shell script
        cmd = ["bash", run_pretrain_script] + extra_args

        print(f"Executing: {' '.join(cmd)}")
        sys.stdout.flush()

        @omnihub.tools.profile()
        def run_cmd():
            return subprocess.run(cmd, env=env)

        result = run_cmd()

        # Exit with the same return code as the script
        sys.exit(result.returncode)
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
        # Clean up temporary config file if it was created
        if temp_config_created and os.path.exists(final_exp_config):
            try:
                os.unlink(final_exp_config)
                print(f"Cleaned up temporary config: {final_exp_config}")
            except Exception as e:
                print(
                    f"Warning: Failed to clean up temporary config: {e}",
                    file=sys.stderr,
                )


@omnihub.run.entrypoint
def run(*args, **kwargs):
    run_primus_pretrain(*args, **kwargs)


if __name__ == "__main__":
    # When run directly (not through omnihub-run), parse simple args
    import argparse

    parser = argparse.ArgumentParser(description="Primus Pretrain Wrapper")
    parser.add_argument(
        "--exp-config",
        "--EXP",
        dest="exp_config",
        required=True,
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--primus-path",
        default=get_default_primus_path(),
        help="Path to Primus installation (default: ${OMNIHUB_SRC_DIR}/../Primus)",
    )
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node-rank", type=int, default=0, help="Current node rank")
    parser.add_argument(
        "--gpus-per-node", type=int, default=8, help="Number of GPUs per node"
    )
    parser.add_argument(
        "--master-addr", default="localhost", help="Master node address"
    )
    parser.add_argument(
        "--master-port", type=int, default=1234, help="Master node port"
    )
    parser.add_argument(
        "--hipblaslt-tuning-stage", type=int, help="HipBLASLt tuning stage: 0/1/2/3"
    )
    parser.add_argument(
        "--data-path",
        dest="data_path",
        help="Path to data directory (fallback if OMNIHUB_DATA_DIR not set)",
    )
    parser.add_argument(
        "--primus-workspace",
        dest="primus_workspace",
        help="Path to Primus workspace/results directory (fallback if OMNIHUB_RESULTS_DIR not set)",
    )
    parser.add_argument(
        "--models-path",
        dest="models_path",
        help="Path to models directory (fallback if OMNIHUB_MODELS_DIR not set)",
    )

    args, extra = parser.parse_known_args()

    config = {
        "exp_config": args.exp_config,
        "primus_path": args.primus_path,
        "nnodes": args.nnodes,
        "node_rank": args.node_rank,
        "gpus_per_node": args.gpus_per_node,
        "master_addr": args.master_addr,
        "master_port": args.master_port,
    }

    if args.hipblaslt_tuning_stage is not None:
        config["hipblaslt_tuning_stage"] = args.hipblaslt_tuning_stage

    # Add optional path arguments if provided
    if args.data_path is not None:
        config["data_path"] = args.data_path
    if args.primus_workspace is not None:
        config["primus_workspace"] = args.primus_workspace
    if args.models_path is not None:
        config["models_path"] = args.models_path

    run_primus_pretrain(extra, config)
