#!/usr/bin/env python3
"""
Primus Pretrain Wrapper for Omnihub

Runs Primus pretrain via examples/run_pretrain_cli.sh by default (primus-cli direct).
With run_mode: single the wrapper sets RUN_MODE=single so primus-cli-direct uses
python3 per process (no inner torchrun), matching OmniHub --runner manual (one Slurm
task per GPU) or --runner torchrun (outer torchrun). Legacy examples/run_pretrain.sh
always invokes torchrun and ignores RUN_MODE; set use_primus_cli: false only if needed.
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


# Env vars that the wrapper sets for distributed runs; when run_mode is not set we
# unset these so Primus's torchrun can set them (no duplication).
DISTRIBUTED_ENV_VARS = (
    "RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "NNODES",
    "NODE_RANK",
    "GPUS_PER_NODE",
)


def _get_master_addr_from_slurm() -> Optional[str]:
    """
    Derive MASTER_ADDR from Slurm when not set by the job script.
    Returns the first hostname from SLURM_NODELIST via scontrol show hostnames, or None.
    """
    nodelist = os.getenv("SLURM_NODELIST")
    if not nodelist:
        return None
    try:
        result = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout:
            first_line = result.stdout.strip().splitlines()[0].strip()
            if first_line:
                return first_line
    except (subprocess.TimeoutExpired, FileNotFoundError, IndexError):
        pass
    return None


def _resolve_hf_snapshot(repo_id: str, hf_home: str) -> Optional[str]:
    """Return the local snapshot path for *repo_id* under ``$HF_HOME``, or None."""
    cache_name = "models--" + repo_id.replace("/", "--")
    snapshots_dir = os.path.join(hf_home, "hub", cache_name, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None

    refs_main = os.path.join(hf_home, "hub", cache_name, "refs", "main")
    if os.path.isfile(refs_main):
        with open(refs_main) as f:
            target_hash = f.read().strip()
        candidate = os.path.join(snapshots_dir, target_hash)
        if os.path.isdir(candidate):
            return candidate

    for entry in sorted(os.listdir(snapshots_dir)):
        candidate = os.path.join(snapshots_dir, entry)
        if os.path.isdir(candidate) and os.path.isfile(
            os.path.join(candidate, "tokenizer.json")
        ):
            return candidate

    return None


def _resolve_hf_assets_from_hf_home(
    merged_config: Dict[str, Any], primus_path: str
) -> None:
    """Resolve HuggingFace repo IDs to local snapshots using the $HF_HOME cache.

    First checks ``modules.<mod>.overrides.model.hf_assets_path`` (explicit
    override already present in the OmniHub config).  If no repo ID is found
    there, falls back to reading the Primus model YAML file referenced by
    ``modules.<mod>.model`` + ``modules.<mod>.framework`` at
    ``<primus_path>/primus/configs/models/<framework>/<model_file>``.

    When a repo ID is found and a matching snapshot exists under
    ``$HF_HOME/hub/models--<org>--<model>/snapshots/``, the resolved local
    path is written into ``overrides.model.hf_assets_path`` so that Primus's
    prepare hook sees a local directory and skips the download.
    """
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        return

    def _try_resolve_from_overrides(overrides: Dict[str, Any]) -> bool:
        """Resolve ``model.hf_assets_path`` if it already exists as a dict."""
        model_section = overrides.get("model")
        if not isinstance(model_section, dict):
            return False
        val = model_section.get("hf_assets_path")
        if not val or not isinstance(val, str):
            return False
        if "/" not in val or os.path.isdir(val):
            return False
        resolved = _resolve_hf_snapshot(val, hf_home)
        if resolved:
            model_section["hf_assets_path"] = resolved
            print(f"HF assets resolved from $HF_HOME: {val} -> {resolved}")
            return True
        return False

    def _try_resolve_from_model_yaml(
        mod_val: Dict[str, Any],
    ) -> bool:
        """Read the Primus model YAML to discover ``model.hf_assets_path``."""
        model_file = mod_val.get("model")
        framework = mod_val.get("framework")
        if not isinstance(model_file, str) or not isinstance(framework, str):
            return False

        model_yaml_path = os.path.join(
            primus_path, "primus", "configs", "models", framework, model_file
        )
        if not os.path.isfile(model_yaml_path):
            return False

        try:
            with open(model_yaml_path, "r") as f:
                model_cfg = yaml.safe_load(f)
        except Exception:
            return False

        model_section = model_cfg.get("model") if isinstance(model_cfg, dict) else None
        if not isinstance(model_section, dict):
            return False
        repo_id = model_section.get("hf_assets_path")
        if not repo_id or not isinstance(repo_id, str):
            return False
        if "/" not in repo_id or os.path.isdir(repo_id):
            return False

        resolved = _resolve_hf_snapshot(repo_id, hf_home)
        if not resolved:
            return False

        overrides = mod_val.setdefault("overrides", {})
        model_overrides = overrides.setdefault("model", {})
        model_overrides["hf_assets_path"] = resolved
        print(
            f"HF assets resolved from $HF_HOME (via {model_file}): "
            f"{repo_id} -> {resolved}"
        )
        return True

    modules = merged_config.get("modules", {})
    for mod_name, mod_val in modules.items():
        if not isinstance(mod_val, dict):
            continue
        overrides = mod_val.get("overrides")
        if isinstance(overrides, dict) and _try_resolve_from_overrides(overrides):
            continue
        _try_resolve_from_model_yaml(mod_val)


def run_primus_pretrain(extra_args, config):
    """
    Execute Primus pretrain (default: examples/run_pretrain_cli.sh via primus-cli direct).

    Args:
        extra_args: Additional command-line arguments
        config: Configuration dictionary from YAML file

    Expected config keys:
        - primus_path: Path to Primus installation (default: ${OMNIHUB_SRC_DIR}/../Primus)
        - exp_config: Path to experiment config file (required)
        - use_primus_cli: If true (default), require run_pretrain_cli.sh; if false, use run_pretrain.sh
        - run_mode: e.g. single → sets RUN_MODE=single (no inner torchrun with CLI path)
        - nnodes: Number of nodes (optional, default: from env or 1)
        - node_rank: Current node rank (optional, default: from env or 0)
        - gpus_per_node: Number of GPUs per node (optional, default: from env or 8)
        - master_addr: Master node address (optional, default: from env or localhost)
        - master_port: Master node port for the inner Primus torchrun (optional,
          default: outer MASTER_PORT + 1 to avoid EADDRINUSE with the parent process)
        - dry_run: If true, pass --dry-run to primus-cli (CLI path only; legacy path warns)
        - hipblaslt_tuning_stage: HipBLASLt tuning stage 0/1/2/3 (optional, default: 0)

    Any additional keys in the config will be treated as overrides for the exp_config file.
    These override parameters will be recursively merged with the exp_config, with the
    config parameters taking precedence over those in the exp_config file.
    """

    # Get Primus path from config or use default based on OMNIHUB_SRC_DIR
    primus_path = config.get("primus_path", get_default_primus_path())

    # Resolve distributed settings from any of: config, OmniHub env (RANK, WORLD_SIZE,
    # MASTER_ADDR, MASTER_PORT, LOCAL_RANK), or Primus/Slurm (NNODES, NODE_RANK,
    # GPUS_PER_NODE, SLURM_*). We need gpus_per_node and helpers before choosing script.
    gpus_per_node = config.get("gpus_per_node") or os.getenv("GPUS_PER_NODE")
    if gpus_per_node is not None:
        gpus_per_node = int(gpus_per_node)
    else:
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                gpus_per_node = torch.cuda.device_count()
        except Exception:
            pass
        gpus_per_node = gpus_per_node or 8

    def _get_nnodes():
        """Number of nodes: config, then NNODES, then Slurm, then WORLD_SIZE (OmniHub)."""
        if config.get("nnodes") is not None:
            return str(config["nnodes"])
        if os.getenv("NNODES"):
            return os.getenv("NNODES")
        if os.getenv("SLURM_NNODES"):
            return os.getenv("SLURM_NNODES")
        if os.getenv("SLURM_JOB_NUM_NODES"):
            return os.getenv("SLURM_JOB_NUM_NODES")
        if os.getenv("WORLD_SIZE") and gpus_per_node:
            return str(int(os.getenv("WORLD_SIZE")) // gpus_per_node)
        return "1"

    def _get_node_rank():
        """Node rank: config, then NODE_RANK, then Slurm, then RANK (OmniHub) // gpus_per_node."""
        if config.get("node_rank") is not None:
            return str(config["node_rank"])
        if os.getenv("NODE_RANK"):
            return os.getenv("NODE_RANK")
        if os.getenv("SLURM_NODEID") is not None:
            return os.getenv("SLURM_NODEID")
        if os.getenv("RANK") is not None and gpus_per_node:
            return str(int(os.getenv("RANK")) // gpus_per_node)
        return "0"

    try:
        nnodes_for_script = int(_get_nnodes())
    except (TypeError, ValueError):
        nnodes_for_script = 1

    use_primus_cli = bool(config.get("use_primus_cli", True))
    run_pretrain_cli_path = os.path.join(primus_path, "examples/run_pretrain_cli.sh")
    run_pretrain_legacy_path = os.path.join(primus_path, "examples/run_pretrain.sh")

    if use_primus_cli:
        if not os.path.isfile(run_pretrain_cli_path):
            print(
                "Error: use_primus_cli is true (default) but Primus CLI script was not found:\n"
                f"  {run_pretrain_cli_path}\n"
                "Install a Primus tree that includes examples/run_pretrain_cli.sh, or set "
                "use_primus_cli: false to use examples/run_pretrain.sh (inner torchrun; "
                "avoid OmniHub --runner manual with multiple tasks per node).",
                file=sys.stderr,
            )
            sys.exit(1)
        run_pretrain_script = run_pretrain_cli_path
    else:
        if not os.path.isfile(run_pretrain_legacy_path):
            print(
                f"Error: use_primus_cli is false but legacy script not found: {run_pretrain_legacy_path}",
                file=sys.stderr,
            )
            sys.exit(1)
        run_pretrain_script = run_pretrain_legacy_path

    # Legacy run_pretrain.sh uses inner torchrun: one Slurm task per node (uniform job).
    if not use_primus_cli and os.getenv("SLURM_JOB_ID"):
        _nt = os.getenv("SLURM_NTASKS")
        _nn = os.getenv("SLURM_NNODES") or os.getenv("SLURM_JOB_NUM_NODES")
        if _nt is not None and _nn is not None:
            try:
                if int(_nt) != int(_nn):
                    print(
                        "Error: Legacy Primus pretrain (use_primus_cli: false, "
                        "examples/run_pretrain.sh) requires exactly one Slurm task per node "
                        f"(SLURM_NTASKS must equal SLURM_NNODES). Got SLURM_NTASKS={_nt}, "
                        f"SLURM_NNODES={_nn}.\n"
                        "Fix the job allocation (for example omnihub-generate-job "
                        "--tasks-per-node 1 with --runner manual, or --runner torchrun), "
                        "or set use_primus_cli: true with a layout suited to run_pretrain_cli.sh.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            except ValueError:
                pass

    # run_mode: unset = let Primus / outer torchrun own distributed env (see below).
    run_mode_val = config.get("run_mode")

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
        "use_primus_cli",
        "run_mode",
        "dry_run",
        "primus_log_level",
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

        _resolve_hf_assets_from_hf_home(merged_config, primus_path)

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

    # Distributed training settings. NNODES/NODE_RANK/MASTER_* resolved from config,
    # OmniHub (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, LOCAL_RANK), or
    # Primus/Slurm (NNODES, NODE_RANK, GPUS_PER_NODE, SLURM_*).
    env["NNODES"] = _get_nnodes()
    env["NODE_RANK"] = _get_node_rank()
    env["GPUS_PER_NODE"] = str(
        config.get("gpus_per_node", os.getenv("GPUS_PER_NODE", str(gpus_per_node)))
    )
    # MASTER_ADDR: config or env, else derive from Slurm (first host in SLURM_NODELIST)
    master_addr = config.get("master_addr") or os.getenv("MASTER_ADDR")
    if not master_addr:
        master_addr = _get_master_addr_from_slurm() or "localhost"
    env["MASTER_ADDR"] = str(master_addr)

    # MASTER_PORT for the inner Primus torchrun. Must differ from the outer port
    # (OmniHub's Distributed / outer torchrun already binds the outer MASTER_PORT).
    # Default: outer_port + 1 to avoid EADDRINUSE while staying near a validated port.
    _outer_port = int(os.getenv("MASTER_PORT", "29400"))
    env["MASTER_PORT"] = str(config.get("master_port", _outer_port + 1))

    # Per-process env vars for distributed training. Set from node-level vars when
    # not already set (e.g. OmniHub --manual-launch-ddp or outer torchrun).
    nnodes_int = int(env["NNODES"])
    node_rank_int = int(env["NODE_RANK"])
    gpus_int = int(env["GPUS_PER_NODE"])
    world_size = nnodes_int * gpus_int
    if os.getenv("WORLD_SIZE") is None:
        env["WORLD_SIZE"] = str(world_size)
    if os.getenv("RANK") is None:
        # One process per node: this process represents the first rank on the node.
        env["RANK"] = str(node_rank_int * gpus_int)
    if os.getenv("LOCAL_RANK") is None:
        env["LOCAL_RANK"] = "0"
    if os.getenv("LOCAL_WORLD_SIZE") is None:
        env["LOCAL_WORLD_SIZE"] = str(gpus_int)

    # run_pretrain_cli.sh + RUN_MODE=single: python3 per invocation (OmniHub manual
    # uses one task per GPU; SLURM_NTASKS > SLURM_NNODES is expected). Legacy path is
    # checked earlier (fatal if not one task per node).
    if use_primus_cli:
        using_cli = run_pretrain_script.endswith("run_pretrain_cli.sh")
        skip_ntasks_layout_warning = using_cli and run_mode_val == "single"
        slurm_ntasks = os.getenv("SLURM_NTASKS")
        slurm_nnodes = os.getenv("SLURM_NNODES") or os.getenv("SLURM_JOB_NUM_NODES")
        if (
            not skip_ntasks_layout_warning
            and slurm_ntasks is not None
            and slurm_nnodes is not None
        ):
            try:
                ntasks = int(slurm_ntasks)
                nnodes = int(slurm_nnodes)
                if nnodes > 0 and ntasks != nnodes:
                    print(
                        "Warning: inner Primus torchrun expects one Slurm task per node "
                        "(SLURM_NTASKS should equal SLURM_NNODES). Current job has "
                        "SLURM_NTASKS={} and SLURM_NNODES={}. "
                        "Use run_pretrain_cli.sh with run_mode: single for one task per GPU "
                        "(OmniHub --runner manual), or one task per node with inner torchrun.".format(
                            slurm_ntasks, slurm_nnodes
                        ),
                        file=sys.stderr,
                    )
            except ValueError:
                pass

    if run_mode_val == "single":
        env["RUN_MODE"] = "single"

    # Primus runner logging: set PRIMUS_LOG_LEVEL=INFO (or DEBUG) to enable LOG_INFO_RANK0
    # output from primus-cli-direct.sh and other runner scripts. RANK0 logs only appear on
    # the process with NODE_RANK=0.
    primus_log_level = config.get("primus_log_level") or os.getenv("PRIMUS_LOG_LEVEL")
    if primus_log_level is not None:
        env["PRIMUS_LOG_LEVEL"] = str(primus_log_level).upper()

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
    if "NNODES" in env:
        print(f"NNODES: {env['NNODES']}")
    if "NODE_RANK" in env:
        print(f"NODE_RANK: {env['NODE_RANK']}")
    if "GPUS_PER_NODE" in env:
        print(f"GPUS_PER_NODE: {env['GPUS_PER_NODE']}")
    if "MASTER_ADDR" in env:
        print(f"MASTER_ADDR: {env['MASTER_ADDR']}")
    if "MASTER_PORT" in env:
        print(f"MASTER_PORT: {env['MASTER_PORT']}")
    if "WORLD_SIZE" in env:
        print(f"WORLD_SIZE: {env['WORLD_SIZE']}")
    if "RANK" in env:
        print(f"RANK: {env['RANK']}")
    if "LOCAL_RANK" in env:
        print(f"LOCAL_RANK: {env['LOCAL_RANK']}")
    if "LOCAL_WORLD_SIZE" in env:
        print(f"LOCAL_WORLD_SIZE: {env['LOCAL_WORLD_SIZE']}")
    if "RUN_MODE" in env:
        print(f"RUN_MODE: {env['RUN_MODE']}")
    if "DATA_PATH" in env:
        print(f"DATA_PATH: {env['DATA_PATH']}")
    if "PRIMUS_WORKSPACE" in env:
        print(f"PRIMUS_WORKSPACE: {env['PRIMUS_WORKSPACE']}")
    if "BACKEND_PATH" in env:
        print(f"BACKEND_PATH: {env['BACKEND_PATH']}")
    if "PRIMUS_HIPBLASLT_TUNING_STAGE" in env:
        print(f"PRIMUS_HIPBLASLT_TUNING_STAGE: {env['PRIMUS_HIPBLASLT_TUNING_STAGE']}")
    if "PRIMUS_LOG_LEVEL" in env:
        print(f"PRIMUS_LOG_LEVEL: {env['PRIMUS_LOG_LEVEL']}")
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

        # Build the launch command. For the CLI path, dry_run is passed as
        # --dry-run to primus-cli (global flag before the mode subcommand).
        # run_pretrain_cli.sh hardcodes $@ after --config, so we call primus-cli
        # directly when dry_run is requested.
        dry_run = config.get("dry_run", False)
        if dry_run and use_primus_cli:
            primus_cli_path = os.path.join(primus_path, "runner/primus-cli")
            cmd = [
                "bash",
                primus_cli_path,
                "--dry-run",
                "direct",
                "--",
                "train",
                "pretrain",
                "--config",
                final_exp_config,
            ] + extra_args
        elif dry_run and not use_primus_cli:
            print(
                "Warning: dry_run is not supported with use_primus_cli: false "
                "(run_pretrain.sh has no dry-run mode). Running normally.",
                file=sys.stderr,
            )
            cmd = ["bash", run_pretrain_script] + extra_args
        else:
            cmd = ["bash", run_pretrain_script] + extra_args

        print(f"Executing: {' '.join(cmd)}")
        sys.stdout.flush()

        @omnihub.tools.profile()
        def run_cmd():
            my_env = env.copy()
            return subprocess.run(cmd, env=my_env)

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
        "--master-port",
        type=int,
        default=None,
        help="Master node port for inner torchrun (default: outer MASTER_PORT + 1)",
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
