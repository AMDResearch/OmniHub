#!/usr/bin/env python3
"""Dist-init smoke entrypoint for OmniHub.

Builds the **same** child environment that ``pretrain_wrapper.py`` would hand
to Primus's ``run_pretrain.sh`` / ``run_pretrain_cli.sh``, but instead of
launching training it runs a tiny ``primus_child_dist_smoke.py`` script that
only does ``torch.distributed.init_process_group`` + one collective.

Launch via ``omnihub-generate-job`` with the matching app config
(``config-dist-smoke.yaml``) using the **same** ``--runner``, ``--num-nodes``,
``--partition``, and ``--tasks-per-node`` as the failing Primus job, then
``sbatch`` the generated script.

Config keys (in the YAML app config):
    smoke_mode : str
        ``"torchrun"`` (default) — launch the child via ``torchrun``, matching
        what ``run_pretrain.sh`` does internally.
        ``"direct"`` — launch the child via ``python3`` directly, to isolate
        whether the hang is in the torchrun rendezvous layer.

All other keys accepted by ``pretrain_wrapper.py`` (``primus_path``,
``exp_config``, ``master_port``, ``gpus_per_node``, etc.) are honoured so the
env matches.
"""

import os
import subprocess
import sys
from typing import Any, Dict, List

try:
    import omnihub.run
except ImportError:

    class omnihub:  # type: ignore[no-redef]
        class run:
            @staticmethod
            def entrypoint(func):
                return func


def _get_default_primus_path() -> str:
    return os.path.join(os.getenv("OMNIHUB_SRC_DIR", "."), "..", "Primus")


def _get_master_addr_from_slurm():
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
            first = result.stdout.strip().splitlines()[0].strip()
            if first:
                return first
    except (subprocess.TimeoutExpired, FileNotFoundError, IndexError):
        pass
    return None


def _build_child_env(config: Dict[str, Any]) -> Dict[str, str]:
    """Replicate the env dict that ``run_primus_pretrain`` would pass to the subprocess."""
    env = os.environ.copy()

    primus_path = config.get("primus_path", _get_default_primus_path())
    env["PRIMUS_PATH"] = primus_path

    exp_config = config.get("exp_config") or config.get("EXP") or ""
    env["EXP"] = exp_config

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

    def _nnodes() -> str:
        if config.get("nnodes") is not None:
            return str(config["nnodes"])
        for v in ("NNODES", "SLURM_NNODES", "SLURM_JOB_NUM_NODES"):
            if os.getenv(v):
                return os.getenv(v)  # type: ignore[return-value]
        if os.getenv("WORLD_SIZE") and gpus_per_node:
            return str(int(os.getenv("WORLD_SIZE")) // gpus_per_node)  # type: ignore[arg-type]
        return "1"

    def _node_rank() -> str:
        if config.get("node_rank") is not None:
            return str(config["node_rank"])
        if os.getenv("NODE_RANK"):
            return os.getenv("NODE_RANK")  # type: ignore[return-value]
        if os.getenv("SLURM_NODEID") is not None:
            return os.getenv("SLURM_NODEID")  # type: ignore[return-value]
        if os.getenv("RANK") is not None and gpus_per_node:
            return str(int(os.getenv("RANK")) // gpus_per_node)  # type: ignore[arg-type]
        return "0"

    env["NNODES"] = _nnodes()
    env["NODE_RANK"] = _node_rank()
    env["GPUS_PER_NODE"] = str(
        config.get("gpus_per_node", os.getenv("GPUS_PER_NODE", str(gpus_per_node)))
    )

    master_addr = config.get("master_addr") or os.getenv("MASTER_ADDR")
    if not master_addr:
        master_addr = _get_master_addr_from_slurm() or "localhost"
    env["MASTER_ADDR"] = str(master_addr)

    _outer_port = int(os.getenv("MASTER_PORT", "29400"))
    env["MASTER_PORT"] = str(config.get("master_port", _outer_port + 1))

    nnodes_int = int(env["NNODES"])
    node_rank_int = int(env["NODE_RANK"])
    gpus_int = int(env["GPUS_PER_NODE"])
    world_size = nnodes_int * gpus_int

    if os.getenv("WORLD_SIZE") is None:
        env["WORLD_SIZE"] = str(world_size)
    if os.getenv("RANK") is None:
        env["RANK"] = str(node_rank_int * gpus_int)
    if os.getenv("LOCAL_RANK") is None:
        env["LOCAL_RANK"] = "0"
    if os.getenv("LOCAL_WORLD_SIZE") is None:
        env["LOCAL_WORLD_SIZE"] = str(gpus_int)

    run_mode_val = config.get("run_mode")
    if run_mode_val == "single":
        env["RUN_MODE"] = "single"

    return env


def _find_child_script() -> str:
    """Locate ``primus_child_dist_smoke.py`` relative to this file or OMNIHUB_SRC_DIR."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "primus_child_dist_smoke.py"),
        os.path.join(
            os.getenv("OMNIHUB_SRC_DIR", ""),
            "applications",
            "primus-pretrain",
            "primus_child_dist_smoke.py",
        ),
    ]
    for c in candidates:
        p = os.path.normpath(c)
        if os.path.isfile(p):
            return p
    print("Error: cannot find primus_child_dist_smoke.py", file=sys.stderr)
    sys.exit(1)


def _print_env_diff(parent_env: Dict[str, str], child_env: Dict[str, str]) -> None:
    """Show env vars that differ between the parent (omnihub-run) and the child."""
    keys = sorted(
        set(
            [
                "MASTER_ADDR",
                "MASTER_PORT",
                "RANK",
                "WORLD_SIZE",
                "LOCAL_RANK",
                "LOCAL_WORLD_SIZE",
                "NNODES",
                "NODE_RANK",
                "GPUS_PER_NODE",
                "RUN_MODE",
                "DRY_RUN_MODE",
                "EXP",
                "PRIMUS_PATH",
            ]
        )
    )
    print("=" * 60)
    print("Env diff: parent (omnihub-run) vs child (inner torchrun)")
    print(f"  {'Variable':<20s} {'Parent':<25s} {'Child':<25s}")
    print(f"  {'-'*20:<20s} {'-'*25:<25s} {'-'*25:<25s}")
    for k in keys:
        p = parent_env.get(k, "<unset>")
        c = child_env.get(k, "<unset>")
        marker = " <<" if p != c else ""
        print(f"  {k:<20s} {str(p):<25s} {str(c):<25s}{marker}")
    print("=" * 60)
    sys.stdout.flush()


def run_smoke(extra_args: List[str], config: Dict[str, Any]) -> None:
    parent_snapshot = os.environ.copy()
    child_env = _build_child_env(config)
    child_script = _find_child_script()
    smoke_mode = config.get("smoke_mode", "torchrun")

    _print_env_diff(parent_snapshot, child_env)

    primus_path = config.get("primus_path", _get_default_primus_path())
    cwd = primus_path if os.path.isdir(primus_path) else None

    if smoke_mode == "torchrun":
        cmd = [
            "torchrun",
            "--nproc_per_node",
            child_env["GPUS_PER_NODE"],
            "--nnodes",
            child_env["NNODES"],
            "--node_rank",
            child_env["NODE_RANK"],
            "--master_addr",
            child_env["MASTER_ADDR"],
            "--master_port",
            child_env["MASTER_PORT"],
            child_script,
        ]
    elif smoke_mode == "direct":
        cmd = [sys.executable, child_script]
    else:
        print(
            f"Error: unknown smoke_mode '{smoke_mode}' (expected torchrun or direct)",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"dist_init_smoke: mode={smoke_mode}")
    print(f"dist_init_smoke: cmd={' '.join(cmd)}")
    print(f"dist_init_smoke: cwd={cwd}")
    sys.stdout.flush()

    result = subprocess.run(cmd, env=child_env, cwd=cwd)
    sys.exit(result.returncode)


@omnihub.run.entrypoint
def run(*args, **kwargs):
    run_smoke(*args, **kwargs)
