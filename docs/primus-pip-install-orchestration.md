# Primus pip install orchestration (direct vs legacy)

How Primus runs prepare steps and pip installs in **direct mode** (`primus-cli direct` / `run_pretrain_cli.sh`) vs **legacy** (`run_pretrain.sh`), and why concurrent pip installs cause OSErrors.

## Two entry paths

| Entry | Prepare step | Emerging-Optimizers pip? |
|-------|--------------|--------------------------|
| **Direct** | `prepare_experiment.sh` → **runner/.../megatron/prepare.py** | **No** (only `make` in Megatron dataset dir) |
| **Legacy** | `run_prepare_experiment` → **examples/scripts/prepare_experiment.py** → **examples/megatron/prepare.py** | **Yes** (`pip install -e Emerging-Optimizers`) |

If you see **"Running backend prepare: .../examples/megatron/prepare.py"**, the legacy path is used. OmniHub multinode typically uses direct (`run_pretrain_cli.sh`).

## Direct mode: primus-cli direct

1. **primus-cli-direct.sh** runs `pip install -qq -r requirements.txt` (every process that runs the script).
2. **execute_hooks** runs **prepare_experiment.sh** → **runner/helpers/hooks/train/pretrain/megatron/prepare.py**. That script does dataset prep (rank 0 only in Python), then `make` in Megatron dataset dir. It does **not** run `pip install -e Emerging-Optimizers`.
3. There is **no** "run hook only on rank 0" in the shell; every srun task runs the hook. So with 2 nodes, 2 tasks, both run the runner prepare (but runner prepare has no shared pip -e).

## Legacy: run_pretrain.sh

1. **Every task** runs `pip install -r requirements.txt` then **run_prepare_experiment**.
2. **run_prepare_experiment** runs **examples/scripts/prepare_experiment.py**, which runs **examples/megatron/prepare.py**.
3. **examples/megatron/prepare.py** in `build_megatron_helper()` does:
   - `make` in Megatron dataset dir
   - `pip install -e primus_path/third_party/Emerging-Optimizers`
4. **No rank guard:** every srun task runs this. N tasks → N concurrent `pip install -e` into the same venv → OSError 17/2 and corrupted dist-info.

## Where "Building Emerging Optimizers" comes from

- **File:** `Primus/examples/megatron/prepare.py`
- **Function:** `build_megatron_helper()` (lines ~265–268)
- Only used on the **legacy** path (examples prepare). The **runner** hook prepare does not call this.

## Recommendations

1. Prefer **direct mode** so the runner prepare is used (no Emerging-Optimizers pip at runtime).
2. If using legacy: run prepare **once** (e.g. rank 0 only or a separate single-task job), or pre-install Emerging-Optimizers in the container image.
3. In **examples/megatron/prepare.py**, guard the Emerging-Optimizers install by `get_node_rank() == 0` and sync other ranks (or run install in a single-process step before training).
