---
name: omnihub-agent
description: Runs and configures the OmniHub Agent service from the aaji/agent branch (start, stop, status, config, LLM/tunnel). Use when the user asks about omnihub-agent, the aaji/agent branch, the agent service, or configuring the OmniHub agent.
---

# OmniHub Agent

**Availability:** The OmniHub Agent exists only on branch **aaji/agent** (not on main). Check out that branch to use `omnihub-agent` and the `omnihub_agent` package.

## Commands

- **omnihub-agent start** — Start the agent daemon (acquires lock, writes PID file, runs orchestrator loop).
- **omnihub-agent stop** — Send SIGTERM using the PID file.
- **omnihub-agent status** — Show whether the agent is running and optionally state.
- **omnihub-agent run-once** — Run a single orchestrator tick (no daemon).

Only one instance per user; the lock file prevents a second `start` from succeeding.

## Config and environment

- **Config file:** `~/.omnihub_agent/config.yaml` (or path in `OMNIHUB_AGENT_CONFIG`).
- **Env overrides:** Options can be overridden with `OMNIHUB_AGENT_*` (e.g. `OMNIHUB_AGENT_LLM_BASE_URL`).
- **PID / lock:** Default `~/.omnihub_agent/agent.pid` and `agent.lock`; override with `pid_file` / `lock_file` in config or `OMNIHUB_AGENT_PID_FILE` / `OMNIHUB_AGENT_LOCK_FILE`.

## CPU affinity

Optional. In config: `cpu_cores: "0,1,2"` or `"0-3"`. Or set `OMNIHUB_AGENT_CPU_CORES=0,1,2`. Applied at startup on Linux via `sched_setaffinity`.

## LLM and tunnel (public HPC → internal gateway)

No internal hostnames are hardcoded. You set a base URL or proxy in config that points at your tunnel or proxy.

- **llm_base_url** — Base URL for the LLM API (e.g. `http://localhost:8000/v1`). Agent calls `{base}/chat/completions` (OpenAI-compatible).
- **llm_proxy_url** — Optional HTTP(S) proxy; or, if set alone, used as the request target.
- **llm_model** — Optional model name in the request body.
- **tunnel_command** — Optional; for reference only. The agent does **not** run it; you run the tunnel separately.

Example: after starting a tunnel on localhost:8000, set in `~/.omnihub_agent/config.yaml`:

```yaml
llm_base_url: "http://localhost:8000/v1"
llm_model: "my-model"
```

If neither `llm_base_url` nor `llm_proxy_url` is set, the Recommendation agent uses only rule-based recommendations.

## systemd (user unit)

On the aaji/agent branch, see **docs/omnihub-agent.service.example**. Install under `~/.config/systemd/user/omnihub-agent.service`, then:

```bash
systemctl --user daemon-reload
systemctl --user enable --now omnihub-agent
```

Use `CPUAffinity` or `Environment=OMNIHUB_AGENT_CPU_CORES=...` in the unit if desired.

## Kubernetes backend

On aaji/agent, the agent can use a Kubernetes backend instead of SLURM. Config: `backend: kubernetes`, `k8s_namespace`, `k8s_label_selector`, optional `k8s_results_base_path`. See **docs/omnihub-agent.md** on that branch for details.

## Reference

Full lifecycle, tunnel options, and Kubernetes behavior: **docs/omnihub-agent.md** on branch **aaji/agent**.
