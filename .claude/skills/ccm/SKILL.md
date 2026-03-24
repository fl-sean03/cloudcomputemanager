---
name: ccm
description: CloudComputeManager — GPU cloud management platform for running any workload on Vast.ai with checkpointing, preemption recovery, multi-stage pipelines, environment management, benchmarking, and cost optimization. Use when working with CCM code, submitting cloud jobs, managing instances, managing environments, or building on the CCM platform.
argument-hint: "[topic or task]"
---

# CloudComputeManager (CCM) Agent Skill

You are working with **CloudComputeManager** at `/home/sf2/Workspace/main/46-CCM`.

CCM manages GPU workloads on Vast.ai spot instances. It handles the full lifecycle: instance provisioning, environment setup, job execution, progress monitoring, checkpoint/sync, preemption recovery, and cost tracking. It is **workload-agnostic** — works for LAMMPS, OpenMM, PyTorch, GROMACS, custom scripts, or anything with a CLI.

**User request**: $ARGUMENTS

If a specific topic or task is given, focus on that. If no argument, give a concise overview and ask what they need. For any code changes, **always read the relevant source files first** — do not guess at the implementation.

---

## Key References

| Document | What It Covers |
|----------|---------------|
| `docs/usage.md` | Complete usage guide: architecture, YAML schema, CLI, SDK, benchmarks |
| `docs/ENVIRONMENT_DESIGN.md` | Environment management design and strategy |
| `AGENTS.md` | Current sprint status, what is done, what is remaining |
| `CLAUDE.md` | Build, test, key conventions |

## Architecture

```
src/cloudcomputemanager/
├── agents/sdk.py         — Async Python SDK (CloudComputeManagerAgent)
├── api/                  — FastAPI REST API
├── benchmarks/           — GPU cost-performance benchmark framework
├── checkpoint/           — Checkpoint detection (LAMMPS, PyTorch, generic)
├── cli/
│   ├── main.py           — All command registration
│   ├── jobs.py           — submit, list, status, wait, reconnect, recover
│   ├── jobs_exec.py      — exec, upload, ssh by job_id
│   ├── batch.py          — Batch submit with matrix expansion
│   ├── environments.py   — ccm env export/pack/list
│   └── benchmarks.py     — Benchmark CLI
├── core/
│   ├── environment.py    — Environment parsing, validation, setup commands
│   ├── models.py         — Job, Instance, Checkpoint models
│   ├── wrapper.py        — SIGTERM-aware wrapper (single source of truth)
│   ├── templates.py      — YAML loading, merging, ${VAR} substitution
│   └── config.py         — Settings (all CCM_ env vars)
├── daemon/monitor.py     — Main loop: health, completion, stages, progress
├── providers/
│   ├── base.py           — CloudProvider ABC, wait_for_ready (with sentinel)
│   └── vast.py           — Vast.ai: SSH retry, rsync, heartbeat, onstart
├── sync/engine.py        — Rsync-based data synchronization
└── templates/            — 6 built-in YAML templates
```

## Job YAML Quick Reference

```yaml
name: my-job
project: my-project
image: nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Environment: auto-selects fastest strategy
environment:
  conda_pack: ./env.tar.gz       # Fastest: pre-packed conda env (~2 min)
  # OR conda_env: ./environment.yml  # Reproducible (~10-15 min)
  # OR packages:                     # Inline package list
  #      conda: [numpy, scipy]
  #      pip: [requests]
  #      apt: [libfftw3-dev]
  # OR requirements: ./requirements.txt
  # OR docker_image: myuser/myimage:latest  # Override image

setup: |                          # Additional setup (runs in onstart)
  export MY_VAR=value

command: python3 train.py         # Job command

resources:
  gpu_type: RTX_4090
  gpu_count: 1
  gpu_memory_min: 24
  disk_gb: 50

budget:
  max_hourly_rate: 0.50
  max_hours: 24
  max_cost_usd: 10.0

upload:
  source: ./input_files/
  destination: /workspace

sync:
  enabled: true
  source: /workspace/results
  interval_minutes: 15
  include_patterns: ["*.log", "*.dat"]

checkpoint:
  enabled: true
  patterns: ["*.pt", "*.restart"]

stages:                           # Multi-stage pipelines
  - name: equilibration
    command: run_equil.sh
  - name: production
    command: run_prod.sh

terminate_on_complete: true
```

## CLI Quick Reference

```bash
# Job submission
ccm jobs submit job.yaml
ccm jobs submit job.yaml --set KEY=VALUE     # Variable substitution
ccm jobs submit job.yaml --template pytorch-train --wait

# Job monitoring
ccm jobs list [--project X] [--status running]
ccm jobs status <job_id>
ccm jobs logs <job_id> [--tail 50] [--follow]
ccm jobs wait <job_id>

# Instance interaction
ccm exec <job_id> "command"
ccm ssh <job_id>
ccm upload <job_id> ./file /workspace/

# Batch operations
ccm batch submit sweep.yaml --max-parallel 5
ccm batch status --project my-project
ccm batch wait --project my-project

# Environment management (NEW)
ccm env export -n myenv -o environment.yml   # Export conda env
ccm env pack -n myenv -o env.tar.gz          # Create conda-pack
ccm env list                                  # List local envs

# GPU search and benchmarks
ccm search RTX_4090 --max-price 0.50
ccm benchmark run benchmark.yaml
ccm benchmark results

# Daemon (background monitor)
ccm daemon start
ccm daemon status
ccm daemon stop

# Maintenance
ccm cleanup [--execute]
ccm config show
```

## Environment Management

CCM supports 5 environment strategies, auto-selecting the fastest:

| Strategy | Setup Time | When to Use |
|----------|-----------|-------------|
| `docker_image` | <1 min | Pre-built image with everything installed |
| `conda_pack` | 1-2 min | Complex conda envs (OpenMM, LAMMPS, etc.) |
| `conda_env` | 5-15 min | Reproducible from environment.yml |
| `packages` | 5-15 min | Simple inline package list |
| `requirements` | 2-5 min | pip-only requirements.txt |

**Recommended workflow for complex environments:**
```bash
# 1. Create conda-pack from local environment
conda activate myenv
conda install conda-pack
conda-pack -n myenv -o myenv.tar.gz

# 2. Reference in job YAML
# environment:
#   conda_pack: ./myenv.tar.gz

# 3. Submit job
ccm jobs submit job.yaml --set MOLECULE=benzene
```

The conda-pack tarball is uploaded to the instance, unpacked, and activated
before the job command runs. No conda solve needed on the cloud.

## Key Implementation Details

- **Wrapper script** (`core/wrapper.py`): All jobs run through a SIGTERM-aware
  wrapper that catches spot preemption, sends checkpoint signals, and writes
  exit codes for daemon detection.

- **Setup sentinel** (`/tmp/.ccm_setup_done`): CCM waits for this file before
  declaring an instance ready, ensuring pip/conda installs complete.

- **Heartbeat** (`/workspace/.ccm_heartbeat`): Written every 60s by a background
  process on the instance, allowing CCM to detect when an instance was last alive.

- **Provisioning timeout**: Auto-adjusted based on environment strategy
  (300s for Docker, 600s for conda-pack, 1200s for conda env).

## Known Issues

- **Vast.ai CLI (v0.5.0)**: `show_instance` crashes when `start_date` is None
  (instance booting). CCM handles this by retrying in `get_instance()`.

- **Large conda-pack uploads**: 1-2 GB tarballs may take several minutes to
  upload via rsync. Consider building a Docker image for frequently used envs.

## Build & Test

```bash
cd ~/Workspace/main/46-CCM
pip install -e ".[dev]"
pytest tests/ --ignore=tests/test_e2e_full_lifecycle.py --ignore=tests/test_integration_vast.py
# 368+ tests, ~8s
```
