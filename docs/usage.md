# CCM Usage Guide

CloudComputeManager (CCM) is a GPU cloud management platform for running **any workload** on Vast.ai spot instances. It handles the full lifecycle: instance provisioning, software setup, job execution, progress monitoring, checkpoint/sync, preemption recovery, and cost tracking.

**Workload-agnostic** — works for LAMMPS, PyTorch, GROMACS, NAMD, custom scripts, or anything with a CLI.

**Location**: `/home/sf2/Workspace/main/46-CCM`

---

## Architecture

```
src/cloudcomputemanager/
├── agents/sdk.py         — Async Python SDK (CloudComputeManagerAgent, JobSpec)
├── api/                  — FastAPI REST API (/v1/jobs, /v1/instances, etc.)
├── benchmarks/           — GPU cost-performance benchmark framework
│   ├── engine.py         — Orchestrate benchmarks across GPU tiers
│   └── models.py         — BenchmarkSuite, BenchmarkRun, BenchmarkResult
├── checkpoint/           — Checkpoint detection (LAMMPS, PyTorch, generic patterns)
│   ├── detectors.py      — Application-specific checkpoint finders
│   └── orchestrator.py   — Checkpoint lifecycle management
├── cli/                  — Typer CLI (45+ commands)
│   ├── main.py           — All command registration and top-level shortcuts
│   ├── jobs.py           — submit, list, status, wait, reconnect, recover
│   ├── jobs_exec.py      — exec, upload, ssh by job_id
│   ├── batch.py          — Batch submit with Cartesian matrix expansion
│   ├── benchmarks.py     — Benchmark CLI commands
│   └── metrics.py        — Job performance metrics display
├── core/
│   ├── models.py         — SQLModel tables: Job, Instance, Checkpoint, CostRecord, SyncRecord
│   ├── wrapper.py        — SIGTERM-aware wrapper script builder (single source of truth)
│   ├── templates.py      — YAML template loading, merging, ${VAR} substitution
│   ├── config.py         — pydantic-settings (all CCM_ env vars)
│   ├── database.py       — Async SQLite via aiosqlite (auto-commit sessions)
│   ├── recovery.py       — Preemption recovery: checkpoint → new instance → restore
│   ├── cleanup.py        — Stale job / orphan instance cleanup
│   └── validation.py     — Pre-flight performance validation
├── daemon/
│   ├── monitor.py        — Main loop: health checks, completion, stage advancement,
│   │                       progress parsing, notifications, budget, reconciliation
│   └── service.py        — Daemon lifecycle (start/stop/PID/logs)
├── providers/
│   ├── base.py           — CloudProvider ABC, managed_instance, GPU tier search
│   └── vast.py           — Vast.ai: SSH retry (3x backoff), rsync, heartbeat, onstart
├── sync/engine.py        — Rsync-based continuous data synchronization
└── templates/            — 6 built-in YAML templates (quick-gpu, lammps-gpu, etc.)
```

## Installation

```bash
cd ~/Workspace/main/46-CCM
pip install -e ".[dev]"
ccm config init
echo "your-api-key" > ~/.vast_api_key

# Verify
ccm --version
ccm config show
```

## Job Lifecycle

1. **Submit** — Parse YAML (with template merging + variable substitution), search GPU offers, create instance
2. **Setup** — Run `setup:` commands via Vast.ai `--onstart-cmd` (apt, pip, conda, anything)
3. **Upload** — rsync local files to `/workspace/` on instance
4. **Execute** — Deploy SIGTERM-aware wrapper script (`core/wrapper.py`), launch with `nohup`
5. **Monitor** — Daemon polls `.ccm_exit_code` sentinel, tracks progress via pluggable parsers
6. **Multi-stage** — On stage completion (exit 0), daemon auto-advances to next stage
7. **Completion** — Sync results, fire notification hooks, terminate instance
8. **Preemption** — Wrapper traps SIGTERM → writes `.ccm_preempted` + exit 143 → daemon auto-recovers

## Agent Rules

If you are an AI agent using CCM, follow these rules:

1. **NEVER call `vastai` CLI directly.** Always use `ccm jobs submit` to create jobs and `ccm jobs cancel` to end them. Going around CCM creates ghost instances that cost money and are invisible to monitoring.
2. **ALWAYS set `project:`** in every job YAML — unique per campaign/project (e.g., `project: amaxine-2026`).
3. **ALWAYS include a `progress:` block** if you want the dashboard to show progress, rate, and ETA. Without it, those columns will be empty.
4. **For SSH/commands on a running instance**, use `ccm exec <job_id> "command"` or `ccm ssh <job_id>`.
5. **To check status**: `ccm jobs list --project your-project`
6. **If CCM lacks a capability you need**, ask the user rather than going around CCM.

### Required vs Optional Fields

| Field | Required? | What Happens If Missing |
|-------|-----------|------------------------|
| `name` | **Yes** | Job submission fails |
| `image` | **Yes** | Job submission fails |
| `command` | **Yes** | Job submission fails |
| `project` | Strongly recommended | Jobs can't be filtered by project; all agents see each other's jobs |
| `resources.gpu_type` | Recommended | Any available GPU (may get something expensive) |
| `budget.max_hourly_rate` | Recommended | No price cap; could get expensive instances |
| `progress` | Recommended | Dashboard shows no progress, rate, or ETA |
| `setup` | Optional | No pre-job software installation |
| `stages` | Optional | Job runs as single stage |
| `upload` | Optional | No files uploaded to instance |
| `sync` | Optional | No continuous result sync (final sync still happens) |
| `checkpoint` | Optional | No checkpoint detection for recovery |
| `notifications` | Optional | No alerts on completion/failure |

### Progress Regex Examples by Workload

The `progress:` block tells the daemon how to extract a progress number from your job's output. Adapt the regex to your specific workload:

```yaml
# LAMMPS (molecular dynamics)
progress:
  type: regex_parse
  file: /workspace/output/npt.log
  regex: '^\s*(\d+)\s+'
  total: 5000000

# PyTorch / ML training
progress:
  type: regex_parse
  file: /workspace/train.log
  regex: 'Epoch (\d+)'
  total: 100

# OpenMM (molecular dynamics)
progress:
  type: regex_parse
  file: /workspace/output.csv
  regex: '(\d+),'
  total: 50000000

# GROMACS
progress:
  type: regex_parse
  file: /workspace/md.log
  regex: 'Step\s+(\d+)'
  total: 10000000

# Generic percentage output
progress:
  type: regex_parse
  file: /workspace/job.log
  regex: '(\d+)%'
  total: 100

# Custom command (count output files, lines, etc.)
progress:
  type: custom_command
  command: "ls /workspace/output/*.dat 2>/dev/null | wc -l"
  total: 500

# File growth (no regex needed — tracks output directory size)
progress:
  type: file_growth
# Requires resources.expected_output_size_mb to compute percentage
```

### What Shows on the Dashboard

| Column | Source | Always Available? |
|--------|--------|-------------------|
| Status | Job status (RUNNING, RECOVERING, etc.) | Yes |
| Project | `project:` in YAML | Yes, if you set it |
| Name | `name:` in YAML | Yes |
| GPU | Auto-detected from Vast.ai instance | Yes (~30s after creation) |
| $/hr | Auto-detected from Vast.ai instance | Yes (~30s after creation) |
| Cost | Computed live: elapsed hours × $/hr | Yes |
| Runtime | Computed live: now - started_at | Yes |
| Progress | From `progress:` config → daemon parsing | **Only if `progress:` is configured** |

Progress column shows combined info when available: `67% · 145/s · 4.2h left`. For multi-stage jobs without progress config, shows `Stage 1/2`.

## Job YAML Reference

```yaml
name: my-job                          # Required
project: my-project                   # IMPORTANT: set a unique project name per agent/campaign
image: ubuntu:22.04                   # Docker image
setup: |                              # Pre-job setup (runs before job starts)
  apt-get update && apt-get install -y python3-pip
  pip install numpy torch
command: python3 train.py             # Job command to run

# ${VAR} substitution — set via: ccm jobs submit job.yaml --set KEY=VALUE
# Built-in vars: ${TIMESTAMP}, ${DATE}, ${RANDOM}

resources:
  gpu_type: RTX_4090                  # null = any available GPU
  gpu_count: 1
  gpu_memory_min: 16                  # GB VRAM
  cpu_cores: 8
  disk_gb: 50

budget:
  max_cost_usd: 10.0
  max_hours: 24
  max_hourly_rate: 0.50

provisioning:
  timeout: 600                        # Seconds to wait for SSH-ready (default: 300)

upload:
  source: ./input_files/
  destination: /workspace/input

sync:
  enabled: true
  source: /workspace/results
  interval_minutes: 15
  include_patterns: ["*.log", "*.dat", "*.dump"]

checkpoint:
  strategy: application               # application | filesystem
  interval_minutes: 30
  patterns: ["*.restart", "*.pt", "*.ckpt"]

# Multi-stage pipelines (daemon auto-advances on exit 0)
stages:
  - name: equilibration
    command: mpirun -np 16 lmp -in equil.inp > equil.log 2>&1
  - name: production
    command: mpirun -np 16 lmp -in prod.inp > prod.log 2>&1
  - name: analysis
    command: python3 analyze.py

# Progress tracking (daemon extracts metrics from running job)
progress:
  type: regex_parse                   # regex_parse | file_growth | custom_command
  file: /workspace/output.log         # File to tail
  regex: 'Step\s+(\d+)'              # Extract current value
  total: 5000000                      # Known total for % calculation

# Notification hooks (shell commands with variable substitution)
# Variables: ${JOB_ID}, ${JOB_NAME}, ${EXIT_CODE}, ${STATUS}, ${INSTANCE_ID}, ${PROJECT}
notifications:
  on_complete: "echo 'Done: ${JOB_NAME}' >> ~/ccm_alerts.log"
  on_failure: "curl -X POST https://hooks.slack.com/... -d '{\"text\":\"FAILED: ${JOB_NAME}\"}'"
  on_budget_exceeded: "echo 'BUDGET: ${JOB_NAME}' >> ~/ccm_alerts.log"
```

## CLI Reference

```bash
# === Job Management ===
ccm jobs submit job.yaml                        # Submit job
ccm jobs submit job.yaml --set KEY=VALUE        # With variable substitution
ccm jobs submit job.yaml --template lammps-gpu  # Using template
ccm jobs list [--project X] [--status running]  # List jobs
ccm jobs status <job_id> [--watch]              # Detailed status
ccm jobs logs <job_id> [--tail 100] [--follow]  # View logs
ccm jobs wait <job_id> [--timeout 3600]         # Wait for completion
ccm jobs cancel <job_id>                        # Cancel job
ccm jobs reconnect [job_id]                     # Reconnect after downtime
ccm jobs recover [job_id]                       # Retry preempted/failed jobs
ccm jobs metrics <job_id>                       # Show performance metrics

# === Instance Interaction (by job_id) ===
ccm exec <job_id> "tail -20 output.log"         # Run command on instance
ccm upload <job_id> ./file.py /workspace/        # Upload file mid-flight
ccm ssh <job_id>                                 # Interactive SSH session

# === Batch Operations ===
ccm batch submit jobs/*.yaml --parallel 5        # Submit multiple configs
ccm batch submit sweep.yaml                      # Matrix expansion (Cartesian product)
ccm batch status --project my-project            # Aggregate status
ccm batch wait --project my-project              # Wait for all jobs
ccm batch cancel --project X --force             # Cancel all in project

# === Benchmarks (cost-performance analysis) ===
ccm benchmark run benchmark.yaml                 # Run suite across GPU tiers
ccm benchmark run benchmark.yaml --gpu RTX_3060  # Limit to one GPU type
ccm benchmark results [suite_id]                 # Show results table

# === Infrastructure ===
ccm instances search --gpu RTX_4090 --max-price 0.50
ccm instances list
ccm instances terminate <instance_id>
ccm daemon start [--foreground]                  # Start background monitor
ccm daemon stop
ccm daemon status
ccm config show
ccm config init
ccm templates list
ccm templates show lammps-gpu
ccm cleanup [--execute]                          # Clean stale jobs/orphan instances
```

## Batch Matrix Expansion

A matrix YAML expands to N jobs via Cartesian product:

```yaml
# sweep.yaml
template: job-template.yaml
matrix:
  STRESS: [50, 100, 150, 200, 250, 300]
  DIRECTION: [X, Y]
# → 12 jobs (6 x 2), each with ${STRESS} and ${DIRECTION} substituted
```

```bash
ccm batch submit sweep.yaml            # Creates 12 jobs automatically
ccm batch submit sweep.yaml --dry-run  # Preview without submitting
```

## Python SDK

```python
from cloudcomputemanager.agents import CloudComputeManagerAgent, JobSpec

async with CloudComputeManagerAgent() as vm:
    # Submit a job
    job = await vm.submit(JobSpec(
        name="simulation",
        command="mpirun lmp -in input.in",
        gpu_type="RTX_3060",
        max_hourly_rate=0.10,
    ))

    # Get direct SSH credentials (raw host/port/key, not passthrough)
    creds = await vm.get_ssh_credentials(job.job_id)
    # → {"host": "ssh1.vast.ai", "port": 22, "user": "root", "key_path": "..."}

    # Wait for completion
    result = await vm.wait_for_completion(job.job_id)
    # → JobResult(success=True, total_cost_usd=1.44, output_location="...")

    # Or stream events
    async for event in vm.watch_jobs([job.job_id]):
        if event.type == "job.completed":
            results = await vm.get_results(event.job_id)

    # Batch submit
    jobs = await vm.submit_batch([spec1, spec2, spec3], max_concurrent=5)

    # Search GPUs
    offers = await vm.search_gpus(gpu_type="RTX_4090", max_price=0.50)
```

## Benchmark YAML

```yaml
name: my-workload-bench
description: Compare GPU cost-performance for my workload
workload:
  image: ubuntu:22.04
  setup: apt-get install -y my-software
  command: my-software --benchmark --threads ${NCPUS}
  files:
    - benchmarks/input.data
matrix:
  gpu_type: [RTX_3060, RTX_4070, RTX_4090]
metrics:
  - name: throughput
    source: stdout                    # stdout | stderr
    regex: 'throughput:\s+([\d.]+)'   # Capture group 1 = metric value
    unit: ops/s
repetitions: 3
timeout: 300
budget:
  max_per_instance: 0.50
```

## Resilience & Recovery

| Scenario | What Happens |
|----------|-------------|
| **Laptop sleeps/closes** | Job keeps running (nohup). Run `ccm reconnect` when back. |
| **Instance preempted** | Wrapper catches SIGTERM, sends SIGUSR1 for app checkpoint, writes exit 143. Daemon auto-recovers from checkpoint. |
| **Daemon was down** | On restart, `_reconcile_stale_jobs()` detects completed/dead jobs and handles them. |
| **Instance dies, daemon down** | `ccm reconnect` checks Vast.ai API, marks for recovery. Then `ccm recover`. |
| **Budget exceeded** | Daemon auto-terminates job, syncs results first, fires `on_budget_exceeded` notification. |

## Multi-Agent / Multi-Project Usage

CCM uses a **single daemon, single database, single Vast.ai API key**. Multiple agents share this infrastructure safely. Isolation is via the `project` field.

**Rules for agents:**
1. **NEVER call `vastai` CLI directly** for creating or destroying instances. Always use `ccm jobs submit` to create and `ccm jobs cancel` to terminate. Going around CCM creates ghost instances that cost money and are invisible to monitoring.
2. **Always set a unique `project` name** in every job YAML
3. **Always filter by your project** when listing/waiting: `ccm jobs list --project my-project`
4. **The daemon monitors ALL jobs** — don't start your own daemon
5. **Each job gets its own instance** — no resource conflicts
6. **For SSH/commands on a running instance**, use `ccm exec <job_id> "command"` or `ccm ssh <job_id>`. These are safe — they don't change infrastructure state.
7. **If CCM lacks a capability you need**, ask the user rather than going around CCM

```yaml
# Agent A (Amaxine project)
project: amaxine-2026

# Agent B (Pt-catalysis)
project: pt-catalysis

# Agent C (ML-solvent)
project: ml-solvent
```

```bash
# Each agent scopes to its own project:
ccm jobs list --project amaxine-2026
ccm batch wait --project pt-catalysis

# ccm reconnect and ccm daemon check ALL projects — correct behavior
```

## Dashboard

```bash
ccm dashboard                    # Open web dashboard on port 8765
ccm dashboard --port 9000        # Custom port
ccm dashboard --no-browser       # Start without opening browser
```

The dashboard shows a single-page view at `http://localhost:8765/dashboard` with:
- **Alert banner** — failed jobs, stalled progress, budget warnings
- **Summary cards** — active jobs, today's spend, weekly spend, recoveries
- **Active jobs table** — status, project, progress bars, rate, ETA, cost, sync status, action buttons
- **Events feed** — completions, failures, preemptions, syncs from the last 24h
- **Cost breakdown** — per-project and per-GPU-type aggregation
- **Finished jobs** — collapsible section for recently completed jobs

Live updates via Server-Sent Events — no manual refresh needed. Actions (cancel, sync, view logs) available directly from the table. Reads from the same SQLite database as the daemon.

## Built-in Templates

| Name | Image | GPU | Checkpoint | Use Case |
|------|-------|-----|------------|----------|
| `quick-gpu` | nvidia/cuda | RTX_3060 | No | Quick experiments |
| `lammps-gpu` | NGC LAMMPS | RTX_3060 | Yes (restart.*) | MD simulations |
| `namd-production` | NGC NAMD | RTX_3060 | Yes (*.restart.*) | NAMD production |
| `pytorch-train` | PyTorch | RTX_4090 | Yes (*.pt) | ML training |
| `jupyter-dev` | Jupyter | RTX_3060 | No | Interactive dev |
| `llm-inference` | vLLM | RTX_4090 | No | LLM serving |

## Environment Management

CCM supports deploying complex software environments to cloud instances via the
`environment:` field in job YAML. Five strategies are supported, auto-selected
by priority:

| Strategy | Field | Setup Time | Best For |
|----------|-------|-----------|----------|
| Docker image | `docker_image:` | <1 min | Pre-built images |
| Conda pack | `conda_pack:` | 1-2 min | Complex conda envs (OpenMM, LAMMPS) |
| Conda env file | `conda_env:` | 5-15 min | Reproducibility from environment.yml |
| Inline packages | `packages:` | 5-15 min | Simple package lists |
| Requirements | `requirements:` | 2-5 min | pip-only projects |

### Example: Conda Pack (Recommended for Complex Environments)

```bash
# 1. Create pack from local conda environment
conda activate myenv
conda install conda-pack
conda-pack -n myenv -o myenv.tar.gz

# 2. Reference in job YAML
# environment:
#   conda_pack: ./myenv.tar.gz

# 3. Submit
ccm jobs submit job.yaml
```

### Example: Inline Packages

```yaml
environment:
  packages:
    apt:
      - libfftw3-dev
    conda:
      - openmm
      - openmmtools
    pip:
      - alchemlyb
  channels: ["conda-forge"]
```

### Example: Docker Image Override

```yaml
environment:
  docker_image: myuser/openmm-cuda12:latest
```

### How It Works

1. Instance boots with base Docker image
2. User setup commands run via onstart (basic apt/pip)
3. Instance becomes SSH-ready, CCM detects sentinel
4. Regular files uploaded (scripts, input data)
5. Environment files uploaded (conda-pack tarball, env.yml)
6. Environment setup runs via SSH (unpack, conda create, etc.)
7. Job wrapper deployed with activation prefix
8. Job runs in the configured environment

### CLI Commands

```bash
ccm env export -n myenv -o environment.yml   # Export conda env
ccm env pack -n myenv -o env.tar.gz          # Create conda-pack tarball
ccm env list                                  # List local conda envs
```

### Known Limitations

**Conda-pack and CUDA:** Conda-pack tarballs contain pre-compiled CUDA libraries
(PTX code) for the GPU architecture where the pack was created. A pack built on
an RTX 5080 (sm_120) will fail on an RTX 3060 (sm_86) with
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. For GPU workloads like OpenMM, LAMMPS, or
PyTorch, use `conda_env` strategy (slower but architecture-independent) or a
pre-built Docker image. Conda-pack works fine for CPU-only and pure Python packages.

**Provisioning timeout:** Complex environments (conda env with 20+ packages) can
take 10-15 minutes to set up. CCM auto-adjusts the timeout based on strategy,
but if you hit timeouts, increase `provisioning.timeout` in the job YAML.

### Implementation

- `core/environment.py` — Parsing, validation, setup command generation
- `cli/environments.py` — CLI commands for export, pack, list
- `cli/jobs.py` — Integration into submit flow
- `docs/ENVIRONMENT_DESIGN.md` — Full design document with future plans

---

## Key Files for Development

| File | What It Controls |
|------|-----------------|
| `core/models.py` | All data models — Job fields, enums, CostRecord |
| `core/wrapper.py` | SIGTERM-aware wrapper (single source, used by 3 call sites) |
| `core/templates.py` | YAML loading, template merging, `${VAR}` substitution |
| `core/environment.py` | Environment parsing, validation, setup command generation |
| `daemon/monitor.py` | Main loop: completion, health, stages, progress, notifications, budget |
| `cli/jobs.py` | Job submission flow (with environment integration), wait, reconnect |
| `cli/batch.py` | Matrix expansion + parallel batch submission |
| `cli/environments.py` | `ccm env` commands: export, pack, list |
| `providers/vast.py` | Vast.ai integration: SSH retry, rsync, onstart, heartbeat |
| `providers/base.py` | CloudProvider ABC, wait_for_ready (with setup sentinel check) |
| `agents/sdk.py` | CloudComputeManagerAgent — async Python API for agents |
| `benchmarks/engine.py` | Benchmark orchestration across GPU tiers |
