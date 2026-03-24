---
name: ccm
description: CloudComputeManager — GPU cloud management platform for running any workload on Vast.ai with checkpointing, preemption recovery, multi-stage pipelines, benchmarking, and cost optimization. Use when working with CCM code, submitting cloud jobs, managing instances, or building on the CCM platform.
argument-hint: "[topic or task]"
---

# CloudComputeManager (CCM) Agent Skill

You are working with **CloudComputeManager** at `/home/sf2/Workspace/main/46-CCM`.

CCM manages GPU workloads on Vast.ai spot instances. It handles the full lifecycle: instance provisioning, software setup, job execution, progress monitoring, checkpoint/sync, preemption recovery, and cost tracking. It is **workload-agnostic** — works for LAMMPS, PyTorch, GROMACS, custom scripts, or anything with a CLI.

**User request**: $ARGUMENTS

If a specific topic or task is given, focus on that. If no argument, give a concise overview and ask what they need. For any code changes, **always read the relevant source files first** — do not guess at the implementation.

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

## Job Lifecycle

1. **Submit** — Parse YAML (with template merging + variable substitution), search GPU offers, create instance
2. **Setup** — Run `setup:` commands via Vast.ai `--onstart-cmd` (apt, pip, conda, anything)
3. **Upload** — rsync local files to `/workspace/` on instance
4. **Execute** — Deploy SIGTERM-aware wrapper script (`core/wrapper.py`), launch with `nohup`
5. **Monitor** — Daemon polls `.ccm_exit_code` sentinel, tracks progress via pluggable parsers
6. **Multi-stage** — On stage completion (exit 0), daemon auto-advances to next stage
7. **Completion** — Sync results, fire notification hooks, terminate instance
8. **Preemption** — Wrapper traps SIGTERM → writes `.ccm_preempted` + exit 143 → daemon auto-recovers

## Job YAML Reference

```yaml
name: my-job                          # Required
project: my-project                   # For batch grouping
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

## CLI Quick Reference

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
ccm batch submit sweep.yaml         # Creates 12 jobs automatically
ccm batch submit sweep.yaml --dry-run  # Preview without submitting
```

## Python SDK (for Agents)

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

    # Get direct SSH credentials (not passthrough — raw host/port/key)
    creds = await vm.get_ssh_credentials(job.job_id)
    # → {"host": "ssh1.vast.ai", "port": 22, "user": "root", "key_path": "/home/..."}

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

Key files for resilience:
- `core/wrapper.py` — SIGTERM trap, exit code 143, .ccm_preempted marker
- `daemon/monitor.py` → `_reconcile_stale_jobs()` — startup reconciliation
- `daemon/monitor.py` → `handle_job_completion()` — exit 143 → auto-recovery
- `cli/jobs.py` → `reconnect_jobs()` — manual reconnection without daemon
- `providers/vast.py` — instance heartbeat (writes .ccm_heartbeat every 60s)

## Key Files for Development

| File | What It Controls |
|------|-----------------|
| `AGENTS.md` | Current project status and sprint tracker |
| `docs/SPRINT_2026-03-23.md` | Full sprint plan and design decisions |
| `core/models.py` | All data models — Job fields, enums, CostRecord |
| `core/wrapper.py` | SIGTERM-aware wrapper (single source, used by 3 call sites) |
| `core/templates.py` | YAML loading, template merging, `${VAR}` substitution |
| `daemon/monitor.py` | Main loop: completion, health, stages, progress, notifications, budget |
| `cli/jobs.py` | Job submission flow, wait, reconnect, recover |
| `cli/batch.py` | Matrix expansion + parallel batch submission |
| `providers/vast.py` | Vast.ai integration: SSH retry, rsync, onstart, heartbeat |
| `agents/sdk.py` | CloudComputeManagerAgent — async Python API for agents |
| `benchmarks/engine.py` | Benchmark orchestration across GPU tiers |

## Environment

```bash
cd ~/Workspace/main/46-CCM
pip install -e ".[dev]"
ccm config init
echo "your-api-key" > ~/.vast_api_key

# Tests (314 passing)
pytest tests/ --ignore=tests/test_e2e_full_lifecycle.py --ignore=tests/test_integration_vast.py

# Integration tests (requires API key, costs money)
pytest tests/test_integration_vast.py --run-integration
```

## Built-in Templates

| Name | Image | GPU | Checkpoint | Use Case |
|------|-------|-----|------------|----------|
| `quick-gpu` | nvidia/cuda | RTX_3060 | No | Quick experiments |
| `lammps-gpu` | NGC LAMMPS | RTX_3060 | Yes (restart.*) | MD simulations |
| `namd-production` | NGC NAMD | RTX_3060 | Yes (*.restart.*) | NAMD production |
| `pytorch-train` | PyTorch | RTX_4090 | Yes (*.pt) | ML training |
| `jupyter-dev` | Jupyter | RTX_3060 | No | Interactive dev |
| `llm-inference` | vLLM | RTX_4090 | No | LLM serving |
