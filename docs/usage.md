# CCM Usage Guide

CloudComputeManager (CCM) is a GPU cloud management platform for running **any workload** on Vast.ai spot instances. It handles the full lifecycle: instance provisioning, software setup, job execution, progress monitoring, checkpoint/sync, preemption recovery, and cost tracking.

**Workload-agnostic** — works for LAMMPS, PyTorch, GROMACS, NAMD, custom scripts, or anything with a CLI.

**Location**: `/home/sf2/Workspace/main/46-CCM`

---

## Architecture

```
src/cloudcomputemanager/
├── agents/sdk.py          — Async Python SDK (CloudComputeManagerAgent, JobSpec)
├── api/                   — FastAPI REST API (/v1/jobs, /v1/instances, etc.)
├── benchmarks/            — GPU cost-performance benchmark framework
├── checkpoint/            — Checkpoint detection + restart adapters (8 app types)
├── cli/
│   ├── main.py            — All command registration (50+ commands)
│   ├── jobs.py            — submit, list, status, wait, reconnect, recover
│   ├── jobs_exec.py       — exec, upload, ssh by job_id
│   ├── batch.py           — Batch submit with Cartesian matrix expansion
│   ├── environments.py    — ccm env export/pack/list
│   └── benchmarks.py      — Benchmark CLI commands
├── core/
│   ├── models.py          — SQLModel: Job, Instance, Checkpoint, CostRecord
│   ├── instances.py       — Instance labels, sync from Vast.ai API, safe termination
│   ├── wrapper.py         — SIGTERM-aware wrapper script builder
│   ├── environment.py     — Environment parsing, setup command generation
│   ├── templates.py       — YAML loading, merging, ${VAR} substitution
│   ├── config.py          — pydantic-settings (all CCM_ env vars)
│   ├── database.py        — Async SQLite via aiosqlite
│   ├── recovery.py        — Preemption recovery orchestration (restart adapter chain)
│   └── cleanup.py         — Stale job / orphan instance cleanup
├── daemon/
│   ├── monitor.py         — Main loop: health, completion, stages, progress, budget
│   └── service.py         — Daemon lifecycle (start/stop/PID/logs)
├── dashboard/             — Web UI (FastAPI + Jinja2 + HTMX + SSE)
│   ├── routes.py          — Page, partial, SSE, and action endpoints
│   ├── data.py            — Data aggregation queries
│   └── templates/         — HTML templates + static assets
├── providers/
│   ├── base.py            — CloudProvider ABC, GPU tier search
│   └── vast.py            — Vast.ai: SSH retry, rsync, labels, heartbeat
├── sync/engine.py         — Rsync-based data synchronization
└── templates/             — 6 built-in YAML templates
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
| `restart` | Optional | CCM auto-detects app type from command (NAMD, GROMACS, LAMMPS, etc.) |
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

# Restart override (for apps not auto-detected by CCM)
# If omitted, CCM auto-detects from the command string (NAMD, GROMACS, LAMMPS, etc.)
restart:
  command: "my-simulator --resume --from checkpoint.dat"
  detect_checkpoint: "checkpoint.dat"  # Only use restart command if this glob matches

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

# === Dashboard & Infrastructure ===
ccm dashboard [--port 8765]                      # Web dashboard
ccm reconnect [job_id]                           # Rehydrate after downtime
ccm instances search --gpu RTX_4090 --max-price 0.50
ccm instances list
ccm daemon start [--foreground]                  # Background monitor
ccm daemon stop
ccm daemon status
ccm config show
ccm templates list
ccm templates show lammps-gpu
ccm cleanup [--execute]                          # Clean stale jobs

# === Environment Management ===
ccm env export -n myenv -o environment.yml       # Export conda env
ccm env pack -n myenv -o env.tar.gz              # Create conda-pack
ccm env list                                     # List local envs
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
| **Instance preempted (SIGTERM)** | Wrapper catches SIGTERM, sends SIGUSR1 for app checkpoint, writes exit 143. Daemon auto-detects app type, generates restart command via adapter chain, recovers on new instance. |
| **Instance disappears** | Daemon detects `instance_not_found`, syncs last data, provisions new instance, uploads checkpoints, restart adapter prepares resume command, job continues from checkpoint. |
| **GPU crash (SIGSEGV/SIGABRT)** | If exit code in `recover_on_exit_codes`, auto-recovers on different instance. Bad instance offer blacklisted for 24h. |
| **Daemon was down** | On restart, `_reconcile_stale_jobs()` detects completed/dead jobs and handles them. |
| **Budget exceeded** | Daemon auto-terminates job, syncs results first, fires `on_budget_exceeded` notification. |

### Long-Running Jobs (>4 hours)

Vast.ai instances can disappear at any time — even on-demand. For jobs longer than 4 hours, **checkpoint-restart is mandatory**. See [`docs/SPOT_INSTANCE_SURVIVAL_GUIDE.md`](SPOT_INSTANCE_SURVIVAL_GUIDE.md) for the complete guide.

Key settings for long jobs:
```yaml
sync:
  interval_minutes: 5                      # Sync checkpoints every 5 min
checkpoint:
  patterns: ["checkpoint.*", "*.restart.*"] # Your app's checkpoint files
retry:
  max_attempts: 10                          # Allow many recoveries
  recover_on_exit_codes: [139, 137, 134]    # GPU crashes → auto-retry
resources:
  cuda_version_min: 12.6                    # For NGC containers
```

### Automatic Restart Adapters

When a job is preempted and recovered onto a new instance, CCM **auto-detects the application type** from the command string and uses the appropriate restart strategy. No user configuration needed for supported applications.

The recovery flow:
1. Daemon detects preemption (exit code 143 or instance disappears)
2. Syncs latest checkpoint files from old instance to local disk
3. Provisions a new instance, uploads original files + synced checkpoints
4. **Restart adapter chain** determines the correct resume command
5. Job starts on new instance and resumes from checkpoint

#### Supported Applications

| Application | Detection | Checkpoint Files | Restart Strategy |
|-------------|-----------|-----------------|-----------------|
| **NAMD** | `namd` in command | `.restart.xsc/.coor/.vel` | Generates full restart config (cooling/production aware) |
| **GROMACS** | `gmx` or `mdrun` | `*.cpt` | Injects `-cpi state.cpt` flag |
| **LAMMPS** | `lmp` or `lammps` | `restart.*.bin` | Generates wrapper passing restart file as variable |
| **Quantum ESPRESSO** | `pw.x`, `ph.x` | `*.save/` dirs | Rewrites `restart_mode = 'restart'` in input |
| **VASP** | `vasp` | `WAVECAR` + `CONTCAR` | Copies CONTCAR → POSCAR (WAVECAR auto-detected) |
| **PyTorch Lightning** | `lightning` or `trainer.fit` | `*.ckpt` | Appends `--ckpt_path last` |
| **HF Trainer** | `--do_train` or `run_clm` | `checkpoint-*/` dirs | Appends `--resume_from_checkpoint True` |
| **Generic** | Everything else | (none checked) | Re-runs original command (industry standard) |

The adapter chain is ordered by specificity — first match wins, Generic is always last.

#### How It Works (Priority Order)

1. **User-defined restart** (highest priority) — if `restart:` is set in job YAML and checkpoint files match `detect_checkpoint`, use `restart.command`
2. **Auto-detected adapter** — match command string against adapter chain, call `prepare_restart()` with synced checkpoint files
3. **Original command** (fallback) — if no adapter matches or no checkpoint files found, re-run the original command

#### Self-Healing Applications

Some applications automatically resume if checkpoint files are present:
- **GROMACS** with `-cpi` flag: reads checkpoint on startup
- **VASP**: auto-reads WAVECAR if it exists (ISTART default)
- **PyTorch scripts** that check for `checkpoint.pt` on startup

For these, CCM's periodic sync ensures checkpoint files are on the new instance. The adapter adds any missing flags; the application does the rest.

#### Custom Applications (restart: YAML)

For applications not in the adapter list, add a `restart:` section to your job YAML:

```yaml
name: my-custom-job
command: "my-simulator --config run.conf"

restart:
  command: "my-simulator --config run.conf --resume-from latest"
  detect_checkpoint: "checkpoint_*.dat"  # Only use restart command if these exist
```

CCM checks `restart.command` first (highest priority). If `detect_checkpoint` is set, the restart command is only used when matching files exist in the sync directory — otherwise it falls back to the original command.

**Tip**: If your application already checks for checkpoint files on startup (like the PyTorch example above), you don't need a `restart:` section at all. CCM syncs all files to `/workspace` on the new instance, and your script picks them up automatically.

#### NAMD Deep Restart (Built-In)

NAMD requires the most complex restart handling — a full config file must be generated with `binCoordinates`/`binVelocities`/`extendedSystem`, correct `firsttimestep`, and awareness of cooling vs production phases. CCM handles this automatically:

1. Finds synced `.restart.coor/.vel/.xsc` files
2. Parses step number from `.xsc`
3. Detects cooling vs production phase
4. Generates a restart config that resumes from the exact step
5. Preserves existing DCD as `simulation_before_{step}.dcd`
6. Resumes from the last checkpoint

This was battle-tested during a 700+ job campaign (20-sample ensemble, 14 days, ~$100 total).

See [`docs/NAMD_CHECKPOINT_RESTART_DESIGN.md`](NAMD_CHECKPOINT_RESTART_DESIGN.md) for NAMD-specific implementation details.

#### Implementation

The adapter system lives in `checkpoint/restart_adapters.py`:
- `RestartAdapter` ABC with `detect(command)` and `prepare_restart(command, sync_dir, job_id)`
- 8 adapter implementations in a priority-ordered registry
- `get_restart_adapter(command)` returns the first matching adapter
- `recovery.py` calls the adapter chain during `recover_job()`
- 76 unit tests + end-to-end validated on real Vast.ai infrastructure

## Multi-Agent / Multi-Project Usage

CCM uses a **single daemon, single database, single Vast.ai API key**. Multiple agents share this infrastructure safely. Isolation is via the `project` field. See [Agent Rules](#agent-rules) above for the complete rule set.

```yaml
# Each agent uses a unique project name:
project: amaxine-2026      # Agent A
project: pt-catalysis       # Agent B
project: ml-solvent         # Agent C
```

```bash
# Each agent scopes commands to its own project:
ccm jobs list --project amaxine-2026
ccm batch wait --project pt-catalysis

# ccm reconnect and ccm daemon check ALL projects — correct behavior
```

Every instance created through CCM is labeled on Vast.ai: `ccm|{job_id}|{project}|{name}`. This ensures instances can be matched to jobs even if the local database is wiped. Instances without `ccm|` labels show as "Unmanaged" on the dashboard with a warning.

## Dashboard

```bash
ccm dashboard                    # Open web dashboard on port 8765
ccm dashboard --port 9000        # Custom port
ccm dashboard --no-browser       # Start without opening browser
```

The dashboard shows a single-page view at `http://localhost:8765/dashboard` with:
- **Summary cards** — active jobs, today's spend (live), weekly spend, burn rate, recoveries
- **Unmanaged instances warning** — instances on Vast.ai not created through CCM (with terminate button)
- **Active jobs table** — status, project, name, GPU, $/hr, cost (live), runtime, progress, actions (Cancel, SSH, Logs)
- **Events feed** — completions, failures, preemptions from daemon log (24h)
- **Cost breakdown** — per-project totals + per-GPU-type breakdown (live computed from elapsed × hourly rate)
- **Finished jobs** — collapsible section for recently completed jobs

Live updates via Server-Sent Events — no manual refresh needed. All costs are computed live (elapsed × hourly rate), not stored values. Instance data (GPU type, $/hr, SSH info) is auto-synced from the Vast.ai API.

## Best Practices & Tips

### Instance Selection (Critical for Long Jobs)

```yaml
resources:
  reliability_min: 0.99         # REQUIRED for jobs > 4 hours
  cuda_version_min: 12.6        # For NGC containers (NAMD, PyTorch NGC)
```

**`reliability_min: 0.99` is the single most important setting.** From 469 jobs in the hydrogenation campaign:
- Without filter: **1% survival** (158 dead, 1 alive)
- With `reliability_min: 0.99`: **66% survival** (23 alive, 12 dead from other causes)

The reliability score is Vast.ai's measure of host uptime history. Hosts with 0.99+ have proven thousands of hours of stable operation. Price does NOT correlate with reliability — the cheapest 0.99+ RTX 4090 is $0.27/hr, cheaper than many unreliable hosts.

**You do NOT need a price floor.** The reliability filter handles quality. Just set `max_hourly_rate` to cap the upper end. CCM picks the cheapest offer matching all filters.

**`min_duration_hours` is optional.** Most reliable hosts already have long duration limits (1000+ hours). Only needed if you specifically want to exclude short-rental hosts.

### Docker Image Strategy

- **Use common base images** (`nvidia/cuda`, `pytorch/pytorch`, `ubuntu`) — these are pre-cached on most hosts and start in seconds
- **NGC HPC images** (`nvcr.io/hpc/namd`, `nvcr.io/hpc/gromacs`) are 5-8 GB and NOT cached on most hosts — provisioning takes 15-30 minutes
- **If using a large image**: set `provisioning.timeout: 1800` (30 min) to allow time for image pull
- **Alternative**: use `setup:` commands to install software on a cached base image

### Budget Configuration

```yaml
budget:
  max_cost_usd: 10.0            # Per-job total cost limit
  max_hours: 9999               # Don't time-kill recovered jobs
  max_hourly_rate: 0.55         # Allows reliable datacenter hosts
```

- **Don't set `max_hours` too low** — recovered jobs inherit the original creation time, so a 48h max_hours will kill a job that's been through recovery cycles even if it's only run for 10 actual hours
- **`max_hourly_rate`** filters at search time — set it high enough to reach reliable hosts ($0.40-0.60 for RTX 4090)

### Retry & Recovery

```yaml
retry:
  max_attempts: 999             # Infinite retries for production jobs
  recover_on_exit_codes: [139, 137, 134]  # GPU crashes → auto-retry
```

- **Set `max_attempts` high** for production campaigns — instances will fail and need recovery
- **`recover_on_exit_codes`** catches instance-specific GPU crashes (SIGSEGV, OOM, SIGABRT) that would succeed on a different instance

### Sync & Checkpointing

```yaml
sync:
  interval_minutes: 5           # Sync every 5 min — max lost progress on failure
checkpoint:
  patterns: ["*.restart.*"]     # Your app's checkpoint files
```

- **Sync frequently** — if a host dies, you lose everything since the last sync
- **5 minutes** is the recommended interval for production jobs (balance between overhead and data safety)
- **Ensure your application writes checkpoint files** that can be used to resume (NAMD `restartfreq`, LAMMPS `restart`, PyTorch `torch.save`)

### Common Pitfalls

1. **Cheap ≠ Good**: $0.04/hr RTX 3060 instances are 10x cheaper but die 10x more often. Net cost is often higher due to wasted compute and recovery overhead.
2. **"On-demand" doesn't mean reliable**: On Vast.ai, on-demand prevents bidding preemption but NOT host failures. The host machine itself can still go offline.
3. **Recovery blocks monitoring**: Fixed in v0.2.0 (#26). If running an older version, recovery processing blocks the entire daemon — no syncs or health checks until recovery completes.
4. **Ghost jobs**: If a host disappears between daemon poll cycles, the DB shows "RUNNING" but the instance is dead. The daemon detects this on the next cycle and routes to recovery.
5. **DCD on restart**: NAMD creates a new DCD file on restart (doesn't append). Use CatDCD or MDAnalysis to merge segments.

### Cost Estimation (with `reliability_min: 0.99`)

| GPU | Cheapest reliable | ns/day (50K) | Hours for 15ns | $/job | Verdict |
|-----|-------------------|-------------|----------------|-------|---------|
| RTX 4070 Ti | $0.08 | 22 | 16.5 | $1.32 | Best value (if 16h is OK) |
| RTX 4080S | $0.14 | 30 | 12.1 | $1.71 | Good balance |
| RTX 4090 | $0.27 | 35 | 10.4 | $2.78 | Safest (completes fastest) |
| A100 SXM | $0.77 | 55 | 6.6 | $5.07 | Overkill for 50K atoms |

**Recommendation**: RTX 4090 with `reliability_min: 0.99` for production campaigns. The cheapest reliable RTX 4090 is ~$0.27/hr — no need for a price floor, the reliability filter handles quality.

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
| `core/models.py` | All data models — Job fields (incl. `restart_json`), enums, CostRecord |
| `core/recovery.py` | Preemption recovery — adapter chain, user-defined restart, instance replacement |
| `core/wrapper.py` | SIGTERM-aware wrapper (single source, used by 3 call sites) |
| `core/templates.py` | YAML loading, template merging, `${VAR}` substitution |
| `core/environment.py` | Environment parsing, validation, setup command generation |
| `checkpoint/restart_adapters.py` | RestartAdapter ABC + 8 implementations (NAMD, GROMACS, LAMMPS, QE, VASP, PL, HF, Generic) |
| `checkpoint/namd_restart.py` | NAMD-specific restart config generation (cooling/production phase aware) |
| `daemon/monitor.py` | Main loop: completion, health, stages, progress, notifications, budget |
| `cli/jobs.py` | Job submission flow (with environment integration), wait, reconnect |
| `cli/batch.py` | Matrix expansion + parallel batch submission |
| `cli/environments.py` | `ccm env` commands: export, pack, list |
| `providers/vast.py` | Vast.ai integration: SSH retry, rsync, onstart, heartbeat |
| `providers/base.py` | CloudProvider ABC, wait_for_ready (with setup sentinel check) |
| `agents/sdk.py` | CloudComputeManagerAgent — async Python API for agents |
| `benchmarks/engine.py` | Benchmark orchestration across GPU tiers |
