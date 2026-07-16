# CloudComputeManager - Agent Context

## Project Overview

**CloudComputeManager** (CCM) is a cloud compute management platform for scientific computing on Vast.ai spot instances. Core capabilities:
- Job submission with custom setup commands and multi-stage pipelines
- Automatic checkpointing and preemption recovery
- Continuous data synchronization with local storage
- LAMMPS-aware progress monitoring
- Per-job budget enforcement and cost tracking
- Agent-native APIs (CLI, REST, Python SDK)
- Automated GPU cost-performance benchmarking

## Current Status (2026-04-09)

**Version**: 0.2.0-dev (Sprint Phases 1-4 COMPLETE + Resilience Hardening + Restart Adapters)
**Tests**: 390 unit tests passing (76 adapter + 314 existing), real Vast.ai e2e preemption recovery validated
  *(2026-07-09 correction: the default run — `pytest tests/ --ignore=tests/test_e2e_full_lifecycle.py --ignore=tests/test_integration_vast.py` — now collects 449 tests; measured result: 445 passed, 4 failed in 7.04s (2 in `tests/test_dashboard.py::TestDashboardRoutes`, 2 in `tests/test_restart_adapters.py::TestNAMDRestartAdapter` — see `docs/audits/2026-07-09-full-audit.md` in the parent workspace). Run via a fresh `pip install -e ".[dev]"` venv on Python 3.12.3; no persistent Python env for CCM exists on this machine — create a venv per CLAUDE.md to run the suite.)*
  *(2026-07-16 update: the 2 NAMD adapter failures are fixed on main (`51fb6ad`, owner-directed) — the cooling-phase gate was campaign-specific logic applied to all NAMD jobs; it is now scoped to cooling-protocol commands, the generated config references the discovered checkpoint files, and multi-triplet dirs resume from the furthest checkpoint. Suite now collects 451: 449 passed, 2 failed — only the pre-existing dashboard/Starlette drift remains (test-side: `[r.path for r in app.routes]` hits `_IncludedRouter` without `.path`).)*
**LOC**: ~16,000 source + ~6,500 tests
**Provider**: Vast.ai (via CLI + SSH subprocess)

### Parked work (2026-07-09)

Branch `parked/uncommitted-2026-04` holds a single closeout commit (`4cec574`) capturing ~2026-04 uncommitted WIP: +257/−68 across `checkpoint/namd_restart.py`, `daemon/monitor.py`, `providers/vast.py`, `cli/jobs.py`, `providers/base.py`, plus a log file. It is **unreviewed and unmerged by design** — the owner (Sean) adjudicates it. Do not merge it, rebase it, or build on it.

*(2026-07-16: main's NAMD fix `51fb6ad` touches the same region of `namd_restart.py` the parked branch rewrites — a future merge WILL conflict there, intentionally forcing conscious adjudication. The branches differ in approach: main scopes the cooling gate to campaign jobs and keeps mid-cooling restart-from-scratch; the parked WIP instead reconstructs the remaining 1000→453 K cooling schedule (reassignTemp ramps) so mid-cooling checkpoints resume — that reconstruction is scientifically load-bearing and remains unreviewed. The parked branch's other files (monitor.py +95, vast.py, jobs.py, base.py) are untouched by main's fix.)*

### Sprint Status

Active development sprint started 2026-03-23. See `docs/SPRINT_2026-03-23.md` for full plan.

| Phase | Items | Status |
|-------|-------|--------|
| **Phase 1: Core Workflow** | Setup commands, configurable timeout, SSH passthrough | DONE |
| **Phase 2: Visibility** | Pluggable progress parsing, notification hooks | DONE |
| **Phase 3: Workflows** | Variable substitution, multi-stage jobs, batch matrix | DONE |
| **Phase 4: Benchmarks** | Benchmark engine, historical cost tracking | DONE |
| **Phase 5: Instance Mgmt** | Instance reuse, instance pools | BACKLOG |
| **Phase 6: Quality** | Simplify PackStore, test coverage, docs refresh | PARTIAL (tests done) |

### Why This Sprint

CCM was **abandoned mid-campaign** during MXene Campaign 2026 (see `~/work/research/mxene-shear/docs/CCM_FEEDBACK_FROM_MXENE_CAMPAIGN.md`). The core infrastructure is solid but three critical gaps blocked real usage:
1. No way to run setup commands (e.g., `apt install lammps`) before the job
2. Hard-coded 300s provisioning timeout
3. No mid-flight interaction (can't check progress, upload files, chain stages)

Phase 1 alone would have been enough to use CCM for the entire campaign.

## Architecture

```
src/cloudcomputemanager/
├── agents/         [1,150 LOC] AI agent SDK + SSH credential exposure
├── api/            [905 LOC]   FastAPI REST endpoints
├── benchmarks/     [NEW]       GPU cost-performance benchmark framework
├── checkpoint/     [1,100 LOC] Checkpoint detection + restart adapters (8 app types)
├── cli/            [4,200 LOC] Typer CLI (45+ commands) + exec/upload/ssh by job_id
├── core/           [2,100 LOC] Models (CostRecord, stages, progress), templates (variables), recovery
├── daemon/         [1,100 LOC] Monitor (multi-stage, progress parsing, notifications)
├── packstore/      [1,621 LOC] Scientific package registry (deprioritized)
├── providers/      [955 LOC]   CloudProvider ABC + Vast.ai (setup commands, sentinel)
├── sync/           [442 LOC]   Rsync-based data synchronization engine
└── templates/      [30 LOC]    6 built-in YAML job templates
```

## Key Files

| File | Purpose |
|------|---------|
| `providers/vast.py` | Vast.ai provider: SSH retry (3x exp backoff), rsync retry |
| `providers/base.py` | CloudProvider ABC, managed_instance context manager, GPU tier search |
| `daemon/monitor.py` | Monitor loop: health checks, budget enforcement, progress tracking |
| `core/models.py` | All SQLModel tables: Job, Instance, Checkpoint, SyncRecord |
| `core/recovery.py` | Preemption recovery: restart adapter chain + instance replacement |
| `checkpoint/restart_adapters.py` | RestartAdapter ABC + 8 app-specific adapters |
| `core/validation.py` | Pre-flight performance validation (LAMMPS, NAMD, PyTorch) |
| `core/templates.py` | Template loading, merging, resource key aliasing |
| `agents/sdk.py` | Async Python SDK for AI agents |
| `cli/jobs.py` | Job submit/list/status/wait/sync/recover |
| `cli/batch.py` | Batch submit/status/wait/cancel |

## Implemented Features (v0.1.0)

- Job submission from YAML with template inheritance
- Job lifecycle: PENDING → PROVISIONING → RUNNING → COMPLETED/FAILED
- SSH retry logic (3x exponential backoff on exit code 255)
- rsync upload/download with retry
- LAMMPS/PyTorch/GROMACS/NAMD/QE/VASP checkpoint detection
- Preemption recovery with auto-detected restart adapters (8 app types)
- User-defined restart commands via `restart:` YAML section
- Continuous rsync sync with pattern filtering
- Multi-signal health checks (SSH + process + workspace + disk space)
- Per-job budget enforcement (max_cost_usd, max_hours, max_hourly_rate)
- GPU preference tiers (try cheapest adequate GPU first)
- Pre-flight performance validation
- Background daemon with log rotation
- 6 built-in templates (quick-gpu, lammps-gpu, namd-production, pytorch-train, jupyter-dev, llm-inference)
- Agent SDK with async context manager, event streaming, batch support
- REST API with Swagger docs
- Stale job cleanup, orphan instance cleanup

## New in v0.2.0 (Sprint 2026-03-23)

- **Setup commands**: `setup:` field in job YAML for pre-job software installation
- **Configurable timeout**: `provisioning.timeout` in YAML (no more 300s hard-code)
- **SSH by job_id**: `ccm exec <job_id> "cmd"`, `ccm upload <job_id> ...`, `ccm ssh <job_id>`
- **Agent SSH credentials**: `agent.get_ssh_credentials(job_id)` returns host/port/key
- **Variable substitution**: `--set KEY=VALUE` with `${VAR}` syntax in YAML
- **Multi-stage jobs**: `stages:` list with per-stage commands, daemon auto-advances
- **Batch parameter matrix**: `matrix:` Cartesian product expansion for sweep jobs
- **Pluggable progress parsing**: `progress:` config with regex_parse, file_growth, custom_command
- **Notification hooks**: `notifications:` with on_complete, on_failure, on_budget_exceeded
- **Benchmark framework**: `ccm benchmark run/results` for automated GPU cost-performance analysis
- **CostRecord model**: Historical cost-performance tracking per job
- **SIGTERM-aware wrapper**: Traps preemption signals, writes `.ccm_preempted` marker, sends SIGUSR1 for app checkpoints
- **Shared wrapper builder**: `core/wrapper.py` — single source of truth for all 3 wrapper generation sites
- **Daemon startup reconciliation**: `_reconcile_stale_jobs()` detects completed/dead jobs on restart
- **Exit code 143 detection**: SIGTERM exit code auto-routes to recovery instead of failure
- **`ccm reconnect`**: Rehydrates job state from instances after daemon downtime, syncs results
- **Instance heartbeat**: Background loop writes `/workspace/.ccm_heartbeat` every 60s for liveness detection

- **Web Dashboard**: `ccm dashboard` — single-page view of all jobs, costs, events, alerts with live SSE updates

## Generalized Restart Adapters (2026-04-09)

Replaced NAMD-only hardcoded restart logic with a pluggable adapter chain that auto-detects application type from command string:

| Adapter | Detects | Strategy |
|---------|---------|----------|
| NAMD | `namd` in cmd | Generate restart config from `.xsc` checkpoint |
| GROMACS | `gmx`/`mdrun` | Inject `-cpi state.cpt` flag |
| LAMMPS | `lmp`/`lammps` | Generate wrapper with restart file variable |
| Quantum ESPRESSO | `pw.x`/`ph.x` | Rewrite `restart_mode` in input |
| VASP | `vasp` | Copy CONTCAR → POSCAR (WAVECAR auto-detected) |
| PyTorch Lightning | `lightning`/`trainer.fit` | Append `--ckpt_path last` |
| HF Trainer | `--do_train`/`run_clm` | Append `--resume_from_checkpoint True` |
| Generic | Everything else | Re-run original command (industry standard) |

Priority: user-defined `restart:` YAML > auto-detected adapter > original command.

- `checkpoint/restart_adapters.py` — RestartAdapter ABC + 8 implementations (~350 LOC)
- `core/recovery.py` — Adapter chain replaces `_try_generate_namd_restart()` + `namd_restart: bool`
- `core/models.py` — Added `restart_json` field to Job
- 76 unit tests + end-to-end preemption recovery validated on real Vast.ai instances

## Resilience Hardening (2026-03-25 to 2026-04-03)

20 issues fixed (#7-#28) during a 10-day hydrogenation ensemble campaign (601 job attempts, 18 snapshots):

| Issue | Fix | Impact |
|-------|-----|--------|
| #7 | Nohup SSH timeout handling | False "failed to start" alarms |
| #8 | Auto-recover on configurable exit codes | GPU crashes (SIGSEGV) auto-retry |
| #9 | Case-insensitive status filter | CLI usability |
| #10 | Offer blacklist for bad instances | Avoid known-bad hosts |
| #12 | CUDA version filter | Prevent NGC container segfaults |
| #17 | NAMD checkpoint-restart config generation | Resume from .restart files |
| #19 | Store retry_json in Job record | Auto-recovery was completely broken |
| #20 | Upload original job files during recovery | NAMD couldn't find inputs |
| #21 | Route PROVISIONING failures through recovery | Instance-never-started recovery |
| #22 | Reduce SSH timeout in monitor (60→10s) | Dead instances blocked loop |
| #23 | Skip RECOVERING jobs in health check | Infinite re-preemption loop |
| #24 | Generic upload_json, remove hardcoded paths | Recovery works for any project |
| #25 | Periodic sync during job runtime | Checkpoints never saved before |
| #26 | Non-blocking recovery (async task) | **ROOT CAUSE**: daemon blocked for hours |
| #27 | Reliability + duration filters | Filter unreliable residential hosts |
| #28 | Instance labels during recovery | Recovery instances auto-terminated |

Key learnings:
- `reliability_min: 0.99` is the single most important filter (1% → 66% survival)
- Recovery must be non-blocking (async task) or it freezes the entire daemon
- Instance labels are mandatory — unlabeled instances get auto-terminated
- Periodic sync must run during job lifetime, not just on completion
- See `docs/SPOT_INSTANCE_SURVIVAL_GUIDE.md` and `docs/usage.md` Best Practices

## Remaining Limitations

- Instance reuse across jobs not yet implemented — **Backlog**
- PackStore still over-engineered — **Backlog**
- S3 sync not implemented
- No Windows daemon support
- Recovery uses original job's resources_json (may have outdated price ceilings)

## Development

```bash
cd ~/work/compute/ccm
pip install -e ".[dev]"
pytest tests/ --ignore=tests/test_e2e_full_lifecycle.py --ignore=tests/test_integration_vast.py

# Integration tests (require Vast.ai API key)
pytest tests/test_integration_vast.py --run-integration
```

## Related Resources

- **MXene benchmarks**: `~/work/research/mxene-shear/benchmarks/` (GPU cost analysis, run_comparison.py)
- **MXene campaign feedback**: `~/work/research/mxene-shear/docs/CCM_FEEDBACK_FROM_MXENE_CAMPAIGN.md`

*(Paths above updated 2026-07-01 after the 2026-06-13 workspace reorg: the old AFRL "7-MXenesProject" tree is now `~/work/research/mxene-shear`; CCM itself lives at `~/work/compute/ccm`. All targets verified to exist as of 2026-07-01.)*
- **Sprint plan**: `docs/SPRINT_2026-03-23.md`
- **Design doc**: `DESIGN.md` (original architecture, 400+ lines)
- **Old improvement plan**: `docs/CCM_IMPROVEMENT_PLAN_2026-02-26.md` (GPU query bug, Rich display bug)
