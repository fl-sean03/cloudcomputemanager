# CloudComputeManager Project - Agent Initialization Prompt

**Use this prompt to initialize a new Claude Code session for the CloudComputeManager project.**

---

## Project Overview

Build **CloudComputeManager**: a robust, extensible GPU cloud management platform that handles job submission, automatic checkpointing, spot instance preemption recovery, and continuous data synchronization. The platform should be agent-native (AI agents can programmatically control all operations).

### Why We're Building This

We run LAMMPS molecular dynamics simulations on Vast.ai GPU instances. Current pain points:
- **Spot instance preemption**: Instances can be terminated anytime, losing all work
- **Manual monitoring**: Must SSH in to check job status
- **No automatic recovery**: Jobs must be manually restarted after preemption
- **Data loss risk**: Results not continuously synced to persistent storage

### Strategic Decision: SkyPilot + Custom Layer

After researching existing tools, we decided:
- **Use SkyPilot** for cloud provisioning, instance lifecycle, and basic recovery
- **Build custom layer** on top for:
  - LAMMPS-specific checkpoint detection and auto-restore
  - Continuous rsync-based data sync to local storage
  - Agent-native API wrapper
  - Application-level checkpoint orchestration

**Rationale**: SkyPilot already has Vast.ai integration (~1000 lines), spot recovery, and multi-cloud abstraction. No need to rebuild. But it lacks automatic checkpointing (user responsibility) - that's our value-add.

---

## Project Location

```
/home/sf2/Workspace/main/46-CloudComputeManager/
├── DESIGN.md              # Comprehensive design document (READ THIS FIRST)
├── AGENT_PROMPT.md        # This file
├── skypilot/              # Cloned SkyPilot repo for reference
│   └── sky/
│       ├── clouds/vast.py           # Vast.ai cloud provider
│       ├── provision/vast/          # Instance lifecycle
│       ├── jobs/recovery_strategy.py # Recovery strategies
│       └── jobs/controller.py       # Job controller
└── (implementation files to be created)
```

### Related Directories

```
# Current simulation sync location (reference for what we're protecting)
/home/sf2/AFRL/7-MXenesProject/vast_sync/
├── sync_monitor.sh        # Current manual sync script
├── npt-tri-OH50-F50/      # Synced NPT simulation data
└── shear-couple_xy-F100/  # Synced shear simulation data

# LAMMPS simulation scripts (reference for checkpoint patterns)
/home/sf2/AFRL/7-MXenesProject/shear-strength/grid_sampling/scripts/
/home/sf2/AFRL/7-MXenesProject/equilibration/
```

---

## Credentials & API Keys

### Vast.ai
- **API Key Location**: `~/.vast_api_key` or `~/.config/vastai/vast_api_key`
- **CLI**: `vastai` (pip install vastai)
- **SDK**: `vastai-sdk` (pip install vastai-sdk)
- **Docs**: https://docs.vast.ai/

### SkyPilot (if using)
- **Install**: `pip install skypilot[vast]`
- **Config**: `~/.sky/config.yaml`
- **Vast credential**: Same as above, SkyPilot reads it automatically

### SSH Keys
- Standard SSH keys at `~/.ssh/id_rsa` and `~/.ssh/id_rsa.pub`
- Vast.ai instances accessible via: `ssh -p <port> root@<host>`

---

## Research Summary

### Existing Tools Evaluated

| Tool | Verdict | Notes |
|------|---------|-------|
| **SkyPilot** | USE IT | Vast.ai integration, spot recovery, 8k+ GitHub stars |
| vastai-sdk | Use via SkyPilot | Official SDK, SkyPilot wraps it |
| jjziets/vasttools | Reference only | Community tools |
| Modal/RunPod | Not needed | Different model (serverless) |

### SkyPilot Key Files (in `/home/sf2/Workspace/main/46-CloudComputeManager/skypilot/`)

```
sky/clouds/vast.py              # Cloud provider (342 lines)
sky/provision/vast/instance.py  # Instance lifecycle (293 lines)
sky/provision/vast/utils.py     # Launch logic (272 lines)
sky/jobs/recovery_strategy.py   # FAILOVER, EAGER_NEXT_REGION strategies
sky/jobs/controller.py          # Job monitoring loop
```

### SkyPilot Gaps (What We Must Build)

1. **No automatic checkpointing** - SkyPilot re-runs jobs from scratch on recovery
2. **No application-aware restart** - Doesn't know about LAMMPS restart files
3. **No continuous sync** - Only syncs at job end
4. **No local mirror** - Data only on instance until job completes

---

## Architecture Decision

### Provider Strategy: Vast.ai First, Extensible by Design

```
CloudComputeManager Layer (our code)
    │
    ├── Checkpoint Orchestrator    ← Detects app checkpoints, triggers saves
    ├── Sync Engine                ← Continuous rsync to local/S3
    ├── Agent API                  ← REST/Python SDK for AI agents
    │
    ▼
SkyPilot (existing)
    │
    ├── Job Controller             ← Monitors jobs, detects preemption
    ├── Recovery Strategy          ← EAGER_NEXT_REGION failover
    ├── Vast.ai Adapter            ← Instance lifecycle
    │
    ▼
Vast.ai API / vastai-sdk
```

---

## Implementation Phases

### Phase 1: Foundation (Current Priority)
- [ ] Set up project structure (Python package)
- [ ] Integrate SkyPilot as dependency
- [ ] Create LAMMPS checkpoint detector (parse restart files, log.lammps)
- [ ] Implement continuous rsync wrapper
- [ ] Basic CLI: `cloudcomputemanager submit job.yaml`

### Phase 2: Checkpoint Orchestration
- [ ] Pre-checkpoint hook (signal app to write checkpoint)
- [ ] Checkpoint verification (file exists, size > 0, recent timestamp)
- [ ] Checkpoint upload to S3/local storage
- [ ] Restore from checkpoint on job restart

### Phase 3: Agent API
- [ ] REST API (FastAPI)
- [ ] Python SDK
- [ ] Webhook notifications (job.completed, job.failed, instance.preempted)
- [ ] Async job submission and monitoring

### Phase 4: Polish
- [ ] Cost tracking and budgets
- [ ] Web dashboard (optional)
- [ ] Documentation and examples
- [ ] Multi-provider support (RunPod, Lambda)

---

## LAMMPS-Specific Requirements

### Checkpoint Detection
LAMMPS writes restart files via `write_restart` command:
```
restart 100000 restart.*.bin  # Write every 100k steps
write_restart restart.final.bin  # Write at end
```

Detection logic:
1. Monitor `/workspace/restart.*.bin` files
2. Parse `log.lammps` for current timestep
3. On preemption: save latest restart file + log

### Auto-Restart Logic
```bash
# Check if restart file exists
RESTART=$(ls -t restart.*.bin 2>/dev/null | head -1)
if [ -n "$RESTART" ]; then
    mpirun lmp -in input.in -var restart_file $RESTART
else
    mpirun lmp -in input.in
fi
```

### Input File Pattern
```lammps
# In LAMMPS input file
if "${restart_file} != none" then &
    "read_restart ${restart_file}" &
else &
    "read_data structure.data"
```

---

## Tech Stack

```toml
[project]
dependencies = [
    "skypilot[vast]>=0.11.0",   # Cloud orchestration
    "httpx>=0.27.0",            # Async HTTP
    "pydantic>=2.5.0",          # Data validation
    "typer>=0.9.0",             # CLI
    "rich>=13.0.0",             # Terminal UI
    "fastapi>=0.109.0",         # REST API
    "sqlmodel>=0.0.16",         # Database ORM
    "structlog>=24.1.0",        # Structured logging
]
```

---

## Your Directives

### 1. Research Phase
- [ ] Read `/home/sf2/Workspace/main/46-CloudComputeManager/DESIGN.md` thoroughly
- [ ] Explore SkyPilot codebase at `/home/sf2/Workspace/main/46-CloudComputeManager/skypilot/`
- [ ] Understand how SkyPilot's job controller and recovery work
- [ ] Review current sync script at `/home/sf2/AFRL/7-MXenesProject/vast_sync/sync_monitor.sh`

### 2. Planning Phase
- [ ] Update DESIGN.md with any new insights
- [ ] Create detailed implementation plan
- [ ] Define Python package structure
- [ ] Design checkpoint detection algorithm
- [ ] Design sync architecture

### 3. Implementation Phase
- [ ] Create Python package skeleton
- [ ] Implement SkyPilot integration layer
- [ ] Build checkpoint orchestrator
- [ ] Build sync engine
- [ ] Create CLI interface
- [ ] Write tests

### 4. Validation Phase
- [ ] Test with mock LAMMPS job
- [ ] Test preemption recovery (manual instance termination)
- [ ] Test checkpoint restore
- [ ] Test continuous sync
- [ ] Document usage examples

### 5. Documentation Phase
- [ ] Update DESIGN.md with implementation details
- [ ] Write README.md with quickstart
- [ ] Create example job YAML files
- [ ] Document API endpoints

---

## Success Criteria

1. **Submit a LAMMPS job** via `cloudcomputemanager submit job.yaml`
2. **Job auto-recovers** from spot preemption without data loss
3. **Checkpoints continuously synced** to local storage
4. **Can resume from checkpoint** after recovery
5. **Agent can programmatically** submit, monitor, and retrieve results

---

## Quick Start Commands

```bash
# Navigate to project
cd /home/sf2/Workspace/main/46-CloudComputeManager

# Read the design doc
cat DESIGN.md

# Explore SkyPilot Vast.ai integration
ls -la skypilot/sky/clouds/vast.py
ls -la skypilot/sky/provision/vast/

# Check current sync script (reference)
cat /home/sf2/AFRL/7-MXenesProject/vast_sync/sync_monitor.sh

# Test Vast.ai CLI (if needed)
vastai show instances
vastai search offers 'gpu_name=RTX_4090 num_gpus=1'
```

---

## Context: Current Simulations

There are currently 2 simulations running on Vast.ai (on-demand instances):

1. **NPT tri-OH50-F50** - ssh8.vast.ai:34368 - NPT equilibration (~1% complete)
2. **Shear couple_xy-F100** - ssh9.vast.ai:34370 - Shear grid sampling (1/11 stress levels)

These are being manually synced via the script at `/home/sf2/AFRL/7-MXenesProject/vast_sync/sync_monitor.sh`. The CloudComputeManager project aims to automate and robustify this entire workflow.

---

## Notes

- **Do NOT implement without planning** - Update design doc first
- **Prefer SkyPilot integration** over reimplementing cloud logic
- **Focus on checkpoint/sync layer** - That's the value-add
- **Keep it simple** - Start with LAMMPS, generalize later
- **Agent-native** - Every feature must be API-accessible

---

*This prompt contains everything needed to continue the CloudComputeManager project from scratch. The agent should read DESIGN.md first, explore the SkyPilot codebase, then proceed with implementation.*
