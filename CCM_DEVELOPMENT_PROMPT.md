# CloudComputeManager (CCM) Development Prompt

**Date**: 2026-02-25
**Purpose**: Validate, test, and improve CCM for production use with scientific computing workflows

---

## Background

CloudComputeManager (CCM) is a GPU cloud management platform built on top of Vast.ai that provides:
- Automatic checkpointing for long-running jobs
- Continuous data synchronization
- Preemption recovery
- Agent-native APIs (CLI, REST, Python SDK)

**Why we need this**: We're running LAMMPS molecular dynamics simulations on Vast.ai and currently managing everything manually via SSH/SCP. CCM should streamline this workflow.

---

## Current State

| Metric | Value |
|--------|-------|
| Version | 0.1.0 (Alpha) |
| Lines of Code | ~10,325 |
| Tests | 140 (100% passing) |
| Documentation | Comprehensive (DESIGN.md, README.md) |
| Last Updated | Feb 24, 2026 |

### What's Implemented
- ✅ Vast.ai provider with SSH/rsync
- ✅ Job submission (CLI, API, SDK)
- ✅ Job monitoring daemon
- ✅ LAMMPS checkpoint detection
- ✅ Preemption recovery
- ✅ Batch operations
- ✅ Job templates

### What's Missing
- ❌ Integration tests against real Vast.ai API
- ❌ E2E validation of full workflows
- ❌ Real-world usage validation

---

## Your Goals

### Phase 1: Validation (Priority)
1. Install CCM and verify it runs
2. Test basic commands: `ccm --help`, `ccm config init`
3. Test job submission against a real Vast.ai instance
4. Document any errors encountered

### Phase 2: Bug Fixes
1. Fix any bugs discovered during validation
2. Ensure SSH proxy connections work (sshX.vast.ai)
3. Verify null status handling
4. Test job completion detection

### Phase 3: Integration Tests
1. Create integration tests that run against real Vast.ai
2. Test the full workflow: submit → monitor → sync → complete
3. Add tests for error scenarios (timeout, preemption, etc.)

### Phase 4: MXenes Workflow Template
1. Create a job template for LAMMPS NPT equilibration
2. Create a job template for LAMMPS shear simulations
3. Test end-to-end with actual MXenes data files

---

## Files to Read First

### Essential (Read these first)
```
/home/sf2/Workspace/main/46-VastManager/README.md           # Quickstart guide
/home/sf2/Workspace/main/46-VastManager/AGENTS.md           # Current project state
/home/sf2/Workspace/main/46-VastManager/DESIGN.md           # Architecture (1500 lines)
```

### Core Implementation
```
/home/sf2/Workspace/main/46-VastManager/src/cloudcomputemanager/providers/vast.py   # Vast.ai CLI wrapper
/home/sf2/Workspace/main/46-VastManager/src/cloudcomputemanager/cli/jobs.py         # Job CLI
/home/sf2/Workspace/main/46-VastManager/src/cloudcomputemanager/core/models.py      # Data models
/home/sf2/Workspace/main/46-VastManager/src/cloudcomputemanager/daemon/monitor.py   # Job monitoring
```

### Tests (for understanding expected behavior)
```
/home/sf2/Workspace/main/46-VastManager/tests/                # All 140 tests
```

### Examples
```
/home/sf2/Workspace/main/46-VastManager/examples/             # LAMMPS & PyTorch examples
```

---

## Environment Setup

```bash
# Navigate to project
cd /home/sf2/Workspace/main/46-VastManager

# Install in development mode
pip install -e ".[dev]"

# Set Vast.ai API key (get from https://cloud.vast.ai/cli/)
export CCM_VAST_API_KEY="your-api-key"

# Initialize config
ccm config init

# Verify installation
ccm --help
```

---

## Test Commands

```bash
# Unit tests (should all pass)
pytest tests/ -v

# List available offers
ccm offers search --gpu-name "RTX 4090" --min-ram 32

# Submit a test job (dry run first)
ccm jobs submit examples/lammps-job.yaml --dry-run

# Check job status
ccm jobs list
ccm jobs status <job-id>
```

---

## Known Issues to Investigate

1. **Earlier we hit a bug** when trying to use CCM - need to reproduce and document
2. **SSH proxy connections** - verify sshX.vast.ai format works correctly
3. **vastai CLI dependency** - CCM uses CLI subprocess, consider using vastai-sdk directly
4. **Rate limiting** - no backoff strategy for Vast.ai API limits

---

## Success Criteria

### Minimum (Phase 1)
- [ ] CCM installs without errors
- [ ] `ccm config init` works
- [ ] `ccm offers search` returns results
- [ ] Can submit a simple job to Vast.ai

### Target (Phase 2-3)
- [ ] Full job lifecycle works: submit → run → complete → sync
- [ ] Integration tests pass against real Vast.ai
- [ ] SSH proxy connections work
- [ ] Job completion detection works

### Stretch (Phase 4)
- [ ] LAMMPS NPT template created and tested
- [ ] LAMMPS shear template created and tested
- [ ] MXenes workflow runs end-to-end via CCM

---

## Context from MXenes Project

We're using Vast.ai for LAMMPS simulations:

**Current manual workflow**:
```bash
# 1. Create instance
vastai create instance <offer_id> --image nvidia/cuda:12.2.0-devel-ubuntu22.04

# 2. Upload files
scp -P <port> structure.data script.inp root@sshX.vast.ai:/workspace/

# 3. SSH in, build LAMMPS, run job
ssh -p <port> root@sshX.vast.ai
# ... manual setup ...
mpirun -np 8 lmp -in script.inp > job.log 2>&1 &

# 4. Monitor
ssh -p <port> root@sshX.vast.ai 'tail -f /workspace/job.log'

# 5. Download results
scp -P <port> root@sshX.vast.ai:/workspace/results/* ./local/
```

**Desired CCM workflow**:
```bash
# Single command to submit
ccm jobs submit mxene-npt.yaml --structure F100_dry.data --wait

# Or via Python
from cloudcomputemanager.agents import CloudComputeAgent
async with CloudComputeAgent() as agent:
    job = await agent.submit_job(spec)
    await agent.wait_for_completion(job.id)
    await agent.sync_results(job.id, "./results/")
```

---

## Notes

- This is a personal project repo, feel free to make changes and commit
- Tests should be run before committing
- Update AGENTS.md with any discoveries
- Document any API issues in the codebase

---

## Start Here

1. Read `README.md` and `AGENTS.md`
2. Install with `pip install -e ".[dev]"`
3. Run `pytest tests/ -v` to verify tests pass
4. Try `ccm --help` and `ccm config init`
5. Attempt to search offers: `ccm offers search --gpu-name "RTX 4090"`
6. Document what works and what doesn't
