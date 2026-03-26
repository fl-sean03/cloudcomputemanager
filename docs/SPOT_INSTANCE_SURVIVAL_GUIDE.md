# Spot Instance Survival Guide

**For any agent running long jobs on Vast.ai through CCM.**

## The Problem

Vast.ai instances (even on-demand) can disappear at any time:
- Host machines go offline for maintenance
- Network failures disconnect instances
- On-demand does NOT guarantee survival — it only prevents bidding preemption

If your job takes >4 hours, expect instance failures. Plan for them.

## The Solution: Checkpoint-Restart

Every long-running job must:
1. **Write checkpoint files periodically** (every 5-30 min depending on job size)
2. **Sync checkpoints to local storage** via CCM sync (every 5 min)
3. **Resume from checkpoint on recovery** instead of restarting from scratch

## How It Works in CCM

### 1. Job YAML Configuration

```yaml
name: my-long-job
project: my-project
image: my-image:latest
command: bash /workspace/run_with_recovery.sh

sync:
  enabled: true
  interval_minutes: 5          # Sync every 5 min — max lost progress
  include_patterns:
    - "checkpoint.*"            # Your checkpoint files
    - "*.log"
    - "output.*"

checkpoint:
  enabled: true
  strategy: application
  patterns:
    - "checkpoint.*"
  interval_minutes: 5

retry:
  max_attempts: 10             # Allow many recovery attempts
  backoff_minutes: 1
  recover_on_exit_codes: [139, 137, 134]  # SIGSEGV, OOM, SIGABRT
```

### 2. Make Your Job Checkpoint-Aware

Your job command should:
- Write checkpoint files at regular intervals
- On startup, check if checkpoint files exist and resume from them
- Use different output filenames for each segment (so data isn't overwritten)

**Generic pattern:**
```bash
#!/bin/bash
cd /workspace

if [ -f "checkpoint.state" ]; then
    echo "Resuming from checkpoint..."
    # Your resume logic here
    my_program --resume checkpoint.state
else
    echo "Starting fresh..."
    my_program --from-scratch
fi
```

### 3. Application-Specific Checkpointing

#### NAMD (Molecular Dynamics)
```yaml
# NAMD writes .restart.coor/.vel/.xsc automatically via restartfreq
# CCM has built-in NAMD restart config generation
checkpoint:
  patterns:
    - "simulation.restart.coor"
    - "simulation.restart.vel"
    - "simulation.restart.xsc"
```
CCM's recovery manager auto-generates a NAMD restart config from synced files. See `checkpoint/namd_restart.py`.

#### LAMMPS
```yaml
checkpoint:
  patterns:
    - "restart.*.bin"
```
LAMMPS `restart` command writes periodic checkpoints. Use `read_restart` in your input script to resume.

#### PyTorch / ML Training
```yaml
checkpoint:
  patterns:
    - "checkpoint_*.pt"
    - "model_*.pt"
```
Save `torch.save(model.state_dict(), f'checkpoint_{epoch}.pt')` periodically. On resume, load the latest.

#### GROMACS
```yaml
checkpoint:
  patterns:
    - "state.cpt"
    - "state_prev.cpt"
```
GROMACS writes `.cpt` files. Use `gmx mdrun -cpi state.cpt` to resume.

### 4. What CCM Does Automatically

When a job's instance disappears:
1. **Daemon detects** `instance_not_found` within 30 seconds
2. **Synced files** are already on local disk (from periodic sync)
3. **Recovery manager** provisions a new instance
4. **Uploads** synced files (including checkpoints) to new instance
5. **For NAMD jobs**: generates restart config from `.restart.*` files
6. **Starts** the job — your recovery script detects checkpoints and resumes
7. **Continues syncing** from where it left off

### 5. Key Settings

| Setting | Recommended | Why |
|---------|-------------|-----|
| `sync.interval_minutes` | 5 | Max 5 min lost on failure |
| `retry.max_attempts` | 10 | Allow many recoveries for multi-day jobs |
| `retry.recover_on_exit_codes` | [139, 137, 134] | Auto-recover GPU crashes |
| `checkpoint.interval_minutes` | 5-15 | Match sync frequency |
| `budget.max_hours` | 48-72 | Allow for recovery overhead |
| `resources.cuda_version_min` | 12.6 | For NGC containers (NAMD, PyTorch NGC) |

### 6. Common Failure Modes

| Failure | Exit Code | CCM Behavior |
|---------|-----------|--------------|
| Host disappeared | None | Daemon detects, auto-recovers |
| GPU driver crash (SIGSEGV) | 139 | Auto-recovers if in `recover_on_exit_codes` |
| OOM killed | 137 | Auto-recovers |
| CUDA abort | 134 | Auto-recovers |
| Normal completion | 0 | Syncs results, terminates instance |
| Application error | 1 | Marks FAILED (not recoverable) |
| Preemption (SIGTERM) | 143 | Always auto-recovers |

### 7. Lessons Learned (Hydrogenation Campaign, March 2026)

- **30-hour jobs on $0.05/hr spot instances**: ~100% overnight failure rate
- **On-demand doesn't help**: Vast.ai hosts disappear regardless of rental type
- **CUDA version matters**: NAMD NGC 3.0.1 needs driver ≥560 (CUDA ≥12.6). Filter with `cuda_version_min: 12.6`
- **~33% of RTX 3060 instances segfault on launch**: Bad GPU drivers. The CUDA filter eliminates most of these.
- **Checkpoint-restart is mandatory**: Without it, you lose ALL progress. With it, you lose max 5 min.
- **Budget for retries**: Set `max_cost_usd` to 3-5x expected cost to handle recovery overhead

### 8. Quick Checklist for Any Long Job

- [ ] Job writes checkpoint files periodically
- [ ] Job can detect and resume from checkpoint files on startup
- [ ] CCM sync includes checkpoint file patterns
- [ ] `sync.interval_minutes` ≤ 10 (preferably 5)
- [ ] `retry.max_attempts` ≥ 5
- [ ] `retry.recover_on_exit_codes` includes [139, 137, 134]
- [ ] `budget.max_hours` allows for recovery time
- [ ] If using NGC containers: set `cuda_version_min` appropriately
- [ ] Tested: manually kill instance, verify recovery works
