# NAMD Checkpoint-Restart Recovery Design

**Date**: 2026-03-26
**Status**: Design complete, ready for implementation
**Issue**: Jobs on Vast.ai lose progress when hosts disappear (even on-demand). Need to resume from last checkpoint instead of restarting from scratch.

---

## Problem

Our NAMD cooling+production jobs take ~30 hours. Vast.ai hosts can disappear at any time (even on-demand instances). When that happens, CCM's recovery manager currently:

1. Finds a new instance
2. Uploads the original job files
3. Runs the **same command from scratch** (cooling + production)
4. All progress is lost

We need it to instead:

1. Find synced restart files (`.restart.coor/.vel/.xsc`)
2. Extract the step number from the `.xsc` file
3. Generate a **restart NAMD config** that continues from that step
4. Upload restart files + new config to the new instance
5. Resume production from where it left off

---

## NAMD Restart Mechanics

### Files needed to restart:
```
simulation.restart.coor   — binary coordinates (1.2 MB for 50K atoms)
simulation.restart.vel    — binary velocities (1.2 MB)
simulation.restart.xsc    — extended system info (170 bytes, contains step number)
```

### Step number extraction:
The `.xsc` file has 3 lines:
```
# NAMD extended system configuration restart file
#$LABELS step a_x a_y a_z b_x b_y b_z c_x c_y c_z o_x o_y o_z
500000 72.95 0 0 0 72.95 0 0 0 72.95 36.475 36.475 36.475
```
Step number is the first field on line 3.

### Restart config (vs start-from-scratch):

| Field | From Scratch | Restart |
|-------|-------------|---------|
| `coordinates` | `snapshot_coords.pdb` | Still needed (structure def) |
| `binCoordinates` | Not used | `simulation.restart.coor` |
| `binVelocities` | Not used | `simulation.restart.vel` |
| `extendedSystem` | Not used | `simulation.restart.xsc` |
| `temperature` | `1000` (generates velocities) | **REMOVED** (velocities from file) |
| `firsttimestep` | Not used | `{step_from_xsc}` |
| `cellBasis*` | Set explicitly | **REMOVED** (from extendedSystem) |
| `cellOrigin` | Set explicitly | **REMOVED** (from extendedSystem) |

### DCD behavior on restart:
- NAMD creates a **new DCD file** on restart (doesn't append)
- Old DCD is renamed to `.dcd.BAK`
- Multiple DCD segments must be merged with CatDCD or MDAnalysis
- `firsttimestep` ensures correct frame timestamps in the new DCD

---

## Implementation Plan

### 1. Restart Config Generator (`namd_restart.py`)

A Python module that generates a NAMD restart config given:
- The original config file (for force field settings, output settings)
- The restart files (`.coor`, `.vel`, `.xsc`)
- Total target steps (to compute remaining)

```python
def generate_restart_config(
    original_config_path: str,
    restart_coor: str,
    restart_vel: str,
    restart_xsc: str,
    total_steps: int,           # Total production steps (e.g., 15150000)
    output_path: str,
) -> dict:
    """
    Generate a NAMD restart config.

    Returns:
        dict with keys: config_content, step_number, remaining_steps
    """
```

The generator:
1. Reads `restart_xsc` to extract step number
2. Computes `remaining_steps = total_steps - step_number`
3. Reads the original config for force field/output settings
4. Generates a new config that:
   - Keeps: structure, parameters, PME, switching, cutoff, output settings
   - Replaces: `coordinates` → adds `binCoordinates/binVelocities/extendedSystem`
   - Removes: `temperature`, `cellBasis*`, `cellOrigin`
   - Removes: entire cooling Tcl loop (already done)
   - Sets: `firsttimestep {step_number}`
   - Sets: `run {remaining_steps}`

### 2. Recovery Wrapper Script

When CCM recovers a NAMD job, instead of running the original command, it:

```bash
#!/bin/bash
cd /workspace

# Check if restart files exist
if [ -f "simulation.restart.coor" ] && [ -f "simulation.restart.vel" ] && [ -f "simulation.restart.xsc" ]; then
    echo "=== CHECKPOINT RECOVERY ==="

    # Rename existing DCD to preserve it
    if [ -f "simulation.dcd" ]; then
        STEP=$(awk 'NR==3{print $1}' simulation.restart.xsc)
        mv simulation.dcd "simulation_before_${STEP}.dcd"
        echo "Preserved DCD as simulation_before_${STEP}.dcd"
    fi

    # Generate restart config
    python3 /workspace/generate_restart_config.py

    # Run NAMD with restart config
    namd3 +p8 +devices 0 simulation_restart.namd > job.log 2>&1
else
    echo "=== FRESH START ==="
    # Run original config
    bash validate_namd.sh && namd3 +p8 +devices 0 cooling_production_453K.namd > job.log 2>&1
fi
```

### 3. CCM Recovery Manager Changes

In `recovery.py`:

```python
async def start_recovered_job(self, job, instance_id, has_checkpoint):
    if has_checkpoint and self._is_namd_job(job):
        # Upload restart files from sync dir
        # Generate restart config on instance
        # Run with restart config
    else:
        # Run original command (current behavior)
```

### 4. Sync Configuration

Ensure restart files are synced frequently:

```yaml
sync:
  interval_minutes: 5        # Sync every 5 min (was 10)
  include_patterns:
    - "simulation.restart.*"  # All restart files
    - "simulation.dcd"        # Trajectory
    - "job.log"               # For step tracking
```

### 5. DCD Merging (Post-Processing)

After a job with one or more recoveries completes:

```python
import MDAnalysis as mda

# Load all DCD segments in order
u = mda.Universe("4HPt.psf", [
    "simulation_before_500000.dcd",   # Segment 1 (steps 0-500000)
    "simulation.dcd",                  # Segment 2 (steps 500000-end)
])

# Or use catdcd
# catdcd -o combined.dcd simulation_before_500000.dcd simulation.dcd
```

---

## What Needs to Change in CCM

### File: `core/recovery.py`

1. Add `_is_namd_job(job)` — check if job command contains "namd3"
2. Add `_get_namd_restart_step(sync_dir)` — parse `.xsc` for step number
3. Add `_generate_namd_restart_config(job, sync_dir, instance_id)` — generate and upload restart config
4. Modify `start_recovered_job()` — use restart config for NAMD jobs

### File: `providers/vast.py`

No changes needed.

### File: `daemon/monitor.py`

No changes needed (already syncs restart files).

### New file: `templates/namd_restart.namd.j2` (Jinja2 template)

```
# NAMD Restart Configuration — Generated by CCM Recovery
# Resuming from step {{ step_number }}
# Remaining: {{ remaining_steps }} steps

structure           inputs/4HPt.psf
coordinates         inputs/4HPt.pdb

# Load checkpoint
binCoordinates      simulation.restart.coor
binVelocities       simulation.restart.vel
extendedSystem      simulation.restart.xsc

# Step tracking
firsttimestep       {{ step_number }}

# Force field (copied from original)
paraTypeCharmm      on
parameters          inputs/IFF_CHARMM36_parameters_V9_NEC.prm
exclude             scaled1-4
1-4scaling          1.0
switching           on
switchdist          10.0
cutoff              12.0
pairlistdist        14.0
PME                 yes
PMEGridSpacing      1.0
wrapAll             on
wrapNearest         on

# Integration
timestep            1.0
rigidBonds          water
nonbondedFreq       1
fullElectFrequency  2
stepspercycle       10

# Output
outputName          simulation
binaryoutput        yes
binaryrestart       yes
dcdfreq             10000
outputEnergies      5000
outputTiming        50000
restartfreq         50000
xstFreq             10000

# Thermostat (continues from checkpoint state)
langevin            on
langevinDamping     1.0
langevinTemp        453
langevinHydrogen    off

# Production run — remaining steps only
run                 {{ remaining_steps }}
```

---

## Validation Plan

### Test 1: Restart config generation
- Take a known `.xsc` file with step 500000
- Generate restart config
- Verify: no `temperature` line, has `binCoordinates`, `firsttimestep = 500000`
- Verify: `run` = total_steps - 500000

### Test 2: NAMD restart on local machine
- Take synced restart files from `~/.cloudcomputemanager/sync/job_3d25b306/`
- Generate restart config
- Run NAMD locally for 100 steps
- Verify: starts at step 500000, produces DCD frames starting at correct time

### Test 3: End-to-end recovery on Vast.ai
- Submit a test job
- After it runs for ~30 min (past cooling into production), manually cancel it
- Verify: sync dir has restart files
- Trigger `ccm jobs recover <job_id>`
- Verify: new instance starts, loads restart files, continues from checkpoint
- Verify: DCD segments can be merged

### Test 4: Automatic recovery
- Submit a test job
- Wait for it to reach production
- Manually destroy the Vast.ai instance (simulating host failure)
- Verify: daemon detects failure, auto-triggers recovery
- Verify: job resumes from checkpoint on new instance

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| XSC step number in cooling phase | Recovery would try to restart mid-cooling with production config | Check if step < cooling_steps; if so, restart from scratch |
| Corrupted restart files (mid-write preemption) | NAMD crashes on restart | Use `.old` files as fallback (NAMD writes .old before overwriting current) |
| DCD file corruption | Lost trajectory frames | Rename DCD before restart; validate with MDAnalysis |
| Multiple recoveries | Many DCD segments to merge | Name segments with step numbers; merge in post-processing |
| Sync not completed before preemption | No restart files available | Increase sync frequency to 5 min; fall back to fresh start |

---

## Cost Impact

With checkpoint-restart working:
- Preemption after 20 hours → lose only 5 min of work (last sync interval)
- Recovery takes ~5 min (new instance + upload + start)
- Effective cost increase: <5% (checkpoint + sync overhead)
- **No more lost 20-hour runs**

Without checkpoint-restart:
- Preemption after 20 hours → lose 20 hours of work
- Must restart from scratch
- Effective cost: 2-3x the baseline (from retries)
