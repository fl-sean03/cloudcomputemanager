# CCM Improvement Plan

**Date:** 2026-02-26
**Author:** Claude Code
**Context:** Hydrogenation ensemble simulation batch job failures

---

## Executive Summary

During a 21-job NAMD ensemble simulation batch, CCM exhibited two critical bugs that prevented successful batch submission:

1. **GPU Query Bug**: User-specified GPU type and memory were overridden by hardcoded defaults
2. **Rich Display Bug**: Concurrent job submissions crashed with "Only one live display may be active at once"

This document outlines the root causes, proposed fixes, and testing plan for each issue.

---

## Issue 1: GPU Query Defaults Override User Config

### Symptom
User specifies in job YAML:
```yaml
resources:
  gpu_type: "RTX_3060"
  gpu_memory_gb: 12
```

CCM searches for:
```
gpu_name=RTX_4090 gpu_ram>=16
```

### Root Cause Analysis

**The bug is a cascading default problem across three locations:**

#### Location 1: Class Defaults (core/models.py:94-102)
```python
class Resources(SQLModel, table=True):
    gpu_type: str = Field(default="RTX_4090", ...)  # Line 97
    gpu_memory_min: int = Field(default=16, ...)     # Line 99
```

#### Location 2: Function Defaults (cli/jobs.py:158-161)
```python
offers = await provider.search_offers(
    gpu_type=resources.get("gpu_type", "RTX_4090"),      # Line 158
    gpu_memory_min=resources.get("gpu_memory_min", 16),  # Line 160
)
```

#### Location 3: Recovery Defaults (core/recovery.py:76-79)
```python
offers = await provider.search_offers(
    gpu_type=job.resources.get("gpu_type", "RTX_4090"),  # Line 76
    gpu_memory_min=job.resources.get("gpu_memory_min", 16),  # Line 78
)
```

**Why it happens:**
1. User config loaded correctly via `load_config_with_template()`
2. `deep_merge()` properly applies user overrides to template
3. BUT: `resources.get("gpu_type", "RTX_4090")` ignores user value if key missing from dict
4. When user uses `gpu_memory_gb` (not `gpu_memory_min`), the `.get()` falls back to default

### Proposed Fix

#### Fix 1A: Normalize config keys during loading (templates.py)

Add key normalization after loading config:
```python
def normalize_resources(resources: dict) -> dict:
    """Normalize resource keys to canonical names."""
    key_map = {
        "gpu_memory_gb": "gpu_memory_min",
        "memory_gb": "ram_gb",
        "cpu": "cpu_cores",
    }
    return {key_map.get(k, k): v for k, v in resources.items()}
```

#### Fix 1B: Remove function-level defaults (cli/jobs.py, core/recovery.py)

Replace:
```python
gpu_type=resources.get("gpu_type", "RTX_4090")
```

With:
```python
gpu_type=resources.get("gpu_type")  # None if not specified
```

Then handle None in VastProvider.search_offers() to use sensible auto-detection or raise error.

#### Fix 1C: Add config validation (core/templates.py)

```python
def validate_resources(resources: dict) -> None:
    """Validate required resource fields are present."""
    if not resources.get("gpu_type") and not resources.get("gpu_memory_min"):
        raise ConfigurationError(
            "resources.gpu_type or resources.gpu_memory_min must be specified"
        )
```

### Testing Plan for Issue 1

| Test Case | Input | Expected | File |
|-----------|-------|----------|------|
| Explicit RTX_3060 | `gpu_type: RTX_3060` | Query contains `gpu_name=RTX_3060` | test_gpu_query.py |
| Explicit memory | `gpu_memory_gb: 12` | Query contains `gpu_ram>=12` | test_gpu_query.py |
| Key normalization | `gpu_memory_gb: 8` | Normalized to `gpu_memory_min: 8` | test_templates.py |
| No GPU type | (omitted) | Error or auto-detect | test_validation.py |
| Template override | Template has 4090, user specifies 3060 | User wins | test_templates.py |

---

## Issue 2: Rich Library Display Conflict

### Symptom
```
Error: Only one live display may be active at once
```
Occurs during `ccm batch submit` with `max_parallel > 1`.

### Root Cause Analysis

**The bug is nested Rich displays in concurrent async tasks:**

#### Outer Display (cli/batch.py:87-119)
```python
with Progress(...) as progress:  # One Progress bar for batch
    while pending or running_tasks:
        task_coro = asyncio.create_task(submit_one(config_file))
        running_tasks.append(task_coro)
```

#### Inner Displays (cli/jobs.py, multiple locations)
```python
async def submit_job(...):
    with console.status("Searching..."):   # Line 156 - CONFLICT!
        offers = await provider.search_offers(...)

    with console.status("Creating..."):    # Line 177 - CONFLICT!
        instance = await provider.create_instance(...)
```

**Why it happens:**
- `Progress` and `console.status()` both create Rich `Live` objects
- Rich enforces single active Live per console
- Multiple concurrent `submit_one()` coroutines each try to create displays

### Proposed Fix

#### Fix 2A: Remove inner displays during batch mode

Add a flag to suppress nested displays:
```python
async def submit_job(config_path: str, quiet: bool = False, ...):
    if not quiet:
        with console.status("Searching..."):
            offers = await provider.search_offers(...)
    else:
        offers = await provider.search_offers(...)  # No display
```

Batch submission calls with `quiet=True`:
```python
task_coro = asyncio.create_task(submit_one(config_file, quiet=True))
```

#### Fix 2B: Use shared Progress context (preferred)

Pass the outer Progress object to inner functions:
```python
async def submit_job(config_path: str, progress: Optional[Progress] = None, ...):
    task_id = progress.add_task("Searching...") if progress else None
    offers = await provider.search_offers(...)
    if progress and task_id:
        progress.update(task_id, description="Creating instance...")
```

#### Fix 2C: Use logging instead of Rich displays for batch

For batch mode, log progress instead of displaying:
```python
if batch_mode:
    logger.info(f"Searching offers for {job_name}...")
else:
    with console.status("Searching..."):
        ...
```

### Testing Plan for Issue 2

| Test Case | Setup | Expected | File |
|-----------|-------|----------|------|
| Single job | `ccm jobs submit job.yaml` | Status displays work | test_submit.py |
| Batch serial | `ccm batch submit --max-parallel 1` | Works (no conflict) | test_batch.py |
| Batch parallel | `ccm batch submit --max-parallel 5` | Works (no crash) | test_batch.py |
| Quiet mode | `--quiet` flag | No Rich displays | test_batch.py |

---

## Issue 3: Additional Improvements (Lower Priority)

### 3A: Better error messages for "No suitable offers found"

Current: Generic message
Proposed: Show what was searched and why no matches

```python
if not offers:
    raise NoOffersError(
        f"No offers found matching: {query}\n"
        f"Try: increasing max_hourly_rate, changing gpu_type, or reducing requirements"
    )
```

### 3B: Dry-run mode for batch submit

```bash
ccm batch submit *.yaml --dry-run
```

Shows what would be submitted without actually creating instances.

### 3C: Config validation command

```bash
ccm config validate job.yaml
```

Validates YAML syntax, required fields, and checks if offers exist.

---

## Implementation Plan

### Phase 1: Critical Bug Fixes (Priority: HIGH)

| Step | Task | Files | Est. Time |
|------|------|-------|-----------|
| 1.1 | Add key normalization | core/templates.py | 30 min |
| 1.2 | Remove function defaults | cli/jobs.py, core/recovery.py | 30 min |
| 1.3 | Add config validation | core/templates.py | 30 min |
| 1.4 | Add quiet mode to submit | cli/jobs.py, cli/batch.py | 1 hr |
| 1.5 | Write unit tests | tests/test_*.py | 1 hr |
| 1.6 | Integration test with real Vast.ai | manual | 30 min |

### Phase 2: Testing and Validation

| Step | Task | Method |
|------|------|--------|
| 2.1 | Run existing test suite | `pytest tests/` |
| 2.2 | Test GPU query with RTX_3060 | Manual submit |
| 2.3 | Test batch submit with 5 parallel jobs | `ccm batch submit` |
| 2.4 | Test preemption recovery | Manual instance termination |
| 2.5 | Full ensemble workflow test | 3-job mini-ensemble |

### Phase 3: Documentation and Deployment

| Step | Task |
|------|------|
| 3.1 | Update README with new config keys |
| 3.2 | Add troubleshooting guide |
| 3.3 | Create example job templates for NAMD |
| 3.4 | Tag release version |

---

## File Change Summary

| File | Changes |
|------|---------|
| `core/templates.py` | Add `normalize_resources()`, `validate_resources()` |
| `cli/jobs.py` | Remove defaults from `.get()`, add `quiet` param |
| `cli/batch.py` | Pass `quiet=True` to parallel submissions |
| `core/recovery.py` | Remove defaults from `.get()` |
| `core/models.py` | Consider removing class defaults or making Optional |
| `providers/vast.py` | Handle None gpu_type gracefully |

---

## Success Criteria

1. **GPU Query**: User-specified `gpu_type: RTX_3060` results in query containing `gpu_name=RTX_3060`
2. **Batch Submit**: `ccm batch submit *.yaml --max-parallel 5` completes without Rich errors
3. **Existing Tests**: All 148 existing tests still pass
4. **New Tests**: New tests for GPU query and batch parallel added and passing
5. **Real Workflow**: Successfully submit and monitor a 3-job test ensemble

---

## Appendix: Code Locations Quick Reference

```
/home/sf2/Workspace/main/46-VastManager/
├── src/cloudcomputemanager/
│   ├── cli/
│   │   ├── jobs.py          # Lines 156-163: GPU query defaults
│   │   └── batch.py         # Lines 87-119: Rich Progress
│   ├── core/
│   │   ├── templates.py     # Lines 14-91: Config loading
│   │   ├── models.py        # Lines 94-102: Class defaults
│   │   ├── recovery.py      # Lines 67-101: Recovery defaults
│   │   └── config.py        # Lines 14-201: Settings
│   └── providers/
│       └── vast.py          # Lines 88-121: Query building
└── tests/
    ├── test_templates.py    # Config loading tests
    └── test_batch.py        # Batch submission tests
```
