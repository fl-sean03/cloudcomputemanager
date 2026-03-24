# Instance Integrity System — Design & Implementation Plan

**Date**: 2026-03-24
**Goal**: Make it impossible for ghost instances to exist — every instance on the Vast.ai account must be tracked, labeled, and associated with a CCM job. No orphans, no ghosts, no missing data.

---

## Problem Statement

Currently, CCM has multiple ways instances can become "ghosts":

1. **Agent calls `vastai create instance` directly** — Instance exists on Vast.ai, not in CCM DB
2. **CCM creates instance but crashes before writing DB record** — Instance running, no Job/Instance record
3. **Instance created but label/metadata not set** — Can't identify purpose from Vast.ai side alone
4. **Job cancelled but instance termination fails** — Instance keeps running, costs money
5. **Daemon not running, instance terminates** — DB still shows RUNNING, stale forever
6. **Multiple CCM instances on same machine** — Could see each other's instances

## Design Principles

1. **Vast.ai is the source of truth for what's running** — always sync from API
2. **Labels on Vast.ai are the link** — every CCM-managed instance gets a structured label
3. **Fail-safe: if label is missing, instance is flagged** — dashboard shows unlabeled instances as warnings
4. **Belt and suspenders** — both local DB AND Vast.ai labels track the mapping

---

## Architecture

### Instance Label Format

Every instance created by CCM gets a label in this format:

```
ccm|{job_id}|{project}|{job_name}
```

Example: `ccm|job_abc123|mxene-campaign|shear-xy-F100`

Rules:
- Pipe-delimited (Vast.ai labels are simple strings, pipes are safe)
- Prefix `ccm|` identifies CCM-managed instances
- `job_id` is the unique identifier linking to the Job table
- `project` and `job_name` are human-readable context
- Total length kept under 200 chars (Vast.ai may have limits)

### Instance Lifecycle with Labels

```
1. CCM creates instance
   └── --label "ccm|job_abc123|my-project|my-job" passed at creation time
   └── Instance record created in DB with job_id association
   └── If DB write fails → instance still has label → sync will find it

2. Daemon syncs instances every 60s
   └── Calls vastai show instances --raw
   └── For each instance:
       a. Parse label → extract job_id
       b. If instance in DB → update status, SSH info, hourly_rate
       c. If instance NOT in DB but has ccm| label → create Instance record, link to Job
       d. If instance has NO label → flag as "unmanaged" (show on dashboard as warning)
       e. If instance in DB but NOT on Vast.ai → mark as terminated

3. Dashboard shows ALL instances
   └── "CCM Jobs" section: instances with valid ccm| labels and Job records
   └── "Unmanaged Instances" section: instances without ccm| labels (with warning + cost)
   └── "Terminated" section: instances marked terminated in DB

4. Instance termination
   └── CCM terminates via vastai destroy → marks Instance + Job as terminated
   └── If termination API call fails → retry 3x → if still fails, flag in dashboard
   └── Agent terminates outside CCM → daemon sync detects it's gone → marks terminated
```

### Edge Cases & Mitigations

| Edge Case | What Happens Today | Mitigation |
|-----------|-------------------|------------|
| **Agent calls `vastai create` directly** | Ghost instance, invisible to CCM | Sync discovers it as "unmanaged", shows on dashboard with cost warning |
| **CCM crashes after creating instance, before DB write** | Ghost instance | Label on Vast.ai has job_id → sync recreates Instance record from label |
| **Instance created but label set fails** | No way to identify | Set label AT creation via `--label` flag (atomic). If create succeeds but label is empty, immediately `vastai label instance` as fallback |
| **Job cancelled, instance termination fails** | Instance keeps running | Retry termination 3x. If still fails, dashboard shows "termination failed" alert. Cleanup command can force-terminate |
| **Daemon not running for hours** | Stale DB state | On restart, `_reconcile_stale_jobs()` + `sync_all_instances()` catches everything up |
| **Multiple users sharing Vast.ai account** | See each other's instances | Label prefix `ccm|` identifies CCM instances. Could add user prefix too: `ccm|user|...` |
| **Vast.ai API is down** | Can't sync | Graceful degradation — dashboard shows stale data with "last sync: Xm ago" indicator |
| **Instance rebooted (new SSH port)** | Old SSH info in DB | Sync updates SSH host/port from API on every cycle |
| **Instance bid changed externally** | Rate mismatch | Sync updates hourly_rate from API |

---

## Implementation Plan

### Phase 1: Label on Creation (core change)

**Files to modify:**
- `providers/vast.py` → `create_instance()`: add `--label` flag to vastai create command
- `core/instances.py` → `upsert_instance()`: parse label to extract job_id
- `core/instances.py` → `sync_all_instances()`: use labels to match instances to jobs

**Label generation:**
```python
def build_instance_label(job_id: str, project: str = "", name: str = "") -> str:
    """Build a structured label for Vast.ai instance identification."""
    # Sanitize: remove pipes from user strings
    project = (project or "none").replace("|", "-")[:50]
    name = (name or "unnamed").replace("|", "-")[:50]
    return f"ccm|{job_id}|{project}|{name}"

def parse_instance_label(label: str) -> dict | None:
    """Parse a CCM label into components. Returns None if not a CCM label."""
    if not label or not label.startswith("ccm|"):
        return None
    parts = label.split("|")
    if len(parts) < 2:
        return None
    return {
        "job_id": parts[1],
        "project": parts[2] if len(parts) > 2 else None,
        "name": parts[3] if len(parts) > 3 else None,
    }
```

**In `vast.py` `create_instance()`:**
```python
# Build create command
args = ["create", "instance", offer_id, "--image", image, "--disk", str(disk_gb), "--raw"]

# ALWAYS set label if provided
if label:
    args.extend(["--label", label])
```

**In `cli/jobs.py` `submit_job()`:**
```python
from cloudcomputemanager.core.instances import build_instance_label

label = build_instance_label(job.job_id, config.get("project"), job_name)
create_kwargs["label"] = label
```

### Phase 2: Smart Sync with Label Parsing

**In `core/instances.py` `sync_all_instances()`:**

When syncing instances from Vast.ai API, parse each instance's label:

```python
for pi in all_instances:
    label_data = parse_instance_label(pi.label)  # pi needs label field

    if label_data:
        # CCM-managed instance — link to job via label
        job_id = label_data["job_id"]
        await upsert_instance(pi, job_id=job_id)
    else:
        # Unmanaged instance — still track it, but flag it
        await upsert_instance(pi, job_id=None, unmanaged=True)
```

**ProviderInstance needs label field:**
```python
@dataclass
class ProviderInstance:
    ...
    label: Optional[str] = None  # NEW
```

**In `vast.py` `_parse_instance()`:**
```python
return ProviderInstance(
    ...
    label=data.get("label"),  # NEW — pass through the label from API
)
```

### Phase 3: Dashboard Shows Unmanaged Instances

Add a section to the dashboard between the jobs table and events:

```
┌────────────────────────────────────────────────────────┐
│ ⚠ UNMANAGED INSTANCES (2)                             │
│ These instances are on your Vast.ai account but were   │
│ not created through CCM.                               │
│                                                        │
│ ID        GPU        $/hr    Status   Uptime   Action  │
│ 33479707  RTX 3060   $0.06   running  2h 30m   [Kill]  │
│ 33479853  RTX 3060   $0.01   stopped  1d       [Kill]  │
│                                                        │
│ Total unmanaged cost: $0.07/hr                         │
└────────────────────────────────────────────────────────┘
```

**Data layer addition** (`dashboard/data.py`):
```python
async def get_unmanaged_instances() -> list[dict]:
    """Return instances that exist on Vast.ai but have no CCM job association."""
    async with get_session() as session:
        stmt = select(Instance).where(
            Instance.job_id.is_(None),
            Instance.status.in_([InstanceStatus.RUNNING, InstanceStatus.STARTING]),
        )
        result = await session.execute(stmt)
        instances = result.scalars().all()

    return [{"instance_id": i.instance_id, "gpu_type": i.gpu_type,
             "hourly_rate": i.hourly_rate, "status": i.status.value,
             ...} for i in instances]
```

### Phase 4: Termination Hardening

Make instance termination robust with retries and verification:

```python
async def terminate_instance_safe(provider, instance_id: str, max_retries: int = 3) -> bool:
    """Terminate instance with retries and verification."""
    for attempt in range(max_retries):
        try:
            await provider.terminate_instance(instance_id)
            # Verify it's actually gone
            await asyncio.sleep(5)
            inst = await provider.get_instance(instance_id)
            if inst is None or inst.status in (ProviderStatus.TERMINATED, ProviderStatus.STOPPED):
                return True
        except Exception as e:
            logger.warning("Termination attempt failed", instance_id=instance_id,
                          attempt=attempt+1, error=str(e))
            await asyncio.sleep(2)

    logger.error("Failed to terminate instance after retries", instance_id=instance_id)
    return False
```

### Phase 5: SKILL.md Agent Rules

Add clear rules to SKILL.md that agents must follow:

```markdown
## CRITICAL RULES FOR AGENTS

1. **NEVER call `vastai create instance` or `vastai destroy instance` directly.**
   Always use `ccm jobs submit` to create and `ccm jobs cancel` to terminate.

2. **ALWAYS set a unique `project` name** in every job YAML.

3. **For SSH/commands on a running instance**, use:
   - `ccm exec <job_id> "command"` for one-shot commands
   - `ccm ssh <job_id>` for interactive sessions
   - These are fine to use freely — they don't change infrastructure state.

4. **To check job status**: `ccm jobs list --project my-project`
   Never parse `vastai show instances` directly.

5. **If you need a capability CCM doesn't have**, ask the user rather than
   going around CCM. Going direct creates ghost instances.
```

---

## Implementation Phases & Dependencies

```
Phase 1: Labels on creation                     [independent]
├── Add label param to create_instance()
├── Build label in submit_job() and SDK submit()
├── Parse label in _parse_instance()
└── Validation: create instance, verify label on Vast.ai

Phase 2: Smart sync with labels                  [depends on Phase 1]
├── Add label field to ProviderInstance
├── Update sync_all_instances() to parse labels
├── Match unlabeled instances
└── Validation: sync discovers labeled + unlabeled instances

Phase 3: Dashboard unmanaged section             [depends on Phase 2]
├── Add get_unmanaged_instances() to data.py
├── Create unmanaged.html partial
├── Wire into dashboard.html
└── Validation: dashboard shows unmanaged instances with warnings

Phase 4: Termination hardening                   [independent]
├── Add terminate_instance_safe() with retries
├── Wire into cancel_job, handle_job_completion
├── Add dashboard alert for failed terminations
└── Validation: cancel job → verify instance actually destroyed

Phase 5: Agent rules in SKILL.md                 [independent]
├── Update SKILL.md with rules
├── Update docs/usage.md with same rules
└── Validation: read docs, rules are clear
```

**Phases 1, 4, 5 can run in parallel.**
**Phase 2 depends on Phase 1.**
**Phase 3 depends on Phase 2.**

---

## Testing Strategy

### Unit Tests

```python
class TestInstanceLabels:
    def test_build_label(self):
        label = build_instance_label("job_abc", "my-project", "my-job")
        assert label == "ccm|job_abc|my-project|my-job"

    def test_build_label_sanitizes_pipes(self):
        label = build_instance_label("job_abc", "proj|ect", "na|me")
        assert "|" not in label.split("|", 3)[2]  # project has no extra pipes

    def test_parse_label_valid(self):
        result = parse_instance_label("ccm|job_abc|my-project|my-job")
        assert result["job_id"] == "job_abc"
        assert result["project"] == "my-project"

    def test_parse_label_not_ccm(self):
        assert parse_instance_label("some-other-label") is None
        assert parse_instance_label(None) is None
        assert parse_instance_label("") is None

    def test_parse_label_minimal(self):
        result = parse_instance_label("ccm|job_abc")
        assert result["job_id"] == "job_abc"
        assert result["project"] is None

class TestTerminationSafe:
    async def test_terminate_retries_on_failure(self):
        """Should retry up to max_retries times."""

    async def test_terminate_verifies_destruction(self):
        """Should check instance is actually gone after terminate call."""

class TestSyncWithLabels:
    async def test_sync_matches_labeled_instance_to_job(self):
        """Instance with ccm| label should auto-link to job."""

    async def test_sync_flags_unlabeled_as_unmanaged(self):
        """Instance without ccm| label should be flagged."""

class TestDashboardUnmanaged:
    async def test_unmanaged_instances_shown(self):
        """Dashboard should show unmanaged instances with warnings."""
```

### Integration Tests

- Create instance via `ccm submit` → verify label appears on Vast.ai
- Create instance via `vastai create` directly → verify sync picks it up as unmanaged
- Cancel job → verify instance is actually terminated (check Vast.ai API)
- Kill daemon, create instance, restart daemon → verify sync catches it

### Manual Validation Checklist

- [ ] `ccm submit` sets label on Vast.ai instance
- [ ] `vastai show instances --raw` shows label field with ccm| prefix
- [ ] Dashboard shows labeled instances with full job data
- [ ] Dashboard shows unlabeled instances in "Unmanaged" section
- [ ] Cancel job → instance actually destroyed on Vast.ai
- [ ] Instance created outside CCM → shows as unmanaged on dashboard
- [ ] Daemon restart → all instances synced correctly

---

## Estimated Effort

| Phase | Effort | Parallelizable |
|-------|--------|----------------|
| Phase 1: Labels | 1-2h | Yes |
| Phase 2: Smart sync | 1-2h | No (after Phase 1) |
| Phase 3: Dashboard unmanaged | 1-2h | No (after Phase 2) |
| Phase 4: Termination hardening | 1h | Yes |
| Phase 5: Agent rules | 30min | Yes |
| Tests | 1-2h | After all phases |
| **Total** | **~6-8h** | |
