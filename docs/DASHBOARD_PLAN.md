# CCM Dashboard — Implementation Plan

**Date**: 2026-03-24
**Technology**: FastAPI + Jinja2 + HTMX + SSE (zero new Python dependencies)
**Goal**: Single-page web dashboard showing all CCM jobs, costs, events, and alerts

---

## 1. Architecture Overview

```
Browser (localhost:8765)
    │
    ├── GET /dashboard              → Jinja2 renders full page (dashboard.html)
    ├── GET /dashboard/sse          → Server-Sent Events stream (live updates)
    ├── GET /dashboard/api/summary  → JSON: summary cards data
    ├── GET /dashboard/api/jobs     → JSON: active + recent jobs with metrics
    ├── GET /dashboard/api/events   → JSON: recent events from daemon log
    ├── GET /dashboard/api/costs    → JSON: per-project cost breakdown
    │
    │   HTMX partials (swapped in-place):
    ├── GET /dashboard/partials/alerts    → HTML fragment: alert banner
    ├── GET /dashboard/partials/stats     → HTML fragment: summary cards
    ├── GET /dashboard/partials/jobs      → HTML fragment: jobs table body
    ├── GET /dashboard/partials/events    → HTML fragment: events feed
    └── GET /dashboard/partials/costs     → HTML fragment: cost breakdown

FastAPI app (api/app.py)
    │
    ├── /v1/*           existing API routes (unchanged)
    ├── /dashboard/*    new dashboard routes
    └── /health         existing health check

SQLite DB ←── daemon/monitor.py writes job status, metrics, events
```

**Key design decision**: The dashboard does NOT run its own monitoring loop. It reads from the same SQLite database the daemon writes to. The daemon is the single writer; the dashboard is a read-only viewer.

**SSE flow**: A background asyncio task in the dashboard routes polls the DB every 5 seconds, compares state to the last emission, and pushes HTML fragments via SSE to any connected browsers. HTMX on the frontend listens for SSE events and swaps the relevant DOM elements.

---

## 2. File Structure

```
src/cloudcomputemanager/
  dashboard/
    __init__.py              # Router factory: create_dashboard_router()
    routes.py                # All route handlers + SSE endpoint
    data.py                  # Data aggregation queries (summary, jobs, costs, alerts)
    templates/
      base.html              # HTML skeleton: head, nav, body, HTMX + CSS includes
      dashboard.html         # Main page: extends base, all sections
      partials/
        alerts.html          # Alert banner fragment
        stats.html           # 4 summary cards fragment
        jobs_table.html      # Active jobs table rows fragment
        job_detail.html      # Expanded job detail fragment
        events.html          # Events feed fragment
        costs.html           # Cost breakdown fragment
        finished.html        # Completed/failed jobs fragment
    static/
      htmx.min.js            # Vendored HTMX (14 KB, no CDN dependency)
      style.css              # Dashboard styles (Pico CSS base + custom)
```

---

## 3. Data Layer (`dashboard/data.py`)

This module contains all database queries the dashboard needs. Keeps route handlers thin.

### Functions to Implement

```python
async def get_dashboard_summary() -> dict:
    """Return summary card data.

    Returns:
        {
            "active_jobs": int,        # RUNNING + PROVISIONING + RECOVERING + CHECKPOINTING
            "active_breakdown": dict,  # {"running": 3, "recovering": 1, ...}
            "today_spend": float,      # Sum of cost for jobs active today
            "burn_rate": float,        # Sum of hourly_rate for all active instances
            "week_spend": float,       # Rolling 7-day total cost
            "week_projects": int,      # Distinct projects with spend this week
            "recoveries_24h": int,     # Jobs with attempt_number > 0 changed in 24h
            "recoveries_ok": int,      # Successfully recovered
            "recoveries_failed": int,  # Failed to recover
        }

    Queries:
        - SELECT status, count(*) FROM jobs WHERE status IN (...) GROUP BY status
        - SELECT sum(total_cost_usd) FROM jobs WHERE started_at >= today
        - For burn_rate: join Job → Instance on instance_id, sum hourly_rate
        - SELECT sum(total_cost_usd), count(distinct project) FROM jobs WHERE created_at >= 7d ago
        - SELECT count(*) FROM jobs WHERE attempt_number > 0 AND updated recently
    """

async def get_active_jobs() -> list[dict]:
    """Return all non-terminal jobs with enriched data.

    Returns list of dicts, each with:
        job_id, name, project, status, image, command,
        current_stage, total_stages,     # from stages_json
        progress_percent, steps_per_second, estimated_hours_remaining,  # from metrics_json
        gpu_type, hourly_rate,           # from Instance join
        cost_so_far,                     # total_cost_usd OR computed from hourly_rate * elapsed
        runtime_display,                 # formatted "6h 23m"
        attempt_number,                  # recovery count
        sync_status, last_sync_at,
        ssh_host, ssh_port,              # from Instance join
        started_at,

    Query:
        SELECT j.*, i.gpu_type, i.hourly_rate, i.ssh_host, i.ssh_port, i.gpu_count
        FROM jobs j
        LEFT JOIN instances i ON j.instance_id = i.instance_id
        WHERE j.status IN (RUNNING, PROVISIONING, RECOVERING, CHECKPOINTING, PENDING)
        ORDER BY j.created_at DESC
    """

async def get_recent_events(hours: int = 24) -> list[dict]:
    """Return recent events from daemon log file.

    Reads DaemonService.get_logs() and filters to last N hours.

    Returns list of:
        {
            "timestamp": str,          # formatted "3:42 AM"
            "event_type": str,         # "job_completed", "job_preempted", etc.
            "job_id": str,
            "job_name": str,           # looked up from DB
            "icon": str,               # "●", "⚡", "▲", "↓", etc.
            "color": str,              # "green", "red", "yellow", "blue"
            "detail": str,             # "exit 0", "attempt 3", "42 files, 1.8 GB"
        }
    """

async def get_cost_breakdown(days: int = 7) -> dict:
    """Return per-project and per-GPU cost breakdown.

    Returns:
        {
            "by_project": [
                {"project": str, "job_count": int, "cost": float, "gpu_hours": float, "cost_per_hour": float},
                ...
            ],
            "by_gpu": [
                {"gpu_type": str, "cost": float, "percentage": float, "hours": float, "avg_rate": float},
                ...
            ],
            "total_cost": float,
            "total_hours": float,
        }

    Query:
        Per-project: SELECT project, count(*), sum(total_cost_usd), sum(total_runtime_seconds)
                     FROM jobs WHERE created_at >= N days ago GROUP BY project
        Per-GPU: From CostRecord table or Instance join
    """

async def get_alerts() -> list[dict]:
    """Return actionable alerts.

    Checks:
        1. FAILED jobs in last 24h
        2. BUDGET_EXCEEDED jobs in last 24h
        3. RUNNING jobs with stale progress (metrics.last_updated > 30 min ago)
        4. Budget warnings: projects > 80% of budget
        5. Daemon not running (DaemonService.is_running() == False)

    Returns list of:
        {"severity": "red"|"yellow"|"blue", "message": str, "job_id": str|None, "action": str|None}
    """

async def get_finished_jobs(hours: int = 24, limit: int = 50) -> list[dict]:
    """Return recently completed/failed/cancelled jobs.

    Same fields as get_active_jobs() plus:
        exit_code, output_location, final total_cost_usd, final runtime
    """
```

---

## 4. Routes (`dashboard/routes.py`)

### Page Route

```python
@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Render the full dashboard page."""
    summary = await get_dashboard_summary()
    jobs = await get_active_jobs()
    events = await get_recent_events()
    costs = await get_cost_breakdown()
    alerts = await get_alerts()
    finished = await get_finished_jobs()

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "summary": summary,
        "jobs": jobs,
        "events": events,
        "costs": costs,
        "alerts": alerts,
        "finished": finished,
    })
```

### HTMX Partial Routes

```python
@router.get("/dashboard/partials/alerts", response_class=HTMLResponse)
async def alerts_partial(request: Request):
    alerts = await get_alerts()
    return templates.TemplateResponse("partials/alerts.html", {"request": request, "alerts": alerts})

@router.get("/dashboard/partials/stats", response_class=HTMLResponse)
async def stats_partial(request: Request):
    summary = await get_dashboard_summary()
    return templates.TemplateResponse("partials/stats.html", {"request": request, "summary": summary})

@router.get("/dashboard/partials/jobs", response_class=HTMLResponse)
async def jobs_partial(request: Request):
    jobs = await get_active_jobs()
    return templates.TemplateResponse("partials/jobs_table.html", {"request": request, "jobs": jobs})

# ... same pattern for events, costs, finished
```

### SSE Endpoint

```python
@router.get("/dashboard/sse")
async def dashboard_sse(request: Request):
    """Server-Sent Events stream for live dashboard updates.

    Pushes HTMX-compatible events every 5 seconds.
    Each event contains an HTML fragment that HTMX swaps into the page.
    """
    async def event_generator():
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break

            # Generate updated partials
            alerts = await get_alerts()
            summary = await get_dashboard_summary()
            jobs = await get_active_jobs()

            # Emit as named SSE events that HTMX listens for
            yield {
                "event": "alerts",
                "data": render_template("partials/alerts.html", alerts=alerts)
            }
            yield {
                "event": "stats",
                "data": render_template("partials/stats.html", summary=summary)
            }
            yield {
                "event": "jobs",
                "data": render_template("partials/jobs_table.html", jobs=jobs)
            }

            await asyncio.sleep(5)

    return EventSourceResponse(event_generator())
```

### Action Routes

```python
@router.post("/dashboard/actions/cancel/{job_id}")
async def cancel_job_action(job_id: str):
    """Cancel a job from the dashboard."""
    # Reuse existing API logic
    ...

@router.post("/dashboard/actions/sync/{job_id}")
async def sync_job_action(job_id: str):
    """Trigger sync from the dashboard."""
    ...
```

---

## 5. Frontend Templates

### `base.html` — Skeleton

```html
<!DOCTYPE html>
<html>
<head>
    <title>CCM Dashboard</title>
    <link rel="stylesheet" href="/dashboard/static/style.css">
    <script src="/dashboard/static/htmx.min.js"></script>
</head>
<body>
    <header>
        CCM v{{ version }} | Daemon: {{ daemon_status }} | Last poll: {{ last_poll }}
    </header>
    {% block content %}{% endblock %}
</body>
</html>
```

### `dashboard.html` — Main Page

```html
{% extends "base.html" %}
{% block content %}

<!-- Alert Banner: auto-refreshes via SSE -->
<div id="alerts" hx-ext="sse" sse-connect="/dashboard/sse" sse-swap="alerts">
    {% include "partials/alerts.html" %}
</div>

<!-- Summary Cards: auto-refreshes via SSE -->
<div id="stats" sse-swap="stats">
    {% include "partials/stats.html" %}
</div>

<!-- Active Jobs Table: auto-refreshes via SSE -->
<div id="jobs" sse-swap="jobs">
    {% include "partials/jobs_table.html" %}
</div>

<!-- Two-column: Events + Costs -->
<div class="grid">
    <div id="events" hx-get="/dashboard/partials/events" hx-trigger="every 10s">
        {% include "partials/events.html" %}
    </div>
    <div id="costs" hx-get="/dashboard/partials/costs" hx-trigger="every 30s">
        {% include "partials/costs.html" %}
    </div>
</div>

<!-- Finished Jobs (collapsed) -->
<details>
    <summary>Recently Finished (24h): {{ finished|length }} jobs</summary>
    <div id="finished" hx-get="/dashboard/partials/finished" hx-trigger="every 30s">
        {% include "partials/finished.html" %}
    </div>
</details>

{% endblock %}
```

### HTMX Patterns Used

| Pattern | Where | How |
|---------|-------|-----|
| `sse-connect` + `sse-swap` | Alerts, Stats, Jobs | SSE pushes HTML fragments, HTMX swaps by event name |
| `hx-trigger="every Ns"` | Events, Costs, Finished | Polling fallback for less-critical sections |
| `hx-get` + `hx-target` | Job detail expansion | Click job name → load detail partial into row |
| `hx-post` + `hx-confirm` | Cancel button | Confirm dialog → POST to action route |
| `hx-swap="outerHTML"` | All partials | Replace entire element with response |

---

## 6. CSS Approach

Use **Pico CSS** (10 KB, classless) as the base, with custom overrides for:
- Status badge colors (green/yellow/red/blue)
- Progress bar styling
- Summary card layout (4-column grid)
- Dense table rows (compact padding)
- Alert banner (colored left border)

Vendor Pico CSS into `static/pico.min.css` — no CDN dependency.

---

## 7. CLI Integration

### `ccm dashboard` command

```python
@app.command("dashboard")
def dashboard(
    port: int = typer.Option(8765, "--port", "-p"),
    no_browser: bool = typer.Option(False, "--no-browser"),
):
    """Open the CCM dashboard in your browser."""
    import webbrowser
    from cloudcomputemanager.api.app import create_app
    import uvicorn

    if not no_browser:
        # Open browser after short delay
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}/dashboard")).start()

    app = create_app()  # Dashboard routes are registered in create_app()
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### Dashboard route registration in `create_app()`

```python
def create_app() -> FastAPI:
    app = FastAPI(...)
    # ... existing routes ...

    # Mount dashboard
    from cloudcomputemanager.dashboard import create_dashboard_router
    app.include_router(create_dashboard_router())
    app.mount("/dashboard/static", StaticFiles(directory=STATIC_DIR), name="dashboard-static")

    return app
```

---

## 8. Implementation Phases

### Phase 1: Foundation (can be parallelized)

**Agent A: Data layer** (`dashboard/data.py`)
- Implement all 6 query functions
- Test each against a seeded SQLite database
- Handle edge cases: empty DB, no running jobs, no events
- Validation: unit tests with fixtures

**Agent B: Templates + static assets**
- Create all HTML templates (base, dashboard, 6 partials)
- Vendor htmx.min.js and pico.min.css
- Build the CSS overrides for status badges, progress bars, cards
- Validation: templates render without errors using dummy data

**Completion gate**: Both agents done → data layer returns correct dicts, templates render with dummy data.

### Phase 2: Routes + Integration

**Single agent (depends on Phase 1)**
- Implement `dashboard/routes.py` with all routes
- Implement `dashboard/__init__.py` with `create_dashboard_router()`
- Wire into `api/app.py` `create_app()`
- Implement SSE endpoint
- Add `ccm dashboard` CLI command to `cli/main.py`
- Validation: `ccm dashboard` opens browser, page loads, shows real data from DB

**Completion gate**: Dashboard displays real job data from SQLite. Manual refresh works.

### Phase 3: Live Updates

**Single agent (depends on Phase 2)**
- Implement SSE event generator that polls DB every 5s
- Wire HTMX `sse-connect` + `sse-swap` on frontend
- Add `hx-trigger="every Ns"` for polling fallback sections
- Validation: change a job's status in DB → dashboard updates within 5s without manual refresh

**Completion gate**: Dashboard auto-updates. Start a job via `ccm submit`, watch it appear in dashboard.

### Phase 4: Actions + Polish

**Single agent (depends on Phase 3)**
- Implement action routes (cancel, sync, view logs)
- Add HTMX `hx-post` with `hx-confirm` for cancel buttons
- Add SSH command copy-to-clipboard
- Add log viewer modal/drawer (hx-get loads last 50 lines)
- Add expandable job detail rows
- Validation: cancel a job from dashboard → job actually cancels. Click SSH → copyable command.

**Completion gate**: All interactive features work. Full end-to-end test.

### Phase 5: Tests + Docs

**Can be parallelized:**

**Agent A: Tests**
- Test data.py functions with seeded DB fixtures
- Test route responses (status codes, content type)
- Test SSE endpoint returns valid event stream
- Test alert generation logic (stalled jobs, budget warnings)

**Agent B: Documentation**
- Update `docs/usage.md` with dashboard section
- Update AGENTS.md with dashboard status
- Update README.md with dashboard screenshot/description
- Commit and push

---

## 9. Data Model Changes Required

### 9a. Cache hourly_rate on Job (small, do in Phase 1)

Currently `hourly_rate` lives on the Instance model. The dashboard needs it per-job. Two options:

**Option A (recommended)**: Join in the query — `LEFT JOIN instances ON job.instance_id = instance.instance_id`. No model change needed. The `get_active_jobs()` function does this join.

**Option B**: Add `hourly_rate` field to Job model, set at provisioning time. Avoids join but requires model migration.

**Decision**: Option A. The join is simple, avoids schema changes, and Instance records are already in the same SQLite DB.

### 9b. Event persistence (already solved)

Events are already persisted to `daemon.log` as JSON lines by `DaemonService._handle_event()`. The `get_recent_events()` function reads this file via `DaemonService.get_logs()`. No model change needed.

### 9c. Running cost updates (already partially solved)

`Job.total_cost_usd` is set at completion. For running jobs, the dashboard computes cost client-side:
```python
cost_so_far = job.total_cost_usd + (instance.hourly_rate * hours_since_started)
```
This avoids needing the daemon to update cost every poll cycle. The dashboard just does the math.

---

## 10. Testing Strategy

### Unit Tests (`tests/test_dashboard.py`)

```python
class TestDashboardData:
    """Test data aggregation functions."""

    async def test_summary_empty_db(self):
        """Summary should return zeros on empty DB."""

    async def test_summary_with_running_jobs(self):
        """Summary should count active jobs correctly."""

    async def test_active_jobs_includes_metrics(self):
        """Active jobs should include parsed JobMetrics."""

    async def test_active_jobs_joins_instance(self):
        """Active jobs should include GPU type and hourly_rate from Instance."""

    async def test_alerts_detects_failed_jobs(self):
        """Alerts should include recently failed jobs."""

    async def test_alerts_detects_stalled_progress(self):
        """Alerts should flag jobs with stale metrics."""

    async def test_cost_breakdown_by_project(self):
        """Cost breakdown should aggregate by project."""

    async def test_events_from_daemon_log(self):
        """Events should parse from daemon.log JSON lines."""

class TestDashboardRoutes:
    """Test HTTP endpoints."""

    async def test_dashboard_page_returns_html(self):
        """GET /dashboard should return 200 with text/html."""

    async def test_partials_return_html_fragments(self):
        """GET /dashboard/partials/* should return HTML fragments."""

    async def test_sse_endpoint_returns_event_stream(self):
        """GET /dashboard/sse should return text/event-stream."""

    async def test_cancel_action(self):
        """POST /dashboard/actions/cancel/{job_id} should cancel the job."""
```

### Manual Validation Checklist

- [ ] `ccm dashboard` opens browser to localhost:8765/dashboard
- [ ] Page loads with all 6 sections visible
- [ ] Summary cards show correct counts
- [ ] Active jobs table shows running jobs with progress bars
- [ ] Events feed shows recent completions/failures
- [ ] Cost breakdown shows per-project totals
- [ ] SSE updates: change job status → dashboard updates within 5s
- [ ] Cancel button works (with confirmation)
- [ ] SSH command is copyable
- [ ] Log viewer shows job output
- [ ] Alert banner appears when a job fails
- [ ] Alert banner disappears when no problems
- [ ] Page works with 0 jobs (empty state)
- [ ] Page works with 50+ jobs (performance)
- [ ] Works in Chrome, Firefox
- [ ] Works over WSL2 port forwarding

---

## 11. Estimated Effort

| Phase | Effort | Parallelizable |
|-------|--------|----------------|
| Phase 1: Data layer + Templates | 3-4h | Yes (2 agents) |
| Phase 2: Routes + Integration | 2-3h | No (sequential) |
| Phase 3: Live Updates (SSE) | 1-2h | No (sequential) |
| Phase 4: Actions + Polish | 2-3h | No (sequential) |
| Phase 5: Tests + Docs | 2h | Yes (2 agents) |
| **Total** | **~10-14h** | |

---

## 12. Dependencies

**Python packages**: None new. Already have: FastAPI, Jinja2, uvicorn, structlog, SQLModel, aiosqlite.

**Frontend**: Vendored files only (no CDN, no npm):
- `htmx.min.js` — 14 KB, download from htmx.org
- `pico.min.css` — 10 KB, download from picocss.com
- `style.css` — custom, ~200 lines

**Browser**: Any modern browser. No polyfills needed (HTMX and SSE are widely supported).
