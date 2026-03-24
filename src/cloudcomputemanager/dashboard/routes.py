"""Dashboard routes for CCM web interface."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from cloudcomputemanager import __version__
from cloudcomputemanager.daemon.service import DaemonService
from cloudcomputemanager.dashboard.data import (
    get_alerts,
    get_active_jobs,
    get_cost_breakdown,
    get_dashboard_summary,
    get_finished_jobs,
    get_recent_events,
)

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter()


def _daemon_context() -> dict:
    """Get daemon status context for the header."""
    from datetime import datetime
    try:
        # Use is_running() (checks actual PID) not get_status() (reads stale file)
        if DaemonService.is_running():
            status = DaemonService.get_status()
            ts = (status or {}).get("timestamp", "")
            if ts:
                try:
                    last = datetime.fromisoformat(ts)
                    ago = int((datetime.utcnow() - last).total_seconds())
                except (ValueError, TypeError):
                    ago = "?"
            else:
                ago = "?"
            return {"daemon_status": "running", "daemon_last_poll": ago}
        return {"daemon_status": "stopped", "daemon_last_poll": "--"}
    except Exception:
        return {"daemon_status": "unknown", "daemon_last_poll": "--"}


# ============================================================================
# Main page
# ============================================================================


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Render the full dashboard page."""
    summary = await get_dashboard_summary()
    jobs = await get_active_jobs()
    events = await get_recent_events()
    costs = await get_cost_breakdown()
    alerts = await get_alerts()
    finished = await get_finished_jobs()

    context = {
        "request": request,
        "version": __version__,
        **_daemon_context(),
        "summary": summary,
        "jobs": jobs,
        "events": events,
        "costs": costs,
        "alerts": alerts,
        "finished": finished,
    }
    return templates.TemplateResponse("dashboard.html", context)


# ============================================================================
# HTMX partial endpoints (for polling fallback)
# ============================================================================


@router.get("/dashboard/partials/alerts", response_class=HTMLResponse)
async def alerts_partial(request: Request):
    """Return alerts banner HTML fragment."""
    alerts = await get_alerts()
    return templates.TemplateResponse("partials/alerts.html", {"request": request, "alerts": alerts})


@router.get("/dashboard/partials/stats", response_class=HTMLResponse)
async def stats_partial(request: Request):
    """Return summary cards HTML fragment."""
    summary = await get_dashboard_summary()
    return templates.TemplateResponse("partials/stats.html", {"request": request, "summary": summary})


@router.get("/dashboard/partials/jobs", response_class=HTMLResponse)
async def jobs_partial(request: Request):
    """Return active jobs table HTML fragment."""
    jobs = await get_active_jobs()
    return templates.TemplateResponse("partials/jobs_table.html", {"request": request, "jobs": jobs})


@router.get("/dashboard/partials/events", response_class=HTMLResponse)
async def events_partial(request: Request):
    """Return events feed HTML fragment."""
    events = await get_recent_events()
    return templates.TemplateResponse("partials/events.html", {"request": request, "events": events})


@router.get("/dashboard/partials/costs", response_class=HTMLResponse)
async def costs_partial(request: Request):
    """Return cost breakdown HTML fragment."""
    costs = await get_cost_breakdown()
    return templates.TemplateResponse("partials/costs.html", {"request": request, "costs": costs})


@router.get("/dashboard/partials/finished", response_class=HTMLResponse)
async def finished_partial(request: Request):
    """Return finished jobs table HTML fragment."""
    finished = await get_finished_jobs()
    return templates.TemplateResponse("partials/finished.html", {"request": request, "finished": finished})


# ============================================================================
# SSE endpoint for live updates
# ============================================================================


@router.get("/dashboard/sse")
async def dashboard_sse(request: Request):
    """Server-Sent Events stream for live dashboard updates.

    Pushes named events every 5 seconds. HTMX's sse-swap attribute
    matches event names to DOM element IDs and swaps the content.
    """
    from starlette.responses import StreamingResponse

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break

            try:
                # Fetch fresh data
                alerts = await get_alerts()
                summary = await get_dashboard_summary()
                jobs = await get_active_jobs()

                # Render partials to strings
                alerts_html = templates.get_template("partials/alerts.html").render(
                    alerts=alerts
                )
                stats_html = templates.get_template("partials/stats.html").render(
                    summary=summary
                )
                jobs_html = templates.get_template("partials/jobs_table.html").render(
                    jobs=jobs
                )

                # Emit as named SSE events
                for event_name, html in [
                    ("alerts", alerts_html),
                    ("stats", stats_html),
                    ("jobs", jobs_html),
                ]:
                    # SSE format: event name + data (newlines replaced for SSE protocol)
                    lines = html.replace("\n", "\n data: ")
                    yield f"event: {event_name}\ndata: {lines}\n\n"

            except Exception:
                # Don't crash the SSE stream on errors
                yield f"event: error\ndata: Dashboard update failed\n\n"

            await asyncio.sleep(5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# Action endpoints
# ============================================================================


@router.post("/dashboard/actions/cancel/{job_id}", response_class=HTMLResponse)
async def cancel_job_action(job_id: str, request: Request):
    """Cancel a job and return updated jobs table."""
    from cloudcomputemanager.core.database import get_session
    from cloudcomputemanager.core.models import Job, JobStatus
    from cloudcomputemanager.providers.vast import VastProvider
    from sqlmodel import select

    provider = VastProvider()

    async with get_session() as session:
        stmt = select(Job).where(Job.job_id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if job and job.status in (JobStatus.RUNNING, JobStatus.PROVISIONING, JobStatus.RECOVERING):
            if job.instance_id:
                try:
                    await provider.terminate_instance(job.instance_id)
                except Exception:
                    pass
            job.status = JobStatus.CANCELLED
            session.add(job)

    # Return updated jobs table
    jobs = await get_active_jobs()
    return templates.TemplateResponse("partials/jobs_table.html", {"request": request, "jobs": jobs})


@router.post("/dashboard/actions/sync/{job_id}", response_class=HTMLResponse)
async def sync_job_action(job_id: str):
    """Trigger immediate sync for a job."""
    from cloudcomputemanager.core.database import get_session
    from cloudcomputemanager.core.models import Job
    from cloudcomputemanager.core.config import get_settings
    from cloudcomputemanager.providers.vast import VastProvider
    from sqlmodel import select

    provider = VastProvider()
    settings = get_settings()

    async with get_session() as session:
        stmt = select(Job).where(Job.job_id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

    if job and job.instance_id:
        sync_config = job.sync_config or {}
        sync_dir = settings.sync_local_path / job.job_id
        sync_dir.mkdir(parents=True, exist_ok=True)
        await provider.rsync_download(
            job.instance_id,
            sync_config.get("source", "/workspace") + "/",
            str(sync_dir) + "/",
        )
        return HTMLResponse("<span class='badge badge-green'>Synced</span>")

    return HTMLResponse("<span class='badge badge-red'>Failed</span>")


@router.get("/dashboard/actions/logs/{job_id}", response_class=HTMLResponse)
async def job_logs_action(job_id: str):
    """Return last 50 lines of job log."""
    from cloudcomputemanager.core.database import get_session
    from cloudcomputemanager.core.models import Job
    from cloudcomputemanager.providers.vast import VastProvider
    from sqlmodel import select

    provider = VastProvider()

    async with get_session() as session:
        stmt = select(Job).where(Job.job_id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

    if job and job.instance_id:
        try:
            rc, stdout, stderr = await provider.execute_command(
                job.instance_id,
                "tail -50 /workspace/job.log 2>/dev/null || echo 'No logs available'",
                timeout=15,
            )
            log_text = stdout if rc == 0 else f"Error: {stderr}"
        except Exception as e:
            log_text = f"Cannot connect to instance: {e}"
    else:
        log_text = "No instance associated with this job"

    # Return as pre-formatted HTML for the log modal
    import html
    escaped = html.escape(log_text)
    return HTMLResponse(
        f'<dialog open id="log-dialog">'
        f'<article>'
        f'<header><strong>Logs: {job_id}</strong>'
        f'<button onclick="this.closest(\'dialog\').remove()" style="float:right">&times;</button></header>'
        f'<pre style="max-height:60vh;overflow:auto;font-size:0.8rem">{escaped}</pre>'
        f'</article></dialog>'
    )
