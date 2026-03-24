"""Dashboard data layer for CloudComputeManager.

Provides async functions that query the CCM database and daemon logs
to produce dashboard-ready data structures. All functions handle empty
databases gracefully (returning empty lists / zero counts, never crashing).
"""

import json
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import func
from sqlmodel import select

from cloudcomputemanager.core.database import get_session
from cloudcomputemanager.core.models import (
    CostRecord,
    Instance,
    Job,
    JobMetrics,
    JobStatus,
)
from cloudcomputemanager.daemon.service import DaemonService


# Statuses considered "terminal" (not active)
_TERMINAL_STATUSES = {
    JobStatus.COMPLETED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
    JobStatus.BUDGET_EXCEEDED,
}

# Event icon / colour mapping
_EVENT_STYLE = {
    "job_started":          ("●", "green"),
    "job_completed":        ("●", "green"),
    "job_failed":           ("●", "red"),
    "job_preempted":        ("⚡", "yellow"),
    "sync_completed":       ("↓", "blue"),
    "instance_terminated":  ("○", "gray"),
}


def _format_runtime(seconds: float) -> str:
    """Format seconds as 'Xh Ym'."""
    if seconds <= 0:
        return "0h 0m"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def _format_event_time(dt: datetime) -> str:
    """Format a datetime as '3:42 AM'."""
    return dt.strftime("%-I:%M %p")


# ============================================================================
# 1. Dashboard summary
# ============================================================================


async def get_dashboard_summary() -> dict:
    """Return high-level dashboard summary.

    Keys:
        jobs_by_status  – dict[str, int]
        today_spend     – float (USD)
        burn_rate       – float (USD/hr for all RUNNING jobs)
        week_spend      – float (USD)
        recovery_stats  – dict with recovered_24h count
    """
    now = datetime.utcnow()
    today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago = now - timedelta(days=7)
    day_ago = now - timedelta(hours=24)

    jobs_by_status: dict[str, int] = {}
    today_spend: float = 0.0
    burn_rate: float = 0.0
    week_spend: float = 0.0
    recovered_24h: int = 0

    try:
        async with get_session() as session:
            # --- Jobs by status (active statuses) ---
            for status in (
                JobStatus.RUNNING,
                JobStatus.PROVISIONING,
                JobStatus.RECOVERING,
                JobStatus.CHECKPOINTING,
            ):
                stmt = (
                    select(func.count())
                    .select_from(Job)
                    .where(Job.status == status)
                )
                result = await session.execute(stmt)
                jobs_by_status[status.value] = result.scalar_one_or_none() or 0

            # --- Today's spend ---
            stmt = (
                select(func.coalesce(func.sum(Job.total_cost_usd), 0.0))
                .where(Job.started_at >= today_midnight)
            )
            result = await session.execute(stmt)
            today_spend = float(result.scalar_one_or_none() or 0.0)

            # --- Burn rate (sum hourly_rate for running jobs' instances) ---
            stmt = (
                select(func.coalesce(func.sum(Instance.hourly_rate), 0.0))
                .where(
                    Instance.job_id.in_(
                        select(Job.job_id).where(Job.status == JobStatus.RUNNING)
                    )
                )
            )
            result = await session.execute(stmt)
            burn_rate = float(result.scalar_one_or_none() or 0.0)

            # --- Week spend ---
            stmt = (
                select(func.coalesce(func.sum(Job.total_cost_usd), 0.0))
                .where(Job.created_at >= week_ago)
            )
            result = await session.execute(stmt)
            week_spend = float(result.scalar_one_or_none() or 0.0)

            # --- Recovery stats (jobs with attempt_number > 0 in last 24h) ---
            stmt = (
                select(func.count())
                .select_from(Job)
                .where(Job.attempt_number > 0, Job.created_at >= day_ago)
            )
            result = await session.execute(stmt)
            recovered_24h = result.scalar_one_or_none() or 0

    except Exception:
        pass  # graceful fallback – return zeros

    # Compute aggregates the template expects
    active_statuses = {JobStatus.RUNNING.value, JobStatus.PROVISIONING.value,
                       JobStatus.RECOVERING.value, JobStatus.CHECKPOINTING.value}
    active_jobs = sum(v for k, v in jobs_by_status.items() if k in active_statuses)
    active_breakdown = {k: v for k, v in jobs_by_status.items() if k in active_statuses and v > 0}

    # Count distinct projects with spend this week
    try:
        async with get_session() as session:
            stmt = (
                select(func.count(func.distinct(Job.project)))
                .where(Job.created_at >= week_ago, Job.project.isnot(None))
            )
            result = await session.execute(stmt)
            week_projects = result.scalar_one_or_none() or 0
    except Exception:
        week_projects = 0

    return {
        "active_jobs": active_jobs,
        "active_breakdown": active_breakdown,
        "jobs_by_status": jobs_by_status,
        "today_spend": today_spend,
        "burn_rate": burn_rate,
        "week_spend": week_spend,
        "week_projects": week_projects,
        "recoveries_24h": recovered_24h,
        "recoveries_ok": recovered_24h,  # TODO: distinguish ok vs failed
        "recoveries_failed": 0,
    }


# ============================================================================
# 2. Active jobs
# ============================================================================


async def get_active_jobs() -> list[dict]:
    """Return details for every non-terminal job, joined with instance data."""
    now = datetime.utcnow()
    results: list[dict] = []

    try:
        async with get_session() as session:
            stmt = (
                select(Job, Instance)
                .outerjoin(Instance, Job.instance_id == Instance.instance_id)
                .where(Job.status.notin_(_TERMINAL_STATUSES))
                .order_by(Job.created_at.desc())
            )
            rows = await session.execute(stmt)

            for job, instance in rows.all():
                # Parse metrics
                metrics = job.get_metrics()

                # Parse stages
                stages = job.get_stages()
                total_stages = len(stages) if stages else 0

                # Compute cost so far from elapsed time + hourly rate
                cost_so_far = job.total_cost_usd
                hourly_rate: Optional[float] = None
                gpu_type: Optional[str] = None
                gpu_count: Optional[int] = None
                ssh_host: Optional[str] = None
                ssh_port: Optional[int] = None

                if instance is not None:
                    hourly_rate = instance.hourly_rate
                    gpu_type = instance.gpu_type
                    gpu_count = instance.gpu_count
                    ssh_host = instance.ssh_host
                    ssh_port = instance.ssh_port
                    if job.started_at and hourly_rate:
                        elapsed_hours = (now - job.started_at).total_seconds() / 3600.0
                        cost_so_far = elapsed_hours * hourly_rate

                # Format runtime
                runtime_seconds = 0.0
                if job.started_at:
                    runtime_seconds = (now - job.started_at).total_seconds()
                runtime_display = _format_runtime(runtime_seconds)

                results.append(
                    {
                        "job_id": job.job_id,
                        "name": job.name,
                        "project": job.project,
                        "status": job.status.value if isinstance(job.status, JobStatus) else job.status,
                        "current_stage": job.current_stage,
                        "total_stages": total_stages,
                        "progress_percent": metrics.progress_percent,
                        "steps_per_second": metrics.steps_per_second,
                        "eta_hours": metrics.estimated_hours_remaining,
                        "gpu_type": gpu_type,
                        "gpu_count": gpu_count,
                        "hourly_rate": hourly_rate,
                        "cost_so_far": round(cost_so_far, 4),
                        "runtime_display": runtime_display,
                        "attempt_number": job.attempt_number,
                        "sync_status": job.sync_status,
                        "last_sync_at": job.last_sync_at.isoformat() if job.last_sync_at else None,
                        "ssh_host": ssh_host,
                        "ssh_port": ssh_port,
                        "started_at": job.started_at.isoformat() if job.started_at else None,
                        "instance_id": job.instance_id,
                    }
                )
    except Exception:
        pass  # graceful fallback

    return results


# ============================================================================
# 3. Recent events
# ============================================================================


async def get_recent_events(hours: int = 24) -> list[dict]:
    """Return daemon log events from the last *hours*, enriched with job names."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    results: list[dict] = []

    try:
        log_entries = DaemonService.get_logs(lines=200)
    except Exception:
        log_entries = []

    if not log_entries:
        return results

    # Collect unique job_ids so we can batch-lookup names
    job_ids = {e.get("job_id") for e in log_entries if e.get("job_id")}
    job_name_map: dict[str, str] = {}
    if job_ids:
        try:
            async with get_session() as session:
                stmt = select(Job.job_id, Job.name).where(Job.job_id.in_(list(job_ids)))
                rows = await session.execute(stmt)
                for jid, jname in rows.all():
                    job_name_map[jid] = jname
        except Exception:
            pass

    for entry in log_entries:
        ts_str = entry.get("timestamp")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            continue
        if ts < cutoff:
            continue

        event_type = entry.get("event", "unknown")
        icon, color = _EVENT_STYLE.get(event_type, ("?", "gray"))
        job_id = entry.get("job_id")

        results.append(
            {
                "timestamp": _format_event_time(ts),
                "event_type": event_type,
                "job_id": job_id,
                "job_name": job_name_map.get(job_id, ""),
                "icon": icon,
                "color": color,
                "detail": entry.get("data", {}),
            }
        )

    return results


# ============================================================================
# 4. Cost breakdown
# ============================================================================


async def get_cost_breakdown(days: int = 7) -> dict:
    """Return cost breakdown by project and by GPU type over the last *days*.

    Returns:
        {
            "by_project": [{"project": str, "job_count": int,
                            "total_cost": float, "total_hours": float}, ...],
            "by_gpu":     [{"gpu_type": str, "total_cost": float,
                            "total_hours": float}, ...],
            "total_cost":  float,
            "total_hours": float,
        }
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    by_project: list[dict] = []
    by_gpu: list[dict] = []
    total_cost: float = 0.0
    total_hours: float = 0.0

    try:
        async with get_session() as session:
            # --- Per project ---
            stmt = (
                select(
                    Job.project,
                    func.count().label("job_count"),
                    func.coalesce(func.sum(Job.total_cost_usd), 0.0).label("total_cost"),
                    func.coalesce(func.sum(Job.total_runtime_seconds), 0).label("total_seconds"),
                )
                .where(Job.created_at >= cutoff)
                .group_by(Job.project)
            )
            rows = await session.execute(stmt)
            for project, job_count, cost, seconds in rows.all():
                by_project.append(
                    {
                        "project": project or "(none)",
                        "job_count": int(job_count),
                        "total_cost": round(float(cost), 4),
                        "total_hours": round(int(seconds) / 3600.0, 2),
                    }
                )

            # --- Per GPU type via CostRecord first ---
            stmt_cr = (
                select(
                    CostRecord.gpu_type,
                    func.coalesce(func.sum(CostRecord.total_cost_usd), 0.0).label("total_cost"),
                    func.coalesce(func.sum(CostRecord.total_runtime_seconds), 0).label("total_seconds"),
                )
                .where(CostRecord.created_at >= cutoff)
                .group_by(CostRecord.gpu_type)
            )
            rows_cr = await session.execute(stmt_cr)
            cr_results = rows_cr.all()

            if cr_results:
                for gpu_type, cost, seconds in cr_results:
                    by_gpu.append(
                        {
                            "gpu_type": gpu_type,
                            "total_cost": round(float(cost), 4),
                            "total_hours": round(int(seconds) / 3600.0, 2),
                        }
                    )
            else:
                # Fallback: join jobs -> instances
                stmt_ji = (
                    select(
                        Instance.gpu_type,
                        func.coalesce(func.sum(Job.total_cost_usd), 0.0).label("total_cost"),
                        func.coalesce(func.sum(Job.total_runtime_seconds), 0).label("total_seconds"),
                    )
                    .join(Instance, Job.instance_id == Instance.instance_id)
                    .where(Job.created_at >= cutoff)
                    .group_by(Instance.gpu_type)
                )
                rows_ji = await session.execute(stmt_ji)
                for gpu_type, cost, seconds in rows_ji.all():
                    by_gpu.append(
                        {
                            "gpu_type": gpu_type,
                            "total_cost": round(float(cost), 4),
                            "total_hours": round(int(seconds) / 3600.0, 2),
                        }
                    )

            # --- Totals ---
            stmt_total = select(
                func.coalesce(func.sum(Job.total_cost_usd), 0.0).label("tc"),
                func.coalesce(func.sum(Job.total_runtime_seconds), 0).label("ts"),
            ).where(Job.created_at >= cutoff)
            row = await session.execute(stmt_total)
            tc, ts = row.one()
            total_cost = round(float(tc), 4)
            total_hours = round(int(ts) / 3600.0, 2)

    except Exception:
        pass  # graceful fallback

    return {
        "by_project": by_project,
        "by_gpu": by_gpu,
        "total_cost": total_cost,
        "total_hours": total_hours,
    }


# ============================================================================
# 5. Alerts
# ============================================================================


async def get_alerts() -> list[dict]:
    """Return active alerts for the dashboard."""
    alerts: list[dict] = []
    now = datetime.utcnow()
    day_ago = now - timedelta(hours=24)
    stall_threshold = now - timedelta(minutes=30)

    try:
        # Check daemon health first (no DB needed)
        if not DaemonService.is_running():
            alerts.append(
                {"severity": "red", "message": "Daemon is not running", "job_id": None}
            )
    except Exception:
        alerts.append(
            {"severity": "red", "message": "Unable to check daemon status", "job_id": None}
        )

    try:
        async with get_session() as session:
            # --- Failed jobs in last 24h ---
            stmt = select(Job).where(
                Job.status == JobStatus.FAILED,
                Job.completed_at >= day_ago,
            )
            rows = await session.execute(stmt)
            for (job,) in rows.all():
                alerts.append(
                    {
                        "severity": "red",
                        "message": f"Job '{job.name}' failed: {job.error_message or 'unknown error'}",
                        "job_id": job.job_id,
                    }
                )

            # --- Budget exceeded jobs in last 24h ---
            stmt = select(Job).where(
                Job.status == JobStatus.BUDGET_EXCEEDED,
                Job.completed_at >= day_ago,
            )
            rows = await session.execute(stmt)
            for (job,) in rows.all():
                alerts.append(
                    {
                        "severity": "red",
                        "message": f"Job '{job.name}' exceeded its budget",
                        "job_id": job.job_id,
                    }
                )

            # --- Stalled running jobs (metrics not updated for > 30 min) ---
            stmt = select(Job).where(Job.status == JobStatus.RUNNING)
            rows = await session.execute(stmt)
            for (job,) in rows.all():
                metrics = job.get_metrics()
                if metrics.last_updated and metrics.last_updated < stall_threshold:
                    alerts.append(
                        {
                            "severity": "yellow",
                            "message": f"Job '{job.name}' appears stalled (no metrics update for >30 min)",
                            "job_id": job.job_id,
                        }
                    )

            # --- Budget consumption warnings (per project) ---
            stmt = (
                select(
                    Job.project,
                    func.sum(Job.total_cost_usd).label("spent"),
                )
                .where(Job.status.notin_(_TERMINAL_STATUSES))
                .group_by(Job.project)
            )
            rows = await session.execute(stmt)
            for project, spent in rows.all():
                if project is None:
                    continue
                # Look up budget cap from active jobs in that project
                stmt_budget = select(Job.budget_json).where(
                    Job.project == project,
                    Job.status.notin_(_TERMINAL_STATUSES),
                )
                budget_rows = await session.execute(stmt_budget)
                total_budget = 0.0
                for (budget_json,) in budget_rows.all():
                    try:
                        budget_data = json.loads(budget_json)
                        total_budget += budget_data.get("max_cost_usd", 0.0)
                    except (ValueError, TypeError):
                        pass
                if total_budget > 0 and spent and float(spent) / total_budget > 0.80:
                    pct = round(float(spent) / total_budget * 100, 1)
                    alerts.append(
                        {
                            "severity": "yellow",
                            "message": f"Project '{project}' at {pct}% of budget (${float(spent):.2f} / ${total_budget:.2f})",
                            "job_id": None,
                        }
                    )
    except Exception:
        pass  # graceful fallback

    return alerts


# ============================================================================
# 6. Finished jobs
# ============================================================================


async def get_finished_jobs(hours: int = 24, limit: int = 50) -> list[dict]:
    """Return recently finished jobs (completed, failed, cancelled, budget_exceeded)."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    now = datetime.utcnow()
    results: list[dict] = []

    try:
        async with get_session() as session:
            stmt = (
                select(Job, Instance)
                .outerjoin(Instance, Job.instance_id == Instance.instance_id)
                .where(
                    Job.status.in_(_TERMINAL_STATUSES),
                    Job.completed_at >= cutoff,
                )
                .order_by(Job.completed_at.desc())
                .limit(limit)
            )
            rows = await session.execute(stmt)

            for job, instance in rows.all():
                metrics = job.get_metrics()
                stages = job.get_stages()
                total_stages = len(stages) if stages else 0

                hourly_rate: Optional[float] = None
                gpu_type: Optional[str] = None
                gpu_count: Optional[int] = None
                ssh_host: Optional[str] = None
                ssh_port: Optional[int] = None

                cost_so_far = job.total_cost_usd
                if instance is not None:
                    hourly_rate = instance.hourly_rate
                    gpu_type = instance.gpu_type
                    gpu_count = instance.gpu_count
                    ssh_host = instance.ssh_host
                    ssh_port = instance.ssh_port

                runtime_seconds = job.total_runtime_seconds or 0
                if not runtime_seconds and job.started_at and job.completed_at:
                    runtime_seconds = int(
                        (job.completed_at - job.started_at).total_seconds()
                    )
                runtime_display = _format_runtime(runtime_seconds)

                results.append(
                    {
                        "job_id": job.job_id,
                        "name": job.name,
                        "project": job.project,
                        "status": job.status.value if isinstance(job.status, JobStatus) else job.status,
                        "current_stage": job.current_stage,
                        "total_stages": total_stages,
                        "progress_percent": metrics.progress_percent,
                        "steps_per_second": metrics.steps_per_second,
                        "eta_hours": metrics.estimated_hours_remaining,
                        "gpu_type": gpu_type,
                        "gpu_count": gpu_count,
                        "hourly_rate": hourly_rate,
                        "cost_so_far": round(cost_so_far, 4),
                        "runtime_display": runtime_display,
                        "attempt_number": job.attempt_number,
                        "sync_status": job.sync_status,
                        "last_sync_at": job.last_sync_at.isoformat() if job.last_sync_at else None,
                        "ssh_host": ssh_host,
                        "ssh_port": ssh_port,
                        "started_at": job.started_at.isoformat() if job.started_at else None,
                        "instance_id": job.instance_id,
                        # Finished-job-specific fields
                        "exit_code": job.exit_code,
                        "output_location": job.output_location,
                        "error_message": job.error_message,
                    }
                )
    except Exception:
        pass  # graceful fallback

    return results
