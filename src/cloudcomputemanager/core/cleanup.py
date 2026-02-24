"""Cleanup utilities for stale jobs and instances."""

from typing import Optional
from datetime import datetime, timedelta

from sqlmodel import select

from cloudcomputemanager.core.database import get_session
from cloudcomputemanager.core.models import Job, JobStatus
from cloudcomputemanager.providers.vast import VastProvider


async def find_stale_jobs(
    provider: Optional[VastProvider] = None,
) -> list[tuple[Job, str]]:
    """
    Find jobs with stale state (pointing to dead/non-existent instances).

    Returns list of (job, reason) tuples.
    """
    if provider is None:
        provider = VastProvider()

    stale_jobs = []

    async with get_session() as session:
        # Find jobs that are supposedly running
        stmt = select(Job).where(
            Job.status.in_([
                JobStatus.RUNNING,
                JobStatus.PROVISIONING,
                JobStatus.RECOVERING,
                JobStatus.CHECKPOINTING,
            ])
        )
        result = await session.execute(stmt)
        jobs = result.scalars().all()

        for job in jobs:
            if not job.instance_id:
                stale_jobs.append((job, "no_instance_id"))
                continue

            # Check if instance still exists
            try:
                instance = await provider.get_instance(job.instance_id)
                if instance is None:
                    stale_jobs.append((job, "instance_not_found"))
                elif instance.status.value in ["terminated", "error", "stopped"]:
                    stale_jobs.append((job, f"instance_{instance.status.value}"))
            except Exception as e:
                stale_jobs.append((job, f"check_failed:{str(e)[:50]}"))

    return stale_jobs


async def cleanup_stale_jobs(
    provider: Optional[VastProvider] = None,
    dry_run: bool = True,
    mark_as: JobStatus = JobStatus.FAILED,
) -> list[tuple[str, str, str]]:
    """
    Clean up stale jobs by marking them as failed/cancelled.

    Args:
        provider: VastProvider instance (creates new if None)
        dry_run: If True, only report what would be cleaned (no changes)
        mark_as: Status to set for stale jobs (default: FAILED)

    Returns:
        List of (job_id, old_status, reason) tuples for cleaned jobs
    """
    if provider is None:
        provider = VastProvider()

    stale_jobs = await find_stale_jobs(provider)
    cleaned = []

    if dry_run:
        return [(job.job_id, job.status.value, reason) for job, reason in stale_jobs]

    async with get_session() as session:
        for job, reason in stale_jobs:
            # Refresh job from database
            stmt = select(Job).where(Job.job_id == job.job_id)
            result = await session.execute(stmt)
            db_job = result.scalar_one_or_none()

            if db_job:
                old_status = db_job.status.value
                db_job.status = mark_as
                db_job.error_message = f"Cleaned up: {reason}"
                db_job.completed_at = datetime.utcnow()
                session.add(db_job)
                cleaned.append((db_job.job_id, old_status, reason))

        await session.commit()

    return cleaned


async def cleanup_orphan_instances(
    provider: Optional[VastProvider] = None,
    dry_run: bool = True,
    label_prefix: Optional[str] = None,
) -> list[tuple[str, str]]:
    """
    Find and terminate instances not associated with any job.

    Args:
        provider: VastProvider instance
        dry_run: If True, only report what would be terminated
        label_prefix: Only consider instances with this label prefix

    Returns:
        List of (instance_id, label) tuples for terminated instances
    """
    if provider is None:
        provider = VastProvider()

    # Get all running instances from provider
    instances = await provider.list_instances()

    # Get all job instance IDs from database
    async with get_session() as session:
        stmt = select(Job.instance_id).where(Job.instance_id.isnot(None))
        result = await session.execute(stmt)
        job_instance_ids = {row[0] for row in result.all()}

    orphans = []
    for instance in instances:
        # Skip if associated with a job
        if instance.instance_id in job_instance_ids:
            continue

        # Skip if doesn't match label prefix
        label = getattr(instance, 'label', '') or ''
        if label_prefix and not label.startswith(label_prefix):
            continue

        orphans.append((instance.instance_id, label))

    if dry_run:
        return orphans

    # Terminate orphan instances
    terminated = []
    for instance_id, label in orphans:
        try:
            await provider.terminate_instance(instance_id)
            terminated.append((instance_id, label))
        except Exception:
            pass  # Ignore termination failures

    return terminated


async def get_cleanup_summary(provider: Optional[VastProvider] = None) -> dict:
    """Get summary of what needs cleanup."""
    if provider is None:
        provider = VastProvider()

    stale_jobs = await find_stale_jobs(provider)
    orphan_instances = await cleanup_orphan_instances(provider, dry_run=True)

    # Count by reason
    reasons = {}
    for _, reason in stale_jobs:
        reasons[reason] = reasons.get(reason, 0) + 1

    return {
        "stale_jobs": len(stale_jobs),
        "stale_job_reasons": reasons,
        "orphan_instances": len(orphan_instances),
        "orphan_instance_ids": [i[0] for i in orphan_instances],
    }
