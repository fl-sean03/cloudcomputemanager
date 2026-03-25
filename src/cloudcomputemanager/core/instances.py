"""Instance record management — syncs Vast.ai state to local database.

This module ensures the Instance table stays in sync with reality,
regardless of how instances were created (ccm submit, agent SDK,
direct vastai CLI, or the Vast.ai web console).
"""

from datetime import datetime
from typing import Optional

import structlog
from sqlmodel import select

from cloudcomputemanager.core.database import get_session
from cloudcomputemanager.core.models import Instance, InstanceStatus, RentalType, Job
from cloudcomputemanager.providers.base import ProviderInstance, ProviderStatus

logger = structlog.get_logger(__name__)


# ============================================================================
# Label helpers
# ============================================================================

LABEL_PREFIX = "ccm"
LABEL_SEP = "|"


def build_instance_label(job_id: str, project: str = "", name: str = "") -> str:
    """Build a structured label for Vast.ai instance identification.

    Format: ccm|{job_id}|{project}|{name}
    Max 120 chars to stay well under Vast.ai's 1024 limit.
    """
    project = (project or "none").replace(LABEL_SEP, "-")[:50]
    name = (name or "unnamed").replace(LABEL_SEP, "-")[:50]
    return f"{LABEL_PREFIX}{LABEL_SEP}{job_id}{LABEL_SEP}{project}{LABEL_SEP}{name}"


def parse_instance_label(label: str) -> Optional[dict]:
    """Parse a CCM label into components.

    Returns None if not a CCM-managed label.
    Returns {"job_id": str, "project": str|None, "name": str|None} on success.
    """
    if not label or not label.startswith(f"{LABEL_PREFIX}{LABEL_SEP}"):
        return None
    parts = label.split(LABEL_SEP)
    if len(parts) < 2 or not parts[1]:
        return None
    return {
        "job_id": parts[1],
        "project": parts[2] if len(parts) > 2 else None,
        "name": parts[3] if len(parts) > 3 else None,
    }


# Map provider status to our InstanceStatus enum
_STATUS_MAP = {
    ProviderStatus.PENDING: InstanceStatus.CREATING,
    ProviderStatus.STARTING: InstanceStatus.STARTING,
    ProviderStatus.RUNNING: InstanceStatus.RUNNING,
    ProviderStatus.STOPPING: InstanceStatus.STOPPING,
    ProviderStatus.STOPPED: InstanceStatus.STOPPED,
    ProviderStatus.TERMINATED: InstanceStatus.TERMINATED,
    ProviderStatus.ERROR: InstanceStatus.ERROR,
}


async def upsert_instance(provider_inst: ProviderInstance, job_id: Optional[str] = None) -> Instance:
    """Create or update an Instance record from a ProviderInstance.

    If an Instance with this instance_id already exists, update its status
    and networking info (SSH may change on restart). If not, create it.

    Args:
        provider_inst: Instance data from the cloud provider
        job_id: Optional job_id to associate with this instance

    Returns:
        The created/updated Instance record
    """
    async with get_session() as session:
        stmt = select(Instance).where(Instance.instance_id == provider_inst.instance_id)
        result = await session.execute(stmt)
        db_inst = result.scalar_one_or_none()

        new_status = _STATUS_MAP.get(provider_inst.status, InstanceStatus.CREATING)

        if db_inst:
            # Update existing record
            db_inst.status = new_status
            db_inst.ssh_host = provider_inst.ssh_host
            db_inst.ssh_port = provider_inst.ssh_port
            db_inst.hourly_rate = provider_inst.hourly_rate
            db_inst.last_health_check = datetime.utcnow()
            db_inst.health_status = new_status.value
            if provider_inst.jupyter_url:
                db_inst.jupyter_url = provider_inst.jupyter_url
            if job_id and not db_inst.job_id:
                db_inst.job_id = job_id
            # Mark terminated if status changed
            if new_status in (InstanceStatus.TERMINATED, InstanceStatus.STOPPED, InstanceStatus.ERROR):
                if not db_inst.terminated_at:
                    db_inst.terminated_at = datetime.utcnow()
            if new_status == InstanceStatus.RUNNING and not db_inst.started_at:
                db_inst.started_at = datetime.utcnow()
            session.add(db_inst)
        else:
            # Create new record
            db_inst = Instance(
                instance_id=provider_inst.instance_id,
                provider=provider_inst.provider,
                status=new_status,
                gpu_type=provider_inst.gpu_type,
                gpu_count=provider_inst.gpu_count,
                gpu_memory_gb=provider_inst.gpu_memory_gb,
                cpu_cores=provider_inst.cpu_cores,
                memory_gb=provider_inst.memory_gb,
                disk_gb=provider_inst.disk_gb,
                ssh_host=provider_inst.ssh_host,
                ssh_port=provider_inst.ssh_port,
                ssh_user=provider_inst.ssh_user,
                hourly_rate=provider_inst.hourly_rate,
                rental_type=RentalType.INTERRUPTIBLE if provider_inst.interruptible else RentalType.ON_DEMAND,
                jupyter_url=provider_inst.jupyter_url,
                job_id=job_id,
                health_status=new_status.value,
                last_health_check=datetime.utcnow(),
            )
            if new_status == InstanceStatus.RUNNING:
                db_inst.started_at = datetime.utcnow()
            session.add(db_inst)

        return db_inst


async def sync_all_instances(provider) -> dict:
    """Sync ALL instances from the provider API into the local database.

    Calls provider.list_instances() to get every instance on the account,
    then upserts each into the Instance table. Also matches unassociated
    instances to Job records by instance_id.

    Returns summary: {"synced": int, "new": int, "updated": int, "terminated": int, "unmanaged": int}
    """
    try:
        all_instances = await provider.list_instances()
    except Exception as e:
        logger.warning("Failed to list instances from provider", error=str(e))
        return {"synced": 0, "new": 0, "updated": 0, "terminated": 0, "unmanaged": 0}

    stats = {"synced": len(all_instances), "new": 0, "updated": 0, "terminated": 0, "unmanaged": 0}

    for pi in all_instances:
        # Check if this instance is already in DB
        async with get_session() as session:
            stmt = select(Instance).where(Instance.instance_id == pi.instance_id)
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()

        # Find matching job — try label first (most reliable), then instance_id
        job_id = None
        label_data = parse_instance_label(getattr(pi, 'label', None))
        if label_data:
            job_id = label_data["job_id"]
        else:
            # Fallback: match by instance_id in Job table
            async with get_session() as session:
                stmt = select(Job.job_id).where(Job.instance_id == pi.instance_id)
                result = await session.execute(stmt)
                row = result.first()
                if row:
                    job_id = row[0]

        # Detect unlabeled/unmanaged running instances
        is_running = pi.status in (ProviderStatus.RUNNING, ProviderStatus.STARTING)
        if is_running and not label_data and not job_id:
            stats["unmanaged"] += 1
            logger.warning(
                "UNMANAGED INSTANCE DETECTED: running instance with no CCM label or job. "
                "An agent may have called vastai directly instead of using CCM.",
                instance_id=pi.instance_id,
                gpu_type=pi.gpu_type,
                hourly_rate=pi.hourly_rate,
            )

        await upsert_instance(pi, job_id=job_id)

        if existing:
            stats["updated"] += 1
            if pi.status in (ProviderStatus.TERMINATED, ProviderStatus.STOPPED):
                stats["terminated"] += 1
        else:
            stats["new"] += 1

    # Also check for instances in our DB that are no longer on the provider
    # (fully destroyed, removed from Vast.ai's listing)
    async with get_session() as session:
        stmt = select(Instance).where(
            Instance.status.not_in([InstanceStatus.TERMINATED, InstanceStatus.STOPPED, InstanceStatus.ERROR])
        )
        result = await session.execute(stmt)
        db_instances = result.scalars().all()

    live_ids = {pi.instance_id for pi in all_instances}
    for db_inst in db_instances:
        if db_inst.instance_id not in live_ids:
            # Instance is gone from provider — mark as terminated
            async with get_session() as session:
                stmt = select(Instance).where(Instance.instance_id == db_inst.instance_id)
                result = await session.execute(stmt)
                inst = result.scalar_one_or_none()
                if inst:
                    inst.status = InstanceStatus.TERMINATED
                    inst.terminated_at = datetime.utcnow()
                    inst.health_status = "terminated"
                    session.add(inst)
                    stats["terminated"] += 1
            logger.info("Instance no longer on provider, marked terminated",
                        instance_id=db_inst.instance_id)

    # Auto-terminate unmanaged instances that have been running without a label
    # and have no job association. This prevents rogue agents from burning money.
    if stats["unmanaged"] > 0:
        for pi in all_instances:
            is_running = pi.status in (ProviderStatus.RUNNING, ProviderStatus.STARTING)
            has_label = parse_instance_label(getattr(pi, 'label', None)) is not None

            if is_running and not has_label:
                # Check if this instance has a job_id in DB
                has_job = False
                async with get_session() as session:
                    stmt = select(Instance.job_id).where(Instance.instance_id == pi.instance_id)
                    result = await session.execute(stmt)
                    row = result.first()
                    if row and row[0]:
                        has_job = True

                if not has_job:
                    logger.warning(
                        "Auto-terminating unmanaged instance (no label, no job)",
                        instance_id=pi.instance_id,
                        gpu_type=pi.gpu_type,
                        hourly_rate=pi.hourly_rate,
                    )
                    try:
                        await provider.terminate_instance(pi.instance_id)
                        # Update DB
                        async with get_session() as session:
                            stmt = select(Instance).where(Instance.instance_id == pi.instance_id)
                            result = await session.execute(stmt)
                            inst = result.scalar_one_or_none()
                            if inst:
                                inst.status = InstanceStatus.TERMINATED
                                inst.terminated_at = datetime.utcnow()
                                session.add(inst)
                    except Exception as e:
                        logger.error("Failed to auto-terminate unmanaged instance",
                                     instance_id=pi.instance_id, error=str(e))

    if stats["new"] > 0 or stats["terminated"] > 0 or stats["unmanaged"] > 0:
        logger.info("Instance sync complete", **stats)

    return stats


async def terminate_instance_safe(provider, instance_id: str, max_retries: int = 3) -> bool:
    """Terminate an instance with retries and verification.

    Retries up to max_retries times, then verifies the instance is
    actually gone from the provider. Updates Instance record in DB.

    Returns True if instance is confirmed terminated, False otherwise.
    """
    import asyncio

    for attempt in range(max_retries):
        try:
            await provider.terminate_instance(instance_id)
            # Wait briefly, then verify
            await asyncio.sleep(3)
            inst = await provider.get_instance(instance_id)
            if inst is None or inst.status in (ProviderStatus.TERMINATED, ProviderStatus.STOPPED):
                # Confirmed terminated — update DB
                async with get_session() as session:
                    stmt = select(Instance).where(Instance.instance_id == instance_id)
                    result = await session.execute(stmt)
                    db_inst = result.scalar_one_or_none()
                    if db_inst:
                        db_inst.status = InstanceStatus.TERMINATED
                        db_inst.terminated_at = datetime.utcnow()
                        db_inst.health_status = "terminated"
                        session.add(db_inst)
                logger.info("Instance terminated and verified", instance_id=instance_id)
                return True
        except Exception as e:
            logger.warning("Termination attempt failed",
                           instance_id=instance_id, attempt=attempt + 1, error=str(e))
            await asyncio.sleep(2)

    logger.error("Failed to terminate instance after retries", instance_id=instance_id)
    return False
