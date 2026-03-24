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

    Returns summary: {"synced": int, "new": int, "updated": int, "terminated": int}
    """
    try:
        all_instances = await provider.list_instances()
    except Exception as e:
        logger.warning("Failed to list instances from provider", error=str(e))
        return {"synced": 0, "new": 0, "updated": 0, "terminated": 0}

    stats = {"synced": len(all_instances), "new": 0, "updated": 0, "terminated": 0}

    for pi in all_instances:
        # Check if this instance is already in DB
        async with get_session() as session:
            stmt = select(Instance).where(Instance.instance_id == pi.instance_id)
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()

        # Find matching job by instance_id
        job_id = None
        async with get_session() as session:
            stmt = select(Job.job_id).where(Job.instance_id == pi.instance_id)
            result = await session.execute(stmt)
            row = result.first()
            if row:
                job_id = row[0]

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

    if stats["new"] > 0 or stats["terminated"] > 0:
        logger.info("Instance sync complete", **stats)

    return stats
