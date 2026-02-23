"""Sync engine for continuous data synchronization.

The SyncEngine handles:
1. Periodic rsync of results from instances
2. Pattern-based file filtering
3. Progress tracking and reporting
4. Error handling and retry logic
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import structlog

from cloudcomputemanager.core.config import get_settings
from cloudcomputemanager.core.models import SyncConfig, SyncRecord, SyncStatus
from cloudcomputemanager.providers.base import CloudProvider

logger = structlog.get_logger(__name__)


class SyncEngine:
    """Manages continuous data synchronization for jobs.

    Features:
    - Periodic rsync from instances to local storage
    - Pattern-based include/exclude
    - Progress tracking
    - Automatic retry on failures
    """

    def __init__(self, provider: CloudProvider):
        """Initialize the sync engine.

        Args:
            provider: Cloud provider for instance operations
        """
        self._provider = provider
        self._settings = get_settings()
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._sync_records: dict[str, SyncRecord] = {}

    async def sync_once(
        self,
        job_id: str,
        instance_id: str,
        config: SyncConfig,
    ) -> SyncRecord:
        """Perform a single sync operation.

        Args:
            job_id: Job ID
            instance_id: Instance ID
            config: Sync configuration

        Returns:
            SyncRecord with results
        """
        logger.info(
            "Starting sync",
            job_id=job_id,
            instance_id=instance_id,
            source=config.source,
        )

        start_time = datetime.utcnow()

        # Determine local destination
        if config.destination.startswith("s3://"):
            # S3 sync - would use boto3/rclone
            local_dest = self._settings.sync_local_path / job_id
            # TODO: Implement S3 upload after local sync
        else:
            # Local path
            local_dest = Path(config.destination).expanduser()
            if not local_dest.is_absolute():
                local_dest = self._settings.sync_local_path / job_id / config.destination

        local_dest.mkdir(parents=True, exist_ok=True)

        # Create sync record
        record = SyncRecord(
            job_id=job_id,
            instance_id=instance_id,
            source=config.source,
            destination=str(local_dest),
            status=SyncStatus.SYNCING,
            started_at=start_time,
        )

        # Build exclude patterns
        exclude_patterns = config.exclude_patterns.copy()

        # Execute rsync
        try:
            success = await self._provider.rsync_download(
                instance_id,
                config.source + "/",
                str(local_dest) + "/",
                exclude=exclude_patterns,
            )

            if success:
                record.status = SyncStatus.COMPLETED

                # Count synced files
                file_count = sum(1 for _ in local_dest.rglob("*") if _.is_file())
                total_size = sum(f.stat().st_size for f in local_dest.rglob("*") if f.is_file())

                record.files_synced = file_count
                record.bytes_synced = total_size
            else:
                record.status = SyncStatus.FAILED
                record.error_message = "Rsync failed"

        except Exception as e:
            record.status = SyncStatus.FAILED
            record.error_message = str(e)
            logger.error("Sync failed", job_id=job_id, error=str(e))

        record.completed_at = datetime.utcnow()
        record.duration_seconds = (record.completed_at - start_time).seconds

        logger.info(
            "Sync completed",
            job_id=job_id,
            status=record.status,
            files=record.files_synced,
            bytes=record.bytes_synced,
            duration=record.duration_seconds,
        )

        return record

    async def start_periodic_sync(
        self,
        job_id: str,
        instance_id: str,
        config: SyncConfig,
        on_sync: Optional[Callable[[SyncRecord], None]] = None,
    ) -> None:
        """Start periodic sync for a job.

        Args:
            job_id: Job ID
            instance_id: Instance ID
            config: Sync configuration
            on_sync: Callback after each sync
        """
        if not config.enabled:
            logger.debug("Sync disabled for job", job_id=job_id)
            return

        if job_id in self._running_tasks:
            logger.warning("Sync task already running", job_id=job_id)
            return

        async def sync_loop():
            interval_seconds = config.interval_minutes * 60

            while True:
                try:
                    record = await self.sync_once(job_id, instance_id, config)
                    self._sync_records[job_id] = record

                    if on_sync:
                        on_sync(record)

                    await asyncio.sleep(interval_seconds)

                except asyncio.CancelledError:
                    logger.info("Sync loop cancelled", job_id=job_id)
                    # Perform final sync on cancellation
                    try:
                        await self.sync_once(job_id, instance_id, config)
                    except Exception:
                        pass
                    break
                except Exception as e:
                    logger.error("Error in sync loop", job_id=job_id, error=str(e))
                    # Continue running after errors with a shorter retry interval
                    await asyncio.sleep(60)

        task = asyncio.create_task(sync_loop())
        self._running_tasks[job_id] = task
        logger.info(
            "Started periodic sync",
            job_id=job_id,
            interval_minutes=config.interval_minutes,
        )

    async def stop_periodic_sync(
        self,
        job_id: str,
        final_sync: bool = True,
    ) -> Optional[SyncRecord]:
        """Stop periodic sync for a job.

        Args:
            job_id: Job ID
            final_sync: Whether to perform a final sync

        Returns:
            Final sync record if performed
        """
        if job_id not in self._running_tasks:
            return None

        self._running_tasks[job_id].cancel()
        try:
            await self._running_tasks[job_id]
        except asyncio.CancelledError:
            pass

        del self._running_tasks[job_id]
        logger.info("Stopped periodic sync", job_id=job_id)

        return self._sync_records.get(job_id)

    def get_sync_status(self, job_id: str) -> Optional[SyncRecord]:
        """Get the latest sync status for a job.

        Args:
            job_id: Job ID

        Returns:
            Latest sync record or None
        """
        return self._sync_records.get(job_id)

    def is_syncing(self, job_id: str) -> bool:
        """Check if sync is running for a job.

        Args:
            job_id: Job ID

        Returns:
            True if sync is running
        """
        return job_id in self._running_tasks

    def list_synced_files(self, job_id: str) -> list[Path]:
        """List all synced files for a job.

        Args:
            job_id: Job ID

        Returns:
            List of file paths
        """
        sync_dir = self._settings.sync_local_path / job_id
        if not sync_dir.exists():
            return []

        return sorted(
            [f for f in sync_dir.rglob("*") if f.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def get_sync_directory(self, job_id: str) -> Path:
        """Get the sync directory for a job.

        Args:
            job_id: Job ID

        Returns:
            Path to sync directory
        """
        return self._settings.sync_local_path / job_id
