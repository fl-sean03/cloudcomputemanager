"""Checkpoint orchestration for managing save/restore cycles.

The CheckpointOrchestrator coordinates:
1. Periodic checkpoint detection
2. Checkpoint verification
3. Upload to persistent storage
4. Restoration on job recovery
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import structlog

from cloudcomputemanager.checkpoint.detectors import (
    CheckpointDetector,
    CheckpointInfo,
    LAMMPSDetector,
    get_detector,
)
from cloudcomputemanager.core.config import get_settings
from cloudcomputemanager.core.models import (
    Checkpoint,
    CheckpointConfig,
    CheckpointStrategy,
    CheckpointTrigger,
)
from cloudcomputemanager.providers.base import CloudProvider

logger = structlog.get_logger(__name__)


class CheckpointOrchestrator:
    """Manages checkpoint lifecycle for jobs.

    Features:
    - Automatic detection of application checkpoints
    - Periodic sync to persistent storage
    - Verification of checkpoint integrity
    - Restoration on job recovery
    """

    def __init__(
        self,
        provider: CloudProvider,
        detector: Optional[CheckpointDetector] = None,
    ):
        """Initialize the orchestrator.

        Args:
            provider: Cloud provider for instance operations
            detector: Checkpoint detector (defaults to LAMMPS)
        """
        self._provider = provider
        self._detector = detector or LAMMPSDetector()
        self._settings = get_settings()
        self._running_tasks: dict[str, asyncio.Task] = {}

    async def find_checkpoints(
        self,
        instance_id: str,
        checkpoint_path: str = "/workspace",
    ) -> list[CheckpointInfo]:
        """Find checkpoints on an instance.

        Args:
            instance_id: Instance to search
            checkpoint_path: Path to search

        Returns:
            List of found checkpoints
        """

        async def execute_fn(cmd: str) -> tuple[int, str, str]:
            return await self._provider.execute_command(instance_id, cmd)

        return await self._detector.find_checkpoints(execute_fn, checkpoint_path)

    async def get_latest_checkpoint(
        self,
        instance_id: str,
        checkpoint_path: str = "/workspace",
    ) -> Optional[CheckpointInfo]:
        """Get the latest checkpoint from an instance.

        Args:
            instance_id: Instance to search
            checkpoint_path: Path to search

        Returns:
            Latest checkpoint info or None
        """
        checkpoints = await self.find_checkpoints(instance_id, checkpoint_path)
        return checkpoints[0] if checkpoints else None

    async def save_checkpoint(
        self,
        job_id: str,
        instance_id: str,
        config: CheckpointConfig,
        trigger: CheckpointTrigger = CheckpointTrigger.SCHEDULED,
    ) -> Optional[Checkpoint]:
        """Save a checkpoint to persistent storage.

        Args:
            job_id: Job ID
            instance_id: Instance ID
            config: Checkpoint configuration
            trigger: What triggered this checkpoint

        Returns:
            Checkpoint record or None if failed
        """
        logger.info(
            "Saving checkpoint",
            job_id=job_id,
            instance_id=instance_id,
            trigger=trigger,
        )

        start_time = datetime.utcnow()

        # Find latest checkpoint on instance
        checkpoint_info = await self.get_latest_checkpoint(instance_id, config.path)
        if not checkpoint_info:
            logger.warning("No checkpoint found on instance", instance_id=instance_id)
            return None

        # Determine local storage path
        local_dir = self._settings.checkpoint_local_path / job_id
        local_dir.mkdir(parents=True, exist_ok=True)

        # Download checkpoint files
        remote_path = checkpoint_info.path
        local_path = local_dir / Path(remote_path).name

        # If it's a directory pattern, download the whole checkpoint directory
        if config.strategy == CheckpointStrategy.APPLICATION:
            # For LAMMPS, download just the restart file
            success = await self._provider.rsync_download(
                instance_id,
                remote_path,
                str(local_path),
            )
        else:
            # For filesystem strategy, download entire checkpoint directory
            success = await self._provider.rsync_download(
                instance_id,
                config.path + "/",
                str(local_dir) + "/",
                exclude=["*.tmp", "*.temp"],
            )

        if not success:
            logger.error("Failed to download checkpoint", remote=remote_path)
            return None

        duration = (datetime.utcnow() - start_time).seconds

        # Create checkpoint record
        checkpoint = Checkpoint(
            job_id=job_id,
            instance_id=instance_id,
            strategy=config.strategy,
            trigger=trigger,
            location=str(local_path if config.strategy == CheckpointStrategy.APPLICATION else local_dir),
            size_bytes=checkpoint_info.size_bytes,
            file_count=1,  # TODO: Count files for directory checkpoints
            created_at=datetime.utcnow(),
            duration_seconds=duration,
            app_metadata=checkpoint_info.metadata or {},
            verified=False,
        )

        # Verify checkpoint if configured
        if self._settings.checkpoint_verify:
            verified, error = await self.verify_checkpoint(checkpoint)
            checkpoint.verified = verified
            checkpoint.verification_error = error

        logger.info(
            "Checkpoint saved",
            checkpoint_id=checkpoint.checkpoint_id,
            location=checkpoint.location,
            size_bytes=checkpoint.size_bytes,
            verified=checkpoint.verified,
        )

        return checkpoint

    async def verify_checkpoint(
        self,
        checkpoint: Checkpoint,
    ) -> tuple[bool, Optional[str]]:
        """Verify checkpoint integrity.

        Args:
            checkpoint: Checkpoint to verify

        Returns:
            Tuple of (is_valid, error_message)
        """
        path = Path(checkpoint.location)

        # Check file/directory exists
        if not path.exists():
            return False, f"Checkpoint not found: {checkpoint.location}"

        # Check size
        if path.is_file():
            actual_size = path.stat().st_size
            if actual_size == 0:
                return False, "Checkpoint file is empty"
            # Allow some tolerance for size mismatch (rsync might report different)
            if checkpoint.size_bytes > 0 and abs(actual_size - checkpoint.size_bytes) > 1024:
                return False, f"Size mismatch: expected {checkpoint.size_bytes}, got {actual_size}"

        # For LAMMPS, try to verify the restart file format
        if checkpoint.app_metadata.get("type") == "lammps_restart":
            if path.is_file():
                # LAMMPS binary restart files should have a specific header
                # For now, just check it's not empty
                if path.stat().st_size < 100:
                    return False, "LAMMPS restart file too small"

        return True, None

    async def restore_checkpoint(
        self,
        checkpoint: Checkpoint,
        instance_id: str,
        remote_path: str = "/workspace",
    ) -> bool:
        """Restore a checkpoint to an instance.

        Args:
            checkpoint: Checkpoint to restore
            instance_id: Target instance
            remote_path: Destination path on instance

        Returns:
            True if successful
        """
        logger.info(
            "Restoring checkpoint",
            checkpoint_id=checkpoint.checkpoint_id,
            instance_id=instance_id,
            remote_path=remote_path,
        )

        local_path = checkpoint.location
        if not Path(local_path).exists():
            logger.error("Checkpoint not found locally", path=local_path)
            return False

        # Upload checkpoint
        success = await self._provider.rsync_upload(
            instance_id,
            local_path,
            remote_path,
        )

        if success:
            logger.info("Checkpoint restored", instance_id=instance_id)

        return success

    async def start_periodic_checkpoint(
        self,
        job_id: str,
        instance_id: str,
        config: CheckpointConfig,
        on_checkpoint: Optional[Callable[[Checkpoint], None]] = None,
    ) -> None:
        """Start periodic checkpoint monitoring for a job.

        Args:
            job_id: Job ID
            instance_id: Instance ID
            config: Checkpoint configuration
            on_checkpoint: Callback when checkpoint is saved
        """
        if job_id in self._running_tasks:
            logger.warning("Checkpoint task already running", job_id=job_id)
            return

        async def checkpoint_loop():
            interval_seconds = config.interval_minutes * 60
            last_checkpoint_path: Optional[str] = None

            while True:
                try:
                    await asyncio.sleep(interval_seconds)

                    # Check for new checkpoint
                    latest = await self.get_latest_checkpoint(instance_id, config.path)
                    if latest and latest.path != last_checkpoint_path:
                        # New checkpoint detected
                        checkpoint = await self.save_checkpoint(
                            job_id, instance_id, config, CheckpointTrigger.SCHEDULED
                        )

                        if checkpoint:
                            last_checkpoint_path = latest.path
                            if on_checkpoint:
                                on_checkpoint(checkpoint)

                except asyncio.CancelledError:
                    logger.info("Checkpoint loop cancelled", job_id=job_id)
                    break
                except Exception as e:
                    logger.error("Error in checkpoint loop", job_id=job_id, error=str(e))
                    # Continue running after errors

        task = asyncio.create_task(checkpoint_loop())
        self._running_tasks[job_id] = task
        logger.info(
            "Started periodic checkpoint",
            job_id=job_id,
            interval_minutes=config.interval_minutes,
        )

    async def stop_periodic_checkpoint(self, job_id: str) -> None:
        """Stop periodic checkpoint monitoring for a job.

        Args:
            job_id: Job ID
        """
        if job_id in self._running_tasks:
            self._running_tasks[job_id].cancel()
            try:
                await self._running_tasks[job_id]
            except asyncio.CancelledError:
                pass
            del self._running_tasks[job_id]
            logger.info("Stopped periodic checkpoint", job_id=job_id)

    async def trigger_emergency_checkpoint(
        self,
        job_id: str,
        instance_id: str,
        config: CheckpointConfig,
    ) -> Optional[Checkpoint]:
        """Trigger an emergency checkpoint (e.g., on preemption warning).

        This immediately saves the latest checkpoint without waiting for
        the next scheduled interval.

        Args:
            job_id: Job ID
            instance_id: Instance ID
            config: Checkpoint configuration

        Returns:
            Checkpoint record or None
        """
        logger.warning(
            "Triggering emergency checkpoint",
            job_id=job_id,
            instance_id=instance_id,
        )

        # If application supports signal-based checkpoint, send it
        if config.signal:
            # Try to get the main process PID and send signal
            cmd = f"pkill -{config.signal} -f lmp || true"
            await self._provider.execute_command(instance_id, cmd)
            # Wait a moment for checkpoint to be written
            await asyncio.sleep(5)

        # Run pre-checkpoint command if configured
        if config.pre_command:
            await self._provider.execute_command(instance_id, config.pre_command)
            await asyncio.sleep(2)

        return await self.save_checkpoint(
            job_id, instance_id, config, CheckpointTrigger.PREEMPTION
        )

    def list_local_checkpoints(self, job_id: str) -> list[Path]:
        """List all local checkpoints for a job.

        Args:
            job_id: Job ID

        Returns:
            List of checkpoint paths
        """
        checkpoint_dir = self._settings.checkpoint_local_path / job_id
        if not checkpoint_dir.exists():
            return []

        return sorted(checkpoint_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)

    def get_latest_local_checkpoint(self, job_id: str) -> Optional[Path]:
        """Get the latest local checkpoint for a job.

        Args:
            job_id: Job ID

        Returns:
            Path to latest checkpoint or None
        """
        checkpoints = self.list_local_checkpoints(job_id)
        return checkpoints[0] if checkpoints else None
