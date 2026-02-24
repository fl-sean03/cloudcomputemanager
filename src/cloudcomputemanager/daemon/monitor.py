"""Job monitoring for CCM daemon."""

import asyncio
import signal
from datetime import datetime
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import structlog

from sqlmodel import select

from cloudcomputemanager.core.database import init_db, get_session
from cloudcomputemanager.core.config import get_settings
from cloudcomputemanager.core.models import Job, JobStatus
from cloudcomputemanager.providers.vast import VastProvider


logger = structlog.get_logger()


class EventType(Enum):
    """Types of events emitted by the monitor."""
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_PREEMPTED = "job_preempted"
    SYNC_COMPLETED = "sync_completed"
    INSTANCE_TERMINATED = "instance_terminated"
    MONITOR_ERROR = "monitor_error"


@dataclass
class MonitorEvent:
    """Event emitted by the job monitor."""
    event_type: EventType
    job_id: str
    instance_id: Optional[str] = None
    data: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MonitorConfig:
    """Configuration for the job monitor."""
    poll_interval: int = 30  # seconds between status checks
    sync_on_complete: bool = True  # sync results when job completes
    terminate_on_complete: bool = True  # terminate instance when job completes
    preemption_recovery: bool = True  # attempt recovery on preemption
    max_recovery_attempts: int = 3  # max times to recover a preempted job
    health_check_interval: int = 60  # seconds between instance health checks


class JobMonitor:
    """
    Monitors running jobs and handles lifecycle events.

    The monitor runs as an async task and:
    - Polls running jobs for completion
    - Detects instance preemption/failure
    - Triggers result sync on completion
    - Terminates instances after completion
    - Emits events for external consumers (agents, webhooks, etc.)
    """

    def __init__(
        self,
        config: Optional[MonitorConfig] = None,
        provider: Optional[VastProvider] = None,
    ):
        self.config = config or MonitorConfig()
        self.provider = provider or VastProvider()
        self.settings = get_settings()

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._event_handlers: list[Callable[[MonitorEvent], Any]] = []
        self._monitored_jobs: set[str] = set()

    def on_event(self, handler: Callable[[MonitorEvent], Any]) -> None:
        """Register an event handler."""
        self._event_handlers.append(handler)

    def _emit_event(self, event: MonitorEvent) -> None:
        """Emit an event to all registered handlers."""
        for handler in self._event_handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                logger.error("Event handler error", error=str(e), event_type=event.event_type.value)

    async def check_job_completion(self, instance_id: str) -> tuple[bool, Optional[int]]:
        """Check if a job has completed by looking for exit code file."""
        try:
            exit_code, stdout, stderr = await self.provider.execute_command(
                instance_id,
                "cat /workspace/.ccm_exit_code 2>/dev/null || echo 'running'"
            )

            if exit_code == 0 and stdout.strip() != "running":
                try:
                    job_exit_code = int(stdout.strip())
                    return True, job_exit_code
                except ValueError:
                    return False, None
            return False, None
        except Exception as e:
            logger.debug("Job completion check failed", instance_id=instance_id, error=str(e))
            return False, None

    async def check_instance_health(self, instance_id: str) -> tuple[bool, Optional[str]]:
        """
        Check if an instance is healthy and running.

        Returns:
            (healthy, reason) - healthy is True if instance is OK
        """
        try:
            instance = await self.provider.get_instance(instance_id)

            if instance is None:
                return False, "instance_not_found"

            status = instance.status.value.lower()

            if status in ["terminated", "error", "stopped"]:
                return False, f"instance_{status}"

            if status != "running":
                return False, f"instance_{status}"

            # Try SSH health check
            exit_code, stdout, stderr = await self.provider.execute_command(
                instance_id, "echo ok", timeout=10
            )

            if exit_code != 0:
                return False, "ssh_failed"

            return True, None

        except Exception as e:
            return False, f"check_failed:{str(e)[:50]}"

    async def sync_job_results(self, job: Job) -> bool:
        """Sync job results to local storage."""
        if not job.instance_id:
            return False

        sync_config = job.sync_config or {}
        if not sync_config:
            return True  # Nothing to sync

        try:
            sync_dir = self.settings.sync_local_path / job.job_id
            sync_dir.mkdir(parents=True, exist_ok=True)

            success = await self.provider.rsync_download(
                job.instance_id,
                sync_config.get("source", "/workspace") + "/",
                str(sync_dir) + "/",
                exclude=sync_config.get("exclude_patterns", []),
            )

            if success:
                self._emit_event(MonitorEvent(
                    event_type=EventType.SYNC_COMPLETED,
                    job_id=job.job_id,
                    instance_id=job.instance_id,
                    data={"sync_dir": str(sync_dir)},
                ))

            return success
        except Exception as e:
            logger.error("Sync failed", job_id=job.job_id, error=str(e))
            return False

    async def terminate_job_instance(self, job: Job) -> bool:
        """Terminate the instance associated with a job."""
        if not job.instance_id:
            return True

        try:
            await self.provider.terminate_instance(job.instance_id)

            self._emit_event(MonitorEvent(
                event_type=EventType.INSTANCE_TERMINATED,
                job_id=job.job_id,
                instance_id=job.instance_id,
            ))

            return True
        except Exception as e:
            logger.error("Terminate failed", job_id=job.job_id, error=str(e))
            return False

    async def handle_job_completion(
        self,
        job: Job,
        exit_code: int,
    ) -> None:
        """Handle a completed job: update status, sync, terminate."""
        logger.info("Job completed", job_id=job.job_id, exit_code=exit_code)

        # Update job status in database
        async with get_session() as session:
            stmt = select(Job).where(Job.job_id == job.job_id)
            result = await session.execute(stmt)
            db_job = result.scalar_one_or_none()

            if db_job:
                db_job.exit_code = exit_code
                db_job.status = JobStatus.COMPLETED if exit_code == 0 else JobStatus.FAILED
                db_job.completed_at = datetime.utcnow()
                session.add(db_job)

        # Emit event
        event_type = EventType.JOB_COMPLETED if exit_code == 0 else EventType.JOB_FAILED
        self._emit_event(MonitorEvent(
            event_type=event_type,
            job_id=job.job_id,
            instance_id=job.instance_id,
            data={"exit_code": exit_code},
        ))

        # Sync results
        if self.config.sync_on_complete:
            await self.sync_job_results(job)

        # Terminate instance
        if self.config.terminate_on_complete:
            await self.terminate_job_instance(job)

        # Remove from monitored set
        self._monitored_jobs.discard(job.job_id)

    async def handle_preemption(self, job: Job, reason: str) -> None:
        """Handle an instance preemption or failure."""
        logger.warning("Instance preempted/failed", job_id=job.job_id, reason=reason)

        self._emit_event(MonitorEvent(
            event_type=EventType.JOB_PREEMPTED,
            job_id=job.job_id,
            instance_id=job.instance_id,
            data={"reason": reason},
        ))

        # Update job status
        async with get_session() as session:
            stmt = select(Job).where(Job.job_id == job.job_id)
            result = await session.execute(stmt)
            db_job = result.scalar_one_or_none()

            if db_job:
                if self.config.preemption_recovery and db_job.attempt_number < self.config.max_recovery_attempts:
                    db_job.status = JobStatus.RECOVERING
                    db_job.error_message = f"Preempted: {reason}"
                    # Recovery will be handled by recovery module
                else:
                    db_job.status = JobStatus.FAILED
                    db_job.error_message = f"Preempted (max attempts): {reason}"
                    db_job.completed_at = datetime.utcnow()

                session.add(db_job)

        self._monitored_jobs.discard(job.job_id)

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Monitor loop started", poll_interval=self.config.poll_interval)

        # Import recovery manager
        from cloudcomputemanager.core.recovery import RecoveryManager
        recovery_manager = RecoveryManager(
            provider=self.provider,
            max_attempts=self.config.max_recovery_attempts,
        )

        last_recovery_check = 0

        while self._running:
            try:
                # Get running jobs
                async with get_session() as session:
                    stmt = select(Job).where(
                        Job.status == JobStatus.RUNNING
                    )
                    result = await session.execute(stmt)
                    jobs = result.scalars().all()

                logger.debug("Checking jobs", count=len(jobs))

                for job in jobs:
                    if not job.instance_id:
                        continue

                    # Track monitored jobs
                    self._monitored_jobs.add(job.job_id)

                    # Check completion first
                    completed, exit_code = await self.check_job_completion(job.instance_id)
                    if completed:
                        await self.handle_job_completion(job, exit_code)
                        continue

                    # Check instance health
                    healthy, reason = await self.check_instance_health(job.instance_id)
                    if not healthy:
                        await self.handle_preemption(job, reason)
                        continue

                # Handle recovery jobs periodically
                import time
                if self.config.preemption_recovery and time.time() - last_recovery_check > 60:
                    last_recovery_check = time.time()

                    async with get_session() as session:
                        stmt = select(Job).where(Job.status == JobStatus.RECOVERING)
                        result = await session.execute(stmt)
                        recovering_jobs = result.scalars().all()

                    if recovering_jobs:
                        logger.info("Processing recovery jobs", count=len(recovering_jobs))
                        for job in recovering_jobs:
                            try:
                                result = await recovery_manager.recover_job(job)
                                if result.success:
                                    logger.info("Job recovered", job_id=job.job_id, instance=result.new_instance_id)
                                else:
                                    logger.warning("Recovery failed", job_id=job.job_id, error=result.error)
                            except Exception as e:
                                logger.error("Recovery error", job_id=job.job_id, error=str(e))

            except Exception as e:
                logger.error("Monitor loop error", error=str(e))
                self._emit_event(MonitorEvent(
                    event_type=EventType.MONITOR_ERROR,
                    job_id="",
                    data={"error": str(e)},
                ))

            await asyncio.sleep(self.config.poll_interval)

        logger.info("Monitor loop stopped")

    async def start(self) -> None:
        """Start the monitor."""
        if self._running:
            logger.warning("Monitor already running")
            return

        await init_db()

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Job monitor started")

    async def stop(self) -> None:
        """Stop the monitor."""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("Job monitor stopped")

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    @property
    def monitored_job_count(self) -> int:
        """Get number of jobs being monitored."""
        return len(self._monitored_jobs)
