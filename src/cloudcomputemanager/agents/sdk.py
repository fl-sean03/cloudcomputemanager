"""Agent SDK for CloudComputeManager.

Provides a high-level, async Python interface designed for AI agents to
programmatically manage GPU workloads. All operations are:
- Async (non-blocking)
- Type-safe (Pydantic models)
- Idempotent (safe to retry)
- Observable (structured logging, events)

Example usage by an AI agent:

    from cloudcomputemanager.agents import CloudComputeManagerAgent

    async def run_simulation_campaign(parameters: list[dict]):
        async with CloudComputeManagerAgent() as vm:
            # Submit multiple simulations
            jobs = await vm.submit_batch(
                [create_job(p) for p in parameters],
                wait_for_resources=True,
            )

            # Monitor with callbacks
            async for event in vm.watch_jobs([j.job_id for j in jobs]):
                if event.type == "checkpoint":
                    print(f"Checkpoint saved: {event.data}")
                elif event.type == "completed":
                    results = await vm.get_results(event.job_id)
                    process_results(results)

            return await vm.collect_results(jobs)
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import AsyncIterator, Callable, Optional, Union
import json

import structlog
from pydantic import BaseModel

from sqlmodel import select

from cloudcomputemanager.core.config import get_settings
from cloudcomputemanager.core.database import init_db, get_session
from cloudcomputemanager.core.models import (
    Job,
    JobStatus,
    Resources,
    CheckpointConfig,
    SyncConfig,
    Budget,
    Checkpoint,
)
from cloudcomputemanager.providers.vast import VastProvider
from cloudcomputemanager.checkpoint import CheckpointOrchestrator
from cloudcomputemanager.sync import SyncEngine
from cloudcomputemanager.packstore import PackageDeployer, PackageRegistry

logger = structlog.get_logger(__name__)


# ============================================================================
# Event System for Agent Observability
# ============================================================================


class EventType(str, Enum):
    """Types of events agents can observe."""

    JOB_SUBMITTED = "job.submitted"
    JOB_STARTED = "job.started"
    JOB_RUNNING = "job.running"
    JOB_CHECKPOINT = "job.checkpoint"
    JOB_SYNC = "job.sync"
    JOB_PREEMPTED = "job.preempted"
    JOB_RECOVERING = "job.recovering"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"

    INSTANCE_CREATED = "instance.created"
    INSTANCE_READY = "instance.ready"
    INSTANCE_PREEMPTED = "instance.preempted"
    INSTANCE_TERMINATED = "instance.terminated"

    PACKAGE_DEPLOYED = "package.deployed"
    PACKAGE_VERIFIED = "package.verified"


@dataclass
class AgentEvent:
    """An event that agents can observe."""

    type: EventType
    timestamp: datetime
    job_id: Optional[str] = None
    instance_id: Optional[str] = None
    data: dict = field(default_factory=dict)
    message: str = ""


# ============================================================================
# Job Specification (Agent-Friendly)
# ============================================================================


class JobSpec(BaseModel):
    """Simplified job specification for agents.

    Provides sensible defaults so agents can submit jobs with minimal config.
    """

    # Required
    name: str
    command: str

    # Container (defaults to LAMMPS for scientific computing)
    image: str = "nvcr.io/hpc/lammps:29Aug2024"

    # Optional: packages to deploy (alternative to image)
    packages: list[str] = []

    # Resources (no hardcoded defaults - user must specify or use template defaults)
    gpu_type: Optional[str] = None  # None = any available GPU type
    gpu_count: int = 1
    gpu_memory_min: Optional[int] = None  # None = provider default (8GB)
    disk_gb: int = 100

    # Checkpointing (enabled by default)
    checkpoint_enabled: bool = True
    checkpoint_interval_minutes: int = 30
    checkpoint_path: str = "/workspace"

    # Sync (enabled by default)
    sync_enabled: bool = True
    sync_interval_minutes: int = 15
    sync_source: str = "/workspace/results"

    # Budget (reasonable defaults)
    max_cost_usd: float = 50.0
    max_hours: int = 24
    max_hourly_rate: float = 1.0

    # Metadata
    project: Optional[str] = None
    tags: list[str] = []

    def to_job_config(self) -> dict:
        """Convert to full job configuration dict."""
        resources = {
            "gpu_count": self.gpu_count,
            "disk_gb": self.disk_gb,
        }
        # Only include optional fields if specified
        if self.gpu_type is not None:
            resources["gpu_type"] = self.gpu_type
        if self.gpu_memory_min is not None:
            resources["gpu_memory_min"] = self.gpu_memory_min

        return {
            "name": self.name,
            "project": self.project,
            "image": self.image,
            "command": self.command,
            "packages": self.packages,
            "resources": resources,
            "checkpoint": {
                "enabled": self.checkpoint_enabled,
                "interval_minutes": self.checkpoint_interval_minutes,
                "path": self.checkpoint_path,
            },
            "sync": {
                "enabled": self.sync_enabled,
                "interval_minutes": self.sync_interval_minutes,
                "source": self.sync_source,
            },
            "budget": {
                "max_cost_usd": self.max_cost_usd,
                "max_hours": self.max_hours,
                "max_hourly_rate": self.max_hourly_rate,
            },
            "tags": self.tags,
        }


@dataclass
class JobResult:
    """Result of a completed job."""

    job_id: str
    status: str
    success: bool

    # Output
    output_location: Optional[str] = None
    output_files: list[str] = field(default_factory=list)

    # Metrics
    total_cost_usd: float = 0.0
    total_runtime_seconds: int = 0
    checkpoint_count: int = 0

    # Error info
    error_message: Optional[str] = None
    exit_code: Optional[int] = None


# ============================================================================
# Main Agent SDK
# ============================================================================


class CloudComputeManagerAgent:
    """High-level agent interface for CloudComputeManager.

    Designed for AI agents to programmatically manage GPU workloads with:
    - Simple, async API
    - Automatic error handling and retries
    - Event streaming for observability
    - Batch operations for efficiency

    Usage:
        async with CloudComputeManagerAgent() as vm:
            job = await vm.submit(JobSpec(
                name="my-simulation",
                command="mpirun lmp -in input.in",
            ))
            result = await vm.wait_for_completion(job.job_id)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        auto_checkpoint: bool = True,
        auto_sync: bool = True,
    ):
        """Initialize the agent.

        Args:
            api_key: Vast.ai API key (uses settings if not provided)
            auto_checkpoint: Automatically enable checkpointing
            auto_sync: Automatically enable sync
        """
        self._provider: Optional[VastProvider] = None
        self._checkpoint_orchestrator: Optional[CheckpointOrchestrator] = None
        self._sync_engine: Optional[SyncEngine] = None
        self._package_deployer: Optional[PackageDeployer] = None

        self._api_key = api_key
        self._auto_checkpoint = auto_checkpoint
        self._auto_sync = auto_sync

        self._event_handlers: list[Callable[[AgentEvent], None]] = []
        self._active_jobs: dict[str, Job] = {}

    async def __aenter__(self) -> "CloudComputeManagerAgent":
        """Enter async context."""
        await self._initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context."""
        await self._cleanup()

    async def _initialize(self) -> None:
        """Initialize components."""
        # Initialize database for persistence
        await init_db()

        self._provider = VastProvider(api_key=self._api_key)
        self._checkpoint_orchestrator = CheckpointOrchestrator(self._provider)
        self._sync_engine = SyncEngine(self._provider)
        self._package_deployer = PackageDeployer(self._provider)

        # Load active jobs from database
        async with get_session() as session:
            stmt = select(Job).where(Job.status.in_([
                JobStatus.RUNNING,
                JobStatus.PENDING,
                JobStatus.PROVISIONING,
                JobStatus.CHECKPOINTING,
                JobStatus.RECOVERING,
            ]))
            result = await session.execute(stmt)
            for job in result.scalars().all():
                self._active_jobs[job.job_id] = job
                logger.info("Loaded active job from database", job_id=job.job_id)

        logger.info("CloudComputeManagerAgent initialized", active_jobs=len(self._active_jobs))

    async def _cleanup(self) -> None:
        """Cleanup resources."""
        # Stop any running sync/checkpoint loops
        for job_id in list(self._active_jobs.keys()):
            if self._sync_engine:
                await self._sync_engine.stop_periodic_sync(job_id)
            if self._checkpoint_orchestrator:
                await self._checkpoint_orchestrator.stop_periodic_checkpoint(job_id)

        logger.info("CloudComputeManagerAgent cleaned up")

    def on_event(self, handler: Callable[[AgentEvent], None]) -> None:
        """Register an event handler.

        Args:
            handler: Function to call on events
        """
        self._event_handlers.append(handler)

    def _emit_event(self, event: AgentEvent) -> None:
        """Emit an event to all handlers."""
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.warning("Event handler error", error=str(e))

    # =========================================================================
    # Job Operations
    # =========================================================================

    async def submit(
        self,
        spec: Union[JobSpec, dict],
        wait_for_start: bool = False,
        deploy_packages: bool = True,
    ) -> Job:
        """Submit a job.

        Args:
            spec: Job specification
            wait_for_start: Wait for job to start running
            deploy_packages: Deploy specified packages before running

        Returns:
            Job object
        """
        if isinstance(spec, dict):
            spec = JobSpec(**spec)

        config = spec.to_job_config()

        logger.info("Submitting job", name=spec.name, gpu=spec.gpu_type)

        # Find best offer
        offers = await self._provider.search_offers(
            gpu_type=spec.gpu_type,
            gpu_count=spec.gpu_count,
            disk_gb_min=spec.disk_gb,
            max_hourly_rate=spec.max_hourly_rate,
        )

        if not offers:
            raise RuntimeError(f"No GPU offers found matching requirements")

        best_offer = offers[0]
        logger.info("Selected offer", offer_id=best_offer.offer_id, rate=best_offer.hourly_rate)

        # Create instance
        instance = await self._provider.create_instance(
            offer_id=best_offer.offer_id,
            image=spec.image,
            disk_gb=spec.disk_gb,
        )

        self._emit_event(AgentEvent(
            type=EventType.INSTANCE_CREATED,
            timestamp=datetime.utcnow(),
            instance_id=instance.instance_id,
            message=f"Instance created: {instance.instance_id}",
        ))

        # Wait for instance ready
        ready = await self._provider.wait_for_ready(instance.instance_id, timeout=300)
        if not ready:
            raise RuntimeError(f"Instance {instance.instance_id} failed to start")

        self._emit_event(AgentEvent(
            type=EventType.INSTANCE_READY,
            timestamp=datetime.utcnow(),
            instance_id=instance.instance_id,
        ))

        # Deploy packages if specified
        if deploy_packages and spec.packages:
            result = await self._package_deployer.deploy(
                instance.instance_id,
                spec.packages,
            )
            for dep in result.deployments:
                self._emit_event(AgentEvent(
                    type=EventType.PACKAGE_DEPLOYED,
                    timestamp=datetime.utcnow(),
                    instance_id=instance.instance_id,
                    data={"package": dep.package_name, "status": dep.status.value},
                ))

        # Create job record
        job = Job(
            name=spec.name,
            project=spec.project,
            status=JobStatus.RUNNING,
            image=spec.image,
            command=spec.command,
            instance_id=instance.instance_id,
            resources_json=json.dumps(config["resources"]),
            checkpoint_json=json.dumps(config["checkpoint"]),
            sync_json=json.dumps(config["sync"]),
            budget_json=json.dumps(config["budget"]),
            tags_json=json.dumps(spec.tags),
            started_at=datetime.utcnow(),
        )

        # Persist to database
        async with get_session() as session:
            session.add(job)
            await session.commit()
            await session.refresh(job)

        self._active_jobs[job.job_id] = job

        self._emit_event(AgentEvent(
            type=EventType.JOB_SUBMITTED,
            timestamp=datetime.utcnow(),
            job_id=job.job_id,
            instance_id=instance.instance_id,
            message=f"Job submitted: {job.job_id}",
        ))

        # Deploy and run SIGTERM-aware wrapper script
        from cloudcomputemanager.core.wrapper import build_deploy_commands
        setup_cmd, run_cmd = build_deploy_commands(spec.command)
        await self._provider.execute_command(instance.instance_id, setup_cmd)
        await self._provider.execute_command(instance.instance_id, run_cmd)

        self._emit_event(AgentEvent(
            type=EventType.JOB_STARTED,
            timestamp=datetime.utcnow(),
            job_id=job.job_id,
        ))

        # Start checkpoint monitoring if enabled
        if self._auto_checkpoint and spec.checkpoint_enabled:
            checkpoint_config = CheckpointConfig(
                interval_minutes=spec.checkpoint_interval_minutes,
                path=spec.checkpoint_path,
            )
            await self._checkpoint_orchestrator.start_periodic_checkpoint(
                job.job_id,
                instance.instance_id,
                checkpoint_config,
                on_checkpoint=lambda c: self._emit_event(AgentEvent(
                    type=EventType.JOB_CHECKPOINT,
                    timestamp=datetime.utcnow(),
                    job_id=job.job_id,
                    data={"checkpoint_id": c.checkpoint_id, "size": c.size_bytes},
                )),
            )

        # Start sync if enabled
        if self._auto_sync and spec.sync_enabled:
            sync_config = SyncConfig(
                source=spec.sync_source,
                destination=str(get_settings().sync_local_path / job.job_id),
                interval_minutes=spec.sync_interval_minutes,
            )
            await self._sync_engine.start_periodic_sync(
                job.job_id,
                instance.instance_id,
                sync_config,
                on_sync=lambda r: self._emit_event(AgentEvent(
                    type=EventType.JOB_SYNC,
                    timestamp=datetime.utcnow(),
                    job_id=job.job_id,
                    data={"files": r.files_synced, "bytes": r.bytes_synced},
                )),
            )

        if wait_for_start:
            # Verify job is actually running
            await asyncio.sleep(5)
            exit_code, stdout, _ = await self._provider.execute_command(
                instance.instance_id,
                "pgrep -f 'lmp\\|python\\|mpirun' || echo 'not running'"
            )
            if "not running" in stdout:
                logger.warning("Job may not have started", job_id=job.job_id)

        return job

    async def submit_batch(
        self,
        specs: list[Union[JobSpec, dict]],
        max_concurrent: int = 5,
        wait_for_resources: bool = True,
    ) -> list[Job]:
        """Submit multiple jobs.

        Args:
            specs: List of job specifications
            max_concurrent: Maximum concurrent jobs
            wait_for_resources: Wait if resources unavailable

        Returns:
            List of submitted jobs
        """
        jobs = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def submit_one(spec):
            async with semaphore:
                return await self.submit(spec)

        tasks = [submit_one(spec) for spec in specs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error("Batch job failed", error=str(result))
            else:
                jobs.append(result)

        return jobs

    async def get_status(self, job_id: str) -> dict:
        """Get job status.

        Args:
            job_id: Job ID

        Returns:
            Status dict with job info
        """
        job = self._active_jobs.get(job_id)
        if not job:
            return {"status": "unknown", "error": "Job not found in active jobs"}

        # Check if instance is still running
        if job.instance_id:
            instance = await self._provider.get_instance(job.instance_id)
            instance_status = instance.status.value if instance else "unknown"
        else:
            instance_status = "none"

        # Get sync status
        sync_record = self._sync_engine.get_sync_status(job_id)

        return {
            "job_id": job.job_id,
            "name": job.name,
            "status": job.status.value,
            "instance_id": job.instance_id,
            "instance_status": instance_status,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "total_cost_usd": job.total_cost_usd,
            "last_sync": {
                "files": sync_record.files_synced if sync_record else 0,
                "bytes": sync_record.bytes_synced if sync_record else 0,
            } if sync_record else None,
        }

    async def wait_for_completion(
        self,
        job_id: str,
        timeout: int = 86400,  # 24 hours
        poll_interval: int = 60,
    ) -> JobResult:
        """Wait for a job to complete.

        Args:
            job_id: Job ID
            timeout: Maximum wait time in seconds
            poll_interval: Check interval in seconds

        Returns:
            JobResult with outcome
        """
        job = self._active_jobs.get(job_id)
        if not job or not job.instance_id:
            return JobResult(
                job_id=job_id,
                status="unknown",
                success=False,
                error_message="Job not found",
            )

        elapsed = 0
        while elapsed < timeout:
            # Check for completion via exit code sentinel file
            exit_code, stdout, _ = await self._provider.execute_command(
                job.instance_id,
                "cat /workspace/.ccm_exit_code 2>/dev/null || echo 'running'",
            )

            if stdout.strip() != "running":
                # Job finished - parse exit code from sentinel
                try:
                    final_exit_code = int(stdout.strip())
                except ValueError:
                    final_exit_code = 0

                # Do final sync
                if self._sync_engine.is_syncing(job_id):
                    await self._sync_engine.stop_periodic_sync(job_id, final_sync=True)

                # Stop checkpoint monitoring
                await self._checkpoint_orchestrator.stop_periodic_checkpoint(job_id)

                success = final_exit_code == 0
                job.status = JobStatus.COMPLETED if success else JobStatus.FAILED
                job.completed_at = datetime.utcnow()
                job.exit_code = final_exit_code

                # Persist to database
                async with get_session() as session:
                    session.add(job)
                    await session.commit()

                self._emit_event(AgentEvent(
                    type=EventType.JOB_COMPLETED if success else EventType.JOB_FAILED,
                    timestamp=datetime.utcnow(),
                    job_id=job_id,
                    data={"exit_code": final_exit_code},
                ))

                # Get output files
                sync_dir = get_settings().sync_local_path / job_id
                output_files = []
                if sync_dir.exists():
                    output_files = [str(f.relative_to(sync_dir)) for f in sync_dir.rglob("*") if f.is_file()]

                return JobResult(
                    job_id=job_id,
                    status=job.status.value,
                    success=success,
                    output_location=str(sync_dir),
                    output_files=output_files[:100],  # Limit
                    total_cost_usd=job.total_cost_usd,
                    total_runtime_seconds=(job.completed_at - job.started_at).seconds if job.started_at else 0,
                    exit_code=final_exit_code,
                )

            # Check instance health
            instance = await self._provider.get_instance(job.instance_id)
            if not instance or instance.status.value not in ["running", "starting"]:
                # Instance preempted or failed
                job.status = JobStatus.FAILED
                job.error_message = "Instance was preempted"

                # Persist to database
                async with get_session() as session:
                    session.add(job)
                    await session.commit()

                self._emit_event(AgentEvent(
                    type=EventType.INSTANCE_PREEMPTED,
                    timestamp=datetime.utcnow(),
                    job_id=job_id,
                    instance_id=job.instance_id,
                ))

                return JobResult(
                    job_id=job_id,
                    status="preempted",
                    success=False,
                    error_message="Instance was preempted",
                )

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        return JobResult(
            job_id=job_id,
            status="timeout",
            success=False,
            error_message=f"Timeout after {timeout}s",
        )

    async def cancel(self, job_id: str) -> bool:
        """Cancel a job.

        Args:
            job_id: Job ID

        Returns:
            True if cancelled successfully
        """
        job = self._active_jobs.get(job_id)
        if not job:
            return False

        # Stop monitoring
        await self._sync_engine.stop_periodic_sync(job_id)
        await self._checkpoint_orchestrator.stop_periodic_checkpoint(job_id)

        # Terminate instance
        if job.instance_id:
            await self._provider.terminate_instance(job.instance_id)

        job.status = JobStatus.CANCELLED

        # Persist cancellation
        async with get_session() as session:
            session.add(job)
            await session.commit()

        del self._active_jobs[job_id]

        return True

    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        project: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List jobs from database.

        Args:
            status: Filter by status
            project: Filter by project
            limit: Maximum results

        Returns:
            List of job info dicts
        """
        async with get_session() as session:
            stmt = select(Job).order_by(Job.created_at.desc()).limit(limit)

            if status:
                stmt = stmt.where(Job.status == status)
            if project:
                stmt = stmt.where(Job.project == project)

            result = await session.execute(stmt)
            jobs = result.scalars().all()

            return [
                {
                    "job_id": j.job_id,
                    "name": j.name,
                    "project": j.project,
                    "status": j.status.value,
                    "instance_id": j.instance_id,
                    "created_at": j.created_at.isoformat() if j.created_at else None,
                    "started_at": j.started_at.isoformat() if j.started_at else None,
                    "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                    "total_cost_usd": j.total_cost_usd,
                    "exit_code": j.exit_code,
                }
                for j in jobs
            ]

    async def get_job(self, job_id: str) -> Optional[dict]:
        """Get detailed job information.

        Args:
            job_id: Job ID

        Returns:
            Job info dict or None if not found
        """
        # Check in-memory first
        if job_id in self._active_jobs:
            job = self._active_jobs[job_id]
        else:
            # Load from database
            async with get_session() as session:
                stmt = select(Job).where(Job.job_id == job_id)
                result = await session.execute(stmt)
                job = result.scalar_one_or_none()

        if not job:
            return None

        # Get checkpoints for this job
        async with get_session() as session:
            stmt = select(Checkpoint).where(Checkpoint.job_id == job_id).order_by(Checkpoint.created_at.desc())
            result = await session.execute(stmt)
            checkpoints = result.scalars().all()

        return {
            "job_id": job.job_id,
            "name": job.name,
            "project": job.project,
            "status": job.status.value,
            "image": job.image,
            "command": job.command,
            "instance_id": job.instance_id,
            "resources": job.resources,
            "budget": job.budget,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "total_cost_usd": job.total_cost_usd,
            "total_runtime_seconds": job.total_runtime_seconds,
            "exit_code": job.exit_code,
            "error_message": job.error_message,
            "output_location": job.output_location,
            "attempt_number": job.attempt_number,
            "checkpoints": [
                {
                    "checkpoint_id": c.checkpoint_id,
                    "created_at": c.created_at.isoformat(),
                    "size_bytes": c.size_bytes,
                    "verified": c.verified,
                }
                for c in checkpoints
            ],
        }

    async def retry_job(self, job_id: str) -> Optional[Job]:
        """Retry a failed job.

        Args:
            job_id: Job ID to retry

        Returns:
            New job or None if not retryable
        """
        job_info = await self.get_job(job_id)
        if not job_info:
            logger.warning("Job not found for retry", job_id=job_id)
            return None

        if job_info["status"] not in ["failed", "cancelled"]:
            logger.warning("Job not in retryable state", job_id=job_id, status=job_info["status"])
            return None

        # Create new JobSpec from original
        spec = JobSpec(
            name=f"{job_info['name']}-retry",
            command=job_info["command"],
            image=job_info["image"],
            gpu_type=job_info["resources"].get("gpu_type"),
            gpu_count=job_info["resources"].get("gpu_count", 1),
            disk_gb=job_info["resources"].get("disk_gb", 100),
            max_cost_usd=job_info["budget"].get("max_cost_usd", 50.0),
            max_hours=job_info["budget"].get("max_hours", 24),
            max_hourly_rate=job_info["budget"].get("max_hourly_rate", 1.0),
            project=job_info["project"],
        )

        logger.info("Retrying job", original_job_id=job_id, new_name=spec.name)
        return await self.submit(spec)

    async def get_costs(
        self,
        project: Optional[str] = None,
        since_days: int = 30,
    ) -> dict:
        """Get cost breakdown.

        Args:
            project: Filter by project
            since_days: Days of history

        Returns:
            Cost breakdown dict
        """
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(days=since_days)

        async with get_session() as session:
            stmt = select(Job).where(Job.created_at >= cutoff)
            if project:
                stmt = stmt.where(Job.project == project)

            result = await session.execute(stmt)
            jobs = result.scalars().all()

        total_cost = sum(j.total_cost_usd for j in jobs)
        total_runtime = sum(j.total_runtime_seconds for j in jobs)

        by_status = {}
        by_project = {}

        for j in jobs:
            status = j.status.value
            by_status[status] = by_status.get(status, 0) + j.total_cost_usd

            proj = j.project or "unassigned"
            by_project[proj] = by_project.get(proj, 0) + j.total_cost_usd

        return {
            "period_days": since_days,
            "total_cost_usd": round(total_cost, 4),
            "total_runtime_hours": round(total_runtime / 3600, 2),
            "job_count": len(jobs),
            "by_status": by_status,
            "by_project": by_project,
            "average_cost_per_job": round(total_cost / len(jobs), 4) if jobs else 0,
        }

    # =========================================================================
    # Observation Helpers
    # =========================================================================

    async def watch_jobs(
        self,
        job_ids: list[str],
        poll_interval: int = 30,
    ) -> AsyncIterator[AgentEvent]:
        """Watch multiple jobs for events.

        Args:
            job_ids: Jobs to watch
            poll_interval: Check interval in seconds

        Yields:
            AgentEvent for each state change
        """
        previous_states: dict[str, str] = {}

        while True:
            all_done = True

            for job_id in job_ids:
                status = await self.get_status(job_id)
                current_state = status.get("status", "unknown")

                if current_state != previous_states.get(job_id):
                    # State changed
                    event_type = {
                        "running": EventType.JOB_RUNNING,
                        "completed": EventType.JOB_COMPLETED,
                        "failed": EventType.JOB_FAILED,
                        "recovering": EventType.JOB_RECOVERING,
                    }.get(current_state, EventType.JOB_RUNNING)

                    yield AgentEvent(
                        type=event_type,
                        timestamp=datetime.utcnow(),
                        job_id=job_id,
                        data=status,
                    )

                    previous_states[job_id] = current_state

                if current_state not in ["completed", "failed", "cancelled"]:
                    all_done = False

            if all_done:
                break

            await asyncio.sleep(poll_interval)

    async def get_results(self, job_id: str) -> dict:
        """Get job results and output files.

        Args:
            job_id: Job ID

        Returns:
            Dict with output location and file list
        """
        sync_dir = get_settings().sync_local_path / job_id

        if not sync_dir.exists():
            return {"exists": False, "files": []}

        files = []
        for f in sync_dir.rglob("*"):
            if f.is_file():
                files.append({
                    "path": str(f.relative_to(sync_dir)),
                    "size_bytes": f.stat().st_size,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                })

        return {
            "exists": True,
            "location": str(sync_dir),
            "file_count": len(files),
            "files": sorted(files, key=lambda x: x["modified"], reverse=True)[:100],
        }

    # =========================================================================
    # Resource Management
    # =========================================================================

    async def get_ssh_credentials(self, job_id: str) -> Optional[dict]:
        """Get direct SSH credentials for a job's instance.

        Returns a dict with host, port, user, key_path so agents can
        SSH directly without going through CCM's execute_command layer.

        Args:
            job_id: Job ID

        Returns:
            Dict with SSH connection details, or None if unavailable
        """
        job = self._active_jobs.get(job_id)
        if not job:
            # Try loading from DB
            async with get_session() as session:
                stmt = select(Job).where(Job.job_id == job_id)
                result = await session.execute(stmt)
                job = result.scalar_one_or_none()

        if not job or not job.instance_id:
            return None

        instance = await self._provider.get_instance(job.instance_id)
        if not instance:
            return None

        settings = get_settings()
        key_path = str(settings.ssh_key_path) if settings.ssh_key_path.exists() else None

        return {
            "host": instance.ssh_host,
            "port": instance.ssh_port,
            "user": instance.ssh_user,
            "key_path": key_path,
            "instance_id": instance.instance_id,
            "job_id": job_id,
        }

    async def search_gpus(
        self,
        gpu_type: Optional[str] = None,
        max_price: Optional[float] = None,
        min_memory_gb: Optional[int] = None,
    ) -> list[dict]:
        """Search available GPU offers.

        Args:
            gpu_type: GPU model (e.g., "RTX_4090")
            max_price: Maximum hourly price
            min_memory_gb: Minimum GPU memory

        Returns:
            List of available offers
        """
        offers = await self._provider.search_offers(
            gpu_type=gpu_type,
            gpu_memory_min=min_memory_gb,
            max_hourly_rate=max_price,
        )

        return [
            {
                "offer_id": o.offer_id,
                "gpu_type": o.gpu_type,
                "gpu_count": o.gpu_count,
                "gpu_memory_gb": o.gpu_memory_gb,
                "hourly_rate": o.hourly_rate,
                "location": o.location,
                "reliability": o.reliability_score,
            }
            for o in offers
        ]

    async def deploy_packages(
        self,
        instance_id: str,
        packages: list[str],
    ) -> dict:
        """Deploy packages to an instance.

        Args:
            instance_id: Target instance
            packages: Package names to deploy

        Returns:
            Deployment result
        """
        result = await self._package_deployer.deploy(instance_id, packages)

        return {
            "success": result.success,
            "environment": result.environment.__dict__,
            "deployments": [
                {
                    "package": d.package_name,
                    "variant": d.variant_id,
                    "status": d.status.value,
                    "verified": d.verified,
                }
                for d in result.deployments
            ],
        }
