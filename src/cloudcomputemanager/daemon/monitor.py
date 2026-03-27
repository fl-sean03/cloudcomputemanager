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
        self._last_recovery_check: float = 0

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
                "cat /workspace/.ccm_exit_code 2>/dev/null || echo 'running'",
                timeout=10,  # Short timeout — don't block monitor loop
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

    async def check_instance_health(
        self, instance_id: str, job: Optional[Job] = None,
    ) -> tuple[bool, Optional[str]]:
        """Multi-signal health check for an instance.

        Checks:
        1. Instance status via provider API
        2. SSH connectivity
        3. Process liveness (if process_pattern configured on job)
        4. Workspace existence
        5. Disk space (warns if <1GB free)

        Returns:
            (healthy, reason) — healthy is True if instance is OK
        """
        try:
            # Signal 1: Instance status from provider
            instance = await self.provider.get_instance(instance_id)

            if instance is None:
                return False, "instance_not_found"

            status = instance.status.value.lower()

            if status in ["terminated", "error", "stopped"]:
                return False, f"instance_{status}"

            if status != "running":
                return False, f"instance_{status}"

            # Signal 2-5: Combined SSH probe (single SSH call for efficiency)
            health_cmd_parts = [
                # SSH connectivity + workspace existence
                "[ -d /workspace ] && echo 'WS_OK' || echo 'WS_MISSING'",
                # Exit code sentinel check
                "cat /workspace/.ccm_exit_code 2>/dev/null || echo 'no_exit'",
                # Disk space check
                "df -BG /workspace 2>/dev/null | tail -1 | awk '{print $4}' || echo 'unknown'",
            ]

            # Add process check if job has a process pattern
            process_pattern = None
            if job:
                job_config = job.resources or {}
                process_pattern = job_config.get("process_pattern")
            if process_pattern:
                health_cmd_parts.append(
                    f"pgrep -f {process_pattern!r} > /dev/null && echo 'PROC_OK' || echo 'PROC_STOPPED'"
                )

            health_cmd = "; ".join(health_cmd_parts)
            exit_code, stdout, stderr = await self.provider.execute_command(
                instance_id, health_cmd, timeout=15
            )

            if exit_code == 255:
                # SSH connection failed entirely
                return False, "ssh_failed"

            lines = stdout.strip().split("\n")

            # Parse workspace status
            ws_status = lines[0].strip() if lines else "unknown"
            if ws_status == "WS_MISSING":
                return False, "workspace_missing"

            # Parse disk space
            if len(lines) >= 3:
                disk_free = lines[2].strip()
                if disk_free not in ("unknown", "") and disk_free.endswith("G"):
                    try:
                        free_gb = int(disk_free.replace("G", ""))
                        if free_gb < 1:
                            logger.warning("Instance disk nearly full", instance_id=instance_id, free_gb=free_gb)
                    except ValueError:
                        pass

            # Parse process status
            if process_pattern and len(lines) >= 4:
                proc_status = lines[3].strip()
                if proc_status == "PROC_STOPPED":
                    # Process not running but no exit code — might be a problem
                    exit_status = lines[1].strip() if len(lines) >= 2 else "no_exit"
                    if exit_status == "no_exit":
                        return False, "process_stopped_no_exit_code"

            return True, None

        except Exception as e:
            return False, f"check_failed:{str(e)[:50]}"

    async def check_job_budget(self, job: Job) -> tuple[bool, Optional[str]]:
        """Check if job has exceeded its budget.

        Returns:
            (within_budget, reason) — within_budget is True if OK
        """
        budget = job.budget or {}
        if not budget:
            return True, None

        # Calculate elapsed time
        if not job.started_at:
            return True, None

        elapsed_hours = (datetime.utcnow() - job.started_at).total_seconds() / 3600
        hourly_rate = 0  # Job doesn't store hourly_rate; cost tracking is separate
        cost_so_far = hourly_rate * elapsed_hours

        # Check max hours
        max_hours = budget.get("max_hours")
        if max_hours and elapsed_hours >= max_hours:
            return False, f"max_hours_exceeded:{elapsed_hours:.1f}h/{max_hours}h"

        # Check max cost
        max_cost = budget.get("max_cost_usd")
        if max_cost and cost_so_far >= max_cost:
            return False, f"max_cost_exceeded:${cost_so_far:.2f}/${max_cost}"

        # Warn at 80% threshold
        if max_cost and cost_so_far >= max_cost * 0.8:
            logger.warning("Job approaching budget limit", job_id=job.job_id, cost=cost_so_far, max=max_cost)

        return True, None

    async def sync_job_results(self, job: Job) -> bool:
        """Sync job results to local storage and update sync tracking."""
        if not job.instance_id:
            return False

        sync_config = job.sync_config or {}
        if not sync_config:
            return True  # Nothing to sync

        # Update sync status to syncing
        async with get_session() as session:
            stmt = select(Job).where(Job.job_id == job.job_id)
            result = await session.execute(stmt)
            db_job = result.scalar_one_or_none()
            if db_job:
                db_job.sync_status = "syncing"
                session.add(db_job)
                await session.commit()

        try:
            sync_dir = self.settings.sync_local_path / job.job_id
            sync_dir.mkdir(parents=True, exist_ok=True)

            success = await self.provider.rsync_download(
                job.instance_id,
                sync_config.get("source", "/workspace") + "/",
                str(sync_dir) + "/",
                exclude=sync_config.get("exclude_patterns", []),
            )

            # Update sync tracking in DB
            async with get_session() as session:
                stmt = select(Job).where(Job.job_id == job.job_id)
                result = await session.execute(stmt)
                db_job = result.scalar_one_or_none()
                if db_job:
                    if success:
                        db_job.sync_status = "synced"
                        db_job.last_sync_at = datetime.utcnow()
                        # Count synced files/bytes
                        if sync_dir.exists():
                            files = list(sync_dir.rglob("*"))
                            db_job.synced_files = sum(1 for f in files if f.is_file())
                            db_job.synced_bytes = sum(f.stat().st_size for f in files if f.is_file())
                    else:
                        db_job.sync_status = "sync_failed"
                    session.add(db_job)
                    await session.commit()

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
            # Mark sync as failed
            async with get_session() as session:
                stmt = select(Job).where(Job.job_id == job.job_id)
                result = await session.execute(stmt)
                db_job = result.scalar_one_or_none()
                if db_job:
                    db_job.sync_status = "sync_failed"
                    session.add(db_job)
                    await session.commit()
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
        """Handle a completed job: update status, sync, terminate.

        Exit code 143 (SIGTERM) indicates preemption — route to recovery instead.
        """
        from cloudcomputemanager.core.wrapper import PREEMPTION_EXIT_CODE
        import json

        # Exit code 143 = SIGTERM = preemption. Route to recovery, not failure.
        if exit_code == PREEMPTION_EXIT_CODE:
            logger.warning("Job preempted (SIGTERM, exit 143)", job_id=job.job_id)
            await self.handle_preemption(job, "sigterm_preemption")
            return

        # Check if this exit code is configured as recoverable
        # (e.g., SIGSEGV=139 on bad GPU instances, OOM-kill=137)
        # Fixes: https://github.com/fl-sean03/cloudcomputemanager/issues/8
        if exit_code != 0:
            retry_config = json.loads(job.retry_json) if job.retry_json else {}
            recoverable_codes = retry_config.get("recover_on_exit_codes", [])
            if exit_code in recoverable_codes:
                logger.warning(
                    "Job failed with recoverable exit code, routing to recovery",
                    job_id=job.job_id,
                    exit_code=exit_code,
                    recoverable_codes=recoverable_codes,
                )
                await self.handle_preemption(job, f"recoverable_exit_{exit_code}")
                return

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
                await session.commit()

        # Emit event
        event_type = EventType.JOB_COMPLETED if exit_code == 0 else EventType.JOB_FAILED
        self._emit_event(MonitorEvent(
            event_type=event_type,
            job_id=job.job_id,
            instance_id=job.instance_id,
            data={"exit_code": exit_code},
        ))

        # Run notification hooks
        notification_event = "on_complete" if exit_code == 0 else "on_failure"
        await self.run_notification(job, notification_event)

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

        # Blacklist the offer that produced this failure (for recoverable exits)
        if reason.startswith("recoverable_exit_"):
            try:
                from cloudcomputemanager.core.recovery import blacklist_offer
                # Look up the offer_id from the instance
                instance = await self.provider.get_instance(job.instance_id)
                if instance and hasattr(instance, 'offer_id') and instance.offer_id:
                    blacklist_offer(instance.offer_id, reason, job.project or "")
            except Exception as e:
                logger.debug("Could not blacklist offer", error=str(e))

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
                # Use per-job retry config if available, fall back to daemon defaults
                import json as _json
                job_retry = _json.loads(db_job.retry_json) if db_job.retry_json else {}
                max_attempts = job_retry.get("max_attempts", self.config.max_recovery_attempts)

                if self.config.preemption_recovery and db_job.attempt_number < max_attempts:
                    db_job.status = JobStatus.RECOVERING
                    db_job.error_message = f"Preempted: {reason}"
                    logger.info("Job set to RECOVERING",
                               job_id=job.job_id, attempt=db_job.attempt_number, max=max_attempts)
                else:
                    db_job.status = JobStatus.FAILED
                    db_job.error_message = f"Preempted (max attempts {max_attempts}): {reason}"
                    db_job.completed_at = datetime.utcnow()
                    logger.info("Job FAILED (max attempts reached)",
                               job_id=job.job_id, attempt=db_job.attempt_number, max=max_attempts)

                session.add(db_job)
                await session.commit()

        self._monitored_jobs.discard(job.job_id)

    async def update_job_progress(self, job: Job) -> None:
        """Update job progress metrics using pluggable progress parsers.

        Supports three modes (configured via job's progress config):
        - regex_parse: Tail a file, extract a value via regex, compute progress
        - custom_command: Run a command, parse stdout as numeric progress
        - file_growth: Track output file size growth (fallback/default)
        """
        if not job.instance_id:
            return

        import re
        import json as _json

        progress_config = job.get_progress_config()
        progress_type = progress_config.get("type", "file_growth")

        current_value = None
        total_value = progress_config.get("total")
        rate_value = None

        try:
            if progress_type == "regex_parse" and progress_config.get("file"):
                # Tail the specified file, apply regex to extract current value
                target_file = progress_config["file"]
                pattern = progress_config.get("regex", r"(\d+)")
                tail_lines = progress_config.get("tail_lines", 20)

                exit_code, stdout, _ = await self.provider.execute_command(
                    job.instance_id,
                    f"tail -n {tail_lines} {target_file} 2>/dev/null",
                    timeout=10,
                )
                if exit_code == 0 and stdout.strip():
                    # Find all matches, use the last one (most recent)
                    matches = re.findall(pattern, stdout, re.MULTILINE)
                    if matches:
                        try:
                            current_value = float(matches[-1])
                        except (ValueError, TypeError):
                            pass

            elif progress_type == "custom_command" and progress_config.get("command"):
                # Run an arbitrary command and parse stdout as a number
                exit_code, stdout, _ = await self.provider.execute_command(
                    job.instance_id,
                    progress_config["command"],
                    timeout=15,
                )
                if exit_code == 0 and stdout.strip():
                    # Extract first number from output
                    match = re.search(r"([\d.]+)", stdout.strip())
                    if match:
                        current_value = float(match.group(1))

            else:
                # Default: file_growth — track output directory size
                resources = job.resources or {}
                check_cmd = "du -sm /workspace 2>/dev/null | awk '{print $1}'"
                exit_code, stdout, _ = await self.provider.execute_command(
                    job.instance_id, check_cmd, timeout=10
                )
                if exit_code == 0:
                    try:
                        output_size_mb = float(stdout.strip().split("\n")[0])
                        expected_size_mb = resources.get("expected_output_size_mb")
                        if expected_size_mb and expected_size_mb > 0:
                            current_value = output_size_mb
                            total_value = expected_size_mb
                    except (ValueError, IndexError):
                        pass

            # Calculate progress percentage and update metrics
            progress_pct = None
            if current_value is not None and total_value and total_value > 0:
                progress_pct = min(100.0, (current_value / total_value) * 100)

            # Update metrics in DB
            if current_value is not None or progress_pct is not None:
                async with get_session() as session:
                    stmt = select(Job).where(Job.job_id == job.job_id)
                    result = await session.execute(stmt)
                    db_job = result.scalar_one_or_none()
                    if db_job:
                        metrics = _json.loads(db_job.metrics_json) if db_job.metrics_json != "{}" else {}

                        # Compute rate (steps/second) from change since last update
                        prev_step = metrics.get("current_step")
                        prev_time = metrics.get("last_updated")
                        steps_per_second = None
                        if current_value is not None and prev_step is not None and prev_time:
                            try:
                                prev_dt = datetime.fromisoformat(prev_time)
                                elapsed_secs = (datetime.utcnow() - prev_dt).total_seconds()
                                if elapsed_secs > 5:  # Avoid division by tiny intervals
                                    step_delta = current_value - prev_step
                                    if step_delta > 0:
                                        steps_per_second = step_delta / elapsed_secs
                            except (ValueError, TypeError):
                                pass

                        if current_value is not None:
                            metrics["current_step"] = current_value
                        if total_value:
                            metrics["total_steps"] = total_value
                        if progress_pct is not None:
                            metrics["progress_percent"] = round(progress_pct, 1)
                        if steps_per_second is not None:
                            metrics["steps_per_second"] = round(steps_per_second, 1)
                            # ETA: remaining steps / rate
                            if total_value and current_value is not None and steps_per_second > 0:
                                remaining = total_value - current_value
                                eta_hours = (remaining / steps_per_second) / 3600
                                metrics["estimated_hours_remaining"] = round(eta_hours, 1)
                        metrics["last_updated"] = datetime.utcnow().isoformat()
                        db_job.metrics_json = _json.dumps(metrics)
                        session.add(db_job)
                        await session.commit()

        except Exception as e:
            logger.debug("Progress check failed", job_id=job.job_id, error=str(e))

    async def advance_job_stage(self, job: Job, exit_code: int) -> bool:
        """Check if a multi-stage job should advance to the next stage.

        Returns True if the job advanced (still running), False if done or failed.
        """
        stages = job.get_stages()
        if not stages:
            return False  # Not a multi-stage job

        current_idx = job.current_stage
        current_stage = stages[current_idx] if current_idx < len(stages) else None

        if not current_stage:
            return False

        # If current stage failed, don't advance
        if exit_code != 0:
            logger.warning("Stage failed", job_id=job.job_id, stage=current_stage.get("name"), exit_code=exit_code)
            return False

        next_idx = current_idx + 1
        if next_idx >= len(stages):
            return False  # All stages complete

        # Advance to next stage
        next_stage = stages[next_idx]
        next_command = next_stage.get("command", "")

        if not next_command:
            logger.error("Next stage has no command", job_id=job.job_id, stage=next_stage.get("name"))
            return False

        logger.info("Advancing to next stage",
                     job_id=job.job_id,
                     stage_name=next_stage.get("name"),
                     stage_index=next_idx)

        # Remove the exit code sentinel so the next stage can write its own
        await self.provider.execute_command(
            job.instance_id,
            "rm -f /workspace/.ccm_exit_code",
            timeout=10,
        )

        # Deploy and run SIGTERM-aware wrapper for next stage
        from cloudcomputemanager.core.wrapper import build_deploy_commands
        setup_cmd, run_cmd = build_deploy_commands(
            next_command,
            stage_name=next_stage.get("name"),
            script_path="/workspace/run_stage.sh",
        )
        await self.provider.execute_command(job.instance_id, setup_cmd, timeout=15)
        await self.provider.execute_command(job.instance_id, run_cmd, timeout=15)

        # Update job in DB
        async with get_session() as session:
            stmt = select(Job).where(Job.job_id == job.job_id)
            result = await session.execute(stmt)
            db_job = result.scalar_one_or_none()
            if db_job:
                db_job.current_stage = next_idx
                db_job.command = next_command
                session.add(db_job)
                await session.commit()

        return True

    async def run_notification(self, job: Job, event: str) -> None:
        """Run notification hooks for a job event.

        Args:
            job: The job
            event: Event name (on_complete, on_failure, on_budget_exceeded)
        """
        import shlex

        notifications = job.get_notifications()
        command = notifications.get(event)
        if not command:
            return

        # Substitute variables (shell-escaped to prevent injection)
        replacements = {
            "${JOB_ID}": shlex.quote(job.job_id),
            "${JOB_NAME}": shlex.quote(job.name),
            "${EXIT_CODE}": str(job.exit_code or 0),
            "${INSTANCE_ID}": shlex.quote(job.instance_id or ""),
            "${PROJECT}": shlex.quote(job.project or ""),
            "${STATUS}": shlex.quote(job.status.value),
        }
        for key, val in replacements.items():
            command = command.replace(key, val)

        try:
            import asyncio as _asyncio
            proc = await _asyncio.create_subprocess_shell(
                command,
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.PIPE,
            )
            await _asyncio.wait_for(proc.communicate(), timeout=30)
            logger.info("Notification sent", hook=event, job_id=job.job_id)
        except asyncio.TimeoutError:
            logger.warning("Notification timed out", hook=event, job_id=job.job_id)
        except Exception as e:
            logger.warning("Notification failed", hook=event, job_id=job.job_id, error=str(e))

    async def _reconcile_stale_jobs(self) -> None:
        """On daemon startup, reconcile jobs that changed while daemon was down.

        Checks all RUNNING/CHECKPOINTING jobs:
        - Instance gone → handle_preemption
        - Instance stopped/error → handle_preemption
        - Exit code exists → handle_completion (with multi-stage awareness)
        - Still running → continue monitoring (no action)
        """
        logger.info("Reconciling stale jobs after daemon restart")

        # First, sync all instances from provider to populate Instance table
        try:
            from cloudcomputemanager.core.instances import sync_all_instances
            stats = await sync_all_instances(self.provider)
            logger.info("Instance sync on startup", **stats)
        except Exception as e:
            logger.warning("Instance sync failed on startup", error=str(e))

        async with get_session() as session:
            stmt = select(Job).where(
                Job.status.in_([JobStatus.RUNNING, JobStatus.CHECKPOINTING])
            )
            result = await session.execute(stmt)
            jobs = result.scalars().all()

        if not jobs:
            logger.info("No stale jobs to reconcile")
            return

        logger.info("Found jobs to reconcile", count=len(jobs))

        for job in jobs:
            if not job.instance_id:
                continue

            try:
                # Check instance status via provider API
                instance = await self.provider.get_instance(job.instance_id)

                if instance is None:
                    logger.warning("Instance gone for running job",
                                   job_id=job.job_id, instance_id=job.instance_id)
                    await self.handle_preemption(job, "instance_gone_on_restart")
                    continue

                from cloudcomputemanager.providers.base import ProviderStatus
                if instance.status not in (ProviderStatus.RUNNING, ProviderStatus.STARTING):
                    logger.warning("Instance not running on restart",
                                   job_id=job.job_id, status=instance.status.value)
                    await self.handle_preemption(job, f"instance_{instance.status.value}_on_restart")
                    continue

                # Instance is running — check for completion
                completed, exit_code = await self.check_job_completion(job.instance_id)
                if completed:
                    logger.info("Job completed while daemon was down",
                                job_id=job.job_id, exit_code=exit_code)
                    if exit_code == 0 and await self.advance_job_stage(job, exit_code):
                        continue
                    await self.handle_job_completion(job, exit_code)
                else:
                    logger.info("Job still running, resuming monitoring", job_id=job.job_id)

            except Exception as e:
                logger.error("Error reconciling job", job_id=job.job_id, error=str(e))

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Monitor loop started", poll_interval=self.config.poll_interval)

        # Import recovery manager
        from cloudcomputemanager.core.recovery import RecoveryManager
        recovery_manager = RecoveryManager(
            provider=self.provider,
            max_attempts=self.config.max_recovery_attempts,
        )

        # Reconcile any jobs that changed while daemon was down
        await self._reconcile_stale_jobs()

        self._last_recovery_check = 0

        self._last_instance_sync: float = 0

        while self._running:
            try:
                # Sync all instances from provider every 60s
                # This discovers instances created outside CCM and detects terminated ones
                now = asyncio.get_event_loop().time()
                if now - self._last_instance_sync > 60:
                    try:
                        from cloudcomputemanager.core.instances import sync_all_instances
                        await sync_all_instances(self.provider)
                        self._last_instance_sync = now
                    except Exception as e:
                        logger.debug("Instance sync failed", error=str(e))

                # Get ALL active jobs (not just RUNNING — also PROVISIONING, RECOVERING, CHECKPOINTING)
                async with get_session() as session:
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

                logger.debug("Checking jobs", count=len(jobs))

                for job in jobs:
                    if not job.instance_id:
                        continue

                    # Track monitored jobs
                    self._monitored_jobs.add(job.job_id)

                    # For PROVISIONING/RECOVERING: only check if instance is alive
                    # Don't check for exit code (job hasn't started yet)
                    if job.status in (JobStatus.PROVISIONING, JobStatus.RECOVERING):
                        healthy, reason = await self.check_instance_health(job.instance_id, job=job)
                        if not healthy:
                            logger.warning("Instance dead for non-running job",
                                           job_id=job.job_id, status=job.status.value, reason=reason)
                            # Route through handle_preemption so retry logic applies
                            await self.handle_preemption(job, f"instance_lost_during_{job.status.value}")
                        continue

                    # Check completion first (RUNNING/CHECKPOINTING jobs only)
                    completed, exit_code = await self.check_job_completion(job.instance_id)
                    if completed:
                        # For multi-stage jobs, try to advance to next stage
                        if exit_code == 0 and await self.advance_job_stage(job, exit_code):
                            # Job advanced to next stage, still running
                            continue
                        await self.handle_job_completion(job, exit_code)
                        continue

                    # Check budget
                    within_budget, budget_reason = await self.check_job_budget(job)
                    if not within_budget:
                        logger.warning("Job budget exceeded", job_id=job.job_id, reason=budget_reason)
                        await self.run_notification(job, "on_budget_exceeded")
                        # Sync results before terminating
                        await self.sync_job_results(job)
                        # Update job status
                        async with get_session() as session:
                            stmt = select(Job).where(Job.job_id == job.job_id)
                            result = await session.execute(stmt)
                            db_job = result.scalar_one_or_none()
                            if db_job:
                                db_job.status = JobStatus.BUDGET_EXCEEDED
                                db_job.error_message = f"Budget exceeded: {budget_reason}"
                                db_job.completed_at = datetime.utcnow()
                                session.add(db_job)
                                await session.commit()
                        # Terminate instance
                        await self.terminate_job_instance(job)
                        self._monitored_jobs.discard(job.job_id)
                        continue

                    # Check instance health (multi-signal)
                    healthy, reason = await self.check_instance_health(job.instance_id, job=job)
                    if not healthy:
                        await self.handle_preemption(job, reason)
                        continue

                    # Update progress metrics
                    await self.update_job_progress(job)

                # Handle recovery jobs periodically
                if self.config.preemption_recovery and asyncio.get_event_loop().time() - self._last_recovery_check > 60:
                    self._last_recovery_check = asyncio.get_event_loop().time()

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
