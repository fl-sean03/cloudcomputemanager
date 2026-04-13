"""Preemption recovery for CCM jobs."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
import structlog

from sqlmodel import select

from cloudcomputemanager.core.database import get_session
from cloudcomputemanager.core.config import get_settings
from cloudcomputemanager.core.models import Job, JobStatus, Checkpoint
from cloudcomputemanager.providers.vast import VastProvider

logger = structlog.get_logger()


# =============================================================================
# Offer Blacklist — remembers bad instances to avoid on retry
# =============================================================================

_BLACKLIST_FILE = Path.home() / ".cloudcomputemanager" / "offer_blacklist.json"
_BLACKLIST_EXPIRY_HOURS = 24


def _load_blacklist() -> dict:
    """Load offer blacklist from disk. Returns {offer_id: {reason, expires, project}}."""
    if not _BLACKLIST_FILE.exists():
        return {}
    try:
        data = json.loads(_BLACKLIST_FILE.read_text())
        # Prune expired entries
        now = datetime.utcnow().isoformat()
        pruned = {k: v for k, v in data.items() if v.get("expires", "") > now}
        if len(pruned) < len(data):
            _BLACKLIST_FILE.write_text(json.dumps(pruned, indent=2))
        return pruned
    except Exception:
        return {}


def blacklist_offer(offer_id: str, reason: str, project: str = "") -> None:
    """Add an offer to the blacklist."""
    bl = _load_blacklist()
    bl[str(offer_id)] = {
        "reason": reason,
        "project": project,
        "added": datetime.utcnow().isoformat(),
        "expires": (datetime.utcnow() + timedelta(hours=_BLACKLIST_EXPIRY_HOURS)).isoformat(),
    }
    _BLACKLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    _BLACKLIST_FILE.write_text(json.dumps(bl, indent=2))
    logger.info("Offer blacklisted", offer_id=offer_id, reason=reason)


def get_blacklisted_offers() -> set[str]:
    """Get set of currently blacklisted offer IDs."""
    return set(_load_blacklist().keys())


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    success: bool
    job_id: str
    new_instance_id: Optional[str] = None
    checkpoint_restored: bool = False
    error: Optional[str] = None
    attempt_number: int = 0


class RecoveryManager:
    """
    Manages job recovery after preemption or failure.

    Recovery flow:
    1. Find jobs in RECOVERING state
    2. Locate latest checkpoint (if any)
    3. Search for new instance matching requirements
    4. Create new instance and deploy files
    5. Restore checkpoint if available
    6. Resume job execution
    """

    def __init__(
        self,
        provider: Optional[VastProvider] = None,
        max_attempts: int = 5,
        backoff_minutes: int = 5,
    ):
        self.provider = provider or VastProvider()
        self.settings = get_settings()
        self.max_attempts = max_attempts
        self.backoff_minutes = backoff_minutes

    async def get_latest_checkpoint(self, job_id: str) -> Optional[Checkpoint]:
        """Get the latest verified checkpoint for a job."""
        async with get_session() as session:
            stmt = (
                select(Checkpoint)
                .where(Checkpoint.job_id == job_id)
                .where(Checkpoint.verified == True)
                .order_by(Checkpoint.created_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def find_recovery_instance(self, job: Job) -> Optional[str]:
        """
        Find a new instance for job recovery.

        Returns offer_id if found, None otherwise.
        """
        resources = job.resources or {}
        budget = job.budget or {}

        # Use user-specified values without hardcoded defaults
        # The vast.py provider will handle None values with sensible defaults
        gpu_type = resources.get("gpu_type")
        gpu_count = resources.get("gpu_count", 1)
        gpu_memory_min = resources.get("gpu_memory_min")
        disk_gb = resources.get("disk_gb", 50)
        max_hourly_rate = budget.get("max_hourly_rate")
        cuda_version_min = resources.get("cuda_version_min")
        reliability_min = resources.get("reliability_min")
        min_duration_hours = resources.get("min_duration_hours")

        logger.info(
            "Searching for recovery instance",
            job_id=job.job_id,
            gpu_type=gpu_type,
            gpu_memory_min=gpu_memory_min,
        )

        # Exclude blacklisted offers (bad instances from prior failures)
        excluded = get_blacklisted_offers()

        offers = await self.provider.search_offers(
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            gpu_memory_min=gpu_memory_min,
            disk_gb_min=disk_gb,
            max_hourly_rate=max_hourly_rate,
            exclude_offer_ids=excluded if excluded else None,
            cuda_version_min=cuda_version_min,
            reliability_min=reliability_min,
            min_duration_hours=min_duration_hours,
        )

        if not offers:
            logger.warning("No recovery offers found", job_id=job.job_id, gpu_type=gpu_type)
            return None

        # Return best offer
        return offers[0].offer_id

    async def create_recovery_instance(
        self,
        job: Job,
        offer_id: str,
    ) -> Optional[str]:
        """Create a new instance for recovery."""
        resources = job.resources or {}

        # CRITICAL: Set label so sync_all_instances doesn't auto-terminate as "unmanaged"
        from cloudcomputemanager.core.instances import build_instance_label
        label = build_instance_label(job.job_id, job.project or "", job.name)

        instance = await self.provider.create_instance(
            offer_id=offer_id,
            image=job.image,
            disk_gb=resources.get("disk_gb", 50),
            label=label,
        )

        if not instance:
            return None

        # Wait for instance to be ready — use provisioning_timeout from job or default 1800s
        timeout = getattr(job, 'provisioning_timeout', None) or 1800
        ready = await self.provider.wait_for_ready(
            instance.instance_id,
            timeout=timeout,
        )

        if not ready:
            logger.error(
                "Recovery instance failed to start",
                job_id=job.job_id,
                instance_id=instance.instance_id,
            )
            # Clean up failed instance
            try:
                await self.provider.terminate_instance(instance.instance_id)
            except Exception:
                pass
            return None

        return instance.instance_id

    async def restore_checkpoint(
        self,
        job: Job,
        instance_id: str,
        checkpoint: Checkpoint,
    ) -> bool:
        """Restore a checkpoint to the new instance."""
        logger.info(
            "Restoring checkpoint",
            job_id=job.job_id,
            checkpoint_id=checkpoint.checkpoint_id,
            instance_id=instance_id,
        )

        # Upload checkpoint files
        checkpoint_path = self.settings.checkpoint_local_path / job.job_id
        if not checkpoint_path.exists():
            logger.error("Checkpoint path not found", path=str(checkpoint_path))
            return False

        # Upload to instance
        success = await self.provider.rsync_upload(
            instance_id,
            str(checkpoint_path) + "/",
            "/workspace/",
        )

        return success

    async def upload_job_files(
        self,
        job: Job,
        instance_id: str,
        upload_config: dict,
    ) -> bool:
        """Upload job files to new instance."""
        source = upload_config.get("source")
        if not source:
            return True  # Nothing to upload

        destination = upload_config.get("destination", "/workspace")
        exclude = upload_config.get("exclude", [])

        from pathlib import Path
        source_path = Path(source).expanduser()
        if not source_path.exists():
            logger.warning("Upload source not found", source=str(source_path))
            return False

        return await self.provider.rsync_upload(
            instance_id,
            str(source_path) + "/",
            destination + "/",
            exclude=exclude,
        )

    async def start_recovered_job(
        self,
        job: Job,
        instance_id: str,
        has_checkpoint: bool,
        command_override: Optional[str] = None,
    ) -> bool:
        """Start the job on the recovered instance.

        Args:
            job: The job being recovered
            instance_id: New instance to run on
            has_checkpoint: Whether a checkpoint was restored
            command_override: If set, use this command instead of job.command
                (typically set by a restart adapter or user-defined restart config)
        """
        import base64

        command = command_override or job.command or ""
        if command_override:
            logger.info("Using restart adapter command", job_id=job.job_id,
                       command=command[:120])

        # Wrap with exit code tracking — must use set +e so crashes
        # still write exit code (set -e would kill bash before $? capture)
        wrapper_script = f'''#!/bin/bash
# CCM Recovery Job Wrapper
cd /workspace

# Run job in subshell with set +e so we always capture exit code
set +e
(
{command}
) &
JOB_PID=$!
wait $JOB_PID
JOB_EXIT_CODE=$?
set -e

# Write exit code — MUST execute even on crash
echo $JOB_EXIT_CODE > /workspace/.ccm_exit_code
echo "Job completed with exit code $JOB_EXIT_CODE" >> /workspace/job.log
exit $JOB_EXIT_CODE
'''

        script_b64 = base64.b64encode(wrapper_script.encode()).decode()

        # Write and execute script
        setup_cmd = f"echo {script_b64} | base64 -d > /workspace/run_job.sh && chmod +x /workspace/run_job.sh"
        exit_code, stdout, stderr = await self.provider.execute_command(instance_id, setup_cmd)

        if exit_code != 0:
            logger.error("Failed to write recovery script",
                        job_id=job.job_id, exit_code=exit_code,
                        stderr=stderr[:200] if stderr else "")
            return False

        # Create checkpoint marker if restoring
        if has_checkpoint:
            await self.provider.execute_command(
                instance_id,
                "touch /workspace/.ccm_checkpoint_marker"
            )

        # Run in background with timeout — nohup can hang on slow SSH
        run_cmd = "cd /workspace && nohup bash /workspace/run_job.sh > /workspace/job.log 2>&1 &"
        exit_code, stdout, stderr = await self.provider.execute_command(
            instance_id, run_cmd, timeout=15
        )

        # nohup timeout (exit -1) is OK — verify process started
        if exit_code != 0:
            verify_code, verify_out, _ = await self.provider.execute_command(
                instance_id,
                "pgrep -f run_job.sh > /dev/null 2>&1 && echo RUNNING || echo NOT_RUNNING",
                timeout=15,
            )
            if "RUNNING" in verify_out:
                logger.info("Recovery job started (verified after nohup timeout)",
                           job_id=job.job_id)
                return True
            logger.error("Failed to start recovery job",
                        job_id=job.job_id, exit_code=exit_code,
                        stderr=stderr[:200] if stderr else "")
            return False

        return True

    async def _increment_attempt(self, job: Job) -> None:
        """Increment attempt_number in DB on failed recovery."""
        async with get_session() as session:
            stmt = select(Job).where(Job.job_id == job.job_id)
            result = await session.execute(stmt)
            db_job = result.scalar_one_or_none()
            if db_job:
                db_job.attempt_number += 1
                session.add(db_job)
                await session.commit()

    async def recover_job(self, job: Job) -> RecoveryResult:
        """
        Attempt to recover a single job.

        Returns RecoveryResult with success status and details.
        """
        logger.info(
            "Starting job recovery",
            job_id=job.job_id,
            attempt=job.attempt_number + 1,
        )

        # Check attempt limit — use per-job config if available, else daemon default
        job_retry = json.loads(job.retry_json) if job.retry_json else {}
        effective_max_attempts = job_retry.get("max_attempts", self.max_attempts)

        if job.attempt_number >= effective_max_attempts:
            return RecoveryResult(
                success=False,
                job_id=job.job_id,
                error=f"Max attempts ({effective_max_attempts}) exceeded",
                attempt_number=job.attempt_number,
            )

        # Get latest checkpoint
        checkpoint = await self.get_latest_checkpoint(job.job_id)
        has_checkpoint = checkpoint is not None

        # Find new instance
        offer_id = await self.find_recovery_instance(job)
        if not offer_id:
            # Increment attempt_number in DB so we track retries
            await self._increment_attempt(job)
            return RecoveryResult(
                success=False,
                job_id=job.job_id,
                error="No suitable offers found",
                attempt_number=job.attempt_number + 1,
            )

        # Create instance
        instance_id = await self.create_recovery_instance(job, offer_id)
        if not instance_id:
            # Blacklist this offer — it failed to provision
            blacklist_offer(offer_id, "provisioning_failed", job.project or "")
            await self._increment_attempt(job)
            return RecoveryResult(
                success=False,
                job_id=job.job_id,
                error="Failed to create recovery instance",
                attempt_number=job.attempt_number + 1,
            )

        logger.info(
            "Recovery instance created",
            job_id=job.job_id,
            instance_id=instance_id,
        )

        # Upload original job files (PSF, PRM, NAMD config, etc.)
        # Uses upload_json stored during job submission — no hardcoded paths
        upload_config = json.loads(job.upload_json) if hasattr(job, 'upload_json') and job.upload_json else {}
        upload_source = upload_config.get("source")
        upload_dest = upload_config.get("destination", "/workspace")
        upload_exclude = upload_config.get("exclude", [])

        if upload_source and Path(upload_source).exists():
            logger.info("Uploading original job files",
                       source=upload_source, dest=upload_dest)
            success = await self.provider.rsync_upload(
                instance_id,
                str(upload_source) + "/",
                upload_dest + "/",
                exclude=upload_exclude,
            )
            if not success:
                logger.error("Failed to upload original job files",
                           job_id=job.job_id, source=upload_source)
        elif upload_source:
            logger.warning("Upload source not found for recovery",
                         job_id=job.job_id, source=upload_source)
        else:
            logger.debug("No upload source configured for job",
                        job_id=job.job_id)

        # Then overlay sync directory (has latest restart files, DCD, logs)
        sync_dir = self.settings.sync_local_path / job.job_id

        # --- Determine restart command via adapter chain ---
        from cloudcomputemanager.checkpoint.restart_adapters import (
            RestartResult as _RestartResult,
            get_restart_adapter,
        )

        restart_result: Optional[_RestartResult] = None

        # Priority 1: User-defined restart command in job YAML
        user_restart = (
            json.loads(job.restart_json)
            if hasattr(job, "restart_json") and job.restart_json
            else {}
        )
        if user_restart.get("command") and sync_dir.exists():
            detect_pattern = user_restart.get("detect_checkpoint")
            has_checkpoint_file = True
            if detect_pattern:
                has_checkpoint_file = bool(list(sync_dir.glob(detect_pattern)))
            if has_checkpoint_file:
                restart_result = _RestartResult(
                    command=user_restart["command"],
                    description="User-defined restart command",
                )
                logger.info(
                    "Using user-defined restart command",
                    job_id=job.job_id,
                    command=user_restart["command"][:120],
                )

        # Priority 2: Auto-detected adapter
        if restart_result is None and sync_dir.exists():
            adapter = get_restart_adapter(job.command or "")
            logger.info(
                "Restart adapter selected",
                adapter=adapter.name,
                job_id=job.job_id,
            )
            restart_result = adapter.prepare_restart(
                command=job.command or "",
                sync_dir=sync_dir,
                job_id=job.job_id,
            )
            if restart_result:
                logger.info(
                    "Restart adapter prepared recovery",
                    job_id=job.job_id,
                    adapter=adapter.name,
                    description=restart_result.description,
                    files_written=restart_result.files_written,
                )

        # Upload sync directory (with any files the adapter wrote)
        if sync_dir.exists():
            await self.provider.rsync_upload(
                instance_id,
                str(sync_dir) + "/",
                "/workspace/",
            )

        # Restore checkpoint if available
        checkpoint_restored = False
        if has_checkpoint:
            checkpoint_restored = await self.restore_checkpoint(job, instance_id, checkpoint)

        # Start the job with the adapter's command (or original)
        effective_command = restart_result.command if restart_result else None
        job_started = await self.start_recovered_job(
            job, instance_id, has_checkpoint,
            command_override=effective_command,
        )

        if not job_started:
            # Clean up on failure
            try:
                await self.provider.terminate_instance(instance_id)
            except Exception:
                pass

            return RecoveryResult(
                success=False,
                job_id=job.job_id,
                new_instance_id=instance_id,
                checkpoint_restored=checkpoint_restored,
                error="Failed to start recovered job",
                attempt_number=job.attempt_number + 1,
            )

        # Terminate old instance to prevent orphan billing
        old_instance_id = job.instance_id
        if old_instance_id and old_instance_id != instance_id:
            try:
                await self.provider.terminate_instance(old_instance_id)
                logger.info("Terminated old instance after recovery",
                           job_id=job.job_id, old_instance=old_instance_id)
            except Exception as e:
                logger.debug("Could not terminate old instance", error=str(e))

        # Update job in database
        async with get_session() as session:
            stmt = select(Job).where(Job.job_id == job.job_id)
            result = await session.execute(stmt)
            db_job = result.scalar_one_or_none()

            if db_job:
                db_job.instance_id = instance_id
                db_job.status = JobStatus.RUNNING
                db_job.attempt_number += 1
                db_job.error_message = None
                session.add(db_job)
                await session.commit()

        logger.info(
            "Job recovery successful",
            job_id=job.job_id,
            new_instance_id=instance_id,
            checkpoint_restored=checkpoint_restored,
        )

        return RecoveryResult(
            success=True,
            job_id=job.job_id,
            new_instance_id=instance_id,
            checkpoint_restored=checkpoint_restored,
            attempt_number=job.attempt_number + 1,
        )

    async def recover_all_pending(self) -> list[RecoveryResult]:
        """Recover all jobs in RECOVERING state."""
        async with get_session() as session:
            stmt = select(Job).where(Job.status == JobStatus.RECOVERING)
            result = await session.execute(stmt)
            jobs = result.scalars().all()

        results = []
        for job in jobs:
            try:
                result = await self.recover_job(job)
                results.append(result)

                # Backoff between recovery attempts
                if not result.success:
                    await asyncio.sleep(self.backoff_minutes * 60)

            except Exception as e:
                logger.error("Recovery failed", job_id=job.job_id, error=str(e))
                results.append(RecoveryResult(
                    success=False,
                    job_id=job.job_id,
                    error=str(e),
                ))

        return results
