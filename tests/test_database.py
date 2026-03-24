"""Tests for database commit paths using in-memory SQLite.

Verifies that jobs are actually persisted through the database layer,
including creation, status updates, recovery, cleanup, and sync tracking.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import AsyncGenerator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel, select

from cloudcomputemanager.core.models import Job, JobStatus, SyncStatus


def _make_session_factory():
    """Create a fresh in-memory engine and session factory for one test."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
    )
    factory = sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    return engine, factory


@asynccontextmanager
async def _session(factory) -> AsyncGenerator[AsyncSession, None]:
    """Mimic the production get_session: auto-commit on success, rollback on error."""
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def _init_tables(engine):
    """Create all tables in the in-memory database."""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


def _make_job(**overrides) -> Job:
    """Build a Job with sensible defaults; caller can override any field."""
    defaults = dict(
        name="test-job",
        image="python:3.11",
        command="python -c 'print(1)'",
    )
    defaults.update(overrides)
    return Job(**defaults)


# ---------------------------------------------------------------------------
# 1. Job creation -> commit -> read back
# ---------------------------------------------------------------------------

class TestJobCreation:

    async def test_create_and_read_back(self):
        """Insert a job, commit, then SELECT it back and verify all fields."""
        engine, factory = _make_session_factory()
        await _init_tables(engine)

        # Write
        async with _session(factory) as session:
            job = _make_job(name="persist-check", project="proj-a")
            session.add(job)

        # Read in a separate session to prove persistence
        async with _session(factory) as session:
            stmt = select(Job).where(Job.name == "persist-check")
            result = await session.execute(stmt)
            loaded = result.scalar_one()

            assert loaded.name == "persist-check"
            assert loaded.project == "proj-a"
            assert loaded.status == JobStatus.PENDING
            assert loaded.image == "python:3.11"
            assert loaded.command == "python -c 'print(1)'"
            assert loaded.id is not None  # auto-generated PK

        await engine.dispose()

    async def test_multiple_jobs_persist(self):
        """Create several jobs in one session and verify count."""
        engine, factory = _make_session_factory()
        await _init_tables(engine)

        async with _session(factory) as session:
            for i in range(5):
                session.add(_make_job(name=f"batch-{i}"))

        async with _session(factory) as session:
            result = await session.execute(select(Job))
            jobs = result.scalars().all()
            assert len(jobs) == 5

        await engine.dispose()

    async def test_job_id_auto_generated(self):
        """Verify that job_id is auto-generated and unique across jobs."""
        engine, factory = _make_session_factory()
        await _init_tables(engine)

        async with _session(factory) as session:
            j1 = _make_job(name="a")
            j2 = _make_job(name="b")
            session.add(j1)
            session.add(j2)

        async with _session(factory) as session:
            result = await session.execute(select(Job))
            jobs = result.scalars().all()
            ids = [j.job_id for j in jobs]
            assert len(set(ids)) == 2  # unique
            for jid in ids:
                assert jid.startswith("job_")

        await engine.dispose()


# ---------------------------------------------------------------------------
# 2. Job status update -> commit -> verify persisted
# ---------------------------------------------------------------------------

class TestJobStatusUpdate:

    async def test_update_status_persists(self):
        """Change status from PENDING to RUNNING and read it back."""
        engine, factory = _make_session_factory()
        await _init_tables(engine)

        async with _session(factory) as session:
            job = _make_job(name="status-test")
            session.add(job)

        # Update in a new session
        async with _session(factory) as session:
            result = await session.execute(
                select(Job).where(Job.name == "status-test")
            )
            job = result.scalar_one()
            assert job.status == JobStatus.PENDING
            job.status = JobStatus.RUNNING
            job.started_at = datetime(2026, 1, 15, 12, 0, 0)
            session.add(job)

        # Verify
        async with _session(factory) as session:
            result = await session.execute(
                select(Job).where(Job.name == "status-test")
            )
            job = result.scalar_one()
            assert job.status == JobStatus.RUNNING
            assert job.started_at == datetime(2026, 1, 15, 12, 0, 0)

        await engine.dispose()

    async def test_status_transition_to_completed(self):
        """Full lifecycle: PENDING -> RUNNING -> COMPLETED with exit code."""
        engine, factory = _make_session_factory()
        await _init_tables(engine)

        async with _session(factory) as session:
            session.add(_make_job(name="lifecycle"))

        # PENDING -> RUNNING
        async with _session(factory) as session:
            job = (await session.execute(
                select(Job).where(Job.name == "lifecycle")
            )).scalar_one()
            job.status = JobStatus.RUNNING
            job.instance_id = "inst_001"
            session.add(job)

        # RUNNING -> COMPLETED
        async with _session(factory) as session:
            job = (await session.execute(
                select(Job).where(Job.name == "lifecycle")
            )).scalar_one()
            job.status = JobStatus.COMPLETED
            job.exit_code = 0
            job.completed_at = datetime.utcnow()
            job.total_cost_usd = 1.25
            job.total_runtime_seconds = 3600
            session.add(job)

        # Verify final state
        async with _session(factory) as session:
            job = (await session.execute(
                select(Job).where(Job.name == "lifecycle")
            )).scalar_one()
            assert job.status == JobStatus.COMPLETED
            assert job.exit_code == 0
            assert job.instance_id == "inst_001"
            assert job.total_cost_usd == 1.25
            assert job.total_runtime_seconds == 3600

        await engine.dispose()

    async def test_update_does_not_create_duplicate(self):
        """Updating a job should not create a second row."""
        engine, factory = _make_session_factory()
        await _init_tables(engine)

        async with _session(factory) as session:
            session.add(_make_job(name="single"))

        # Update
        async with _session(factory) as session:
            job = (await session.execute(
                select(Job).where(Job.name == "single")
            )).scalar_one()
            job.status = JobStatus.RUNNING
            session.add(job)

        # Count
        async with _session(factory) as session:
            result = await session.execute(select(Job))
            assert len(result.scalars().all()) == 1

        await engine.dispose()


# ---------------------------------------------------------------------------
# 3. Recovery commit path
# ---------------------------------------------------------------------------

class TestRecoveryCommitPath:

    async def test_set_recovering_and_read_back(self):
        """Simulate preemption: set status to RECOVERING, bump attempt, persist."""
        engine, factory = _make_session_factory()
        await _init_tables(engine)

        async with _session(factory) as session:
            job = _make_job(name="recovery-job", status=JobStatus.RUNNING)
            job.instance_id = "inst_old"
            job.attempt_number = 1
            session.add(job)

        # Simulate preemption detection -> RECOVERING
        async with _session(factory) as session:
            job = (await session.execute(
                select(Job).where(Job.name == "recovery-job")
            )).scalar_one()
            job.status = JobStatus.RECOVERING
            job.instance_id = None
            job.attempt_number += 1
            session.add(job)

        # Verify persisted
        async with _session(factory) as session:
            job = (await session.execute(
                select(Job).where(Job.name == "recovery-job")
            )).scalar_one()
            assert job.status == JobStatus.RECOVERING
            assert job.instance_id is None
            assert job.attempt_number == 2

        await engine.dispose()

    async def test_recovery_to_running_with_new_instance(self):
        """After recovery completes, job moves to RUNNING on a new instance."""
        engine, factory = _make_session_factory()
        await _init_tables(engine)

        async with _session(factory) as session:
            job = _make_job(
                name="recover-run",
                status=JobStatus.RECOVERING,
            )
            job.attempt_number = 2
            session.add(job)

        # Recovery succeeds
        async with _session(factory) as session:
            job = (await session.execute(
                select(Job).where(Job.name == "recover-run")
            )).scalar_one()
            job.status = JobStatus.RUNNING
            job.instance_id = "inst_new"
            job.last_checkpoint_id = "ckpt_abc123"
            session.add(job)

        async with _session(factory) as session:
            job = (await session.execute(
                select(Job).where(Job.name == "recover-run")
            )).scalar_one()
            assert job.status == JobStatus.RUNNING
            assert job.instance_id == "inst_new"
            assert job.last_checkpoint_id == "ckpt_abc123"
            assert job.attempt_number == 2

        await engine.dispose()

    async def test_recovery_exceeds_max_attempts_marks_failed(self):
        """When recovery attempts are exhausted, the job should be marked FAILED."""
        engine, factory = _make_session_factory()
        await _init_tables(engine)

        max_attempts = 5

        async with _session(factory) as session:
            job = _make_job(
                name="exhaust-retry",
                status=JobStatus.RECOVERING,
            )
            job.attempt_number = max_attempts
            session.add(job)

        # Recovery logic checks and fails
        async with _session(factory) as session:
            job = (await session.execute(
                select(Job).where(Job.name == "exhaust-retry")
            )).scalar_one()
            assert job.attempt_number >= max_attempts
            job.status = JobStatus.FAILED
            job.error_message = f"Max recovery attempts ({max_attempts}) exceeded"
            job.completed_at = datetime.utcnow()
            session.add(job)

        async with _session(factory) as session:
            job = (await session.execute(
                select(Job).where(Job.name == "exhaust-retry")
            )).scalar_one()
            assert job.status == JobStatus.FAILED
            assert "Max recovery attempts" in job.error_message
            assert job.completed_at is not None

        await engine.dispose()


# ---------------------------------------------------------------------------
# 4. Cleanup flow with real DB
# ---------------------------------------------------------------------------

class TestCleanupFlow:

    async def test_mark_stale_job_as_failed(self):
        """Create a stale job (RUNNING but no instance), mark FAILED, verify."""
        engine, factory = _make_session_factory()
        await _init_tables(engine)

        # Create stale job
        async with _session(factory) as session:
            stale = _make_job(
                name="stale-job",
                status=JobStatus.RUNNING,
            )
            stale.instance_id = None  # no instance -> stale
            session.add(stale)

        # Cleanup: find stale and mark failed
        async with _session(factory) as session:
            stmt = select(Job).where(
                Job.status.in_([
                    JobStatus.RUNNING,
                    JobStatus.PROVISIONING,
                    JobStatus.RECOVERING,
                    JobStatus.CHECKPOINTING,
                ]),
                Job.instance_id.is_(None),
            )
            result = await session.execute(stmt)
            stale_jobs = result.scalars().all()

            assert len(stale_jobs) == 1
            for job in stale_jobs:
                job.status = JobStatus.FAILED
                job.error_message = "Cleaned up: no_instance_id"
                job.completed_at = datetime.utcnow()
                session.add(job)

        # Verify cleanup persisted
        async with _session(factory) as session:
            job = (await session.execute(
                select(Job).where(Job.name == "stale-job")
            )).scalar_one()
            assert job.status == JobStatus.FAILED
            assert "no_instance_id" in job.error_message

        await engine.dispose()

    async def test_cleanup_leaves_healthy_jobs_alone(self):
        """Jobs with valid instance_id should not be affected by cleanup."""
        engine, factory = _make_session_factory()
        await _init_tables(engine)

        async with _session(factory) as session:
            healthy = _make_job(name="healthy", status=JobStatus.RUNNING)
            healthy.instance_id = "inst_good"
            stale = _make_job(name="stale", status=JobStatus.RUNNING)
            stale.instance_id = None
            session.add(healthy)
            session.add(stale)

        # Cleanup stale only
        async with _session(factory) as session:
            stmt = select(Job).where(
                Job.status == JobStatus.RUNNING,
                Job.instance_id.is_(None),
            )
            result = await session.execute(stmt)
            for job in result.scalars().all():
                job.status = JobStatus.FAILED
                job.error_message = "Cleaned up: no_instance_id"
                session.add(job)

        # Verify healthy untouched, stale cleaned
        async with _session(factory) as session:
            healthy = (await session.execute(
                select(Job).where(Job.name == "healthy")
            )).scalar_one()
            stale = (await session.execute(
                select(Job).where(Job.name == "stale")
            )).scalar_one()

            assert healthy.status == JobStatus.RUNNING
            assert healthy.instance_id == "inst_good"
            assert stale.status == JobStatus.FAILED

        await engine.dispose()

    async def test_cleanup_multiple_stale_jobs(self):
        """Cleanup should handle multiple stale jobs in a single pass."""
        engine, factory = _make_session_factory()
        await _init_tables(engine)

        async with _session(factory) as session:
            for i in range(4):
                job = _make_job(
                    name=f"stale-{i}",
                    status=JobStatus.RUNNING,
                )
                job.instance_id = None
                session.add(job)

        async with _session(factory) as session:
            stmt = select(Job).where(
                Job.status == JobStatus.RUNNING,
                Job.instance_id.is_(None),
            )
            result = await session.execute(stmt)
            cleaned = 0
            for job in result.scalars().all():
                job.status = JobStatus.FAILED
                job.error_message = "Cleaned up: no_instance_id"
                session.add(job)
                cleaned += 1
            assert cleaned == 4

        async with _session(factory) as session:
            result = await session.execute(
                select(Job).where(Job.status == JobStatus.FAILED)
            )
            assert len(result.scalars().all()) == 4

        await engine.dispose()


# ---------------------------------------------------------------------------
# 5. Sync status tracking fields
# ---------------------------------------------------------------------------

class TestSyncStatusTracking:

    async def test_default_sync_fields(self):
        """New jobs should have default sync field values."""
        engine, factory = _make_session_factory()
        await _init_tables(engine)

        async with _session(factory) as session:
            session.add(_make_job(name="sync-defaults"))

        async with _session(factory) as session:
            job = (await session.execute(
                select(Job).where(Job.name == "sync-defaults")
            )).scalar_one()
            assert job.sync_status == "not_synced"
            assert job.last_sync_at is None
            assert job.synced_bytes == 0
            assert job.synced_files == 0

        await engine.dispose()

    async def test_update_sync_status_to_syncing(self):
        """Transition sync_status from not_synced to syncing."""
        engine, factory = _make_session_factory()
        await _init_tables(engine)

        async with _session(factory) as session:
            session.add(_make_job(name="sync-start"))

        async with _session(factory) as session:
            job = (await session.execute(
                select(Job).where(Job.name == "sync-start")
            )).scalar_one()
            job.sync_status = "syncing"
            session.add(job)

        async with _session(factory) as session:
            job = (await session.execute(
                select(Job).where(Job.name == "sync-start")
            )).scalar_one()
            assert job.sync_status == "syncing"

        await engine.dispose()

    async def test_sync_completion_updates_all_fields(self):
        """After sync completes, all tracking fields should be updated."""
        engine, factory = _make_session_factory()
        await _init_tables(engine)

        async with _session(factory) as session:
            session.add(_make_job(name="sync-complete"))

        sync_time = datetime(2026, 3, 20, 14, 30, 0)

        async with _session(factory) as session:
            job = (await session.execute(
                select(Job).where(Job.name == "sync-complete")
            )).scalar_one()
            job.sync_status = "synced"
            job.last_sync_at = sync_time
            job.synced_bytes = 1_048_576  # 1 MiB
            job.synced_files = 42
            session.add(job)

        async with _session(factory) as session:
            job = (await session.execute(
                select(Job).where(Job.name == "sync-complete")
            )).scalar_one()
            assert job.sync_status == "synced"
            assert job.last_sync_at == sync_time
            assert job.synced_bytes == 1_048_576
            assert job.synced_files == 42

        await engine.dispose()

    async def test_sync_failure_records_status(self):
        """Sync failure should persist sync_failed status."""
        engine, factory = _make_session_factory()
        await _init_tables(engine)

        async with _session(factory) as session:
            session.add(_make_job(name="sync-fail"))

        async with _session(factory) as session:
            job = (await session.execute(
                select(Job).where(Job.name == "sync-fail")
            )).scalar_one()
            job.sync_status = "sync_failed"
            job.error_message = "rsync: connection timed out"
            session.add(job)

        async with _session(factory) as session:
            job = (await session.execute(
                select(Job).where(Job.name == "sync-fail")
            )).scalar_one()
            assert job.sync_status == "sync_failed"
            assert "timed out" in job.error_message

        await engine.dispose()

    async def test_incremental_sync_accumulates_bytes_and_files(self):
        """Successive syncs should accumulate totals."""
        engine, factory = _make_session_factory()
        await _init_tables(engine)

        async with _session(factory) as session:
            session.add(_make_job(name="sync-incr"))

        # First sync
        async with _session(factory) as session:
            job = (await session.execute(
                select(Job).where(Job.name == "sync-incr")
            )).scalar_one()
            job.sync_status = "synced"
            job.synced_bytes = 500_000
            job.synced_files = 10
            job.last_sync_at = datetime(2026, 3, 20, 10, 0, 0)
            session.add(job)

        # Second sync adds more
        async with _session(factory) as session:
            job = (await session.execute(
                select(Job).where(Job.name == "sync-incr")
            )).scalar_one()
            job.synced_bytes += 300_000
            job.synced_files += 5
            job.last_sync_at = datetime(2026, 3, 20, 10, 15, 0)
            session.add(job)

        async with _session(factory) as session:
            job = (await session.execute(
                select(Job).where(Job.name == "sync-incr")
            )).scalar_one()
            assert job.synced_bytes == 800_000
            assert job.synced_files == 15
            assert job.last_sync_at == datetime(2026, 3, 20, 10, 15, 0)

        await engine.dispose()


# ---------------------------------------------------------------------------
# 6. Rollback on error
# ---------------------------------------------------------------------------

class TestRollbackBehavior:

    async def test_failed_session_does_not_persist(self):
        """If an exception occurs in the session, changes should roll back."""
        engine, factory = _make_session_factory()
        await _init_tables(engine)

        # Seed a job
        async with _session(factory) as session:
            session.add(_make_job(name="rollback-test"))

        # Try to update, but raise before commit
        with pytest.raises(RuntimeError):
            async with _session(factory) as session:
                job = (await session.execute(
                    select(Job).where(Job.name == "rollback-test")
                )).scalar_one()
                job.status = JobStatus.FAILED
                session.add(job)
                raise RuntimeError("simulated crash")

        # Original status should be intact
        async with _session(factory) as session:
            job = (await session.execute(
                select(Job).where(Job.name == "rollback-test")
            )).scalar_one()
            assert job.status == JobStatus.PENDING

        await engine.dispose()
