"""Tests for cleanup module."""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from cloudcomputemanager.core.cleanup import (
    find_stale_jobs,
    cleanup_stale_jobs,
    find_orphan_instances,
    cleanup_orphan_instances,
    get_cleanup_summary,
)
from cloudcomputemanager.core.models import Job, JobStatus


class MockInstance:
    """Mock Vast.ai instance."""
    def __init__(self, instance_id, label=None, status="running"):
        self.instance_id = str(instance_id)
        self.label = label


class MockProvider:
    """Mock VastProvider for testing."""

    def __init__(self, instances=None):
        self.instances = instances or []
        self.terminated = []

    async def get_instance(self, instance_id):
        for inst in self.instances:
            if inst.instance_id == str(instance_id):
                return inst
        return None

    async def list_instances(self):
        return self.instances

    async def terminate_instance(self, instance_id):
        self.terminated.append(instance_id)


class TestFindStaleJobs:
    """Tests for finding stale jobs."""

    @pytest_asyncio.fixture
    async def setup_db(self, test_db):
        """Setup test database."""
        yield

    @pytest.mark.asyncio
    async def test_find_job_with_dead_instance(self, setup_db, test_db):
        """Test finding job pointing to non-existent instance."""
        from cloudcomputemanager.core.database import get_session

        # Create job with instance that doesn't exist
        job = Job(
            name="test-job",
            status=JobStatus.RUNNING,
            instance_id="99999",
            image="test:latest",
        )
        async with get_session() as session:
            session.add(job)

        # Mock provider returns no instances
        provider = MockProvider([])

        stale = await find_stale_jobs(provider)
        assert len(stale) == 1
        assert stale[0][0].job_id == job.job_id
        assert "not_found" in stale[0][1]

    @pytest.mark.asyncio
    async def test_healthy_job_not_stale(self, setup_db, test_db):
        """Test that job with running instance is not stale."""
        from cloudcomputemanager.core.database import get_session

        job = Job(
            name="test-job",
            status=JobStatus.RUNNING,
            instance_id="12345",
            image="test:latest",
        )
        async with get_session() as session:
            session.add(job)

        # Mock provider returns the instance
        provider = MockProvider([MockInstance("12345")])

        stale = await find_stale_jobs(provider)
        assert len(stale) == 0


class TestCleanupStaleJobs:
    """Tests for cleaning up stale jobs."""

    @pytest_asyncio.fixture
    async def setup_db(self, test_db):
        """Setup test database."""
        yield

    @pytest.mark.asyncio
    async def test_cleanup_marks_as_failed(self, setup_db, test_db):
        """Test that cleanup marks stale jobs as failed."""
        from cloudcomputemanager.core.database import get_session
        from sqlmodel import select

        job = Job(
            name="test-job",
            status=JobStatus.RUNNING,
            instance_id="99999",
            image="test:latest",
        )
        async with get_session() as session:
            session.add(job)

        provider = MockProvider([])
        cleaned = await cleanup_stale_jobs(provider, dry_run=False)

        assert len(cleaned) == 1

        # Verify job status changed
        async with get_session() as session:
            result = await session.execute(
                select(Job).where(Job.job_id == job.job_id)
            )
            updated_job = result.scalar_one()
            assert updated_job.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_dry_run_doesnt_modify(self, setup_db, test_db):
        """Test that dry run doesn't modify jobs."""
        from cloudcomputemanager.core.database import get_session
        from sqlmodel import select

        job = Job(
            name="test-job",
            status=JobStatus.RUNNING,
            instance_id="99999",
            image="test:latest",
        )
        async with get_session() as session:
            session.add(job)

        provider = MockProvider([])
        cleaned = await cleanup_stale_jobs(provider, dry_run=True)

        assert len(cleaned) == 1

        # Verify job status unchanged
        async with get_session() as session:
            result = await session.execute(
                select(Job).where(Job.job_id == job.job_id)
            )
            unchanged_job = result.scalar_one()
            assert unchanged_job.status == JobStatus.RUNNING


class TestFindOrphanInstances:
    """Tests for finding orphan instances."""

    @pytest_asyncio.fixture
    async def setup_db(self, test_db):
        """Setup test database."""
        yield

    @pytest.mark.asyncio
    async def test_find_instance_without_job(self, setup_db, test_db):
        """Test finding instance with no matching job."""
        provider = MockProvider([
            MockInstance("12345", "orphan-instance")
        ])

        orphans = await find_orphan_instances(provider)
        assert len(orphans) == 1
        assert orphans[0].instance_id == "12345"

    @pytest.mark.asyncio
    async def test_instance_with_job_not_orphan(self, setup_db, test_db):
        """Test that instance with job is not orphan."""
        from cloudcomputemanager.core.database import get_session

        # Create job with instance
        job = Job(
            name="test-job",
            status=JobStatus.RUNNING,
            instance_id="12345",
            image="test:latest",
        )
        async with get_session() as session:
            session.add(job)

        provider = MockProvider([MockInstance("12345")])

        orphans = await find_orphan_instances(provider)
        assert len(orphans) == 0


class TestCleanupOrphanInstances:
    """Tests for cleaning up orphan instances."""

    @pytest_asyncio.fixture
    async def setup_db(self, test_db):
        """Setup test database."""
        yield

    @pytest.mark.asyncio
    async def test_cleanup_terminates_orphans(self, setup_db, test_db):
        """Test that cleanup terminates orphan instances."""
        provider = MockProvider([
            MockInstance("12345", "orphan-1"),
            MockInstance("67890", "orphan-2"),
        ])

        terminated = await cleanup_orphan_instances(provider, dry_run=False)

        assert len(terminated) == 2
        assert "12345" in provider.terminated
        assert "67890" in provider.terminated

    @pytest.mark.asyncio
    async def test_dry_run_doesnt_terminate(self, setup_db, test_db):
        """Test that dry run doesn't terminate instances."""
        provider = MockProvider([MockInstance("12345", "orphan")])

        terminated = await cleanup_orphan_instances(provider, dry_run=True)

        assert len(terminated) == 1
        assert len(provider.terminated) == 0


class TestGetCleanupSummary:
    """Tests for cleanup summary."""

    @pytest_asyncio.fixture
    async def setup_db(self, test_db):
        """Setup test database."""
        yield

    @pytest.mark.asyncio
    async def test_summary_structure(self, setup_db, test_db):
        """Test that summary has expected structure."""
        provider = MockProvider([])
        summary = await get_cleanup_summary(provider)

        assert "stale_jobs" in summary
        assert "orphan_instances" in summary
        assert "stale_job_reasons" in summary
        assert isinstance(summary["stale_jobs"], int)
        assert isinstance(summary["orphan_instances"], int)
        assert isinstance(summary["stale_job_reasons"], dict)
