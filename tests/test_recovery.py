"""Tests for preemption recovery module."""

import pytest
import pytest_asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from cloudcomputemanager.core.recovery import RecoveryManager, RecoveryResult
from cloudcomputemanager.core.models import Job, JobStatus, Checkpoint


class MockOffer:
    """Mock GPU offer."""
    def __init__(self, offer_id="offer123"):
        self.offer_id = offer_id
        self.gpu_type = "RTX_3060"
        self.hourly_rate = 0.05


class MockInstance:
    """Mock instance."""
    def __init__(self, instance_id="inst123"):
        self.instance_id = instance_id


class MockProvider:
    """Mock VastProvider for testing."""

    def __init__(self):
        self.offers = [MockOffer()]
        self.created_instances = []
        self.executed_commands = []
        self.uploaded_files = []
        self.terminated = []
        self._ready = True
        self._execute_return = (0, "", "")

    async def search_offers(self, **kwargs):
        return self.offers

    async def create_instance(self, **kwargs):
        inst = MockInstance(f"inst_{len(self.created_instances)}")
        self.created_instances.append(inst)
        return inst

    async def wait_for_ready(self, instance_id, timeout=300):
        return self._ready

    async def execute_command(self, instance_id, command, timeout=60):
        self.executed_commands.append((instance_id, command))
        return self._execute_return

    async def rsync_upload(self, instance_id, source, dest, exclude=None):
        self.uploaded_files.append((instance_id, source, dest))
        return True

    async def terminate_instance(self, instance_id):
        self.terminated.append(instance_id)


class TestRecoveryResult:
    """Tests for RecoveryResult dataclass."""

    def test_success_result(self):
        """Test successful recovery result."""
        result = RecoveryResult(
            success=True,
            job_id="job123",
            new_instance_id="inst456",
            checkpoint_restored=True,
            attempt_number=1
        )
        assert result.success
        assert result.new_instance_id == "inst456"
        assert result.checkpoint_restored

    def test_failure_result(self):
        """Test failed recovery result."""
        result = RecoveryResult(
            success=False,
            job_id="job123",
            error="No offers found",
            attempt_number=2
        )
        assert not result.success
        assert result.error == "No offers found"


class TestRecoveryManager:
    """Tests for RecoveryManager."""

    @pytest_asyncio.fixture
    async def setup_db(self, test_db):
        """Setup test database."""
        yield

    def test_init_defaults(self):
        """Test RecoveryManager initialization."""
        rm = RecoveryManager()
        assert rm.max_attempts == 5
        assert rm.backoff_minutes == 5

    def test_init_custom(self):
        """Test RecoveryManager with custom config."""
        rm = RecoveryManager(max_attempts=3, backoff_minutes=10)
        assert rm.max_attempts == 3
        assert rm.backoff_minutes == 10

    @pytest.mark.asyncio
    async def test_find_recovery_instance_success(self):
        """Test finding recovery instance."""
        provider = MockProvider()
        rm = RecoveryManager(provider=provider)

        job = Job(
            name="test-job",
            status=JobStatus.RECOVERING,
            image="test:latest",
        )
        job.resources = {"gpu_type": "RTX_3060"}

        offer_id = await rm.find_recovery_instance(job)
        assert offer_id == "offer123"

    @pytest.mark.asyncio
    async def test_find_recovery_instance_no_offers(self):
        """Test when no offers available."""
        provider = MockProvider()
        provider.offers = []
        rm = RecoveryManager(provider=provider)

        job = Job(
            name="test-job",
            status=JobStatus.RECOVERING,
            image="test:latest",
        )

        offer_id = await rm.find_recovery_instance(job)
        assert offer_id is None

    @pytest.mark.asyncio
    async def test_create_recovery_instance_success(self):
        """Test creating recovery instance."""
        provider = MockProvider()
        rm = RecoveryManager(provider=provider)

        job = Job(
            name="test-job",
            status=JobStatus.RECOVERING,
            image="test:latest",
        )
        job.resources = {"disk_gb": 50}

        instance_id = await rm.create_recovery_instance(job, "offer123")
        assert instance_id is not None
        assert len(provider.created_instances) == 1

    @pytest.mark.asyncio
    async def test_create_recovery_instance_timeout(self):
        """Test when instance fails to start."""
        provider = MockProvider()
        provider._ready = False
        rm = RecoveryManager(provider=provider)

        job = Job(
            name="test-job",
            status=JobStatus.RECOVERING,
            image="test:latest",
        )
        job.resources = {"disk_gb": 50}

        instance_id = await rm.create_recovery_instance(job, "offer123")
        assert instance_id is None
        # Should terminate failed instance
        assert len(provider.terminated) == 1

    @pytest.mark.asyncio
    async def test_start_recovered_job_success(self):
        """Test starting recovered job."""
        provider = MockProvider()
        rm = RecoveryManager(provider=provider)

        job = Job(
            name="test-job",
            status=JobStatus.RECOVERING,
            command="python train.py",
            image="test:latest",
        )

        success = await rm.start_recovered_job(job, "inst123", has_checkpoint=False)
        assert success
        # Should have setup script and run commands
        assert len(provider.executed_commands) >= 2

    @pytest.mark.asyncio
    async def test_start_recovered_job_with_checkpoint(self):
        """Test starting recovered job with checkpoint marker."""
        provider = MockProvider()
        rm = RecoveryManager(provider=provider)

        job = Job(
            name="test-job",
            status=JobStatus.RECOVERING,
            command="python train.py",
            image="test:latest",
        )

        success = await rm.start_recovered_job(job, "inst123", has_checkpoint=True)
        assert success
        # Should include checkpoint marker command
        checkpoint_cmd = any("checkpoint_marker" in cmd for _, cmd in provider.executed_commands)
        assert checkpoint_cmd

    @pytest.mark.asyncio
    async def test_recover_job_max_attempts_exceeded(self, setup_db, test_db):
        """Test recovery fails when max attempts exceeded."""
        provider = MockProvider()
        rm = RecoveryManager(provider=provider, max_attempts=3)

        job = Job(
            name="test-job",
            status=JobStatus.RECOVERING,
            image="test:latest",
        )
        job.attempt_number = 3

        result = await rm.recover_job(job)
        assert not result.success
        assert "max attempts" in result.error.lower()

    @pytest.mark.asyncio
    async def test_recover_job_no_offers(self, setup_db, test_db):
        """Test recovery fails when no offers available."""
        provider = MockProvider()
        provider.offers = []
        rm = RecoveryManager(provider=provider)

        job = Job(
            name="test-job",
            status=JobStatus.RECOVERING,
            image="test:latest",
        )

        result = await rm.recover_job(job)
        assert not result.success
        assert "offers" in result.error.lower()


class TestRecoveryWithCheckpoints:
    """Tests for recovery with checkpoints."""

    @pytest_asyncio.fixture
    async def setup_db(self, test_db):
        """Setup test database."""
        yield

    @pytest.mark.asyncio
    async def test_get_latest_checkpoint(self, setup_db, test_db):
        """Test getting latest checkpoint for job."""
        from cloudcomputemanager.core.database import get_session

        job = Job(
            name="test-job",
            status=JobStatus.RECOVERING,
            image="test:latest",
        )
        async with get_session() as session:
            session.add(job)

        # Create checkpoints
        ckpt1 = Checkpoint(
            job_id=job.job_id,
            path="/workspace/checkpoint1",
            verified=True,
            created_at=datetime(2024, 1, 1, 10, 0, 0)
        )
        ckpt2 = Checkpoint(
            job_id=job.job_id,
            path="/workspace/checkpoint2",
            verified=True,
            created_at=datetime(2024, 1, 1, 12, 0, 0)  # Later
        )
        async with get_session() as session:
            session.add(ckpt1)
            session.add(ckpt2)

        rm = RecoveryManager()
        latest = await rm.get_latest_checkpoint(job.job_id)

        assert latest is not None
        assert latest.path == "/workspace/checkpoint2"

    @pytest.mark.asyncio
    async def test_get_checkpoint_only_verified(self, setup_db, test_db):
        """Test that only verified checkpoints are returned."""
        from cloudcomputemanager.core.database import get_session

        job = Job(
            name="test-job",
            status=JobStatus.RECOVERING,
            image="test:latest",
        )
        async with get_session() as session:
            session.add(job)

        # Create unverified checkpoint
        ckpt = Checkpoint(
            job_id=job.job_id,
            path="/workspace/checkpoint",
            verified=False,
        )
        async with get_session() as session:
            session.add(ckpt)

        rm = RecoveryManager()
        latest = await rm.get_latest_checkpoint(job.job_id)

        assert latest is None
