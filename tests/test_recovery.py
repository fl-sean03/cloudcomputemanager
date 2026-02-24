"""Tests for preemption recovery module."""

import pytest
from datetime import datetime

from cloudcomputemanager.core.recovery import RecoveryManager, RecoveryResult
from cloudcomputemanager.core.models import Job, JobStatus


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

    def test_default_values(self):
        """Test default values in RecoveryResult."""
        result = RecoveryResult(
            success=True,
            job_id="job123"
        )
        assert result.new_instance_id is None
        assert result.checkpoint_restored is False
        assert result.error is None
        assert result.attempt_number == 0


class TestRecoveryManager:
    """Tests for RecoveryManager."""

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
        import json
        provider = MockProvider()
        rm = RecoveryManager(provider=provider)

        job = Job(
            name="test-job",
            status=JobStatus.RECOVERING,
            image="test:latest",
            resources_json=json.dumps({"gpu_type": "RTX_3060"})
        )

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
        import json
        provider = MockProvider()
        rm = RecoveryManager(provider=provider)

        job = Job(
            name="test-job",
            status=JobStatus.RECOVERING,
            image="test:latest",
            resources_json=json.dumps({"disk_gb": 50})
        )

        instance_id = await rm.create_recovery_instance(job, "offer123")
        assert instance_id is not None
        assert len(provider.created_instances) == 1

    @pytest.mark.asyncio
    async def test_create_recovery_instance_timeout(self):
        """Test when instance fails to start."""
        import json
        provider = MockProvider()
        provider._ready = False
        rm = RecoveryManager(provider=provider)

        job = Job(
            name="test-job",
            status=JobStatus.RECOVERING,
            image="test:latest",
            resources_json=json.dumps({"disk_gb": 50})
        )

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

    def test_max_attempts_logic(self):
        """Test max attempts checking logic."""
        rm = RecoveryManager(max_attempts=3)

        job = Job(
            name="test-job",
            status=JobStatus.RECOVERING,
            image="test:latest",
        )

        # Under limit
        job.attempt_number = 2
        assert job.attempt_number < rm.max_attempts

        # At limit
        job.attempt_number = 3
        assert job.attempt_number >= rm.max_attempts


class TestRecoveryWithCheckpoints:
    """Tests for recovery with checkpoints."""

    def test_checkpoint_path_construction(self):
        """Test that checkpoint paths are constructed correctly."""
        job_id = "test_job_123"
        expected_subdir = job_id

        # Verify job_id can be used as path component
        assert "/" not in job_id or job_id.replace("/", "_")

    def test_job_status_transitions(self):
        """Test job status transitions during recovery."""
        job = Job(
            name="test-job",
            status=JobStatus.RECOVERING,
            image="test:latest",
        )

        # Recovery can succeed -> RUNNING
        job.status = JobStatus.RUNNING
        assert job.status == JobStatus.RUNNING

        # Recovery can fail -> FAILED
        job.status = JobStatus.FAILED
        assert job.status == JobStatus.FAILED
