"""Tests for Agent SDK."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from cloudcomputemanager.agents import (
    CloudComputeManagerAgent,
    JobSpec,
    JobResult,
    AgentEvent,
    EventType,
)


class TestJobSpec:
    """Tests for JobSpec model."""

    def test_minimal_spec(self):
        """Test creating spec with minimal args."""
        spec = JobSpec(
            name="test-job",
            command="echo hello",
        )
        assert spec.name == "test-job"
        assert spec.command == "echo hello"
        assert spec.gpu_type == "RTX_4090"
        assert spec.checkpoint_enabled is True

    def test_custom_spec(self):
        """Test creating spec with custom values."""
        spec = JobSpec(
            name="custom-job",
            command="mpirun lmp -in input.in",
            image="nvcr.io/hpc/lammps:latest",
            gpu_type="A100",
            gpu_count=4,
            disk_gb=200,
            checkpoint_interval_minutes=60,
            max_cost_usd=100.0,
            project="materials-sim",
            tags=["lammps", "production"],
        )
        assert spec.gpu_type == "A100"
        assert spec.gpu_count == 4
        assert spec.disk_gb == 200
        assert spec.project == "materials-sim"

    def test_to_job_config(self):
        """Test converting spec to job config dict."""
        spec = JobSpec(
            name="test",
            command="run.sh",
            gpu_type="RTX_4090",
        )
        config = spec.to_job_config()

        assert config["name"] == "test"
        assert config["command"] == "run.sh"
        assert config["resources"]["gpu_type"] == "RTX_4090"
        assert config["checkpoint"]["enabled"] is True
        assert config["sync"]["enabled"] is True


class TestJobResult:
    """Tests for JobResult model."""

    def test_successful_result(self):
        """Test creating a successful result."""
        result = JobResult(
            job_id="job-123",
            status="completed",
            success=True,
            output_location="/data/results/job-123",
            output_files=["output.dat", "log.lammps"],
            total_cost_usd=2.50,
            total_runtime_seconds=3600,
            exit_code=0,
        )
        assert result.success is True
        assert result.exit_code == 0
        assert len(result.output_files) == 2

    def test_failed_result(self):
        """Test creating a failed result."""
        result = JobResult(
            job_id="job-456",
            status="failed",
            success=False,
            error_message="Out of memory",
            exit_code=137,
        )
        assert result.success is False
        assert result.error_message == "Out of memory"


class TestAgentEvent:
    """Tests for AgentEvent model."""

    def test_event_creation(self):
        """Test creating events."""
        event = AgentEvent(
            type=EventType.JOB_STARTED,
            timestamp=datetime.utcnow(),
            job_id="job-789",
            message="Job started",
        )
        assert event.type == EventType.JOB_STARTED
        assert event.job_id == "job-789"

    def test_event_with_data(self):
        """Test event with data payload."""
        event = AgentEvent(
            type=EventType.JOB_CHECKPOINT,
            timestamp=datetime.utcnow(),
            job_id="job-123",
            data={"checkpoint_id": "chk-1", "size": 1024},
        )
        assert event.data["checkpoint_id"] == "chk-1"


class TestEventType:
    """Tests for EventType enum."""

    def test_all_event_types(self):
        """Test all event types are defined."""
        expected = [
            "JOB_SUBMITTED",
            "JOB_STARTED",
            "JOB_RUNNING",
            "JOB_CHECKPOINT",
            "JOB_SYNC",
            "JOB_PREEMPTED",
            "JOB_RECOVERING",
            "JOB_COMPLETED",
            "JOB_FAILED",
            "INSTANCE_CREATED",
            "INSTANCE_READY",
            "INSTANCE_PREEMPTED",
            "INSTANCE_TERMINATED",
            "PACKAGE_DEPLOYED",
            "PACKAGE_VERIFIED",
        ]
        actual = [e.name for e in EventType]
        for name in expected:
            assert name in actual


class TestCloudComputeManagerAgent:
    """Tests for CloudComputeManagerAgent class."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock provider."""
        provider = AsyncMock()
        provider.search_offers = AsyncMock(return_value=[
            MagicMock(
                offer_id="offer-1",
                gpu_type="RTX_4090",
                hourly_rate=0.50,
            )
        ])
        provider.create_instance = AsyncMock(return_value=MagicMock(
            instance_id="inst-1",
        ))
        provider.wait_for_ready = AsyncMock(return_value=True)
        provider.execute_command = AsyncMock(return_value=(0, "success", ""))
        provider.get_instance = AsyncMock(return_value=MagicMock(
            status=MagicMock(value="running"),
        ))
        return provider

    @pytest.mark.asyncio
    async def test_agent_context_manager(self):
        """Test agent as async context manager."""
        with patch("cloudcomputemanager.agents.sdk.VastProvider"):
            async with CloudComputeManagerAgent() as vm:
                assert vm._provider is not None

    @pytest.mark.asyncio
    async def test_event_handler_registration(self):
        """Test registering event handlers."""
        with patch("cloudcomputemanager.agents.sdk.VastProvider"):
            async with CloudComputeManagerAgent() as vm:
                events = []
                vm.on_event(lambda e: events.append(e))

                # Emit a test event
                vm._emit_event(AgentEvent(
                    type=EventType.JOB_STARTED,
                    timestamp=datetime.utcnow(),
                    job_id="test",
                ))

                assert len(events) == 1
                assert events[0].type == EventType.JOB_STARTED

    def test_search_gpus_params(self):
        """Test search_gpus method parameters."""
        # This tests the method signature/contract
        import inspect
        sig = inspect.signature(CloudComputeManagerAgent.search_gpus)
        params = list(sig.parameters.keys())

        assert "gpu_type" in params
        assert "max_price" in params
        assert "min_memory_gb" in params

    def test_deploy_packages_params(self):
        """Test deploy_packages method parameters."""
        import inspect
        sig = inspect.signature(CloudComputeManagerAgent.deploy_packages)
        params = list(sig.parameters.keys())

        assert "instance_id" in params
        assert "packages" in params
