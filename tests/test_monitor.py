"""Tests for job monitor daemon."""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from cloudcomputemanager.daemon.monitor import (
    JobMonitor,
    MonitorConfig,
    MonitorEvent,
    EventType,
)
from cloudcomputemanager.core.models import Job, JobStatus


class MockInstance:
    """Mock instance."""
    def __init__(self, instance_id, status="running"):
        self.instance_id = instance_id
        self._status = status

    @property
    def status(self):
        return MagicMock(value=self._status)


class MockProvider:
    """Mock VastProvider for testing."""

    def __init__(self):
        self.instances = {}
        self.execute_results = {}
        self.terminated = []
        self.synced = []

    async def get_instance(self, instance_id):
        return self.instances.get(str(instance_id))

    async def execute_command(self, instance_id, command, timeout=60):
        # Try exact match first, then prefix match
        key = (str(instance_id), command)
        if key in self.execute_results:
            return self.execute_results[key]
        # Try prefix matching
        for (inst, cmd), result in self.execute_results.items():
            if inst == str(instance_id) and command.startswith(cmd):
                return result
        return (0, "ok", "")

    async def terminate_instance(self, instance_id):
        self.terminated.append(instance_id)

    async def rsync_download(self, instance_id, source, dest, exclude=None):
        self.synced.append((instance_id, source, dest))
        return True


class TestMonitorConfig:
    """Tests for MonitorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MonitorConfig()
        assert config.poll_interval == 30
        assert config.sync_on_complete is True
        assert config.terminate_on_complete is True
        assert config.preemption_recovery is True
        assert config.max_recovery_attempts == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = MonitorConfig(
            poll_interval=60,
            sync_on_complete=False,
            terminate_on_complete=False,
        )
        assert config.poll_interval == 60
        assert config.sync_on_complete is False
        assert config.terminate_on_complete is False


class TestMonitorEvent:
    """Tests for MonitorEvent."""

    def test_event_creation(self):
        """Test creating monitor event."""
        event = MonitorEvent(
            event_type=EventType.JOB_COMPLETED,
            job_id="job123",
            instance_id="inst456",
            data={"exit_code": 0}
        )
        assert event.event_type == EventType.JOB_COMPLETED
        assert event.job_id == "job123"
        assert event.data["exit_code"] == 0
        assert event.timestamp is not None


class TestJobMonitor:
    """Tests for JobMonitor."""

    def test_init_defaults(self):
        """Test JobMonitor initialization."""
        monitor = JobMonitor()
        assert monitor.config.poll_interval == 30
        assert not monitor.is_running
        assert monitor.monitored_job_count == 0

    def test_init_custom_config(self):
        """Test JobMonitor with custom config."""
        config = MonitorConfig(poll_interval=60)
        monitor = JobMonitor(config=config)
        assert monitor.config.poll_interval == 60

    def test_event_handler_registration(self):
        """Test registering event handlers."""
        monitor = JobMonitor()
        handler = MagicMock()
        monitor.on_event(handler)
        assert handler in monitor._event_handlers

    def test_emit_event_calls_handlers(self):
        """Test that emit_event calls all handlers."""
        monitor = JobMonitor()
        handler1 = MagicMock()
        handler2 = MagicMock()
        monitor.on_event(handler1)
        monitor.on_event(handler2)

        event = MonitorEvent(
            event_type=EventType.JOB_STARTED,
            job_id="job123"
        )
        monitor._emit_event(event)

        handler1.assert_called_once_with(event)
        handler2.assert_called_once_with(event)

    def test_emit_event_handles_errors(self):
        """Test that handler errors don't stop other handlers."""
        monitor = JobMonitor()
        failing_handler = MagicMock(side_effect=Exception("test error"))
        success_handler = MagicMock()
        monitor.on_event(failing_handler)
        monitor.on_event(success_handler)

        event = MonitorEvent(
            event_type=EventType.JOB_STARTED,
            job_id="job123"
        )
        # Should not raise
        monitor._emit_event(event)
        success_handler.assert_called_once()


class TestJobCompletionDetection:
    """Tests for job completion detection."""

    @pytest.mark.asyncio
    async def test_check_job_running(self):
        """Test detecting running job."""
        provider = MockProvider()
        provider.execute_results[("inst123", "cat /workspace/.ccm_exit_code 2>/dev/null || echo 'running'")] = (0, "running", "")

        monitor = JobMonitor(provider=provider)
        completed, exit_code = await monitor.check_job_completion("inst123")

        assert not completed
        assert exit_code is None

    @pytest.mark.asyncio
    async def test_check_job_completed_success(self):
        """Test detecting completed job with success."""
        provider = MockProvider()
        provider.execute_results[("inst123", "cat /workspace/.ccm_exit_code 2>/dev/null || echo 'running'")] = (0, "0", "")

        monitor = JobMonitor(provider=provider)
        completed, exit_code = await monitor.check_job_completion("inst123")

        assert completed
        assert exit_code == 0

    @pytest.mark.asyncio
    async def test_check_job_completed_failed(self):
        """Test detecting completed job with failure."""
        provider = MockProvider()
        provider.execute_results[("inst123", "cat /workspace/.ccm_exit_code 2>/dev/null || echo 'running'")] = (0, "1", "")

        monitor = JobMonitor(provider=provider)
        completed, exit_code = await monitor.check_job_completion("inst123")

        assert completed
        assert exit_code == 1


class TestInstanceHealthCheck:
    """Tests for instance health checking."""

    @pytest.mark.asyncio
    async def test_healthy_instance(self):
        """Test detecting healthy instance."""
        provider = MockProvider()
        provider.instances["inst123"] = MockInstance("inst123", "running")
        provider.execute_results[("inst123", "echo ok")] = (0, "ok", "")

        monitor = JobMonitor(provider=provider)
        healthy, reason = await monitor.check_instance_health("inst123")

        assert healthy
        assert reason is None

    @pytest.mark.asyncio
    async def test_instance_not_found(self):
        """Test detecting non-existent instance."""
        provider = MockProvider()

        monitor = JobMonitor(provider=provider)
        healthy, reason = await monitor.check_instance_health("inst123")

        assert not healthy
        assert "not_found" in reason

    @pytest.mark.asyncio
    async def test_instance_terminated(self):
        """Test detecting terminated instance."""
        provider = MockProvider()
        provider.instances["inst123"] = MockInstance("inst123", "terminated")

        monitor = JobMonitor(provider=provider)
        healthy, reason = await monitor.check_instance_health("inst123")

        assert not healthy
        assert "terminated" in reason

    @pytest.mark.asyncio
    async def test_instance_ssh_failed(self):
        """Test detecting SSH failure."""
        provider = MockProvider()
        provider.instances["inst123"] = MockInstance("inst123", "running")
        provider.execute_results[("inst123", "echo ok")] = (1, "", "Connection refused")

        monitor = JobMonitor(provider=provider)
        healthy, reason = await monitor.check_instance_health("inst123")

        assert not healthy
        assert "ssh_failed" in reason


class TestMonitorLifecycle:
    """Tests for monitor start/stop."""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test starting and stopping monitor."""
        monitor = JobMonitor()

        with patch('cloudcomputemanager.daemon.monitor.init_db', new_callable=AsyncMock):
            await monitor.start()
            assert monitor.is_running

            await monitor.stop()
            assert not monitor.is_running

    @pytest.mark.asyncio
    async def test_double_start(self):
        """Test that double start is handled."""
        monitor = JobMonitor()

        with patch('cloudcomputemanager.daemon.monitor.init_db', new_callable=AsyncMock):
            await monitor.start()
            await monitor.start()  # Should not raise
            assert monitor.is_running

            await monitor.stop()


class TestJobCompletionHandling:
    """Tests for job completion handling."""

    @pytest.mark.asyncio
    async def test_handle_completion_emits_event(self):
        """Test that completion handling emits event."""
        provider = MockProvider()
        monitor = JobMonitor(provider=provider)

        events = []
        monitor.on_event(lambda e: events.append(e))

        job = Job(
            name="test-job",
            status=JobStatus.RUNNING,
            instance_id="inst123",
            image="test:latest",
        )

        config = MonitorConfig(sync_on_complete=False, terminate_on_complete=False)
        monitor.config = config

        # Mock the database session context manager
        with patch('cloudcomputemanager.daemon.monitor.get_session') as mock_session:
            mock_ctx = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_ctx.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))

            await monitor.handle_job_completion(job, exit_code=0)

        assert len(events) == 1
        assert events[0].event_type == EventType.JOB_COMPLETED
        assert events[0].data["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_handle_completion_terminates_instance(self):
        """Test that completion handling terminates instance."""
        provider = MockProvider()
        config = MonitorConfig(sync_on_complete=False, terminate_on_complete=True)
        monitor = JobMonitor(config=config, provider=provider)

        job = Job(
            name="test-job",
            status=JobStatus.RUNNING,
            instance_id="inst123",
            image="test:latest",
        )

        # Mock the database session context manager
        with patch('cloudcomputemanager.daemon.monitor.get_session') as mock_session:
            mock_ctx = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_ctx.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))

            await monitor.handle_job_completion(job, exit_code=0)

        assert "inst123" in provider.terminated


class TestPreemptionHandling:
    """Tests for preemption handling."""

    @pytest.mark.asyncio
    async def test_handle_preemption_emits_event(self):
        """Test that preemption handling emits event."""
        provider = MockProvider()
        monitor = JobMonitor(provider=provider)

        events = []
        monitor.on_event(lambda e: events.append(e))

        job = Job(
            name="test-job",
            status=JobStatus.RUNNING,
            instance_id="inst123",
            image="test:latest",
        )

        # Mock the database session context manager
        with patch('cloudcomputemanager.daemon.monitor.get_session') as mock_session:
            mock_ctx = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_ctx.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))

            await monitor.handle_preemption(job, "instance_terminated")

        assert len(events) == 1
        assert events[0].event_type == EventType.JOB_PREEMPTED
        assert events[0].data["reason"] == "instance_terminated"

    @pytest.mark.asyncio
    async def test_handle_preemption_sets_recovering(self):
        """Test that preemption updates job status."""
        provider = MockProvider()
        config = MonitorConfig(preemption_recovery=True, max_recovery_attempts=3)
        monitor = JobMonitor(config=config, provider=provider)

        job = Job(
            name="test-job",
            status=JobStatus.RUNNING,
            instance_id="inst123",
            image="test:latest",
        )
        job.attempt_number = 0

        # Create a mock job that will be returned by the query
        mock_db_job = MagicMock()
        mock_db_job.attempt_number = 0
        mock_db_job.status = JobStatus.RUNNING

        # Mock the database session context manager
        with patch('cloudcomputemanager.daemon.monitor.get_session') as mock_session:
            mock_ctx = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_ctx.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=mock_db_job)))
            mock_ctx.add = MagicMock()

            await monitor.handle_preemption(job, "instance_terminated")

        # Verify the job status was updated
        assert mock_db_job.status == JobStatus.RECOVERING
