"""Tests for core data models."""

import json
from datetime import datetime

import pytest

from cloudcomputemanager.core.models import (
    Job,
    JobStatus,
    Instance,
    InstanceStatus,
    Checkpoint,
    CheckpointStrategy,
    CheckpointTrigger,
    Resources,
    Budget,
    RetryPolicy,
    CheckpointConfig,
    SyncConfig,
)


class TestResources:
    """Tests for Resources model."""

    def test_default_resources(self):
        """Test default resource values."""
        resources = Resources()

        assert resources.gpu_type == "RTX_4090"
        assert resources.gpu_count == 1
        assert resources.gpu_memory_min == 16
        assert resources.cpu_cores == 8
        assert resources.memory_gb == 32
        assert resources.disk_gb == 50

    def test_custom_resources(self):
        """Test custom resource values."""
        resources = Resources(
            gpu_type="A100",
            gpu_count=2,
            gpu_memory_min=40,
            disk_gb=200,
        )

        assert resources.gpu_type == "A100"
        assert resources.gpu_count == 2
        assert resources.gpu_memory_min == 40
        assert resources.disk_gb == 200


class TestBudget:
    """Tests for Budget model."""

    def test_default_budget(self):
        """Test default budget values."""
        budget = Budget()

        assert budget.max_cost_usd == 50.0
        assert budget.max_hours == 24
        assert budget.max_hourly_rate is None

    def test_custom_budget(self):
        """Test custom budget values."""
        budget = Budget(
            max_cost_usd=100.0,
            max_hours=48,
            max_hourly_rate=1.5,
        )

        assert budget.max_cost_usd == 100.0
        assert budget.max_hours == 48
        assert budget.max_hourly_rate == 1.5


class TestRetryPolicy:
    """Tests for RetryPolicy model."""

    def test_default_retry_policy(self):
        """Test default retry policy values."""
        policy = RetryPolicy()

        assert policy.max_attempts == 5
        assert policy.backoff_minutes == 5
        assert policy.max_backoff_minutes == 60
        assert 42 in policy.recover_on_exit_codes


class TestCheckpointConfig:
    """Tests for CheckpointConfig model."""

    def test_default_checkpoint_config(self):
        """Test default checkpoint config."""
        config = CheckpointConfig()

        assert config.strategy == CheckpointStrategy.APPLICATION
        assert config.interval_minutes == 30
        assert config.path == "/workspace/checkpoints"
        assert "*.bin" in config.patterns

    def test_custom_checkpoint_config(self):
        """Test custom checkpoint config."""
        config = CheckpointConfig(
            strategy=CheckpointStrategy.FILESYSTEM,
            interval_minutes=15,
            path="/data/checkpoints",
            patterns=["*.pt", "*.pth"],
        )

        assert config.strategy == CheckpointStrategy.FILESYSTEM
        assert config.interval_minutes == 15
        assert "*.pt" in config.patterns


class TestSyncConfig:
    """Tests for SyncConfig model."""

    def test_sync_config(self):
        """Test sync config creation."""
        config = SyncConfig(
            source="/workspace/results",
            destination="s3://bucket/results/",
        )

        assert config.enabled is True
        assert config.source == "/workspace/results"
        assert config.destination == "s3://bucket/results/"
        assert config.interval_minutes == 15


class TestJob:
    """Tests for Job model."""

    def test_job_creation(self):
        """Test job creation with defaults."""
        job = Job(
            name="test-job",
            image="python:3.11",
            command="python script.py",
        )

        assert job.name == "test-job"
        assert job.status == JobStatus.PENDING
        assert job.job_id.startswith("job_")
        assert job.attempt_number == 0

    def test_job_status_transitions(self):
        """Test job status values."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"

    def test_job_resources_property(self):
        """Test job resources JSON parsing."""
        resources = Resources(gpu_type="A100", gpu_count=2)
        job = Job(
            name="test",
            image="python:3.11",
            command="python",
            resources_json=resources.model_dump_json(),
        )

        parsed = job.get_resources()
        assert parsed.gpu_type == "A100"
        assert parsed.gpu_count == 2


class TestInstance:
    """Tests for Instance model."""

    def test_instance_creation(self):
        """Test instance creation."""
        instance = Instance(
            instance_id="12345678",
            gpu_type="RTX_4090",
            gpu_memory_gb=24,
            cpu_cores=8,
            memory_gb=32,
            disk_gb=100,
            ssh_host="ssh.vast.ai",
            ssh_port=22345,
            hourly_rate=0.50,
        )

        assert instance.instance_id == "12345678"
        assert instance.status == InstanceStatus.CREATING
        assert instance.gpu_type == "RTX_4090"
        assert instance.hourly_rate == 0.50


class TestCheckpoint:
    """Tests for Checkpoint model."""

    def test_checkpoint_creation(self):
        """Test checkpoint creation."""
        checkpoint = Checkpoint(
            job_id="job_abc123",
            strategy=CheckpointStrategy.APPLICATION,
            trigger=CheckpointTrigger.SCHEDULED,
            location="/checkpoints/job_abc123/restart.1000.bin",
            size_bytes=1024000,
        )

        assert checkpoint.checkpoint_id.startswith("ckpt_")
        assert checkpoint.job_id == "job_abc123"
        assert checkpoint.verified is False
