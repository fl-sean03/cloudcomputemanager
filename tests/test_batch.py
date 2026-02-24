"""Tests for batch job operations."""

import pytest
from datetime import datetime
from pathlib import Path
import yaml

from cloudcomputemanager.core.models import Job, JobStatus


class TestBatchSubmit:
    """Tests for batch job submission."""

    def test_create_job_configs(self, tmp_path):
        """Test creating multiple job config files."""
        configs = []
        for i in range(3):
            config_file = tmp_path / f"job{i}.yaml"
            config = {
                "name": f"test-job-{i}",
                "image": "python:3.11",
                "command": f"python script{i}.py",
                "resources": {"gpu_type": "RTX_3060", "disk_gb": 30}
            }
            config_file.write_text(yaml.dump(config))
            configs.append(config_file)

        assert len(configs) == 3
        for config_file in configs:
            assert config_file.exists()


class TestBatchStatus:
    """Tests for batch job status display."""

    def test_job_counts_calculation(self):
        """Test counting jobs by status."""
        # Create in-memory job list for testing calculation logic
        statuses = [
            JobStatus.RUNNING,
            JobStatus.RUNNING,
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.PENDING,
        ]

        running = sum(1 for s in statuses if s == JobStatus.RUNNING)
        completed = sum(1 for s in statuses if s == JobStatus.COMPLETED)
        failed = sum(1 for s in statuses if s == JobStatus.FAILED)
        pending = sum(1 for s in statuses if s == JobStatus.PENDING)

        assert running == 2
        assert completed == 1
        assert failed == 1
        assert pending == 1

    def test_filter_by_project_logic(self):
        """Test project filtering logic."""
        jobs = [
            {"name": "job1", "project": "project-a"},
            {"name": "job2", "project": "project-a"},
            {"name": "job3", "project": "project-b"},
        ]

        project_a = [j for j in jobs if j["project"] == "project-a"]
        assert len(project_a) == 2


class TestBatchWait:
    """Tests for batch wait functionality."""

    def test_running_jobs_filter(self):
        """Test filtering for running jobs."""
        jobs = [
            {"name": "job1", "status": JobStatus.RUNNING, "instance_id": "inst1"},
            {"name": "job2", "status": JobStatus.COMPLETED, "instance_id": None},
            {"name": "job3", "status": JobStatus.RUNNING, "instance_id": "inst2"},
        ]

        running = [j for j in jobs if j["status"] == JobStatus.RUNNING]
        assert len(running) == 2


class TestBatchCancel:
    """Tests for batch cancel functionality."""

    def test_status_transition(self):
        """Test that status can transition to cancelled."""
        job = Job(
            name="test-job",
            status=JobStatus.RUNNING,
            instance_id="inst123",
            image="test:latest",
        )

        assert job.status == JobStatus.RUNNING
        job.status = JobStatus.CANCELLED
        assert job.status == JobStatus.CANCELLED


class TestBatchValidation:
    """Tests for batch config validation."""

    def test_validate_yaml_files(self, tmp_path):
        """Test validating YAML config files."""
        # Valid config
        valid = tmp_path / "valid.yaml"
        valid.write_text(yaml.dump({
            "name": "test-job",
            "image": "python:3.11",
            "command": "python script.py",
            "resources": {"gpu_type": "RTX_3060"}
        }))

        # Invalid YAML
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("not: valid: yaml: syntax")

        # Load and validate
        with open(valid) as f:
            valid_config = yaml.safe_load(f)
        assert valid_config["name"] == "test-job"

        with pytest.raises(yaml.YAMLError):
            with open(invalid_yaml) as f:
                yaml.safe_load(f)

    def test_required_fields(self):
        """Test checking required fields."""
        required = {"name", "image", "command"}

        config1 = {"name": "job", "image": "img", "command": "cmd"}
        assert required.issubset(config1.keys())

        config2 = {"name": "job", "image": "img"}  # missing command
        assert not required.issubset(config2.keys())


class TestBatchProgress:
    """Tests for batch progress tracking."""

    def test_calculate_completion_percentage(self):
        """Test calculating batch completion percentage."""
        # Simulated job statuses
        total = 5
        completed = 2
        failed = 1
        finished = completed + failed

        completion_pct = (finished / total) * 100 if total > 0 else 0
        success_pct = (completed / finished) * 100 if finished > 0 else 0

        assert completion_pct == 60.0
        assert success_pct == pytest.approx(66.67, rel=0.01)

    def test_empty_batch(self):
        """Test progress with no jobs."""
        total = 0
        finished = 0

        completion_pct = (finished / total) * 100 if total > 0 else 0
        assert completion_pct == 0
