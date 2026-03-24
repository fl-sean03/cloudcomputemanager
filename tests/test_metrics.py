"""Tests for metrics parsing and tracking."""

import pytest
from cloudcomputemanager.core.models import JobMetrics
from cloudcomputemanager.cli.metrics import (
    parse_namd_timing,
    parse_lammps_timing,
    parse_pytorch_training,
    parse_generic_progress,
    parse_log_line,
)


class TestNAMDParser:
    """Test NAMD log parsing."""

    def test_parse_timing_line(self):
        """Parse standard NAMD timing line."""
        line = "TIMING: 1000000  CPU: 6912.3, 0.00691/step  Wall: 9715.2, 0.00972/step, 10.5 hours remaining, 0.000000 MB of memory in use."
        metrics = JobMetrics()

        result = parse_namd_timing(line, metrics)

        assert result is True
        assert metrics.current_step == 1000000
        assert metrics.seconds_per_step == pytest.approx(0.00972, rel=0.01)
        assert metrics.steps_per_second == pytest.approx(102.88, rel=0.01)
        assert metrics.estimated_hours_remaining == pytest.approx(10.5, rel=0.01)

    def test_ignore_non_timing_line(self):
        """Non-timing lines should be ignored."""
        line = "Info: NAMD 3.0.1 for Linux-x86_64-multicore-CUDA"
        metrics = JobMetrics()

        result = parse_namd_timing(line, metrics)

        assert result is False
        assert metrics.current_step is None


class TestLAMMPSParser:
    """Test LAMMPS log parsing."""

    def test_parse_step_line(self):
        """Parse LAMMPS step line."""
        line = "Step 500000 CPU = 1234.5, 0.00247 sec/step"
        metrics = JobMetrics()

        result = parse_lammps_timing(line, metrics)

        assert result is True
        assert metrics.current_step == 500000
        assert metrics.seconds_per_step == pytest.approx(0.00247, rel=0.01)

    def test_parse_performance_line(self):
        """Parse LAMMPS performance line."""
        line = "Performance: 12.34 ns/day, 1.94 hours/ns"
        metrics = JobMetrics()

        result = parse_lammps_timing(line, metrics)

        assert result is True
        assert "12.34 ns/day" in metrics.notes

    def test_parse_loop_time(self):
        """Parse LAMMPS loop time summary."""
        line = "Loop time of 1234.5 on 8 procs for 1000000 steps"
        metrics = JobMetrics()

        result = parse_lammps_timing(line, metrics)

        assert result is True
        assert metrics.total_steps == 1000000


class TestPyTorchParser:
    """Test PyTorch/ML training log parsing."""

    def test_parse_epoch_progress(self):
        """Parse epoch progress."""
        line = "Epoch 10/100, Step 500/1000, Loss: 0.0234"
        metrics = JobMetrics()

        result = parse_pytorch_training(line, metrics)

        assert result is True
        assert metrics.current_step == 500  # Step takes precedence
        assert metrics.total_steps == 1000
        assert metrics.progress_percent == pytest.approx(50.0, rel=0.01)
        assert "0.023400" in metrics.notes

    def test_parse_epoch_only(self):
        """Parse epoch without total."""
        line = "[Epoch 15] train_loss: 0.0123"
        metrics = JobMetrics()

        result = parse_pytorch_training(line, metrics)

        assert result is True
        assert metrics.current_step == 15
        assert "0.012300" in metrics.notes

    def test_parse_iteration_speed(self):
        """Parse iteration speed."""
        line = "Training: 100 it/sec, loss=0.05"
        metrics = JobMetrics()

        result = parse_pytorch_training(line, metrics)

        assert result is True
        assert metrics.steps_per_second == pytest.approx(100.0, rel=0.01)

    def test_parse_samples_speed(self):
        """Parse samples per second."""
        line = "Processed 1500 samples/sec"
        metrics = JobMetrics()

        result = parse_pytorch_training(line, metrics)

        assert result is True
        assert metrics.steps_per_second == pytest.approx(1500.0, rel=0.01)


class TestGenericParser:
    """Test generic progress parsing."""

    def test_parse_percentage(self):
        """Parse percentage progress."""
        line = "Progress: 75.5%"
        metrics = JobMetrics()

        result = parse_generic_progress(line, metrics)

        assert result is True
        assert metrics.progress_percent == pytest.approx(75.5, rel=0.01)

    def test_parse_fraction_brackets(self):
        """Parse fraction in brackets."""
        line = "Processing files [150/300]"
        metrics = JobMetrics()

        result = parse_generic_progress(line, metrics)

        assert result is True
        assert metrics.current_step == 150
        assert metrics.total_steps == 300
        assert metrics.progress_percent == pytest.approx(50.0, rel=0.01)

    def test_parse_fraction_parens(self):
        """Parse fraction in parentheses."""
        line = "Downloading (25/100)"
        metrics = JobMetrics()

        result = parse_generic_progress(line, metrics)

        assert result is True
        assert metrics.current_step == 25
        assert metrics.total_steps == 100

    def test_parse_completed_of(self):
        """Parse 'completed X of Y' pattern."""
        line = "Completed 45 of 90 tasks"
        metrics = JobMetrics()

        result = parse_generic_progress(line, metrics)

        assert result is True
        assert metrics.current_step == 45
        assert metrics.total_steps == 90
        assert metrics.progress_percent == pytest.approx(50.0, rel=0.01)

    def test_parse_eta_hours(self):
        """Parse ETA in hours."""
        line = "ETA: 2.5 hours"
        metrics = JobMetrics()

        result = parse_generic_progress(line, metrics)

        assert result is True
        assert metrics.estimated_hours_remaining == pytest.approx(2.5, rel=0.01)

    def test_parse_eta_minutes(self):
        """Parse ETA in minutes."""
        line = "ETA: 30 minutes"
        metrics = JobMetrics()

        result = parse_generic_progress(line, metrics)

        assert result is True
        assert metrics.estimated_hours_remaining == pytest.approx(0.5, rel=0.01)

    def test_parse_hours_remaining(self):
        """Parse 'X hours remaining' pattern."""
        line = "5.5 hours remaining"
        metrics = JobMetrics()

        result = parse_generic_progress(line, metrics)

        assert result is True
        assert metrics.estimated_hours_remaining == pytest.approx(5.5, rel=0.01)


class TestCombinedParsing:
    """Test the combined log line parser."""

    def test_namd_detected_first(self):
        """NAMD format should be detected."""
        line = "TIMING: 500000 CPU: 100, 0.001/step Wall: 200, 0.002/step, 5.0 hours remaining"
        metrics = JobMetrics()

        result = parse_log_line(line, metrics)

        assert result is True
        assert metrics.current_step == 500000

    def test_generic_fallback(self):
        """Generic parser should catch unrecognized formats."""
        line = "Task progress: 80% complete"
        metrics = JobMetrics()

        result = parse_log_line(line, metrics)

        assert result is True
        assert metrics.progress_percent == pytest.approx(80.0, rel=0.01)

    def test_no_match(self):
        """Lines with no patterns should return False."""
        line = "This is just a regular log message"
        metrics = JobMetrics()

        result = parse_log_line(line, metrics)

        assert result is False

    def test_multiple_lines(self):
        """Parse multiple log lines."""
        lines = [
            "Starting job...",
            "Epoch 5/10, loss=0.05",
            "Progress: 50%",
            "ETA: 2 hours",
        ]
        metrics = JobMetrics()

        for line in lines:
            parse_log_line(line, metrics)

        assert metrics.current_step == 5
        assert metrics.total_steps == 10
        assert metrics.progress_percent == pytest.approx(50.0, rel=0.01)
        assert metrics.estimated_hours_remaining == pytest.approx(2.0, rel=0.01)


class TestJobMetricsModel:
    """Test JobMetrics model."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = JobMetrics()

        assert metrics.is_healthy is True
        assert metrics.steps_per_second is None
        assert metrics.progress_percent is None

    def test_all_fields(self):
        """Test setting all fields."""
        metrics = JobMetrics(
            steps_per_second=100.0,
            seconds_per_step=0.01,
            current_step=5000,
            total_steps=10000,
            progress_percent=50.0,
            estimated_hours_remaining=2.5,
            output_size_mb=512.0,
            gpu_utilization=95.0,
            cpu_utilization=50.0,
            memory_usage_gb=16.0,
            is_healthy=True,
            notes="Running well",
        )

        assert metrics.steps_per_second == 100.0
        assert metrics.progress_percent == 50.0
        assert metrics.gpu_utilization == 95.0
