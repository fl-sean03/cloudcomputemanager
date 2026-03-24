"""Tests for pre-flight performance validation."""

import pytest
from unittest.mock import AsyncMock

from cloudcomputemanager.core.validation import (
    ValidationConfig,
    ValidationResult,
    extract_metric,
    validate_instance,
    LAMMPS_NS_DAY_PATTERN,
    NAMD_NS_DAY_PATTERN,
)


# ---------------------------------------------------------------------------
# 1. Validation skipped when not enabled
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validation_skipped_when_disabled():
    """When enabled=False, validation should pass without running anything."""
    config = ValidationConfig(enabled=False, command="echo hello")
    execute_fn = AsyncMock()

    result = await validate_instance(execute_fn, config)

    assert result.passed is True
    execute_fn.assert_not_awaited()


@pytest.mark.asyncio
async def test_validation_skipped_when_no_command():
    """When command is empty, validation should pass without running anything."""
    config = ValidationConfig(enabled=True, command="")
    execute_fn = AsyncMock()

    result = await validate_instance(execute_fn, config)

    assert result.passed is True
    execute_fn.assert_not_awaited()


# ---------------------------------------------------------------------------
# 2. Metric extraction from LAMMPS output
# ---------------------------------------------------------------------------

LAMMPS_OUTPUT = """\
Loop time of 5.23 on 1 procs for 1000 steps with 32000 atoms
Performance: 16.520 ns/day, 1.453 hours/ns, 191.205 timesteps/s
"""


def test_extract_lammps_metric():
    """Should extract ns/day from LAMMPS performance line."""
    value = extract_metric(LAMMPS_OUTPUT, LAMMPS_NS_DAY_PATTERN)
    assert value == pytest.approx(16.520)


# ---------------------------------------------------------------------------
# 3. Metric extraction from NAMD output
# ---------------------------------------------------------------------------

NAMD_OUTPUT = """\
TIMING: 100  1.234  2.345  0.023  0.012  0.001  12.456
"""


def test_extract_namd_metric():
    """Should extract performance metric from NAMD timing line."""
    value = extract_metric(NAMD_OUTPUT, NAMD_NS_DAY_PATTERN)
    assert value == pytest.approx(12.456)


# ---------------------------------------------------------------------------
# 4. Instance below threshold fails
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_instance_below_threshold_fails():
    """Instance with metric below min_threshold should fail validation."""
    config = ValidationConfig(
        enabled=True,
        command="run_benchmark",
        metric_pattern=LAMMPS_NS_DAY_PATTERN,
        metric_name="ns/day",
        min_threshold=20.0,
    )
    execute_fn = AsyncMock(return_value=(0, LAMMPS_OUTPUT, ""))

    result = await validate_instance(execute_fn, config)

    assert result.passed is False
    assert result.metric_value == pytest.approx(16.520)
    assert result.metric_name == "ns/day"
    assert result.threshold == 20.0


# ---------------------------------------------------------------------------
# 5. Instance above threshold passes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_instance_above_threshold_passes():
    """Instance with metric above min_threshold should pass validation."""
    config = ValidationConfig(
        enabled=True,
        command="run_benchmark",
        metric_pattern=LAMMPS_NS_DAY_PATTERN,
        metric_name="ns/day",
        min_threshold=10.0,
    )
    execute_fn = AsyncMock(return_value=(0, LAMMPS_OUTPUT, ""))

    result = await validate_instance(execute_fn, config)

    assert result.passed is True
    assert result.metric_value == pytest.approx(16.520)
    assert result.metric_name == "ns/day"
    assert result.threshold == 10.0


# ---------------------------------------------------------------------------
# 6. Handling of command failure (non-zero exit code)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_command_failure_nonzero_exit():
    """Non-zero exit code from the benchmark command should fail validation."""
    config = ValidationConfig(
        enabled=True,
        command="run_benchmark",
        metric_pattern=LAMMPS_NS_DAY_PATTERN,
        min_threshold=10.0,
    )
    execute_fn = AsyncMock(return_value=(1, "", "Segmentation fault"))

    result = await validate_instance(execute_fn, config)

    assert result.passed is False
    assert result.error is not None
    assert "exited with code 1" in result.error
    assert "Segmentation fault" in result.raw_output


@pytest.mark.asyncio
async def test_command_raises_exception():
    """If execute_fn raises an exception, validation should fail gracefully."""
    config = ValidationConfig(
        enabled=True,
        command="run_benchmark",
        min_threshold=10.0,
    )
    execute_fn = AsyncMock(side_effect=ConnectionError("SSH connection lost"))

    result = await validate_instance(execute_fn, config)

    assert result.passed is False
    assert result.error is not None
    assert "SSH connection lost" in result.error


# ---------------------------------------------------------------------------
# 7. Handling of unparseable output
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unparseable_output_with_threshold():
    """If metric cannot be extracted and a threshold is set, validation fails."""
    config = ValidationConfig(
        enabled=True,
        command="run_benchmark",
        metric_pattern=LAMMPS_NS_DAY_PATTERN,
        min_threshold=10.0,
    )
    garbage_output = "Some random output with no performance data\nDone."
    execute_fn = AsyncMock(return_value=(0, garbage_output, ""))

    result = await validate_instance(execute_fn, config)

    assert result.passed is False
    assert result.metric_value is None
    assert result.error is not None
    assert "Could not extract metric" in result.error


@pytest.mark.asyncio
async def test_unparseable_output_without_threshold():
    """If metric cannot be extracted but no threshold is set, validation passes."""
    config = ValidationConfig(
        enabled=True,
        command="run_benchmark",
        metric_pattern=LAMMPS_NS_DAY_PATTERN,
        min_threshold=0.0,
    )
    garbage_output = "Some random output with no performance data\nDone."
    execute_fn = AsyncMock(return_value=(0, garbage_output, ""))

    result = await validate_instance(execute_fn, config)

    assert result.passed is True
    assert result.metric_value is None


def test_extract_metric_no_match():
    """extract_metric should return None when pattern does not match."""
    value = extract_metric("no numbers here", r"([\d.]+)\s+ns/day")
    assert value is None
