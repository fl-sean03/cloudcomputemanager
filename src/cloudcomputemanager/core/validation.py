"""Pre-flight performance validation for GPU instances.

Run a short benchmark on a new instance before committing to a full job.
Reject instances that don't meet performance thresholds.
"""

import re
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of a pre-flight validation."""
    passed: bool
    metric_value: Optional[float] = None
    metric_name: Optional[str] = None
    threshold: Optional[float] = None
    raw_output: str = ""
    error: Optional[str] = None


@dataclass
class ValidationConfig:
    """Configuration for pre-flight validation."""
    enabled: bool = False
    command: str = ""  # Short benchmark command
    timeout_seconds: int = 120
    metric_pattern: str = ""  # Regex with a capture group for the numeric value
    metric_name: str = "performance"
    min_threshold: float = 0.0  # Reject instances below this


# Built-in metric extractors
LAMMPS_NS_DAY_PATTERN = r"Performance:\s+([\d.]+)\s+ns/day"
NAMD_NS_DAY_PATTERN = r"TIMING:\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+"
PYTORCH_SAMPLES_SEC_PATTERN = r"([\d.]+)\s+samples/sec"


def extract_metric(output: str, pattern: str) -> Optional[float]:
    """Extract a numeric metric from command output using a regex pattern."""
    match = re.search(pattern, output)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            return None
    return None


async def validate_instance(
    execute_fn: Callable[[str, int], Awaitable[tuple[int, str, str]]],
    config: ValidationConfig,
) -> ValidationResult:
    """Run pre-flight validation on an instance.

    Args:
        execute_fn: async function(command, timeout) -> (exit_code, stdout, stderr)
        config: Validation configuration

    Returns:
        ValidationResult with pass/fail and metrics
    """
    if not config.enabled or not config.command:
        return ValidationResult(passed=True)

    logger.info(
        "Running pre-flight validation",
        command=config.command[:80],
        timeout=config.timeout_seconds,
    )

    try:
        exit_code, stdout, stderr = await execute_fn(config.command, config.timeout_seconds)
    except Exception as e:
        return ValidationResult(
            passed=False,
            error=f"Validation command failed: {e}",
        )

    if exit_code != 0:
        return ValidationResult(
            passed=False,
            raw_output=stdout + stderr,
            error=f"Validation command exited with code {exit_code}",
        )

    combined_output = stdout + "\n" + stderr

    # Extract metric
    if config.metric_pattern:
        metric_value = extract_metric(combined_output, config.metric_pattern)
    else:
        metric_value = None

    if metric_value is None and config.min_threshold > 0:
        return ValidationResult(
            passed=False,
            raw_output=combined_output,
            error=f"Could not extract metric using pattern: {config.metric_pattern}",
        )

    # Check threshold
    if config.min_threshold > 0 and metric_value is not None:
        passed = metric_value >= config.min_threshold
        if not passed:
            logger.warning(
                "Instance failed validation",
                metric=config.metric_name,
                value=metric_value,
                threshold=config.min_threshold,
            )
        else:
            logger.info(
                "Instance passed validation",
                metric=config.metric_name,
                value=metric_value,
                threshold=config.min_threshold,
            )
    else:
        passed = True

    return ValidationResult(
        passed=passed,
        metric_value=metric_value,
        metric_name=config.metric_name,
        threshold=config.min_threshold,
        raw_output=combined_output,
    )
