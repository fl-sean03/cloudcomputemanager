"""Checkpoint detectors for different application types.

Detectors are responsible for:
1. Finding checkpoint files on an instance
2. Determining which checkpoint is the latest/best
3. Extracting metadata about checkpoints (e.g., timestep for LAMMPS)
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CheckpointInfo:
    """Information about a detected checkpoint."""

    path: str
    size_bytes: int
    modified_at: datetime
    iteration: Optional[int] = None
    metadata: Optional[dict] = None


class CheckpointDetector(ABC):
    """Base class for checkpoint detectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Detector name."""
        ...

    @abstractmethod
    async def find_checkpoints(
        self,
        execute_fn,
        checkpoint_path: str = "/workspace",
    ) -> list[CheckpointInfo]:
        """Find checkpoints on the instance.

        Args:
            execute_fn: Async function to execute commands on instance
                       (instance_id, command) -> (exit_code, stdout, stderr)
            checkpoint_path: Path to search for checkpoints

        Returns:
            List of found checkpoints, sorted by recency
        """
        ...

    async def get_latest(
        self,
        execute_fn,
        checkpoint_path: str = "/workspace",
    ) -> Optional[CheckpointInfo]:
        """Get the latest checkpoint.

        Args:
            execute_fn: Async function to execute commands
            checkpoint_path: Path to search

        Returns:
            Latest checkpoint or None
        """
        checkpoints = await self.find_checkpoints(execute_fn, checkpoint_path)
        return checkpoints[0] if checkpoints else None


class LAMMPSDetector(CheckpointDetector):
    """Checkpoint detector for LAMMPS molecular dynamics simulations.

    LAMMPS writes restart files using the write_restart command:
    - restart N file1 file2  -> Writes every N steps to file1/file2
    - write_restart file     -> Writes single restart file

    Common patterns:
    - restart.*.bin (numbered restart files)
    - restart.bin (single restart)
    - *.restart
    """

    @property
    def name(self) -> str:
        return "lammps"

    async def find_checkpoints(
        self,
        execute_fn,
        checkpoint_path: str = "/workspace",
    ) -> list[CheckpointInfo]:
        """Find LAMMPS restart files."""
        checkpoints = []

        # Find all potential restart files
        patterns = ["restart.*.bin", "restart.bin", "*.restart"]
        find_cmd = " -o ".join([f'-name "{p}"' for p in patterns])
        cmd = f'find {checkpoint_path} \\( {find_cmd} \\) -type f 2>/dev/null'

        exit_code, stdout, _ = await execute_fn(cmd)
        if exit_code != 0 or not stdout.strip():
            logger.debug("No LAMMPS restart files found", path=checkpoint_path)
            return []

        # Get details for each file
        for filepath in stdout.strip().split("\n"):
            if not filepath:
                continue

            # Get file info
            stat_cmd = f'stat -c "%s %Y" "{filepath}" 2>/dev/null'
            exit_code, stat_out, _ = await execute_fn(stat_cmd)

            if exit_code != 0:
                continue

            try:
                size_str, mtime_str = stat_out.strip().split()
                size_bytes = int(size_str)
                modified_at = datetime.fromtimestamp(int(mtime_str))
            except (ValueError, IndexError):
                continue

            # Extract iteration from filename if present
            iteration = None
            match = re.search(r"restart\.(\d+)", filepath)
            if match:
                iteration = int(match.group(1))

            checkpoints.append(
                CheckpointInfo(
                    path=filepath,
                    size_bytes=size_bytes,
                    modified_at=modified_at,
                    iteration=iteration,
                    metadata={"type": "lammps_restart"},
                )
            )

        # Sort by iteration (if available) or modification time
        checkpoints.sort(
            key=lambda c: (c.iteration or 0, c.modified_at),
            reverse=True,
        )

        logger.info("Found LAMMPS checkpoints", count=len(checkpoints))
        return checkpoints

    async def get_current_timestep(
        self,
        execute_fn,
        log_path: str = "/workspace/log.lammps",
    ) -> Optional[int]:
        """Get current timestep from LAMMPS log file.

        Args:
            execute_fn: Function to execute commands
            log_path: Path to LAMMPS log file

        Returns:
            Current timestep or None
        """
        cmd = f'grep -E "^[[:space:]]+[0-9]+" "{log_path}" 2>/dev/null | tail -1 | awk \'{{print $1}}\''
        exit_code, stdout, _ = await execute_fn(cmd)

        if exit_code != 0 or not stdout.strip():
            return None

        try:
            return int(stdout.strip())
        except ValueError:
            return None

    async def get_progress(
        self,
        execute_fn,
        log_path: str = "/workspace/log.lammps",
        total_steps: Optional[int] = None,
    ) -> Optional[dict]:
        """Get simulation progress from LAMMPS log.

        Args:
            execute_fn: Function to execute commands
            log_path: Path to LAMMPS log file
            total_steps: Total steps in simulation (for percentage)

        Returns:
            Dict with progress info or None
        """
        current = await self.get_current_timestep(execute_fn, log_path)
        if current is None:
            return None

        result = {"current_step": current}

        if total_steps:
            result["total_steps"] = total_steps
            result["percent_complete"] = min(100.0, (current / total_steps) * 100)

        return result


class FilePatternDetector(CheckpointDetector):
    """Generic checkpoint detector using file patterns.

    Useful for applications that write checkpoint files with predictable
    naming patterns, such as:
    - PyTorch: model_*.pt, checkpoint_*.pth
    - TensorFlow: model.ckpt-*
    - Generic: *.ckpt, *.checkpoint
    """

    def __init__(
        self,
        patterns: Optional[list[str]] = None,
        iteration_regex: Optional[str] = None,
    ):
        """Initialize the detector.

        Args:
            patterns: File patterns to match (glob-style)
            iteration_regex: Regex to extract iteration from filename
        """
        self._patterns = patterns or ["*.ckpt", "*.pt", "*.pth", "checkpoint_*"]
        self._iteration_regex = iteration_regex or r"(\d+)"

    @property
    def name(self) -> str:
        return "file_pattern"

    async def find_checkpoints(
        self,
        execute_fn,
        checkpoint_path: str = "/workspace",
    ) -> list[CheckpointInfo]:
        """Find checkpoint files matching patterns."""
        checkpoints = []

        # Build find command
        find_parts = [f'-name "{p}"' for p in self._patterns]
        find_cmd = " -o ".join(find_parts)
        cmd = f'find {checkpoint_path} \\( {find_cmd} \\) -type f 2>/dev/null'

        exit_code, stdout, _ = await execute_fn(cmd)
        if exit_code != 0 or not stdout.strip():
            return []

        # Get details for each file
        for filepath in stdout.strip().split("\n"):
            if not filepath:
                continue

            stat_cmd = f'stat -c "%s %Y" "{filepath}" 2>/dev/null'
            exit_code, stat_out, _ = await execute_fn(stat_cmd)

            if exit_code != 0:
                continue

            try:
                size_str, mtime_str = stat_out.strip().split()
                size_bytes = int(size_str)
                modified_at = datetime.fromtimestamp(int(mtime_str))
            except (ValueError, IndexError):
                continue

            # Extract iteration
            iteration = None
            if self._iteration_regex:
                match = re.search(self._iteration_regex, Path(filepath).name)
                if match:
                    try:
                        iteration = int(match.group(1))
                    except (ValueError, IndexError):
                        pass

            checkpoints.append(
                CheckpointInfo(
                    path=filepath,
                    size_bytes=size_bytes,
                    modified_at=modified_at,
                    iteration=iteration,
                )
            )

        # Sort by iteration or modification time
        checkpoints.sort(
            key=lambda c: (c.iteration or 0, c.modified_at),
            reverse=True,
        )

        return checkpoints


# Registry of available detectors
DETECTORS: dict[str, type[CheckpointDetector]] = {
    "lammps": LAMMPSDetector,
    "file_pattern": FilePatternDetector,
}


def get_detector(name: str, **kwargs) -> CheckpointDetector:
    """Get a checkpoint detector by name.

    Args:
        name: Detector name
        **kwargs: Additional arguments for detector

    Returns:
        CheckpointDetector instance
    """
    if name not in DETECTORS:
        raise ValueError(f"Unknown detector: {name}. Available: {list(DETECTORS.keys())}")

    return DETECTORS[name](**kwargs)
