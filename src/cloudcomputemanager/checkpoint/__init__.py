"""Checkpoint orchestration for CloudComputeManager.

Provides automatic checkpoint detection, creation, and restoration for
various workload types including LAMMPS molecular dynamics simulations.
Also provides restart adapters for resuming preempted jobs.
"""

from cloudcomputemanager.checkpoint.orchestrator import CheckpointOrchestrator
from cloudcomputemanager.checkpoint.detectors import (
    CheckpointDetector,
    LAMMPSDetector,
    FilePatternDetector,
)
from cloudcomputemanager.checkpoint.restart_adapters import (
    RestartAdapter,
    RestartResult,
    get_restart_adapter,
)

__all__ = [
    "CheckpointOrchestrator",
    "CheckpointDetector",
    "LAMMPSDetector",
    "FilePatternDetector",
    "RestartAdapter",
    "RestartResult",
    "get_restart_adapter",
]
