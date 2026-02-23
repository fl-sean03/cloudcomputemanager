"""CloudComputeManager: A robust GPU cloud management platform.

Features:
- Automatic checkpointing for spot instance recovery
- Continuous data synchronization
- Agent-native APIs for AI integration
- Built on SkyPilot for multi-cloud support

Quick Start for AI Agents:
    from cloudcomputemanager.agents import CloudComputeManagerAgent, JobSpec

    async with CloudComputeManagerAgent() as vm:
        job = await vm.submit(JobSpec(
            name="lammps-sim",
            command="mpirun lmp -in input.in",
        ))
        result = await vm.wait_for_completion(job.job_id)
"""

__version__ = "0.1.0"

from cloudcomputemanager.core.models import (
    Job,
    JobStatus,
    Instance,
    InstanceStatus,
    Checkpoint,
    CheckpointStrategy,
    SyncConfig,
    Resources,
)

# Re-export agent SDK for convenience
from cloudcomputemanager.agents import (
    CloudComputeManagerAgent,
    JobSpec,
    JobResult,
    AgentEvent,
    EventType,
)

# Re-export PackStore
from cloudcomputemanager.packstore import (
    PackageRegistry,
    PackageDeployer,
    DeploymentStrategy,
)

__all__ = [
    # Core models
    "Job",
    "JobStatus",
    "Instance",
    "InstanceStatus",
    "Checkpoint",
    "CheckpointStrategy",
    "SyncConfig",
    "Resources",
    # Agent SDK
    "CloudComputeManagerAgent",
    "JobSpec",
    "JobResult",
    "AgentEvent",
    "EventType",
    # PackStore
    "PackageRegistry",
    "PackageDeployer",
    "DeploymentStrategy",
    # Version
    "__version__",
]
