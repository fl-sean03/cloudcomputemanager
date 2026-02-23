"""Agent integration for CloudComputeManager.

Provides tools for AI agents to programmatically manage GPU workloads.

Key classes:
- CloudComputeManagerAgent: High-level async interface for managing jobs
- JobSpec: Simplified job specification with smart defaults
- JobResult: Structured result object
- AgentEvent: Events for observability

Example:
    from cloudcomputemanager.agents import CloudComputeManagerAgent, JobSpec

    async with CloudComputeManagerAgent() as vm:
        job = await vm.submit(JobSpec(
            name="lammps-simulation",
            command="mpirun -np 4 lmp -in input.in",
            gpu_type="RTX_4090",
        ))
        result = await vm.wait_for_completion(job.job_id)
        print(f"Results at: {result.output_location}")
"""

from cloudcomputemanager.agents.sdk import (
    CloudComputeManagerAgent,
    JobSpec,
    JobResult,
    AgentEvent,
    EventType,
)

__all__ = [
    "CloudComputeManagerAgent",
    "JobSpec",
    "JobResult",
    "AgentEvent",
    "EventType",
]
