"""Core data models for CloudComputeManager.

These models define the structure for jobs, instances, checkpoints, and sync operations.
All models use Pydantic for validation and SQLModel for database integration.
"""

import json as _json
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from sqlmodel import SQLModel, Field


# ============================================================================
# Enums
# ============================================================================


class JobStatus(str, Enum):
    """Status of a managed job."""

    PENDING = "pending"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class InstanceStatus(str, Enum):
    """Status of a GPU instance."""

    CREATING = "creating"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    ERROR = "error"


class CheckpointStrategy(str, Enum):
    """Strategy for checkpointing workloads."""

    APPLICATION = "application"  # App handles its own checkpoints (e.g., LAMMPS restart)
    FILESYSTEM = "filesystem"  # Periodic filesystem-level sync
    CRIU = "criu"  # CRIU transparent checkpoint (future)
    HYBRID = "hybrid"  # Combination of app + filesystem


class CheckpointTrigger(str, Enum):
    """What triggered a checkpoint operation."""

    SCHEDULED = "scheduled"  # Regular interval
    PREEMPTION = "preemption"  # Spot instance about to be terminated
    MANUAL = "manual"  # User-requested
    SIGNAL = "signal"  # Application signaled checkpoint ready


class SyncStatus(str, Enum):
    """Status of a sync operation."""

    IDLE = "idle"
    SYNCING = "syncing"
    COMPLETED = "completed"
    FAILED = "failed"


class RentalType(str, Enum):
    """Type of instance rental."""

    INTERRUPTIBLE = "interruptible"  # Spot instance
    ON_DEMAND = "on_demand"  # Reserved instance


class Priority(str, Enum):
    """Job priority level."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# Value Objects (not persisted, used for configuration)
# ============================================================================


class Resources(SQLModel):
    """Resource requirements for a job."""

    gpu_type: str = Field(default="RTX_4090", description="GPU model name")
    gpu_count: int = Field(default=1, ge=1, le=8, description="Number of GPUs")
    gpu_memory_min: int = Field(default=16, ge=4, description="Minimum GPU VRAM in GB")
    cpu_cores: int = Field(default=8, ge=1, description="Number of CPU cores")
    memory_gb: int = Field(default=32, ge=4, description="System RAM in GB")
    disk_gb: int = Field(default=50, ge=10, description="Disk space in GB")


class Budget(SQLModel):
    """Budget constraints for a job."""

    max_cost_usd: float = Field(default=50.0, gt=0, description="Maximum total cost in USD")
    max_hours: int = Field(default=24, gt=0, description="Maximum runtime in hours")
    max_hourly_rate: Optional[float] = Field(
        default=None, gt=0, description="Maximum hourly rate in USD"
    )


class RetryPolicy(SQLModel):
    """Retry policy for failed jobs."""

    max_attempts: int = Field(default=5, ge=1, le=100, description="Maximum retry attempts")
    backoff_minutes: int = Field(default=5, ge=1, description="Initial backoff in minutes")
    max_backoff_minutes: int = Field(default=60, ge=1, description="Maximum backoff in minutes")
    recover_on_exit_codes: list[int] = Field(
        default_factory=lambda: [42],
        description="Exit codes that trigger recovery instead of failure",
    )


class CheckpointConfig(SQLModel):
    """Configuration for checkpoint behavior."""

    strategy: CheckpointStrategy = Field(
        default=CheckpointStrategy.APPLICATION, description="Checkpoint strategy"
    )
    interval_minutes: int = Field(default=30, ge=5, description="Checkpoint interval in minutes")
    path: str = Field(default="/workspace/checkpoints", description="Checkpoint directory path")
    signal: Optional[str] = Field(
        default=None, description="Signal to send to application for checkpoint (e.g., SIGUSR1)"
    )
    pre_command: Optional[str] = Field(
        default=None, description="Command to run before checkpointing"
    )
    verify_command: Optional[str] = Field(
        default=None, description="Command to verify checkpoint integrity"
    )
    patterns: list[str] = Field(
        default_factory=lambda: ["*.bin", "*.restart", "*.pt", "*.ckpt"],
        description="File patterns to include in checkpoint",
    )


class SyncConfig(SQLModel):
    """Configuration for data synchronization."""

    enabled: bool = Field(default=True, description="Enable continuous sync")
    source: str = Field(default="/workspace/results", description="Source directory on instance")
    destination: str = Field(description="Destination URL (s3://, local path, etc.)")
    interval_minutes: int = Field(default=15, ge=1, description="Sync interval in minutes")
    include_patterns: list[str] = Field(
        default_factory=lambda: ["*.dump", "*.log", "*.dat", "*.out"],
        description="File patterns to include",
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: ["*.tmp", "*.temp", "__pycache__"],
        description="File patterns to exclude",
    )
    delete_remote: bool = Field(
        default=False, description="Delete files at destination not in source"
    )


class InputData(SQLModel):
    """Input data configuration for a job."""

    source: str = Field(description="Source URL or path")
    destination: str = Field(
        default="/workspace/", description="Destination path on instance"
    )


# ============================================================================
# Database Models (persisted)
# ============================================================================


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid4())[:8]


class Job(SQLModel, table=True):
    """A managed computational job."""

    __tablename__ = "jobs"

    id: int | None = Field(default=None, primary_key=True)
    job_id: str = Field(default_factory=lambda: f"job_{generate_id()}", unique=True, index=True)
    name: str = Field(index=True, description="Human-readable job name")
    project: Optional[str] = Field(default=None, index=True, description="Project grouping")
    status: JobStatus = Field(default=JobStatus.PENDING, index=True)

    # Container configuration
    image: str = Field(description="Docker image to use")
    command: str = Field(description="Command to run")
    environment_json: str = Field(default="{}", description="JSON-encoded environment variables")

    # Resource requirements (stored as JSON)
    resources_json: str = Field(default="{}", description="JSON-encoded Resources")
    checkpoint_json: str = Field(default="{}", description="JSON-encoded CheckpointConfig")
    sync_json: str = Field(default="{}", description="JSON-encoded SyncConfig")
    budget_json: str = Field(default="{}", description="JSON-encoded Budget")
    retry_json: str = Field(default="{}", description="JSON-encoded RetryPolicy")
    input_data_json: str = Field(default="[]", description="JSON-encoded list of InputData")

    # Metadata
    priority: Priority = Field(default=Priority.NORMAL)
    tags_json: str = Field(default="[]", description="JSON-encoded tags list")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)

    # Execution state
    instance_id: Optional[str] = Field(default=None, index=True)
    attempt_number: int = Field(default=0)
    last_checkpoint_id: Optional[str] = Field(default=None)

    # Results
    exit_code: Optional[int] = Field(default=None)
    output_location: Optional[str] = Field(default=None)
    total_cost_usd: float = Field(default=0.0)
    total_runtime_seconds: int = Field(default=0)
    error_message: Optional[str] = Field(default=None)

    def get_resources(self) -> Resources:
        """Parse resources from JSON."""
        return Resources.model_validate(_json.loads(self.resources_json))

    def get_checkpoint_config(self) -> CheckpointConfig:
        """Parse checkpoint config from JSON."""
        return CheckpointConfig.model_validate(_json.loads(self.checkpoint_json))

    def get_sync_config(self) -> Optional[SyncConfig]:
        """Parse sync config from JSON."""
        data = _json.loads(self.sync_json)
        return SyncConfig.model_validate(data) if data else None

    def get_budget(self) -> Budget:
        """Parse budget from JSON."""
        return Budget.model_validate(_json.loads(self.budget_json))

    def get_retry_policy(self) -> RetryPolicy:
        """Parse retry policy from JSON."""
        return RetryPolicy.model_validate(_json.loads(self.retry_json))

    def get_environment(self) -> dict:
        """Parse environment from JSON."""
        return _json.loads(self.environment_json)

    def get_tags(self) -> list[str]:
        """Parse tags from JSON."""
        return _json.loads(self.tags_json)

    @property
    def resources(self) -> dict:
        """Get resources as dictionary."""
        return _json.loads(self.resources_json)

    @property
    def budget(self) -> dict:
        """Get budget as dictionary."""
        return _json.loads(self.budget_json)

    @property
    def sync_config(self) -> Optional[dict]:
        """Get sync config as dictionary."""
        data = _json.loads(self.sync_json)
        return data if data else None


class Instance(SQLModel, table=True):
    """A GPU instance managed by CloudComputeManager."""

    __tablename__ = "instances"

    id: int | None = Field(default=None, primary_key=True)
    instance_id: str = Field(unique=True, index=True, description="Provider instance ID")
    provider: str = Field(default="vast", description="Cloud provider name")
    status: InstanceStatus = Field(default=InstanceStatus.CREATING, index=True)

    # Hardware specs
    gpu_type: str = Field(description="GPU model name")
    gpu_count: int = Field(default=1)
    gpu_memory_gb: int = Field(description="GPU VRAM in GB")
    cpu_cores: int = Field(description="Number of CPU cores")
    memory_gb: int = Field(description="System RAM in GB")
    disk_gb: int = Field(description="Disk space in GB")

    # Networking
    ssh_host: str = Field(description="SSH hostname")
    ssh_port: int = Field(description="SSH port")
    ssh_user: str = Field(default="root", description="SSH username")
    jupyter_url: Optional[str] = Field(default=None, description="Jupyter notebook URL")

    # Pricing
    rental_type: RentalType = Field(default=RentalType.INTERRUPTIBLE)
    hourly_rate: float = Field(description="Hourly cost in USD")
    current_bid: Optional[float] = Field(default=None, description="Current bid for spot")

    # Association
    job_id: Optional[str] = Field(default=None, index=True)

    # Health
    last_health_check: Optional[datetime] = Field(default=None)
    health_status: str = Field(default="unknown")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(default=None)
    terminated_at: Optional[datetime] = Field(default=None)


class Checkpoint(SQLModel, table=True):
    """A checkpoint record for a job."""

    __tablename__ = "checkpoints"

    id: int | None = Field(default=None, primary_key=True)
    checkpoint_id: str = Field(
        default_factory=lambda: f"ckpt_{generate_id()}", unique=True, index=True
    )
    job_id: str = Field(index=True, description="Associated job ID")
    instance_id: Optional[str] = Field(default=None, description="Instance that created checkpoint")

    # Checkpoint details
    strategy: CheckpointStrategy = Field(description="Strategy used")
    trigger: CheckpointTrigger = Field(description="What triggered the checkpoint")

    # Storage
    location: str = Field(description="Storage location (s3:// or local path)")
    size_bytes: int = Field(default=0, description="Total size in bytes")
    file_count: int = Field(default=0, description="Number of files")

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    duration_seconds: int = Field(default=0, description="Time to create checkpoint")

    # Application state
    app_metadata_json: str = Field(
        default="{}",
        description="JSON-encoded application-specific metadata (e.g., LAMMPS timestep)",
    )

    # Validity
    verified: bool = Field(default=False, description="Has been verified as restorable")
    verification_error: Optional[str] = Field(default=None)

    def get_app_metadata(self) -> dict:
        """Parse app metadata from JSON."""
        return _json.loads(self.app_metadata_json)


class SyncRecord(SQLModel, table=True):
    """Record of a sync operation."""

    __tablename__ = "sync_records"

    id: int | None = Field(default=None, primary_key=True)
    sync_id: str = Field(
        default_factory=lambda: f"sync_{generate_id()}", unique=True, index=True
    )
    job_id: str = Field(index=True)
    instance_id: Optional[str] = Field(default=None)

    # Sync details
    source: str = Field(description="Source path")
    destination: str = Field(description="Destination path")
    status: SyncStatus = Field(default=SyncStatus.SYNCING)

    # Statistics
    files_synced: int = Field(default=0)
    bytes_synced: int = Field(default=0)
    files_failed: int = Field(default=0)

    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)
    duration_seconds: int = Field(default=0)

    # Error handling
    error_message: Optional[str] = Field(default=None)
