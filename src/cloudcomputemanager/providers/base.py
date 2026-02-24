"""Base classes for cloud provider adapters.

Defines the interface that all provider implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class ProviderStatus(str, Enum):
    """Status of a provider instance."""

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    ERROR = "error"


@dataclass
class ProviderOffer:
    """An available GPU instance offer from a provider."""

    offer_id: str
    provider: str
    gpu_type: str
    gpu_count: int
    gpu_memory_gb: int
    cpu_cores: int
    memory_gb: int
    disk_gb: int
    hourly_rate: float
    location: str
    reliability_score: float = 1.0
    interruptible: bool = True
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None

    @property
    def score(self) -> float:
        """Calculate a composite score for offer comparison.

        Lower is better (cost-focused).
        """
        # Weight factors
        w_cost = 0.6
        w_reliability = 0.3
        w_performance = 0.1

        # Normalize hourly rate (assume $0.10-$5.00 range)
        cost_score = self.hourly_rate / 5.0

        # Reliability already 0-1
        reliability_score = 1.0 - self.reliability_score

        # Performance based on GPU memory (normalize to 80GB max)
        perf_score = 1.0 - (self.gpu_memory_gb * self.gpu_count / 80.0)

        return w_cost * cost_score + w_reliability * reliability_score + w_performance * perf_score


@dataclass
class ProviderInstance:
    """A running instance from a provider."""

    instance_id: str
    provider: str
    status: ProviderStatus
    gpu_type: str
    gpu_count: int
    gpu_memory_gb: int
    cpu_cores: int
    memory_gb: int
    disk_gb: int
    ssh_host: str
    ssh_port: int
    ssh_user: str = "root"
    hourly_rate: float = 0.0
    interruptible: bool = True
    created_at: Optional[datetime] = None
    jupyter_url: Optional[str] = None
    internal_ip: Optional[str] = None
    external_ip: Optional[str] = None


class CloudProvider(ABC):
    """Abstract base class for cloud provider adapters.

    All provider implementations must implement these methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...

    @abstractmethod
    async def search_offers(
        self,
        gpu_type: Optional[str] = None,
        gpu_count: int = 1,
        gpu_memory_min: int = 16,
        disk_gb_min: int = 50,
        max_hourly_rate: Optional[float] = None,
        interruptible: bool = True,
    ) -> list[ProviderOffer]:
        """Search for available GPU offers.

        Args:
            gpu_type: Specific GPU model (e.g., "RTX_4090")
            gpu_count: Number of GPUs required
            gpu_memory_min: Minimum GPU VRAM in GB
            disk_gb_min: Minimum disk space in GB
            max_hourly_rate: Maximum hourly cost in USD
            interruptible: Whether to search for spot/interruptible instances

        Returns:
            List of matching offers, sorted by score (best first)
        """
        ...

    @abstractmethod
    async def create_instance(
        self,
        offer_id: str,
        image: str,
        disk_gb: int = 50,
        ssh_public_key: Optional[str] = None,
        env_vars: Optional[dict[str, str]] = None,
        startup_script: Optional[str] = None,
    ) -> ProviderInstance:
        """Create a new instance from an offer.

        Args:
            offer_id: The offer ID to use
            image: Docker image to run
            disk_gb: Disk space in GB
            ssh_public_key: SSH public key for access
            env_vars: Environment variables to set
            startup_script: Script to run on startup

        Returns:
            The created instance
        """
        ...

    @abstractmethod
    async def get_instance(self, instance_id: str) -> Optional[ProviderInstance]:
        """Get instance details.

        Args:
            instance_id: The instance ID

        Returns:
            Instance details or None if not found
        """
        ...

    @abstractmethod
    async def list_instances(self) -> list[ProviderInstance]:
        """List all instances.

        Returns:
            List of all instances
        """
        ...

    @abstractmethod
    async def start_instance(self, instance_id: str) -> bool:
        """Start a stopped instance.

        Args:
            instance_id: The instance ID

        Returns:
            True if successful
        """
        ...

    @abstractmethod
    async def stop_instance(self, instance_id: str) -> bool:
        """Stop a running instance.

        Args:
            instance_id: The instance ID

        Returns:
            True if successful
        """
        ...

    @abstractmethod
    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate an instance.

        Args:
            instance_id: The instance ID

        Returns:
            True if successful
        """
        ...

    @abstractmethod
    async def execute_command(
        self,
        instance_id: str,
        command: str,
        timeout: int = 60,
    ) -> tuple[int, str, str]:
        """Execute a command on an instance.

        Args:
            instance_id: The instance ID
            command: Command to execute
            timeout: Timeout in seconds

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        ...

    async def wait_for_ready(
        self,
        instance_id: str,
        timeout: int = 300,
        interval: int = 10,
    ) -> bool:
        """Wait for an instance to be ready.

        Args:
            instance_id: The instance ID
            timeout: Maximum wait time in seconds
            interval: Check interval in seconds

        Returns:
            True if instance is ready, False if timeout
        """
        import asyncio
        import time

        start_time = time.monotonic()
        while (time.monotonic() - start_time) < timeout:
            instance = await self.get_instance(instance_id)
            if instance and instance.status == ProviderStatus.RUNNING:
                # Verify SSH access
                try:
                    exit_code, _, _ = await self.execute_command(instance_id, "echo ok", timeout=10)
                    if exit_code == 0:
                        return True
                except Exception:
                    pass
            await asyncio.sleep(interval)
        return False
