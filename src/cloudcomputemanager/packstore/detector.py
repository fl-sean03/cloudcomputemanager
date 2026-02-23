"""Environment detection for PackStore.

Detects GPU capabilities, CUDA version, and driver information on instances.
"""

import re
from dataclasses import dataclass
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


# GPU architecture mapping
GPU_COMPUTE_CAPS = {
    # Consumer GPUs (Ada Lovelace - RTX 40 series)
    "NVIDIA GeForce RTX 4090": ("sm_89", 24),
    "NVIDIA GeForce RTX 4080": ("sm_89", 16),
    "NVIDIA GeForce RTX 4080 SUPER": ("sm_89", 16),
    "NVIDIA GeForce RTX 4070": ("sm_89", 12),
    "NVIDIA GeForce RTX 4070 Ti": ("sm_89", 12),
    "NVIDIA GeForce RTX 4070 SUPER": ("sm_89", 12),
    "NVIDIA GeForce RTX 4060": ("sm_89", 8),
    "NVIDIA GeForce RTX 4060 Ti": ("sm_89", 8),
    "NVIDIA RTX 5080": ("sm_100", 16),  # Blackwell
    "NVIDIA RTX 5090": ("sm_100", 32),  # Blackwell

    # Consumer GPUs (Ampere - RTX 30 series)
    "NVIDIA GeForce RTX 3090": ("sm_86", 24),
    "NVIDIA GeForce RTX 3090 Ti": ("sm_86", 24),
    "NVIDIA GeForce RTX 3080": ("sm_86", 10),
    "NVIDIA GeForce RTX 3080 Ti": ("sm_86", 12),
    "NVIDIA GeForce RTX 3070": ("sm_86", 8),
    "NVIDIA GeForce RTX 3060": ("sm_86", 12),

    # Data center GPUs (Hopper)
    "NVIDIA H100": ("sm_90", 80),
    "NVIDIA H100 PCIe": ("sm_90", 80),
    "NVIDIA H100 SXM5": ("sm_90", 80),
    "NVIDIA H100 NVL": ("sm_90", 94),

    # Data center GPUs (Ampere)
    "NVIDIA A100": ("sm_80", 40),
    "NVIDIA A100 40GB": ("sm_80", 40),
    "NVIDIA A100 80GB": ("sm_80", 80),
    "NVIDIA A100-SXM4-40GB": ("sm_80", 40),
    "NVIDIA A100-SXM4-80GB": ("sm_80", 80),
    "NVIDIA A100-PCIE-40GB": ("sm_80", 40),
    "NVIDIA A100-PCIE-80GB": ("sm_80", 80),
    "NVIDIA A10": ("sm_86", 24),
    "NVIDIA A10G": ("sm_86", 24),
    "NVIDIA A40": ("sm_86", 48),
    "NVIDIA A30": ("sm_80", 24),

    # Data center GPUs (Volta)
    "Tesla V100": ("sm_70", 16),
    "Tesla V100-SXM2-16GB": ("sm_70", 16),
    "Tesla V100-SXM2-32GB": ("sm_70", 32),
    "Tesla V100-PCIE-16GB": ("sm_70", 16),
    "Tesla V100-PCIE-32GB": ("sm_70", 32),
    "NVIDIA V100": ("sm_70", 32),

    # Data center GPUs (Turing)
    "Tesla T4": ("sm_75", 16),
    "NVIDIA T4": ("sm_75", 16),

    # Quadro (various)
    "Quadro RTX 8000": ("sm_75", 48),
    "Quadro RTX 6000": ("sm_75", 24),
    "Quadro RTX 5000": ("sm_75", 16),

    # L-series
    "NVIDIA L4": ("sm_89", 24),
    "NVIDIA L40": ("sm_89", 48),
    "NVIDIA L40S": ("sm_89", 48),
}


@dataclass
class InstanceEnvironment:
    """Environment information for an instance."""

    # GPU information
    gpu_name: str
    gpu_count: int
    gpu_arch: str  # e.g., "sm_89"
    gpu_memory_gb: int

    # Driver and CUDA
    driver_version: str
    cuda_version: str

    # System information
    os_name: str = "linux"
    os_version: str = ""
    cpu_cores: int = 0
    memory_gb: int = 0

    def __str__(self) -> str:
        return (
            f"{self.gpu_name} x{self.gpu_count} ({self.gpu_arch}, {self.gpu_memory_gb}GB), "
            f"Driver {self.driver_version}, CUDA {self.cuda_version}"
        )


class EnvironmentDetector:
    """Detect GPU and CUDA environment on instances."""

    def __init__(self, execute_fn):
        """Initialize the detector.

        Args:
            execute_fn: Async function (command: str) -> (exit_code, stdout, stderr)
        """
        self._execute = execute_fn

    async def detect(self) -> InstanceEnvironment:
        """Detect the full environment."""
        gpu_info = await self._detect_gpu()
        cuda_version = await self._detect_cuda()
        system_info = await self._detect_system()

        return InstanceEnvironment(
            gpu_name=gpu_info["name"],
            gpu_count=gpu_info["count"],
            gpu_arch=gpu_info["arch"],
            gpu_memory_gb=gpu_info["memory_gb"],
            driver_version=gpu_info["driver"],
            cuda_version=cuda_version,
            os_name=system_info.get("os_name", "linux"),
            os_version=system_info.get("os_version", ""),
            cpu_cores=system_info.get("cpu_cores", 0),
            memory_gb=system_info.get("memory_gb", 0),
        )

    async def _detect_gpu(self) -> dict:
        """Detect GPU information using nvidia-smi."""
        # Query GPU info
        cmd = (
            "nvidia-smi --query-gpu=name,driver_version,memory.total,count "
            "--format=csv,noheader,nounits 2>/dev/null | head -1"
        )
        exit_code, stdout, _ = await self._execute(cmd)

        if exit_code != 0 or not stdout.strip():
            logger.warning("nvidia-smi failed, assuming no GPU")
            return {
                "name": "Unknown",
                "count": 0,
                "arch": "sm_00",
                "memory_gb": 0,
                "driver": "0.0",
            }

        parts = [p.strip() for p in stdout.strip().split(",")]
        if len(parts) < 4:
            # Fallback: simpler query
            cmd2 = "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"
            exit_code, stdout2, _ = await self._execute(cmd2)
            parts = [p.strip() for p in stdout2.strip().split(",")]
            if len(parts) >= 2:
                gpu_name = parts[0]
                driver = parts[1]
            else:
                gpu_name = "Unknown"
                driver = "0.0"

            # Get memory separately
            cmd3 = "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"
            _, mem_out, _ = await self._execute(cmd3)
            try:
                memory_mb = int(mem_out.strip())
            except ValueError:
                memory_mb = 0

            # Count GPUs
            cmd4 = "nvidia-smi -L | wc -l"
            _, count_out, _ = await self._execute(cmd4)
            try:
                gpu_count = int(count_out.strip())
            except ValueError:
                gpu_count = 1
        else:
            gpu_name = parts[0]
            driver = parts[1]
            try:
                memory_mb = int(float(parts[2]))
            except ValueError:
                memory_mb = 0
            try:
                gpu_count = int(parts[3])
            except ValueError:
                gpu_count = 1

        # Determine compute capability
        gpu_arch, default_mem = self._get_compute_cap(gpu_name)
        memory_gb = memory_mb // 1024 if memory_mb else default_mem

        return {
            "name": gpu_name,
            "count": gpu_count,
            "arch": gpu_arch,
            "memory_gb": memory_gb,
            "driver": driver,
        }

    def _get_compute_cap(self, gpu_name: str) -> tuple[str, int]:
        """Get compute capability for a GPU name.

        Returns:
            Tuple of (compute_cap, default_memory_gb)
        """
        # Try exact match first
        if gpu_name in GPU_COMPUTE_CAPS:
            return GPU_COMPUTE_CAPS[gpu_name]

        # Try partial match
        gpu_lower = gpu_name.lower()
        for name, (arch, mem) in GPU_COMPUTE_CAPS.items():
            if name.lower() in gpu_lower or gpu_lower in name.lower():
                return (arch, mem)

        # Try to extract from common patterns
        if "4090" in gpu_name or "4080" in gpu_name or "4070" in gpu_name:
            return ("sm_89", 16)
        if "3090" in gpu_name or "3080" in gpu_name:
            return ("sm_86", 24)
        if "a100" in gpu_lower:
            return ("sm_80", 40)
        if "h100" in gpu_lower:
            return ("sm_90", 80)
        if "v100" in gpu_lower:
            return ("sm_70", 16)
        if "t4" in gpu_lower:
            return ("sm_75", 16)

        logger.warning("Unknown GPU, defaulting to sm_80", gpu=gpu_name)
        return ("sm_80", 16)

    async def _detect_cuda(self) -> str:
        """Detect CUDA version."""
        # Try nvcc first
        cmd = "nvcc --version 2>/dev/null | grep 'release' | awk '{print $6}' | tr -d 'V,'"
        exit_code, stdout, _ = await self._execute(cmd)

        if exit_code == 0 and stdout.strip():
            version = stdout.strip()
            # Handle versions like "12.1.105" -> "12.1"
            parts = version.split(".")
            if len(parts) >= 2:
                return f"{parts[0]}.{parts[1]}"
            return version

        # Fallback: nvidia-smi
        cmd2 = "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"
        exit_code, stdout2, _ = await self._execute(cmd2)

        if exit_code == 0 and stdout2.strip():
            # Map driver to approximate CUDA version
            driver = stdout2.strip()
            return self._driver_to_cuda(driver)

        logger.warning("Could not detect CUDA version")
        return "12.1"  # Default assumption

    def _driver_to_cuda(self, driver_version: str) -> str:
        """Map driver version to approximate CUDA version."""
        try:
            major = int(driver_version.split(".")[0])
        except (ValueError, IndexError):
            return "12.1"

        # Approximate mapping
        if major >= 555:
            return "13.0"
        if major >= 545:
            return "12.6"
        if major >= 535:
            return "12.1"
        if major >= 525:
            return "11.8"
        if major >= 515:
            return "11.7"
        if major >= 470:
            return "11.4"
        return "11.0"

    async def _detect_system(self) -> dict:
        """Detect system information."""
        result = {}

        # OS info
        cmd = "cat /etc/os-release 2>/dev/null | grep -E '^(NAME|VERSION)=' | head -2"
        exit_code, stdout, _ = await self._execute(cmd)
        if exit_code == 0:
            for line in stdout.strip().split("\n"):
                if line.startswith("NAME="):
                    result["os_name"] = line.split("=")[1].strip('"')
                elif line.startswith("VERSION="):
                    result["os_version"] = line.split("=")[1].strip('"')

        # CPU cores
        cmd = "nproc 2>/dev/null"
        exit_code, stdout, _ = await self._execute(cmd)
        if exit_code == 0:
            try:
                result["cpu_cores"] = int(stdout.strip())
            except ValueError:
                pass

        # Memory
        cmd = "free -g 2>/dev/null | grep Mem | awk '{print $2}'"
        exit_code, stdout, _ = await self._execute(cmd)
        if exit_code == 0:
            try:
                result["memory_gb"] = int(stdout.strip())
            except ValueError:
                pass

        return result

    async def check_container_runtime(self) -> dict[str, bool]:
        """Check which container runtimes are available."""
        runtimes = {}

        # Docker
        cmd = "docker --version 2>/dev/null"
        exit_code, _, _ = await self._execute(cmd)
        runtimes["docker"] = exit_code == 0

        # Singularity/Apptainer
        cmd = "singularity --version 2>/dev/null || apptainer --version 2>/dev/null"
        exit_code, _, _ = await self._execute(cmd)
        runtimes["singularity"] = exit_code == 0

        # Podman
        cmd = "podman --version 2>/dev/null"
        exit_code, _, _ = await self._execute(cmd)
        runtimes["podman"] = exit_code == 0

        return runtimes

    async def check_package_managers(self) -> dict[str, bool]:
        """Check which package managers are available."""
        managers = {}

        # Conda/Mamba
        cmd = "conda --version 2>/dev/null || mamba --version 2>/dev/null"
        exit_code, _, _ = await self._execute(cmd)
        managers["conda"] = exit_code == 0

        # Spack
        cmd = "spack --version 2>/dev/null"
        exit_code, _, _ = await self._execute(cmd)
        managers["spack"] = exit_code == 0

        # pip
        cmd = "pip --version 2>/dev/null"
        exit_code, _, _ = await self._execute(cmd)
        managers["pip"] = exit_code == 0

        return managers
