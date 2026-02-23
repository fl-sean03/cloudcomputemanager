"""Package deployment for PackStore.

Handles deploying packages to instances using various strategies.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import structlog

from cloudcomputemanager.packstore.registry import (
    PackageRegistry,
    PackageVariant,
    SourceType,
)
from cloudcomputemanager.packstore.detector import EnvironmentDetector, InstanceEnvironment
from cloudcomputemanager.providers.base import CloudProvider

logger = structlog.get_logger(__name__)


class DeploymentStrategy(str, Enum):
    """Strategy for deploying packages."""

    AUTO = "auto"  # Automatically select best strategy
    NGC_CONTAINER = "ngc_container"
    DOCKER = "docker"
    SINGULARITY = "singularity"
    SQUASHFS = "squashfs"
    SPACK = "spack"
    CONDA = "conda"


class DeploymentStatus(str, Enum):
    """Status of a package deployment."""

    PENDING = "pending"
    DETECTING = "detecting"
    PULLING = "pulling"
    INSTALLING = "installing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PackageDeployment:
    """Result of deploying a package."""

    package_name: str
    variant_id: str
    status: DeploymentStatus
    strategy_used: DeploymentStrategy

    # Paths
    install_path: Optional[str] = None
    bin_path: Optional[str] = None

    # Environment
    environment: dict[str, str] = field(default_factory=dict)

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: int = 0

    # Error info
    error_message: Optional[str] = None

    # Verification
    verified: bool = False


@dataclass
class DeploymentResult:
    """Result of deploying multiple packages."""

    deployments: list[PackageDeployment]
    environment: InstanceEnvironment
    total_duration_seconds: int = 0

    @property
    def all_environment(self) -> dict[str, str]:
        """Combined environment variables from all deployments."""
        env = {}
        for dep in self.deployments:
            env.update(dep.environment)
        return env

    @property
    def success(self) -> bool:
        """Whether all deployments succeeded."""
        return all(d.status == DeploymentStatus.COMPLETED for d in self.deployments)


class PackageDeployer:
    """Deploy packages to instances."""

    def __init__(
        self,
        provider: CloudProvider,
        registry: Optional[PackageRegistry] = None,
    ):
        """Initialize the deployer.

        Args:
            provider: Cloud provider for instance operations
            registry: Package registry (uses default if not provided)
        """
        self._provider = provider
        self._registry = registry or PackageRegistry()

    async def deploy(
        self,
        instance_id: str,
        packages: list[str],
        strategy: DeploymentStrategy = DeploymentStrategy.AUTO,
        verify: bool = True,
    ) -> DeploymentResult:
        """Deploy packages to an instance.

        Args:
            instance_id: Instance to deploy to
            packages: List of package names (or variant IDs like "lammps-gpu-kokkos")
            strategy: Deployment strategy to use
            verify: Whether to verify installations

        Returns:
            DeploymentResult with status of all deployments
        """
        start_time = datetime.utcnow()
        deployments = []

        # Create execute function for this instance
        async def execute_fn(cmd: str) -> tuple[int, str, str]:
            return await self._provider.execute_command(instance_id, cmd)

        # Detect environment
        logger.info("Detecting instance environment", instance_id=instance_id)
        detector = EnvironmentDetector(execute_fn)
        env = await detector.detect()
        logger.info("Environment detected", env=str(env))

        # Check available runtimes if auto strategy
        runtimes = {}
        if strategy == DeploymentStrategy.AUTO:
            runtimes = await detector.check_container_runtime()
            logger.debug("Available runtimes", runtimes=runtimes)

        # Deploy each package
        for package_spec in packages:
            dep = await self._deploy_single(
                instance_id,
                package_spec,
                env,
                strategy,
                runtimes,
                verify,
                execute_fn,
            )
            deployments.append(dep)

        total_duration = (datetime.utcnow() - start_time).seconds

        return DeploymentResult(
            deployments=deployments,
            environment=env,
            total_duration_seconds=total_duration,
        )

    async def _deploy_single(
        self,
        instance_id: str,
        package_spec: str,
        env: InstanceEnvironment,
        strategy: DeploymentStrategy,
        runtimes: dict[str, bool],
        verify: bool,
        execute_fn: Callable,
    ) -> PackageDeployment:
        """Deploy a single package."""
        start_time = datetime.utcnow()

        # Parse package spec (can be "lammps" or "lammps-gpu-kokkos")
        if "-" in package_spec:
            # Might be a variant ID
            parts = package_spec.rsplit("-", 2)
            if len(parts) >= 2:
                # Try as variant ID first
                for pkg_name in [package_spec.rsplit("-", 1)[0], package_spec.rsplit("-", 2)[0]]:
                    package = self._registry.get(pkg_name)
                    if package:
                        variant = package.get_variant(package_spec)
                        if variant:
                            break
                else:
                    # Fall back to treating as package name
                    package = self._registry.get(package_spec)
                    variant = None
            else:
                package = self._registry.get(package_spec)
                variant = None
        else:
            package = self._registry.get(package_spec)
            variant = None

        if not package:
            return PackageDeployment(
                package_name=package_spec,
                variant_id="unknown",
                status=DeploymentStatus.FAILED,
                strategy_used=DeploymentStrategy.AUTO,
                error_message=f"Package not found: {package_spec}",
                started_at=start_time,
                completed_at=datetime.utcnow(),
            )

        # Find compatible variant if not specified
        if not variant:
            variant = package.find_compatible_variant(
                env.cuda_version,
                env.driver_version,
                env.gpu_arch,
                env.gpu_memory_gb,
            )

        if not variant:
            return PackageDeployment(
                package_name=package.name,
                variant_id="none",
                status=DeploymentStatus.FAILED,
                strategy_used=DeploymentStrategy.AUTO,
                error_message=f"No compatible variant found for {env}",
                started_at=start_time,
                completed_at=datetime.utcnow(),
            )

        logger.info(
            "Deploying package",
            package=package.name,
            variant=variant.id,
            strategy=strategy,
        )

        # Select deployment strategy
        actual_strategy = self._select_strategy(variant, strategy, runtimes)

        # Execute deployment
        try:
            if actual_strategy in [DeploymentStrategy.NGC_CONTAINER, DeploymentStrategy.DOCKER]:
                result = await self._deploy_container(variant, execute_fn, "docker")
            elif actual_strategy == DeploymentStrategy.SINGULARITY:
                result = await self._deploy_container(variant, execute_fn, "singularity")
            elif actual_strategy == DeploymentStrategy.CONDA:
                result = await self._deploy_conda(variant, execute_fn)
            elif actual_strategy == DeploymentStrategy.SPACK:
                result = await self._deploy_spack(variant, execute_fn)
            else:
                result = {"success": False, "error": f"Unsupported strategy: {actual_strategy}"}

            if not result.get("success"):
                return PackageDeployment(
                    package_name=package.name,
                    variant_id=variant.id,
                    status=DeploymentStatus.FAILED,
                    strategy_used=actual_strategy,
                    error_message=result.get("error", "Unknown error"),
                    started_at=start_time,
                    completed_at=datetime.utcnow(),
                )

            # Verify if requested
            verified = False
            if verify and variant.test_command:
                verified = await self._verify_installation(variant, execute_fn)

            return PackageDeployment(
                package_name=package.name,
                variant_id=variant.id,
                status=DeploymentStatus.COMPLETED,
                strategy_used=actual_strategy,
                install_path=result.get("install_path", variant.install_path),
                environment=variant.environment,
                verified=verified,
                started_at=start_time,
                completed_at=datetime.utcnow(),
                duration_seconds=(datetime.utcnow() - start_time).seconds,
            )

        except Exception as e:
            logger.error("Deployment failed", package=package.name, error=str(e))
            return PackageDeployment(
                package_name=package.name,
                variant_id=variant.id,
                status=DeploymentStatus.FAILED,
                strategy_used=actual_strategy,
                error_message=str(e),
                started_at=start_time,
                completed_at=datetime.utcnow(),
            )

    def _select_strategy(
        self,
        variant: PackageVariant,
        requested: DeploymentStrategy,
        runtimes: dict[str, bool],
    ) -> DeploymentStrategy:
        """Select the best deployment strategy."""
        if requested != DeploymentStrategy.AUTO:
            return requested

        # Get the best source
        source = variant.get_best_source()
        if not source:
            return DeploymentStrategy.SPACK  # Fallback

        # Map source type to strategy
        if source.type == SourceType.NGC_CONTAINER:
            if runtimes.get("docker"):
                return DeploymentStrategy.DOCKER
            elif runtimes.get("singularity"):
                return DeploymentStrategy.SINGULARITY
            else:
                # Try Docker anyway, might work
                return DeploymentStrategy.DOCKER

        elif source.type == SourceType.DOCKER:
            return DeploymentStrategy.DOCKER

        elif source.type == SourceType.SINGULARITY:
            return DeploymentStrategy.SINGULARITY

        elif source.type == SourceType.CONDA:
            return DeploymentStrategy.CONDA

        elif source.type == SourceType.SPACK:
            return DeploymentStrategy.SPACK

        return DeploymentStrategy.DOCKER

    async def _deploy_container(
        self,
        variant: PackageVariant,
        execute_fn: Callable,
        runtime: str = "docker",
    ) -> dict:
        """Deploy using container runtime."""
        source = variant.get_best_source()
        if not source or not source.image:
            return {"success": False, "error": "No container image specified"}

        image = source.full_image

        if runtime == "docker":
            # Pull image
            cmd = f"docker pull {image}"
            logger.info("Pulling Docker image", image=image)
            exit_code, stdout, stderr = await execute_fn(cmd)

            if exit_code != 0:
                return {"success": False, "error": f"Docker pull failed: {stderr}"}

            # Create wrapper script to run commands in container
            wrapper = f"""#!/bin/bash
docker run --rm --gpus all -v /workspace:/workspace -w /workspace {image} "$@"
"""
            wrapper_path = f"/usr/local/bin/{variant.id.replace('-', '_')}"
            cmd = f"echo '{wrapper}' | sudo tee {wrapper_path} && sudo chmod +x {wrapper_path}"
            await execute_fn(cmd)

            return {
                "success": True,
                "install_path": f"container:{image}",
            }

        elif runtime == "singularity":
            # Build/pull singularity image
            sif_path = f"/opt/containers/{variant.id}.sif"
            cmd = f"mkdir -p /opt/containers && singularity pull --dir /opt/containers {variant.id}.sif docker://{image}"
            logger.info("Pulling Singularity image", image=image)
            exit_code, stdout, stderr = await execute_fn(cmd)

            if exit_code != 0:
                return {"success": False, "error": f"Singularity pull failed: {stderr}"}

            return {
                "success": True,
                "install_path": sif_path,
            }

        return {"success": False, "error": f"Unknown runtime: {runtime}"}

    async def _deploy_conda(
        self,
        variant: PackageVariant,
        execute_fn: Callable,
    ) -> dict:
        """Deploy using Conda."""
        source = variant.get_best_source()
        if not source:
            return {"success": False, "error": "No conda source specified"}

        env_name = variant.id.replace("-", "_")

        # Check if conda is available
        exit_code, _, _ = await execute_fn("conda --version")
        if exit_code != 0:
            # Try to install miniconda
            logger.info("Installing Miniconda")
            install_cmd = """
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /opt/conda
rm /tmp/miniconda.sh
/opt/conda/bin/conda init bash
"""
            exit_code, _, stderr = await execute_fn(install_cmd)
            if exit_code != 0:
                return {"success": False, "error": f"Miniconda install failed: {stderr}"}

        # Create environment
        if source.spec:
            cmd = f"/opt/conda/bin/conda create -y -n {env_name} {source.spec} -c conda-forge"
        elif source.env_file:
            cmd = f"/opt/conda/bin/conda env create -y -n {env_name} -f {source.env_file}"
        else:
            return {"success": False, "error": "No conda spec or env file"}

        logger.info("Creating conda environment", env=env_name)
        exit_code, stdout, stderr = await execute_fn(cmd)

        if exit_code != 0:
            return {"success": False, "error": f"Conda create failed: {stderr}"}

        return {
            "success": True,
            "install_path": f"/opt/conda/envs/{env_name}",
        }

    async def _deploy_spack(
        self,
        variant: PackageVariant,
        execute_fn: Callable,
    ) -> dict:
        """Deploy using Spack."""
        source = variant.get_best_source()
        if not source or not source.spec:
            return {"success": False, "error": "No spack spec specified"}

        # Check if spack is available
        exit_code, _, _ = await execute_fn("spack --version")
        if exit_code != 0:
            # Install spack
            logger.info("Installing Spack")
            install_cmd = """
git clone -c feature.manyFiles=true --depth 1 https://github.com/spack/spack.git /opt/spack
echo '. /opt/spack/share/spack/setup-env.sh' >> ~/.bashrc
"""
            exit_code, _, stderr = await execute_fn(install_cmd)
            if exit_code != 0:
                return {"success": False, "error": f"Spack install failed: {stderr}"}

        # Install package
        cmd = f". /opt/spack/share/spack/setup-env.sh && spack install {source.spec}"
        logger.info("Installing with Spack", spec=source.spec)
        exit_code, stdout, stderr = await execute_fn(cmd)

        if exit_code != 0:
            return {"success": False, "error": f"Spack install failed: {stderr}"}

        return {
            "success": True,
            "install_path": "/opt/spack",
        }

    async def _verify_installation(
        self,
        variant: PackageVariant,
        execute_fn: Callable,
    ) -> bool:
        """Verify that a package is correctly installed."""
        if not variant.test_command:
            return True

        logger.info("Verifying installation", variant=variant.id)

        # Set up environment
        env_setup = ""
        for key, value in variant.environment.items():
            env_setup += f"export {key}={value}; "

        cmd = f"{env_setup}{variant.test_command}"
        exit_code, stdout, stderr = await execute_fn(cmd)

        if exit_code != 0:
            logger.warning(
                "Verification failed",
                variant=variant.id,
                exit_code=exit_code,
                stderr=stderr,
            )
            return False

        if variant.expected_output_contains:
            if variant.expected_output_contains not in stdout:
                logger.warning(
                    "Verification output mismatch",
                    variant=variant.id,
                    expected=variant.expected_output_contains,
                    got=stdout[:100],
                )
                return False

        logger.info("Verification passed", variant=variant.id)
        return True

    async def list_installed(
        self,
        instance_id: str,
    ) -> list[str]:
        """List packages that appear to be installed on an instance."""
        installed = []

        async def execute_fn(cmd: str) -> tuple[int, str, str]:
            return await self._provider.execute_command(instance_id, cmd)

        # Check Docker images
        exit_code, stdout, _ = await execute_fn("docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null")
        if exit_code == 0:
            for line in stdout.strip().split("\n"):
                if "lammps" in line.lower():
                    installed.append("lammps")
                if "gromacs" in line.lower():
                    installed.append("gromacs")
                if "namd" in line.lower():
                    installed.append("namd")
                if "quantum_espresso" in line.lower():
                    installed.append("quantum-espresso")

        # Check conda environments
        exit_code, stdout, _ = await execute_fn("conda env list 2>/dev/null | grep -v '^#'")
        if exit_code == 0:
            for line in stdout.strip().split("\n"):
                if "deepmd" in line.lower():
                    installed.append("deepmd-kit")

        return list(set(installed))
