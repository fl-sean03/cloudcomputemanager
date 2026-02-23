"""Tests for PackStore module."""

import pytest
from unittest.mock import AsyncMock, patch

from cloudcomputemanager.packstore import (
    PackageRegistry,
    PackageDeployer,
    DeploymentStrategy,
    EnvironmentDetector,
    InstanceEnvironment,
)
from cloudcomputemanager.packstore.registry import (
    Package,
    PackageVariant,
    PackageSource,
    PackageCategory,
    SourceType,
    Compatibility,
)


class TestPackageRegistry:
    """Tests for PackageRegistry."""

    def test_registry_has_packages(self):
        """Test that registry has built-in packages."""
        registry = PackageRegistry()
        packages = registry.list_packages()

        assert len(packages) > 0
        names = [p.name for p in packages]
        assert "lammps" in names

    def test_get_package_by_name(self):
        """Test getting package by name."""
        registry = PackageRegistry()

        lammps = registry.get("lammps")
        assert lammps is not None
        assert lammps.name == "lammps"
        assert len(lammps.variants) > 0

    def test_get_nonexistent_package(self):
        """Test getting non-existent package."""
        registry = PackageRegistry()
        result = registry.get("nonexistent-package")
        assert result is None

    def test_search_packages(self):
        """Test searching packages."""
        registry = PackageRegistry()

        results = registry.search("molecular")
        assert len(results) > 0

    def test_search_with_cuda_filter(self):
        """Test searching with CUDA version filter."""
        registry = PackageRegistry()

        results = registry.search("lammps", cuda_version="12.1")
        # Should find LAMMPS with CUDA 12 support
        assert len(results) > 0

    def test_list_by_category(self):
        """Test listing packages by category."""
        registry = PackageRegistry()

        md_packages = registry.list_packages(category=PackageCategory.MOLECULAR_DYNAMICS)
        assert len(md_packages) > 0
        for pkg in md_packages:
            assert pkg.category == PackageCategory.MOLECULAR_DYNAMICS

    def test_variant_compatibility_check(self):
        """Test variant compatibility checking."""
        registry = PackageRegistry()
        lammps = registry.get("lammps")

        # Find compatible variant for RTX 4090 (sm_89)
        variant = lammps.find_compatible_variant(
            cuda_version="12.1",
            driver_version="535.0",
            gpu_arch="sm_89",
            gpu_memory_gb=24,
        )
        assert variant is not None


class TestPackageVariant:
    """Tests for PackageVariant."""

    def test_variant_creation(self):
        """Test creating a variant."""
        variant = PackageVariant(
            id="lammps-gpu",
            name="LAMMPS GPU",
            version="2024.1",
            description="GPU-accelerated LAMMPS",
            sources=[
                PackageSource(
                    type=SourceType.NGC_CONTAINER,
                    image="nvcr.io/hpc/lammps",
                    tag="29Aug2024",
                )
            ],
            compatibility=Compatibility(
                cuda_versions=["12.0", "12.1", "12.2"],
                gpu_architectures=["sm_80", "sm_86", "sm_89", "sm_90"],
                min_driver="525.0",
            ),
        )
        assert variant.id == "lammps-gpu"
        assert len(variant.sources) == 1

    def test_get_best_source(self):
        """Test getting best source from variant."""
        variant = PackageVariant(
            id="test",
            name="Test Package",
            version="1.0",
            description="Test",
            sources=[
                PackageSource(
                    type=SourceType.NGC_CONTAINER,
                    image="nvcr.io/hpc/test",
                    tag="latest",
                    priority=1,
                ),
                PackageSource(
                    type=SourceType.SPACK,
                    spec="test@1.0",
                    priority=2,
                ),
            ],
            compatibility=Compatibility(),
        )
        best = variant.get_best_source()
        assert best.type == SourceType.NGC_CONTAINER  # Lower priority = better


class TestPackageSource:
    """Tests for PackageSource."""

    def test_full_image_ngc(self):
        """Test full image name for NGC container."""
        source = PackageSource(
            type=SourceType.NGC_CONTAINER,
            image="nvcr.io/hpc/lammps",
            tag="29Aug2024",
        )
        assert source.full_image == "nvcr.io/hpc/lammps:29Aug2024"

    def test_full_image_docker(self):
        """Test full image name for Docker."""
        source = PackageSource(
            type=SourceType.DOCKER,
            image="pytorch/pytorch",
            tag="2.0.0-cuda11.8-cudnn8-runtime",
        )
        assert source.full_image == "pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime"


class TestEnvironmentDetector:
    """Tests for EnvironmentDetector."""

    @pytest.fixture
    def mock_execute(self):
        """Create mock execute function."""

        async def execute(cmd: str):
            if "nvidia-smi" in cmd and "query-gpu" in cmd:
                return (0, "NVIDIA GeForce RTX 4090, 535.104.05, 24564, 1", "")
            elif "nvcc --version" in cmd:
                return (0, "12.1.105", "")  # Already parsed by awk
            elif "nproc" in cmd:
                return (0, "16", "")
            elif "free -g" in cmd:
                return (0, "64", "")
            elif "cat /etc/os-release" in cmd:
                return (0, 'NAME="Ubuntu"\nVERSION="22.04"', "")
            elif "nvidia-smi -L" in cmd:
                return (0, "GPU 0: NVIDIA GeForce RTX 4090\n", "")
            elif "docker --version" in cmd:
                return (0, "Docker version 24.0.0", "")
            elif "singularity --version" in cmd or "apptainer --version" in cmd:
                return (1, "", "not found")
            else:
                return (0, "", "")

        return execute

    @pytest.mark.asyncio
    async def test_detect_environment(self, mock_execute):
        """Test detecting environment."""
        detector = EnvironmentDetector(mock_execute)
        env = await detector.detect()

        assert "RTX 4090" in env.gpu_name
        assert env.gpu_arch == "sm_89"
        assert env.cuda_version == "12.1"

    @pytest.mark.asyncio
    async def test_check_container_runtime(self, mock_execute):
        """Test checking container runtimes."""
        detector = EnvironmentDetector(mock_execute)
        runtimes = await detector.check_container_runtime()

        assert runtimes["docker"] is True
        assert runtimes["singularity"] is False


class TestDeploymentStrategy:
    """Tests for DeploymentStrategy enum."""

    def test_all_strategies(self):
        """Test all strategies are defined."""
        expected = ["AUTO", "NGC_CONTAINER", "DOCKER", "SINGULARITY", "SQUASHFS", "SPACK", "CONDA"]
        actual = [s.name for s in DeploymentStrategy]

        for name in expected:
            assert name in actual


class TestPackageDeployer:
    """Tests for PackageDeployer."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock provider."""
        provider = AsyncMock()
        provider.execute_command = AsyncMock(side_effect=[
            # nvidia-smi query
            (0, "NVIDIA GeForce RTX 4090, 535.104.05, 24564, 1", ""),
            # nvcc version
            (0, "release 12.1", ""),
            # nproc
            (0, "16", ""),
            # free
            (0, "64", ""),
            # os-release
            (0, 'NAME="Ubuntu"\nVERSION="22.04"', ""),
            # docker version
            (0, "Docker 24.0", ""),
            # singularity version
            (1, "", "not found"),
            # podman version
            (1, "", "not found"),
            # docker pull
            (0, "Pull complete", ""),
            # create wrapper
            (0, "", ""),
            # verification command
            (0, "LAMMPS (29 Aug 2024)", ""),
        ])
        return provider

    def test_deployer_creation(self, mock_provider):
        """Test creating a deployer."""
        deployer = PackageDeployer(mock_provider)
        assert deployer._provider == mock_provider
        assert deployer._registry is not None


class TestInstanceEnvironment:
    """Tests for InstanceEnvironment dataclass."""

    def test_environment_str(self):
        """Test environment string representation."""
        env = InstanceEnvironment(
            gpu_name="NVIDIA GeForce RTX 4090",
            gpu_count=2,
            gpu_arch="sm_89",
            gpu_memory_gb=24,
            driver_version="535.104.05",
            cuda_version="12.1",
        )
        s = str(env)
        assert "RTX 4090" in s
        assert "sm_89" in s
        assert "12.1" in s


class TestGPUMapping:
    """Tests for GPU compute capability mapping."""

    def test_common_gpus(self):
        """Test mapping for common GPUs."""
        from cloudcomputemanager.packstore.detector import GPU_COMPUTE_CAPS

        # RTX 40 series
        assert GPU_COMPUTE_CAPS["NVIDIA GeForce RTX 4090"][0] == "sm_89"
        assert GPU_COMPUTE_CAPS["NVIDIA GeForce RTX 4080"][0] == "sm_89"

        # RTX 30 series
        assert GPU_COMPUTE_CAPS["NVIDIA GeForce RTX 3090"][0] == "sm_86"

        # Data center
        assert GPU_COMPUTE_CAPS["NVIDIA A100"][0] == "sm_80"
        assert GPU_COMPUTE_CAPS["NVIDIA H100"][0] == "sm_90"

    def test_fallback_matching(self):
        """Test fallback matching for GPU names."""
        detector = EnvironmentDetector(AsyncMock())

        # Test partial matching
        arch, mem = detector._get_compute_cap("NVIDIA GeForce RTX 4090 Laptop GPU")
        assert arch == "sm_89"

        # Test pattern extraction
        arch2, _ = detector._get_compute_cap("Some Unknown 4090 GPU")
        assert arch2 == "sm_89"
