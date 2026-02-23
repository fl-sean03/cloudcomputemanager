"""Package registry for PackStore.

Defines the catalog of available packages with their variants, sources,
and compatibility information.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import re

import yaml
import structlog

logger = structlog.get_logger(__name__)


class SourceType(str, Enum):
    """Type of package source."""

    NGC_CONTAINER = "ngc_container"
    DOCKER = "docker"
    SINGULARITY = "singularity"
    SQUASHFS = "squashfs"
    SPACK = "spack"
    CONDA = "conda"
    BINARY = "binary"


class PackageCategory(str, Enum):
    """Category of package."""

    MOLECULAR_DYNAMICS = "molecular_dynamics"
    DFT = "dft"
    ML_FRAMEWORK = "ml_framework"
    ML_POTENTIAL = "ml_potential"
    VISUALIZATION = "visualization"
    UTILITY = "utility"


@dataclass
class PackageSource:
    """A source from which a package can be obtained."""

    type: SourceType
    priority: int = 0  # Lower = try first

    # Container sources
    image: Optional[str] = None
    tag: Optional[str] = None

    # Download sources
    url: Optional[str] = None
    checksum: Optional[str] = None

    # Build sources
    spec: Optional[str] = None  # Spack spec
    env_file: Optional[str] = None  # Conda env file

    # Metadata
    size_mb: int = 0
    pull_time_estimate_seconds: int = 0
    build_time_estimate_minutes: int = 0

    @property
    def full_image(self) -> str:
        """Get full image reference with tag."""
        if self.tag:
            return f"{self.image}:{self.tag}"
        return self.image or ""


@dataclass
class Compatibility:
    """Compatibility requirements for a package variant."""

    cuda_versions: list[str] = field(default_factory=list)
    min_driver: str = "535.0"
    max_driver: Optional[str] = None
    gpu_architectures: list[str] = field(default_factory=list)  # sm_80, sm_89, etc.

    # Optional constraints
    min_gpu_memory_gb: int = 0
    supported_os: list[str] = field(default_factory=lambda: ["linux"])

    def is_compatible(
        self,
        cuda_version: str,
        driver_version: str,
        gpu_arch: str,
        gpu_memory_gb: int = 0,
    ) -> tuple[bool, Optional[str]]:
        """Check if this package is compatible with the given environment.

        Returns:
            Tuple of (is_compatible, reason_if_not)
        """
        # Check CUDA version
        if self.cuda_versions and cuda_version not in self.cuda_versions:
            # Check major version match
            cuda_major = cuda_version.split(".")[0]
            if not any(cv.startswith(cuda_major) for cv in self.cuda_versions):
                return False, f"CUDA {cuda_version} not in supported: {self.cuda_versions}"

        # Check driver version
        if not self._version_gte(driver_version, self.min_driver):
            return False, f"Driver {driver_version} < minimum {self.min_driver}"

        if self.max_driver and self._version_gte(driver_version, self.max_driver):
            return False, f"Driver {driver_version} > maximum {self.max_driver}"

        # Check GPU architecture
        if self.gpu_architectures and gpu_arch not in self.gpu_architectures:
            return False, f"GPU arch {gpu_arch} not in supported: {self.gpu_architectures}"

        # Check GPU memory
        if self.min_gpu_memory_gb and gpu_memory_gb < self.min_gpu_memory_gb:
            return False, f"GPU memory {gpu_memory_gb}GB < minimum {self.min_gpu_memory_gb}GB"

        return True, None

    @staticmethod
    def _version_gte(version: str, min_version: str) -> bool:
        """Check if version >= min_version."""

        def parse_version(v: str) -> list[int]:
            return [int(x) for x in re.findall(r"\d+", v)]

        return parse_version(version) >= parse_version(min_version)


@dataclass
class PackageVariant:
    """A specific variant of a package (e.g., GPU vs CPU build)."""

    id: str
    name: str
    version: str
    description: str

    sources: list[PackageSource] = field(default_factory=list)
    compatibility: Compatibility = field(default_factory=Compatibility)

    # Environment variables to set
    environment: dict[str, str] = field(default_factory=dict)

    # Packages/modules included
    packages_included: list[str] = field(default_factory=list)

    # Verification commands
    test_command: Optional[str] = None
    benchmark_command: Optional[str] = None
    expected_output_contains: Optional[str] = None

    # Installation paths
    install_path: str = "/opt"
    bin_path: Optional[str] = None
    lib_path: Optional[str] = None

    def get_best_source(self, prefer_local: bool = False) -> Optional[PackageSource]:
        """Get the best available source based on priority."""
        if not self.sources:
            return None

        sources = sorted(self.sources, key=lambda s: s.priority)

        if prefer_local:
            # Prefer SquashFS and local sources
            local_sources = [s for s in sources if s.type in [SourceType.SQUASHFS, SourceType.BINARY]]
            if local_sources:
                return local_sources[0]

        return sources[0]


@dataclass
class Package:
    """A scientific computing package with multiple variants."""

    name: str
    display_name: str
    category: PackageCategory
    description: str
    homepage: Optional[str] = None
    license: Optional[str] = None

    variants: list[PackageVariant] = field(default_factory=list)

    def get_variant(self, variant_id: str) -> Optional[PackageVariant]:
        """Get a specific variant by ID."""
        for v in self.variants:
            if v.id == variant_id:
                return v
        return None

    def find_compatible_variant(
        self,
        cuda_version: str,
        driver_version: str,
        gpu_arch: str,
        gpu_memory_gb: int = 0,
        prefer_gpu: bool = True,
    ) -> Optional[PackageVariant]:
        """Find the best compatible variant for the given environment."""
        compatible = []

        for variant in self.variants:
            is_compat, _ = variant.compatibility.is_compatible(
                cuda_version, driver_version, gpu_arch, gpu_memory_gb
            )
            if is_compat:
                compatible.append(variant)

        if not compatible:
            return None

        # Sort by preference (GPU variants first if preferred)
        if prefer_gpu:
            gpu_variants = [v for v in compatible if "gpu" in v.id.lower() or "cuda" in v.id.lower()]
            if gpu_variants:
                return gpu_variants[0]

        return compatible[0]


class PackageRegistry:
    """Registry of available packages."""

    def __init__(self, registry_dir: Optional[Path] = None):
        """Initialize the registry.

        Args:
            registry_dir: Directory containing package YAML files
        """
        self._packages: dict[str, Package] = {}

        if registry_dir and registry_dir.exists():
            self._load_from_directory(registry_dir)
        else:
            self._load_builtin_packages()

    def _load_from_directory(self, registry_dir: Path) -> None:
        """Load packages from YAML files in a directory."""
        for yaml_file in registry_dir.glob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                package = self._parse_package(data)
                self._packages[package.name] = package
                logger.debug("Loaded package", package=package.name, variants=len(package.variants))
            except Exception as e:
                logger.error("Failed to load package", file=str(yaml_file), error=str(e))

    def _load_builtin_packages(self) -> None:
        """Load built-in package definitions."""
        # LAMMPS
        self._packages["lammps"] = Package(
            name="lammps",
            display_name="LAMMPS",
            category=PackageCategory.MOLECULAR_DYNAMICS,
            description="Large-scale Atomic/Molecular Massively Parallel Simulator",
            homepage="https://www.lammps.org/",
            license="GPL-2.0",
            variants=[
                PackageVariant(
                    id="lammps-gpu-kokkos",
                    name="LAMMPS GPU (Kokkos)",
                    version="29Aug2024",
                    description="GPU-accelerated LAMMPS with Kokkos backend",
                    sources=[
                        PackageSource(
                            type=SourceType.NGC_CONTAINER,
                            image="nvcr.io/hpc/lammps",
                            tag="29Aug2024",
                            priority=0,
                            size_mb=2800,
                            pull_time_estimate_seconds=120,
                        ),
                        PackageSource(
                            type=SourceType.SPACK,
                            spec="lammps@20240829 +kokkos +cuda cuda_arch=80,89 +openmp +mpi",
                            priority=10,
                            build_time_estimate_minutes=35,
                        ),
                    ],
                    compatibility=Compatibility(
                        cuda_versions=["12.1", "12.2", "12.4", "12.6", "13.0"],
                        min_driver="535.104",
                        gpu_architectures=["sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90"],
                    ),
                    packages_included=["KOKKOS", "KSPACE", "MANYBODY", "MOLECULE", "REAXFF", "ML-SNAP", "OPENMP", "MPI"],
                    environment={
                        "PATH": "/opt/lammps/bin:$PATH",
                        "LAMMPS_POTENTIALS": "/opt/lammps/potentials",
                    },
                    test_command="lmp -h | head -5",
                    expected_output_contains="LAMMPS",
                ),
                PackageVariant(
                    id="lammps-cpu-omp",
                    name="LAMMPS CPU (OpenMP)",
                    version="29Aug2024",
                    description="CPU-only LAMMPS with OpenMP threading",
                    sources=[
                        PackageSource(
                            type=SourceType.SPACK,
                            spec="lammps@20240829 +openmp +mpi ~cuda",
                            priority=0,
                            build_time_estimate_minutes=20,
                        ),
                    ],
                    compatibility=Compatibility(
                        cuda_versions=[],  # No CUDA required
                        min_driver="0",
                        gpu_architectures=[],  # Any
                    ),
                    packages_included=["KSPACE", "MANYBODY", "MOLECULE", "REAXFF", "OPENMP", "MPI"],
                    environment={
                        "PATH": "/opt/lammps/bin:$PATH",
                        "OMP_NUM_THREADS": "4",
                    },
                ),
            ],
        )

        # Quantum ESPRESSO
        self._packages["quantum-espresso"] = Package(
            name="quantum-espresso",
            display_name="Quantum ESPRESSO",
            category=PackageCategory.DFT,
            description="Integrated suite for electronic-structure calculations",
            homepage="https://www.quantum-espresso.org/",
            license="GPL-2.0",
            variants=[
                PackageVariant(
                    id="qe-gpu",
                    name="Quantum ESPRESSO GPU",
                    version="7.1",
                    description="GPU-accelerated Quantum ESPRESSO",
                    sources=[
                        PackageSource(
                            type=SourceType.NGC_CONTAINER,
                            image="nvcr.io/hpc/quantum_espresso",
                            tag="v7.1",
                            priority=0,
                            size_mb=3200,
                            pull_time_estimate_seconds=150,
                        ),
                    ],
                    compatibility=Compatibility(
                        cuda_versions=["12.1", "12.4"],
                        min_driver="535.104",
                        gpu_architectures=["sm_80", "sm_86", "sm_89"],  # Note: H100 has issues with 7.1
                        min_gpu_memory_gb=16,
                    ),
                    environment={
                        "PATH": "/opt/qe/bin:$PATH",
                        "PSEUDO_DIR": "/opt/qe/pseudo",
                    },
                    test_command="pw.x -h | head -3",
                    expected_output_contains="PWSCF",
                ),
            ],
        )

        # GROMACS
        self._packages["gromacs"] = Package(
            name="gromacs",
            display_name="GROMACS",
            category=PackageCategory.MOLECULAR_DYNAMICS,
            description="Versatile package for molecular dynamics",
            homepage="https://www.gromacs.org/",
            license="LGPL-2.1",
            variants=[
                PackageVariant(
                    id="gromacs-gpu",
                    name="GROMACS GPU",
                    version="2023.2",
                    description="GPU-accelerated GROMACS with CUDA Graphs",
                    sources=[
                        PackageSource(
                            type=SourceType.NGC_CONTAINER,
                            image="nvcr.io/hpc/gromacs",
                            tag="2023.2",
                            priority=0,
                            size_mb=2500,
                            pull_time_estimate_seconds=100,
                        ),
                    ],
                    compatibility=Compatibility(
                        cuda_versions=["12.1", "12.4", "12.6"],
                        min_driver="535.104",
                        gpu_architectures=["sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90"],
                    ),
                    environment={
                        "PATH": "/opt/gromacs/bin:$PATH",
                        "GMX_ENABLE_DIRECT_GPU_COMM": "1",
                    },
                    test_command="gmx -h | head -5",
                    expected_output_contains="GROMACS",
                ),
            ],
        )

        # NAMD
        self._packages["namd"] = Package(
            name="namd",
            display_name="NAMD",
            category=PackageCategory.MOLECULAR_DYNAMICS,
            description="Parallel molecular dynamics for biomolecular systems",
            homepage="https://www.ks.uiuc.edu/Research/namd/",
            license="UIUC",
            variants=[
                PackageVariant(
                    id="namd-gpu-singlenode",
                    name="NAMD GPU (Single Node)",
                    version="3.0-alpha3",
                    description="NAMD 3.0 with GPU acceleration (single node)",
                    sources=[
                        PackageSource(
                            type=SourceType.NGC_CONTAINER,
                            image="nvcr.io/hpc/namd",
                            tag="3.0-alpha3-singlenode",
                            priority=0,
                            size_mb=1800,
                            pull_time_estimate_seconds=80,
                        ),
                    ],
                    compatibility=Compatibility(
                        cuda_versions=["12.1", "12.4"],
                        min_driver="535.104",
                        gpu_architectures=["sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_89"],
                    ),
                    environment={
                        "PATH": "/opt/namd/bin:$PATH",
                    },
                    test_command="namd3 --help | head -5",
                    expected_output_contains="NAMD",
                ),
            ],
        )

        # PyTorch
        self._packages["pytorch"] = Package(
            name="pytorch",
            display_name="PyTorch",
            category=PackageCategory.ML_FRAMEWORK,
            description="Deep learning framework",
            homepage="https://pytorch.org/",
            license="BSD-3-Clause",
            variants=[
                PackageVariant(
                    id="pytorch-gpu-ngc",
                    name="PyTorch NGC",
                    version="24.01",
                    description="NVIDIA-optimized PyTorch container",
                    sources=[
                        PackageSource(
                            type=SourceType.NGC_CONTAINER,
                            image="nvcr.io/nvidia/pytorch",
                            tag="24.01-py3",
                            priority=0,
                            size_mb=8500,
                            pull_time_estimate_seconds=300,
                        ),
                    ],
                    compatibility=Compatibility(
                        cuda_versions=["12.1", "12.2", "12.3"],
                        min_driver="535.104",
                        gpu_architectures=["sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90"],
                    ),
                    environment={
                        "PATH": "/opt/conda/bin:$PATH",
                    },
                    test_command="python -c 'import torch; print(torch.cuda.is_available())'",
                    expected_output_contains="True",
                ),
                PackageVariant(
                    id="pytorch-gpu-conda",
                    name="PyTorch (Conda)",
                    version="2.2.0",
                    description="PyTorch installed via conda-forge",
                    sources=[
                        PackageSource(
                            type=SourceType.CONDA,
                            env_file="pytorch-gpu.yaml",
                            priority=0,
                            build_time_estimate_minutes=10,
                        ),
                    ],
                    compatibility=Compatibility(
                        cuda_versions=["12.1"],
                        min_driver="525.60",
                        gpu_architectures=["sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90"],
                    ),
                ),
            ],
        )

        # DeePMD-kit
        self._packages["deepmd-kit"] = Package(
            name="deepmd-kit",
            display_name="DeePMD-kit",
            category=PackageCategory.ML_POTENTIAL,
            description="Deep learning-based machine learning potential",
            homepage="https://github.com/deepmodeling/deepmd-kit",
            license="LGPL-3.0",
            variants=[
                PackageVariant(
                    id="deepmd-kit-gpu",
                    name="DeePMD-kit GPU",
                    version="2.2.0",
                    description="DeePMD-kit with LAMMPS interface",
                    sources=[
                        PackageSource(
                            type=SourceType.CONDA,
                            spec="deepmd-kit lammps cudatoolkit=11.8",
                            priority=0,
                            build_time_estimate_minutes=15,
                        ),
                    ],
                    compatibility=Compatibility(
                        cuda_versions=["11.8", "12.1"],
                        min_driver="520.61",
                        gpu_architectures=["sm_70", "sm_75", "sm_80", "sm_86", "sm_89"],
                    ),
                    environment={
                        "PATH": "/opt/conda/envs/deepmd/bin:$PATH",
                    },
                    test_command="dp -h | head -3",
                    expected_output_contains="DeePMD-kit",
                ),
            ],
        )

        logger.info("Loaded builtin packages", count=len(self._packages))

    def _parse_package(self, data: dict) -> Package:
        """Parse a package definition from YAML data."""
        # Implementation for loading from YAML
        pkg_data = data.get("package", {})
        variants_data = data.get("variants", [])

        variants = []
        for v_data in variants_data:
            sources = []
            for s_data in v_data.get("sources", []):
                sources.append(PackageSource(
                    type=SourceType(s_data["type"]),
                    image=s_data.get("image"),
                    tag=s_data.get("tag"),
                    url=s_data.get("url"),
                    spec=s_data.get("spec"),
                    priority=s_data.get("priority", 0),
                    size_mb=s_data.get("size_mb", 0),
                ))

            compat_data = v_data.get("compatibility", {})
            compatibility = Compatibility(
                cuda_versions=compat_data.get("cuda_versions", []),
                min_driver=compat_data.get("min_driver", "0"),
                gpu_architectures=compat_data.get("gpu_architectures", []),
            )

            variants.append(PackageVariant(
                id=v_data["id"],
                name=v_data.get("name", v_data["id"]),
                version=v_data.get("version", "latest"),
                description=v_data.get("description", ""),
                sources=sources,
                compatibility=compatibility,
                environment=v_data.get("environment", {}),
                packages_included=v_data.get("packages_included", []),
                test_command=v_data.get("verification", {}).get("test_command"),
                expected_output_contains=v_data.get("verification", {}).get("expected_output_contains"),
            ))

        return Package(
            name=pkg_data.get("name", "unknown"),
            display_name=pkg_data.get("display_name", pkg_data.get("name", "Unknown")),
            category=PackageCategory(pkg_data.get("category", "utility")),
            description=pkg_data.get("description", ""),
            homepage=pkg_data.get("homepage"),
            license=pkg_data.get("license"),
            variants=variants,
        )

    def get(self, name: str) -> Optional[Package]:
        """Get a package by name."""
        return self._packages.get(name)

    def get_variant(self, package_name: str, variant_id: str) -> Optional[PackageVariant]:
        """Get a specific variant of a package."""
        package = self.get(package_name)
        if package:
            return package.get_variant(variant_id)
        return None

    def list_packages(self, category: Optional[PackageCategory] = None) -> list[Package]:
        """List all packages, optionally filtered by category."""
        packages = list(self._packages.values())
        if category:
            packages = [p for p in packages if p.category == category]
        return packages

    def search(
        self,
        query: str = "",
        category: Optional[PackageCategory] = None,
        cuda_version: Optional[str] = None,
        gpu_arch: Optional[str] = None,
    ) -> list[Package]:
        """Search for packages matching criteria."""
        results = []

        for package in self._packages.values():
            # Filter by category
            if category and package.category != category:
                continue

            # Filter by query
            if query:
                query_lower = query.lower()
                if not (
                    query_lower in package.name.lower()
                    or query_lower in package.display_name.lower()
                    or query_lower in package.description.lower()
                ):
                    continue

            # Filter by compatibility
            if cuda_version or gpu_arch:
                has_compatible = False
                for variant in package.variants:
                    is_compat, _ = variant.compatibility.is_compatible(
                        cuda_version or "12.1",
                        "535.104",
                        gpu_arch or "sm_89",
                    )
                    if is_compat:
                        has_compatible = True
                        break
                if not has_compatible:
                    continue

            results.append(package)

        return results

    def find_compatible(
        self,
        package_name: str,
        cuda_version: str,
        driver_version: str,
        gpu_arch: str,
        gpu_memory_gb: int = 0,
    ) -> Optional[PackageVariant]:
        """Find the best compatible variant for a package."""
        package = self.get(package_name)
        if not package:
            return None

        return package.find_compatible_variant(
            cuda_version, driver_version, gpu_arch, gpu_memory_gb
        )
