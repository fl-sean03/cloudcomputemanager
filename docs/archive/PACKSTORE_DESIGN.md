# PackStore: Pre-Built Scientific Computing Package Management

## Executive Summary

PackStore is a module for CloudComputeManager that provides **validated, pre-built scientific computing packages** that can be quickly deployed to GPU cloud instances. Instead of debugging compatibility issues each time, users deploy packages that are **known to work**.

## Problem Statement

Deploying scientific computing software to GPU instances is painful:

1. **CUDA Version Hell**: LAMMPS built with CUDA 12.1 won't work with driver 525
2. **MPI Compatibility**: OpenMPI vs MPICH vs Intel MPI conflicts
3. **Dependency Chains**: QE needs FFTW, ScaLAPACK, ELPA, each with GPU variants
4. **Build Times**: Full LAMMPS build = 30+ minutes; QE = 60+ minutes
5. **Architecture Mismatch**: Binary built for sm_80 (A100) crashes on sm_89 (4090)

## Solution Architecture

```
PackStore System
├── Package Registry          # Catalog of validated packages
│   ├── NGC Containers        # Official NVIDIA containers
│   ├── SquashFS Images       # Pre-built compressed images
│   └── Spack Environments    # Reproducible build specs
│
├── Compatibility Matrix      # What works with what
│   ├── CUDA versions         # 12.1, 12.4, 13.0, etc.
│   ├── GPU architectures     # sm_80, sm_86, sm_89, sm_90
│   └── Driver requirements   # Minimum driver versions
│
├── Deployment Strategies     # How to get packages onto instances
│   ├── Container Pull        # Docker/Singularity pull
│   ├── SquashFS Mount        # Pre-built overlay
│   ├── Volume Attach         # Pre-seeded persistent volume
│   └── Spack Install         # Build from spec (fallback)
│
└── Validation Pipeline       # CI/CD for package testing
    ├── Build automation      # Multi-architecture builds
    ├── GPU testing           # Actual GPU validation
    └── Performance benchmarks # Ensure optimizations work
```

## Package Registry Design

### Package Manifest Schema

```yaml
# packstore/registry/lammps.yaml
package:
  name: lammps
  display_name: "LAMMPS Molecular Dynamics"
  category: molecular_dynamics
  homepage: https://www.lammps.org/

variants:
  - id: lammps-gpu-kokkos
    description: "GPU-accelerated with Kokkos backend"
    version: "29Aug2024"

    sources:
      # Priority 1: NGC container (fastest, most optimized)
      - type: ngc_container
        image: nvcr.io/hpc/lammps:29Aug2024
        size_mb: 2800
        pull_time_estimate: 45s  # on 1Gbps

      # Priority 2: Pre-built SquashFS (fast mount)
      - type: squashfs
        url: s3://packstore/lammps/lammps-29Aug2024-cuda12.4-sm80-sm89.sqsh
        size_mb: 850
        mount_path: /opt/lammps

      # Priority 3: Spack spec (build from source)
      - type: spack
        spec: "lammps@20240829 +kokkos +cuda cuda_arch=80,89 +openmp +mpi"
        build_time_estimate: 35m

    compatibility:
      cuda_versions: ["12.1", "12.4", "12.6", "13.0"]
      min_driver: "535.104"
      gpu_architectures:
        - sm_80  # A100
        - sm_86  # RTX 3090
        - sm_89  # RTX 4090
        - sm_90  # H100

    packages_included:
      - KOKKOS
      - KSPACE
      - MANYBODY
      - MOLECULE
      - REAXFF
      - ML-SNAP
      - OPENMP
      - MPI

    environment:
      PATH: "/opt/lammps/bin:$PATH"
      LAMMPS_POTENTIALS: "/opt/lammps/potentials"
      OMP_NUM_THREADS: "4"

    verification:
      test_command: "lmp -h | head -5"
      benchmark_command: "cd /opt/lammps/bench && mpirun -np 4 lmp -in in.lj"
      expected_output_contains: "LAMMPS"

  - id: lammps-cpu-omp
    description: "CPU-only with OpenMP threading"
    version: "29Aug2024"
    # ... similar structure
```

### Supported Packages (Initial Set)

| Package | Category | GPU | CPU+MPI | Notes |
|---------|----------|-----|---------|-------|
| **LAMMPS** | Molecular Dynamics | Kokkos-CUDA | OpenMP+MPI | Most common MD code |
| **Quantum ESPRESSO** | DFT | CUDA | OpenMP+MPI | Plane-wave DFT |
| **GROMACS** | Molecular Dynamics | CUDA | OpenMP+MPI | Biomolecular MD |
| **NAMD** | Molecular Dynamics | CUDA | Charm++ | Large biomolecular |
| **CP2K** | DFT/MD | CUDA | OpenMP+MPI | Mixed DFT/classical |
| **VASP** | DFT | CUDA | OpenMP+MPI | Licensed, user-supplied |
| **PyTorch** | ML Framework | CUDA | - | Deep learning |
| **TensorFlow** | ML Framework | CUDA | - | Deep learning |
| **JAX** | ML Framework | CUDA | - | Differentiable computing |
| **DeePMD-kit** | ML Potentials | CUDA | - | ML force fields |

## Compatibility Matrix

### CUDA Version ↔ Driver Requirements

```python
CUDA_DRIVER_MATRIX = {
    "13.1": {"min_driver": "555.42", "recommended": "560.35"},
    "13.0": {"min_driver": "550.54", "recommended": "555.42"},
    "12.6": {"min_driver": "535.154", "recommended": "545.23"},
    "12.4": {"min_driver": "535.104", "recommended": "545.23"},
    "12.1": {"min_driver": "530.30", "recommended": "535.104"},
    "11.8": {"min_driver": "520.61", "recommended": "525.85"},
}
```

### GPU Architecture Matrix

```python
GPU_ARCHITECTURES = {
    # Consumer GPUs
    "RTX_4090": {"compute": "sm_89", "cuda_cores": 16384, "vram_gb": 24},
    "RTX_4080": {"compute": "sm_89", "cuda_cores": 9728, "vram_gb": 16},
    "RTX_3090": {"compute": "sm_86", "cuda_cores": 10496, "vram_gb": 24},
    "RTX_3080": {"compute": "sm_86", "cuda_cores": 8704, "vram_gb": 10},

    # Data center GPUs
    "H100_SXM": {"compute": "sm_90", "cuda_cores": 16896, "vram_gb": 80},
    "H100_PCIe": {"compute": "sm_90", "cuda_cores": 14592, "vram_gb": 80},
    "A100_SXM": {"compute": "sm_80", "cuda_cores": 6912, "vram_gb": 80},
    "A100_PCIe": {"compute": "sm_80", "cuda_cores": 6912, "vram_gb": 40},
    "A10": {"compute": "sm_86", "cuda_cores": 9216, "vram_gb": 24},
    "V100": {"compute": "sm_70", "cuda_cores": 5120, "vram_gb": 32},
    "T4": {"compute": "sm_75", "cuda_cores": 2560, "vram_gb": 16},
}
```

## Deployment Strategies

### Strategy 1: NGC Container Pull (Recommended)

**Best for**: First-time setup, maximum optimization

```bash
# Automatic detection and pull
packstore deploy lammps-gpu-kokkos

# Behind the scenes:
docker pull nvcr.io/hpc/lammps:29Aug2024
# or with Singularity:
singularity pull docker://nvcr.io/hpc/lammps:29Aug2024
```

**Pros**:
- Always gets latest security patches
- NVIDIA-optimized builds
- Well-tested containers

**Cons**:
- 2-5 minute pull time (1-3 GB images)
- Requires internet access

### Strategy 2: SquashFS Overlay Mount (Fastest)

**Best for**: Rapid deployment, offline capability

```bash
# Pre-download to local storage
packstore cache lammps-gpu-kokkos

# Creates: ~/.packstore/cache/lammps-29Aug2024-cuda12.4.sqsh (850MB)

# On instance deployment:
mount -t squashfs ~/.packstore/cache/lammps-*.sqsh /opt/lammps -o ro,loop
# Or with overlay for writable layer:
mount -t overlay overlay -o lowerdir=/opt/lammps,upperdir=/tmp/upper,workdir=/tmp/work /opt/lammps
```

**Building SquashFS images**:
```bash
# From NGC container
docker export $(docker create nvcr.io/hpc/lammps:29Aug2024) | \
  tar2sqfs -q lammps.sqsh

# Or from installed directory
mksquashfs /opt/lammps lammps.sqsh -comp zstd -Xcompression-level 19
```

**Pros**:
- 5-10 second mount time
- 50-70% smaller than uncompressed
- 6-10x file access performance vs NFS
- Works offline once cached

**Cons**:
- Requires pre-building images
- Less flexible than containers

### Strategy 3: Pre-Seeded Volumes

**Best for**: Persistent instances, large datasets

```bash
# Create a Vast.ai volume with pre-installed software
packstore volume create scientific-compute --size 200GB

# Volume contains:
# /opt/lammps/       - LAMMPS installation
# /opt/qe/           - Quantum ESPRESSO
# /opt/gromacs/      - GROMACS
# /opt/conda/        - Miniconda with ML packages
# /data/potentials/  - Shared potential files

# Attach to any instance
vast attach volume scientific-compute --mount /opt
```

**Pros**:
- Instant availability (already there)
- Shared across instances
- Can include large datasets

**Cons**:
- Vast.ai volume costs
- Region-specific

### Strategy 4: Spack Build (Fallback)

**Best for**: Custom configurations, debugging

```bash
# Generate Spack environment from registry
packstore spack-env lammps-gpu-kokkos > spack.yaml

# Contents:
spack:
  specs:
    - lammps@20240829 +kokkos +cuda cuda_arch=89 +openmp +mpi ^openmpi
  concretizer:
    unify: true
  config:
    install_tree:
      root: /opt/spack
```

**Pros**:
- Full customization
- Debug symbols available
- Specific architecture tuning

**Cons**:
- 30-60 minute build times
- Requires build dependencies

## Implementation Plan

### Phase 1: Package Registry (Core)

```python
# packstore/registry.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import yaml

class SourceType(Enum):
    NGC_CONTAINER = "ngc_container"
    DOCKER = "docker"
    SQUASHFS = "squashfs"
    SPACK = "spack"
    CONDA = "conda"

@dataclass
class PackageSource:
    type: SourceType
    image: Optional[str] = None      # For containers
    url: Optional[str] = None        # For downloads
    spec: Optional[str] = None       # For Spack
    size_mb: int = 0
    priority: int = 0                # Lower = try first

@dataclass
class Compatibility:
    cuda_versions: list[str]
    min_driver: str
    gpu_architectures: list[str]

    def is_compatible(self, cuda: str, driver: str, gpu_arch: str) -> bool:
        """Check if this package works with given environment."""
        return (
            cuda in self.cuda_versions and
            self._driver_gte(driver, self.min_driver) and
            gpu_arch in self.gpu_architectures
        )

@dataclass
class PackageVariant:
    id: str
    name: str
    version: str
    description: str
    sources: list[PackageSource]
    compatibility: Compatibility
    environment: dict[str, str]
    verification: dict[str, str]

class PackageRegistry:
    """Registry of available packages."""

    def __init__(self, registry_dir: Path):
        self.packages = self._load_registry(registry_dir)

    def find_compatible(
        self,
        package: str,
        cuda_version: str,
        gpu_arch: str,
        driver_version: str,
    ) -> Optional[PackageVariant]:
        """Find best compatible variant for given environment."""
        ...

    def get_deployment_plan(
        self,
        packages: list[str],
        instance_info: InstanceInfo,
    ) -> DeploymentPlan:
        """Generate deployment plan for multiple packages."""
        ...
```

### Phase 2: Deployment Engine

```python
# packstore/deployer.py

class PackageDeployer:
    """Deploy packages to instances."""

    async def deploy(
        self,
        instance_id: str,
        packages: list[str],
        strategy: Optional[DeploymentStrategy] = None,
    ) -> DeploymentResult:
        """Deploy packages using best available strategy."""

        # 1. Detect instance environment
        env = await self._detect_environment(instance_id)

        # 2. Find compatible variants
        variants = [
            self.registry.find_compatible(pkg, env.cuda, env.gpu_arch, env.driver)
            for pkg in packages
        ]

        # 3. Select deployment strategy
        if strategy is None:
            strategy = self._select_strategy(env, variants)

        # 4. Execute deployment
        if strategy == DeploymentStrategy.NGC_CONTAINER:
            return await self._deploy_containers(instance_id, variants)
        elif strategy == DeploymentStrategy.SQUASHFS:
            return await self._deploy_squashfs(instance_id, variants)
        elif strategy == DeploymentStrategy.SPACK:
            return await self._deploy_spack(instance_id, variants)

    async def _detect_environment(self, instance_id: str) -> InstanceEnvironment:
        """Detect CUDA version, GPU architecture, driver version."""

        # nvidia-smi query
        cmd = "nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader"
        _, stdout, _ = await self.provider.execute_command(instance_id, cmd)

        # Parse output
        gpu_name, driver, compute_cap = stdout.strip().split(", ")

        # CUDA version
        cmd = "nvcc --version | grep release | awk '{print $6}' | tr -d 'V'"
        _, cuda_version, _ = await self.provider.execute_command(instance_id, cmd)

        return InstanceEnvironment(
            gpu_name=gpu_name,
            gpu_arch=f"sm_{compute_cap.replace('.', '')}",
            driver_version=driver,
            cuda_version=cuda_version.strip(),
        )
```

### Phase 3: SquashFS Builder

```python
# packstore/builder.py

class SquashFSBuilder:
    """Build optimized SquashFS images from containers."""

    async def build_from_ngc(
        self,
        ngc_image: str,
        output_path: Path,
        compression: str = "zstd",
        compression_level: int = 15,
    ) -> SquashFSImage:
        """Build SquashFS from NGC container."""

        # Pull and export container
        container_id = await self._docker_create(ngc_image)
        tar_path = await self._docker_export(container_id)

        # Convert to SquashFS
        await self._tar_to_squashfs(
            tar_path,
            output_path,
            compression=compression,
            level=compression_level,
        )

        # Generate metadata
        return SquashFSImage(
            path=output_path,
            source_image=ngc_image,
            size_bytes=output_path.stat().st_size,
            compression=compression,
            created_at=datetime.utcnow(),
        )
```

### Phase 4: CLI Commands

```bash
# List available packages
vm packstore list
vm packstore list --category molecular_dynamics

# Search packages
vm packstore search lammps
vm packstore search --gpu-arch sm_89 --cuda 12.4

# Show package details
vm packstore info lammps-gpu-kokkos

# Deploy to instance
vm packstore deploy lammps-gpu-kokkos --instance 12345678
vm packstore deploy lammps qe gromacs --instance 12345678

# Cache packages locally (for offline use)
vm packstore cache lammps-gpu-kokkos
vm packstore cache --all-molecular-dynamics

# Build custom SquashFS
vm packstore build-squashfs nvcr.io/hpc/lammps:29Aug2024 -o lammps.sqsh

# Verify installation
vm packstore verify lammps --instance 12345678
```

## Pre-Built Package Matrix

### Tier 1: NGC Containers (Fastest Setup)

| Package | NGC Image | Size | GPU Arch | Notes |
|---------|-----------|------|----------|-------|
| LAMMPS | `nvcr.io/hpc/lammps:29Aug2024` | 2.8GB | sm70+ | Kokkos backend |
| Quantum ESPRESSO | `nvcr.io/hpc/quantum_espresso:v7.1` | 3.2GB | sm80+ | QE 7.1 |
| GROMACS | `nvcr.io/hpc/gromacs:2023.2` | 2.5GB | sm70+ | CUDA graphs |
| NAMD | `nvcr.io/hpc/namd:3.0-alpha3-singlenode` | 1.8GB | sm60+ | Single-node |
| PyTorch | `nvcr.io/nvidia/pytorch:24.01-py3` | 8.5GB | sm70+ | Full stack |
| TensorFlow | `nvcr.io/nvidia/tensorflow:24.01-tf2-py3` | 9.2GB | sm70+ | TF 2.x |

### Tier 2: SquashFS Images (Fastest Deploy)

Build these from NGC containers for rapid deployment:

```bash
# One-time build (do this on your local machine)
packstore build-squashfs nvcr.io/hpc/lammps:29Aug2024 \
  --output lammps-29Aug2024.sqsh \
  --upload s3://your-bucket/packstore/

# Results in ~850MB compressed image (vs 2.8GB container)
```

### Tier 3: Conda Environments (Most Flexible)

For ML packages and Python-heavy workflows:

```yaml
# packstore/environments/ml-simulation.yaml
name: ml-simulation
channels:
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  - python=3.11
  - pytorch-cuda=12.1
  - deepmd-kit
  - lammps  # with LAMMPS-DeePMD interface
  - ase
  - pymatgen
  - e3nn
```

## Storage Architecture

### Local Cache Structure

```
~/.packstore/
├── cache/
│   ├── squashfs/
│   │   ├── lammps-29Aug2024-sm80.sqsh
│   │   └── qe-7.1-sm89.sqsh
│   ├── containers/
│   │   └── nvcr.io-hpc-lammps-29Aug2024.tar
│   └── conda/
│       └── ml-simulation.tar.gz
├── registry/
│   ├── lammps.yaml
│   ├── qe.yaml
│   └── gromacs.yaml
└── config.yaml
```

### Remote Storage (S3/GCS)

```
s3://packstore-{region}/
├── squashfs/
│   ├── lammps/
│   │   ├── 29Aug2024-cuda12.4-sm80-sm89.sqsh
│   │   └── manifest.json
│   └── qe/
│       └── ...
├── registry/
│   └── packages.json
└── compatibility/
    └── matrix.json
```

## Integration with CloudComputeManager

### Job Configuration

```yaml
# job.yaml
name: lammps-simulation
project: mxenes

# New: specify required packages
packages:
  - lammps-gpu-kokkos
  - deepmd-kit  # optional: ML potential support

# Deployment strategy (optional, auto-detected)
package_strategy: squashfs  # or: ngc, spack, conda

# Rest of job config...
image: ubuntu:22.04  # Base image, packages overlaid
command: |
  source /opt/lammps/env.sh
  mpirun -np 4 lmp -in simulation.in
```

### API Extensions

```python
# New endpoint
POST /v1/packages/deploy
{
    "instance_id": "12345678",
    "packages": ["lammps-gpu-kokkos", "deepmd-kit"],
    "strategy": "auto"
}

# Response
{
    "deployment_id": "dep_abc123",
    "packages": [
        {
            "name": "lammps-gpu-kokkos",
            "status": "deployed",
            "source": "squashfs",
            "mount_path": "/opt/lammps"
        },
        {
            "name": "deepmd-kit",
            "status": "deployed",
            "source": "conda",
            "env_path": "/opt/conda/envs/deepmd"
        }
    ],
    "environment": {
        "PATH": "/opt/lammps/bin:/opt/conda/envs/deepmd/bin:$PATH",
        "LAMMPS_POTENTIALS": "/opt/lammps/potentials"
    }
}
```

## Validation Pipeline

### Automated Testing

```yaml
# .github/workflows/validate-packages.yaml
name: Validate PackStore

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  validate:
    runs-on: [self-hosted, gpu]
    strategy:
      matrix:
        package: [lammps, qe, gromacs]
        gpu: [rtx4090, a100]

    steps:
      - name: Deploy package
        run: vm packstore deploy ${{ matrix.package }}-gpu

      - name: Run validation
        run: vm packstore verify ${{ matrix.package }}

      - name: Run benchmark
        run: vm packstore benchmark ${{ matrix.package }} --report
```

## Success Metrics

1. **Deployment Time**: < 60 seconds for SquashFS, < 5 minutes for NGC pull
2. **Compatibility**: 99% success rate for supported GPU/CUDA combinations
3. **Performance**: Within 5% of native compilation performance
4. **Coverage**: 10+ validated packages for scientific computing

## References

- [NVIDIA NGC Catalog - HPC Collection](https://catalog.ngc.nvidia.com/orgs/hpc/collections/nvidia_hpc)
- [NVIDIA NGC - LAMMPS Container](https://catalog.ngc.nvidia.com/orgs/hpc/containers/lammps)
- [NVIDIA NGC - Quantum ESPRESSO Container](https://catalog.ngc.nvidia.com/orgs/hpc/containers/quantum_espresso)
- [NVIDIA NGC - GROMACS Container](https://catalog.ngc.nvidia.com/orgs/hpc/containers/gromacs)
- [Spack Documentation - CUDA Support](https://spack.readthedocs.io/en/latest/build_systems/cudapackage.html)
- [EasyBuild HPC Package Manager](https://easybuild.io/)
- [Apptainer GPU Support](https://apptainer.org/docs/user/1.0/gpu.html)
- [SquashFS for HPC Containers](https://arxiv.org/pdf/2002.06129)
- [NVIDIA HPC SDK Documentation](https://docs.nvidia.com/hpc-sdk/index.html)
