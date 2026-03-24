# CCM Environment Management Design

## Problem Statement

Scientific workloads often require complex software environments (conda
environments with 20+ packages, specific CUDA versions, compiled libraries).
Currently, CCM supports two approaches:

1. **Docker image:** Specify a public image (fast boot, but requires pre-building)
2. **Setup script:** Install packages at runtime via `setup:` field (flexible, but slow)

Both have significant limitations for real-world use cases:
- Setup scripts take 10-20 minutes for complex environments (conda + pip)
- Pre-built Docker images require Docker Hub accounts and manual image management
- Neither approach captures the user's existing local environment automatically
- Package versions can drift between local and cloud, breaking reproducibility

## Proposed Architecture: Environment Definitions

Add a first-class `environment:` field to the job YAML that supports multiple
strategies for environment setup, chosen automatically based on what the user
provides.

### Job YAML: New `environment:` Field

```yaml
name: my-simulation
image: nvidia/cuda:12.1.0-runtime-ubuntu22.04

# NEW: Environment definition (replaces or supplements `setup:`)
environment:
  # Strategy 1: Conda environment file (exported from local)
  conda_env: ./environment.yml

  # Strategy 2: Requirements file (pip)
  requirements: ./requirements.txt

  # Strategy 3: Pre-built Docker image (fastest)
  # (This overrides the top-level `image:` field)
  docker_image: myuser/openmm-cuda12:latest

  # Strategy 4: Conda pack (pre-packaged tarball)
  conda_pack: ./openmm-env.tar.gz

  # Strategy 5: Inline package list (simple cases)
  packages:
    conda:
      - openmm
      - openmmtools
      - pymbar>=4.0
    pip:
      - alchemlyb
      - openff-toolkit
    apt:
      - libfftw3-dev

  # Common options
  python_version: "3.11"          # Desired Python version
  cuda_version: "12.1"            # Required CUDA version
  channels: ["conda-forge"]        # Conda channels
  cache: true                      # Cache environment between jobs (reuse instances)

command: python run_simulation.py
```

### Strategy Selection (Automatic)

CCM selects the fastest strategy based on what is provided:

| Priority | Strategy | When Used | Setup Time |
|----------|----------|-----------|------------|
| 1 | Docker image | `docker_image:` specified | <1 min |
| 2 | Conda pack | `conda_pack:` specified | 1-2 min (upload + unpack) |
| 3 | Conda env file | `conda_env:` specified | 5-15 min |
| 4 | Inline packages | `packages:` specified | 5-15 min |
| 5 | Requirements.txt | `requirements:` specified | 2-5 min |

If multiple are specified, CCM uses the highest-priority option.

### Implementation Plan

#### Phase 1: Conda Environment Export (Low effort, high value)

Add a CLI command to export the current local conda environment and reference
it in a job YAML:

```bash
# Export current environment to a file
ccm env export --conda-env myenv -o environment.yml

# Or auto-detect from current active environment
ccm env export -o environment.yml
```

The exported file is a standard `conda env export` YAML. On the cloud instance,
CCM installs miniconda (if not present) and runs `conda env create -f`.

Implementation:
- New CLI command in `cli/environments.py`
- In `cli/jobs.py` submit flow: detect `environment.conda_env`, upload the file,
  add conda install to the setup commands
- Automatically prepend `conda activate <env>` to the job command

#### Phase 2: Conda Pack (Medium effort, fastest runtime)

Conda-pack creates a relocatable tarball of a conda environment that can be
unpacked on any Linux machine without conda installed.

```bash
# Create conda pack from local environment
ccm env pack --conda-env alchem -o alchem-env.tar.gz

# Use in job YAML
environment:
  conda_pack: ./alchem-env.tar.gz
```

On the cloud instance, CCM:
1. Uploads the tarball (5-10 GB, but compresses well)
2. Unpacks to `/opt/conda-env/`
3. Runs `source /opt/conda-env/bin/activate`
4. No conda solve needed -- instant activation

Implementation:
- Requires `conda-pack` package locally
- New upload step in submit flow for the tarball
- Activation script prepended to job command

#### Phase 3: Docker Image Builder (Higher effort, best for teams)

Add a command to build and push a Docker image from a conda environment:

```bash
# Build Docker image from conda environment
ccm env build-image --conda-env alchem --tag myuser/openmm-cuda12:latest

# Push to registry
ccm env push-image myuser/openmm-cuda12:latest

# Use in job
environment:
  docker_image: myuser/openmm-cuda12:latest
```

Implementation:
- Generate Dockerfile from conda env spec
- Build with Docker
- Push to Docker Hub or other registry
- Reference in job YAML

#### Phase 4: Environment Cache (Instance Reuse)

If the same environment is used across multiple jobs, avoid re-installing by
reusing the instance:

```yaml
environment:
  conda_env: ./environment.yml
  cache: true  # Keep instance alive between jobs
```

CCM checks if a running instance already has the environment installed (via a
hash of the environment definition) and reuses it instead of creating a new one.

### How This Solves the OpenMM Problem

Currently, deploying our OpenMM alchemical FE jobs fails because:
1. `pip install openmmtools` does not work (requires conda-forge)
2. `conda install` takes 15+ minutes on a fresh instance
3. The provisioning timeout kills the job before setup completes

With the proposed architecture:

**Option A (Recommended): Conda Pack**
```bash
# One-time: create pack from local alchem environment
conda activate alchem
conda install conda-pack
conda-pack -n alchem -o openmm-env.tar.gz
```

```yaml
environment:
  conda_pack: ./openmm-env.tar.gz
```

Setup time: ~2 min (upload 2GB tarball + unpack). No conda solve needed.

**Option B: Docker Image**
```bash
# One-time: build and push
docker build -t myuser/openmm-cuda12:latest -f Dockerfile.openmm .
docker push myuser/openmm-cuda12:latest
```

```yaml
environment:
  docker_image: myuser/openmm-cuda12:latest
```

Setup time: <1 min (Vast.ai caches popular images).

**Option C: Conda Env File**
```bash
conda env export -n alchem > environment.yml
```

```yaml
environment:
  conda_env: ./environment.yml
```

Setup time: 10-15 min (conda solve + install). Timeout must be >=20 min.

### Extensibility Considerations

This design is workload-agnostic and covers common scientific computing patterns:

- **Molecular dynamics:** LAMMPS, GROMACS, NAMD, OpenMM (conda or NGC containers)
- **Machine learning:** PyTorch, TensorFlow, JAX (pip or conda)
- **Quantum chemistry:** Quantum ESPRESSO, ORCA, Gaussian (compiled, need Spack or custom images)
- **Bioinformatics:** Nextflow pipelines (Docker/Singularity)
- **Custom compiled software:** User-built codes (Docker image is the only option)

The priority-based strategy selection means users start simple (inline packages)
and upgrade to faster methods (conda pack, Docker) as their needs grow.

### Database Schema Changes

```python
class Job(SQLModel, table=True):
    # Existing fields...
    environment_json: Optional[str] = Field(default=None)
    # JSON-encoded environment definition:
    # {
    #   "strategy": "conda_pack",
    #   "conda_pack_path": "/path/to/env.tar.gz",
    #   "conda_env_file": null,
    #   "docker_image": null,
    #   "packages": null,
    #   "python_version": "3.11",
    #   "cuda_version": "12.1",
    #   "env_hash": "abc123...",  # For cache matching
    # }
```

### CLI Commands

```bash
# Environment management
ccm env export [-n ENV_NAME] [-o FILE]     # Export conda env to YAML
ccm env pack [-n ENV_NAME] [-o FILE]       # Create conda-pack tarball
ccm env build-image [-n ENV_NAME] [-t TAG] # Build Docker image from env
ccm env push-image TAG                     # Push to registry
ccm env list                               # List cached environments
ccm env clean                              # Remove cached environment data
```

### Migration Path

Existing jobs using `setup:` continue to work unchanged. The `environment:`
field is additive. If both `setup:` and `environment:` are specified, the
environment is set up first, then the setup script runs (for any additional
configuration).

### Estimated Implementation Effort

| Phase | Effort | Value |
|-------|--------|-------|
| Phase 1: Conda env file | 1-2 days | Medium (still slow but reproducible) |
| Phase 2: Conda pack | 2-3 days | High (fast + reproducible) |
| Phase 3: Docker builder | 3-5 days | High (fastest, best for teams) |
| Phase 4: Environment cache | 3-5 days | Medium (optimization) |

**Recommendation:** Implement Phase 2 (conda pack) first. It solves the
immediate problem (fast deployment of complex conda environments) with
moderate effort and works for any conda-based workflow.
