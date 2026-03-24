"""Environment management for CCM.

Handles parsing, validation, and setup command generation for job environments.
Supports multiple strategies:
  1. Docker image (fastest, <1 min)
  2. Conda pack tarball (fast, 1-2 min)
  3. Conda environment file (reproducible, 5-15 min)
  4. Inline packages (simple, 5-15 min)
  5. Requirements.txt (pip-only, 2-5 min)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class EnvironmentStrategy(str, Enum):
    """Strategy for setting up the job environment."""
    DOCKER_IMAGE = "docker_image"
    CONDA_PACK = "conda_pack"
    CONDA_ENV = "conda_env"
    PACKAGES = "packages"
    REQUIREMENTS = "requirements"
    NONE = "none"


@dataclass
class EnvironmentConfig:
    """Parsed environment configuration from job YAML.

    Attributes:
        strategy: Automatically selected based on which fields are set.
        docker_image: Pre-built Docker image to use (overrides job-level image).
        conda_pack: Path to conda-pack tarball (.tar.gz).
        conda_env: Path to conda environment.yml file.
        requirements: Path to pip requirements.txt file.
        packages: Inline package specifications (conda, pip, apt).
        python_version: Desired Python version.
        cuda_version: Required CUDA version.
        channels: Conda channels to use.
    """
    strategy: EnvironmentStrategy = EnvironmentStrategy.NONE
    docker_image: Optional[str] = None
    conda_pack: Optional[str] = None
    conda_env: Optional[str] = None
    requirements: Optional[str] = None
    packages: Optional[dict] = None
    python_version: Optional[str] = None
    cuda_version: Optional[str] = None
    channels: list[str] = field(default_factory=lambda: ["conda-forge"])


def parse_environment(config: dict) -> Optional[EnvironmentConfig]:
    """Parse the `environment:` field from a job YAML config.

    Args:
        config: The full job configuration dict.

    Returns:
        EnvironmentConfig if environment is specified, None otherwise.
    """
    env_raw = config.get("environment")
    if not env_raw:
        return None

    env = EnvironmentConfig(
        docker_image=env_raw.get("docker_image"),
        conda_pack=env_raw.get("conda_pack"),
        conda_env=env_raw.get("conda_env"),
        requirements=env_raw.get("requirements"),
        packages=env_raw.get("packages"),
        python_version=env_raw.get("python_version"),
        cuda_version=env_raw.get("cuda_version"),
        channels=env_raw.get("channels", ["conda-forge"]),
    )

    # Auto-select strategy based on priority
    if env.docker_image:
        env.strategy = EnvironmentStrategy.DOCKER_IMAGE
    elif env.conda_pack:
        env.strategy = EnvironmentStrategy.CONDA_PACK
    elif env.conda_env:
        env.strategy = EnvironmentStrategy.CONDA_ENV
    elif env.packages:
        env.strategy = EnvironmentStrategy.PACKAGES
    elif env.requirements:
        env.strategy = EnvironmentStrategy.REQUIREMENTS
    else:
        env.strategy = EnvironmentStrategy.NONE

    return env


def validate_environment(env: EnvironmentConfig) -> list[str]:
    """Validate an EnvironmentConfig and return a list of errors.

    Returns:
        List of error messages. Empty list means valid.
    """
    errors = []

    if env.strategy == EnvironmentStrategy.CONDA_PACK:
        if not env.conda_pack:
            errors.append("conda_pack path is required for conda_pack strategy")
        else:
            path = Path(env.conda_pack).expanduser()
            if not path.exists():
                errors.append(f"conda_pack file not found: {path}")
            elif not path.suffix == ".gz" and not str(path).endswith(".tar.gz"):
                errors.append(f"conda_pack must be a .tar.gz file: {path}")

    if env.strategy == EnvironmentStrategy.CONDA_ENV:
        if not env.conda_env:
            errors.append("conda_env path is required for conda_env strategy")
        else:
            path = Path(env.conda_env).expanduser()
            if not path.exists():
                errors.append(f"conda_env file not found: {path}")

    if env.strategy == EnvironmentStrategy.REQUIREMENTS:
        if not env.requirements:
            errors.append("requirements path is required for requirements strategy")
        else:
            path = Path(env.requirements).expanduser()
            if not path.exists():
                errors.append(f"requirements file not found: {path}")

    if env.strategy == EnvironmentStrategy.PACKAGES:
        if not env.packages:
            errors.append("packages dict is required for packages strategy")
        else:
            valid_keys = {"conda", "pip", "apt"}
            for key in env.packages:
                if key not in valid_keys:
                    errors.append(f"Unknown package type: {key}. Must be one of: {valid_keys}")

    return errors


def get_setup_commands(env: EnvironmentConfig) -> str:
    """Generate setup shell commands for the environment.

    These commands are prepended to the onstart script to install
    the environment before the job runs.

    Args:
        env: Parsed environment configuration.

    Returns:
        Shell commands as a multiline string.
    """
    if env.strategy == EnvironmentStrategy.NONE:
        return ""

    if env.strategy == EnvironmentStrategy.DOCKER_IMAGE:
        # No setup needed; the image already has everything
        return ""

    if env.strategy == EnvironmentStrategy.CONDA_PACK:
        # Unpack conda-pack tarball to /opt/conda-env and activate
        return "\n".join([
            "mkdir -p /opt/conda-env",
            "tar -xzf /workspace/.ccm_env.tar.gz -C /opt/conda-env",
            "source /opt/conda-env/bin/activate",
            # Fix prefixes (conda-pack relocatability)
            "/opt/conda-env/bin/conda-unpack 2>/dev/null || true",
        ])

    if env.strategy == EnvironmentStrategy.CONDA_ENV:
        # Install miniconda and create environment from file
        # Use curl (more widely available than wget on minimal Docker images)
        channels = " ".join(f"-c {c}" for c in env.channels)
        return "\n".join([
            "curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh",
            "bash /tmp/miniconda.sh -b -p /opt/conda > /dev/null 2>&1",
            "export PATH=/opt/conda/bin:$PATH",
            f"/opt/conda/bin/conda env create -f /workspace/.ccm_env.yml {channels} -q",
            "source /opt/conda/bin/activate ccm_env",
        ])

    if env.strategy == EnvironmentStrategy.PACKAGES:
        commands = []

        # APT packages first
        apt_pkgs = env.packages.get("apt", [])
        if apt_pkgs:
            pkg_str = " ".join(apt_pkgs)
            commands.append(f"apt-get update && apt-get install -y {pkg_str}")

        # Conda packages (requires miniconda)
        conda_pkgs = env.packages.get("conda", [])
        if conda_pkgs:
            channels = " ".join(f"-c {c}" for c in env.channels)
            pkg_str = " ".join(conda_pkgs)
            commands.extend([
                "curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh",
                "bash /tmp/miniconda.sh -b -p /opt/conda > /dev/null 2>&1",
                "export PATH=/opt/conda/bin:$PATH",
                f"/opt/conda/bin/conda install -y {channels} {pkg_str} -q",
            ])

        # Pip packages
        pip_pkgs = env.packages.get("pip", [])
        if pip_pkgs:
            pkg_str = " ".join(pip_pkgs)
            commands.append(f"pip install {pkg_str}")

        return "\n".join(commands)

    if env.strategy == EnvironmentStrategy.REQUIREMENTS:
        return "pip install -r /workspace/.ccm_requirements.txt"

    return ""


def get_command_prefix(env: EnvironmentConfig) -> str:
    """Get the command prefix to activate the environment before running the job.

    Args:
        env: Parsed environment configuration.

    Returns:
        Shell prefix string (e.g., "source /opt/conda-env/bin/activate && ").
        Empty string if no activation needed.
    """
    if env.strategy == EnvironmentStrategy.CONDA_PACK:
        return "source /opt/conda-env/bin/activate && "

    if env.strategy == EnvironmentStrategy.CONDA_ENV:
        return "export PATH=/opt/conda/bin:$PATH && source /opt/conda/bin/activate ccm_env && "

    if env.strategy == EnvironmentStrategy.PACKAGES:
        conda_pkgs = (env.packages or {}).get("conda", [])
        if conda_pkgs:
            return "export PATH=/opt/conda/bin:$PATH && "

    return ""


def get_upload_files(env: EnvironmentConfig) -> list[tuple[str, str]]:
    """Get list of files that need to be uploaded for the environment.

    Returns:
        List of (local_path, remote_path) tuples.
    """
    files = []

    if env.strategy == EnvironmentStrategy.CONDA_PACK and env.conda_pack:
        files.append((str(Path(env.conda_pack).expanduser()), "/workspace/.ccm_env.tar.gz"))

    if env.strategy == EnvironmentStrategy.CONDA_ENV and env.conda_env:
        files.append((str(Path(env.conda_env).expanduser()), "/workspace/.ccm_env.yml"))

    if env.strategy == EnvironmentStrategy.REQUIREMENTS and env.requirements:
        files.append((str(Path(env.requirements).expanduser()), "/workspace/.ccm_requirements.txt"))

    return files


def get_recommended_timeout(env: EnvironmentConfig) -> int:
    """Get recommended provisioning timeout based on environment strategy.

    Returns:
        Timeout in seconds.
    """
    timeouts = {
        EnvironmentStrategy.DOCKER_IMAGE: 300,
        EnvironmentStrategy.CONDA_PACK: 600,
        EnvironmentStrategy.REQUIREMENTS: 600,
        EnvironmentStrategy.PACKAGES: 1200,
        EnvironmentStrategy.CONDA_ENV: 1200,
        EnvironmentStrategy.NONE: 300,
    }
    return timeouts.get(env.strategy, 300)
