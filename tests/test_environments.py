"""Tests for environment management."""

import pytest
import tempfile
from pathlib import Path

from cloudcomputemanager.core.environment import (
    EnvironmentConfig,
    EnvironmentStrategy,
    parse_environment,
    validate_environment,
    get_setup_commands,
    get_command_prefix,
    get_upload_files,
    get_recommended_timeout,
)


class TestParseEnvironment:
    """Tests for parse_environment()."""

    def test_no_environment(self):
        assert parse_environment({}) is None
        assert parse_environment({"image": "ubuntu:22.04"}) is None

    def test_docker_image_strategy(self):
        config = {"environment": {"docker_image": "myuser/myimage:latest"}}
        env = parse_environment(config)
        assert env is not None
        assert env.strategy == EnvironmentStrategy.DOCKER_IMAGE
        assert env.docker_image == "myuser/myimage:latest"

    def test_conda_pack_strategy(self):
        config = {"environment": {"conda_pack": "/path/to/env.tar.gz"}}
        env = parse_environment(config)
        assert env.strategy == EnvironmentStrategy.CONDA_PACK
        assert env.conda_pack == "/path/to/env.tar.gz"

    def test_conda_env_strategy(self):
        config = {"environment": {"conda_env": "./environment.yml"}}
        env = parse_environment(config)
        assert env.strategy == EnvironmentStrategy.CONDA_ENV
        assert env.conda_env == "./environment.yml"

    def test_packages_strategy(self):
        config = {"environment": {"packages": {"conda": ["numpy"], "pip": ["requests"]}}}
        env = parse_environment(config)
        assert env.strategy == EnvironmentStrategy.PACKAGES
        assert env.packages == {"conda": ["numpy"], "pip": ["requests"]}

    def test_requirements_strategy(self):
        config = {"environment": {"requirements": "./requirements.txt"}}
        env = parse_environment(config)
        assert env.strategy == EnvironmentStrategy.REQUIREMENTS

    def test_priority_docker_over_conda_pack(self):
        config = {"environment": {
            "docker_image": "myimage:latest",
            "conda_pack": "/path/to/env.tar.gz",
        }}
        env = parse_environment(config)
        assert env.strategy == EnvironmentStrategy.DOCKER_IMAGE

    def test_priority_conda_pack_over_conda_env(self):
        config = {"environment": {
            "conda_pack": "/path/to/env.tar.gz",
            "conda_env": "./environment.yml",
        }}
        env = parse_environment(config)
        assert env.strategy == EnvironmentStrategy.CONDA_PACK

    def test_custom_channels(self):
        config = {"environment": {
            "packages": {"conda": ["openmm"]},
            "channels": ["conda-forge", "bioconda"],
        }}
        env = parse_environment(config)
        assert env.channels == ["conda-forge", "bioconda"]

    def test_default_channels(self):
        config = {"environment": {"packages": {"pip": ["numpy"]}}}
        env = parse_environment(config)
        assert env.channels == ["conda-forge"]

    def test_python_and_cuda_version(self):
        config = {"environment": {
            "packages": {"pip": ["torch"]},
            "python_version": "3.11",
            "cuda_version": "12.1",
        }}
        env = parse_environment(config)
        assert env.python_version == "3.11"
        assert env.cuda_version == "12.1"


class TestValidateEnvironment:
    """Tests for validate_environment()."""

    def test_valid_conda_pack(self):
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as f:
            env = EnvironmentConfig(
                strategy=EnvironmentStrategy.CONDA_PACK,
                conda_pack=f.name,
            )
            errors = validate_environment(env)
            assert errors == []

    def test_missing_conda_pack_file(self):
        env = EnvironmentConfig(
            strategy=EnvironmentStrategy.CONDA_PACK,
            conda_pack="/nonexistent/file.tar.gz",
        )
        errors = validate_environment(env)
        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_invalid_conda_pack_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".zip") as f:
            env = EnvironmentConfig(
                strategy=EnvironmentStrategy.CONDA_PACK,
                conda_pack=f.name,
            )
            errors = validate_environment(env)
            assert len(errors) == 1
            assert ".tar.gz" in errors[0]

    def test_valid_conda_env(self):
        with tempfile.NamedTemporaryFile(suffix=".yml") as f:
            env = EnvironmentConfig(
                strategy=EnvironmentStrategy.CONDA_ENV,
                conda_env=f.name,
            )
            errors = validate_environment(env)
            assert errors == []

    def test_missing_conda_env_file(self):
        env = EnvironmentConfig(
            strategy=EnvironmentStrategy.CONDA_ENV,
            conda_env="/nonexistent/environment.yml",
        )
        errors = validate_environment(env)
        assert len(errors) == 1

    def test_valid_packages(self):
        env = EnvironmentConfig(
            strategy=EnvironmentStrategy.PACKAGES,
            packages={"conda": ["numpy"], "pip": ["requests"], "apt": ["curl"]},
        )
        errors = validate_environment(env)
        assert errors == []

    def test_invalid_package_type(self):
        env = EnvironmentConfig(
            strategy=EnvironmentStrategy.PACKAGES,
            packages={"npm": ["express"]},
        )
        errors = validate_environment(env)
        assert len(errors) == 1
        assert "npm" in errors[0]

    def test_valid_requirements(self):
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            env = EnvironmentConfig(
                strategy=EnvironmentStrategy.REQUIREMENTS,
                requirements=f.name,
            )
            errors = validate_environment(env)
            assert errors == []

    def test_none_strategy_valid(self):
        env = EnvironmentConfig(strategy=EnvironmentStrategy.NONE)
        errors = validate_environment(env)
        assert errors == []


class TestGetSetupCommands:
    """Tests for get_setup_commands()."""

    def test_none_strategy(self):
        env = EnvironmentConfig(strategy=EnvironmentStrategy.NONE)
        assert get_setup_commands(env) == ""

    def test_docker_image_no_setup(self):
        env = EnvironmentConfig(
            strategy=EnvironmentStrategy.DOCKER_IMAGE,
            docker_image="myimage:latest",
        )
        assert get_setup_commands(env) == ""

    def test_conda_pack_setup(self):
        env = EnvironmentConfig(
            strategy=EnvironmentStrategy.CONDA_PACK,
            conda_pack="/path/to/env.tar.gz",
        )
        cmds = get_setup_commands(env)
        assert "mkdir -p /opt/conda-env" in cmds
        assert "tar -xzf /workspace/.ccm_env.tar.gz" in cmds
        assert "source /opt/conda-env/bin/activate" in cmds

    def test_conda_env_setup(self):
        env = EnvironmentConfig(
            strategy=EnvironmentStrategy.CONDA_ENV,
            conda_env="./env.yml",
            channels=["conda-forge", "bioconda"],
        )
        cmds = get_setup_commands(env)
        assert "miniconda" in cmds.lower()
        assert "conda env create" in cmds
        assert "-c conda-forge" in cmds
        assert "-c bioconda" in cmds

    def test_packages_conda_setup(self):
        env = EnvironmentConfig(
            strategy=EnvironmentStrategy.PACKAGES,
            packages={"conda": ["numpy", "scipy"]},
            channels=["conda-forge"],
        )
        cmds = get_setup_commands(env)
        assert "miniconda" in cmds.lower()
        assert "conda install" in cmds
        assert "numpy" in cmds
        assert "scipy" in cmds

    def test_packages_pip_setup(self):
        env = EnvironmentConfig(
            strategy=EnvironmentStrategy.PACKAGES,
            packages={"pip": ["torch", "numpy"]},
        )
        cmds = get_setup_commands(env)
        assert "pip install" in cmds
        assert "torch" in cmds

    def test_packages_apt_setup(self):
        env = EnvironmentConfig(
            strategy=EnvironmentStrategy.PACKAGES,
            packages={"apt": ["libfftw3-dev", "curl"]},
        )
        cmds = get_setup_commands(env)
        assert "apt-get install" in cmds
        assert "libfftw3-dev" in cmds

    def test_packages_mixed_setup(self):
        env = EnvironmentConfig(
            strategy=EnvironmentStrategy.PACKAGES,
            packages={
                "apt": ["curl"],
                "conda": ["numpy"],
                "pip": ["requests"],
            },
        )
        cmds = get_setup_commands(env)
        assert "apt-get install" in cmds
        assert "conda install" in cmds
        assert "pip install" in cmds

    def test_requirements_setup(self):
        env = EnvironmentConfig(
            strategy=EnvironmentStrategy.REQUIREMENTS,
            requirements="./requirements.txt",
        )
        cmds = get_setup_commands(env)
        assert "pip install -r" in cmds
        assert ".ccm_requirements.txt" in cmds


class TestGetCommandPrefix:
    """Tests for get_command_prefix()."""

    def test_none_no_prefix(self):
        env = EnvironmentConfig(strategy=EnvironmentStrategy.NONE)
        assert get_command_prefix(env) == ""

    def test_docker_no_prefix(self):
        env = EnvironmentConfig(strategy=EnvironmentStrategy.DOCKER_IMAGE)
        assert get_command_prefix(env) == ""

    def test_conda_pack_prefix(self):
        env = EnvironmentConfig(strategy=EnvironmentStrategy.CONDA_PACK)
        prefix = get_command_prefix(env)
        assert "source /opt/conda-env/bin/activate" in prefix

    def test_conda_env_prefix(self):
        env = EnvironmentConfig(strategy=EnvironmentStrategy.CONDA_ENV)
        prefix = get_command_prefix(env)
        assert "activate ccm_env" in prefix

    def test_packages_with_conda_prefix(self):
        env = EnvironmentConfig(
            strategy=EnvironmentStrategy.PACKAGES,
            packages={"conda": ["numpy"]},
        )
        prefix = get_command_prefix(env)
        assert "/opt/conda/bin" in prefix

    def test_packages_pip_only_no_prefix(self):
        env = EnvironmentConfig(
            strategy=EnvironmentStrategy.PACKAGES,
            packages={"pip": ["numpy"]},
        )
        prefix = get_command_prefix(env)
        assert prefix == ""


class TestGetUploadFiles:
    """Tests for get_upload_files()."""

    def test_none_no_uploads(self):
        env = EnvironmentConfig(strategy=EnvironmentStrategy.NONE)
        assert get_upload_files(env) == []

    def test_conda_pack_upload(self):
        env = EnvironmentConfig(
            strategy=EnvironmentStrategy.CONDA_PACK,
            conda_pack="/path/to/env.tar.gz",
        )
        files = get_upload_files(env)
        assert len(files) == 1
        assert files[0] == ("/path/to/env.tar.gz", "/workspace/.ccm_env.tar.gz")

    def test_conda_env_upload(self):
        env = EnvironmentConfig(
            strategy=EnvironmentStrategy.CONDA_ENV,
            conda_env="/path/to/environment.yml",
        )
        files = get_upload_files(env)
        assert len(files) == 1
        assert files[0][1] == "/workspace/.ccm_env.yml"

    def test_requirements_upload(self):
        env = EnvironmentConfig(
            strategy=EnvironmentStrategy.REQUIREMENTS,
            requirements="/path/to/requirements.txt",
        )
        files = get_upload_files(env)
        assert len(files) == 1
        assert files[0][1] == "/workspace/.ccm_requirements.txt"


class TestGetRecommendedTimeout:
    """Tests for get_recommended_timeout()."""

    def test_docker_short_timeout(self):
        env = EnvironmentConfig(strategy=EnvironmentStrategy.DOCKER_IMAGE)
        assert get_recommended_timeout(env) == 300

    def test_conda_pack_medium_timeout(self):
        env = EnvironmentConfig(strategy=EnvironmentStrategy.CONDA_PACK)
        assert get_recommended_timeout(env) == 600

    def test_conda_env_long_timeout(self):
        env = EnvironmentConfig(strategy=EnvironmentStrategy.CONDA_ENV)
        assert get_recommended_timeout(env) == 1200

    def test_packages_long_timeout(self):
        env = EnvironmentConfig(strategy=EnvironmentStrategy.PACKAGES)
        assert get_recommended_timeout(env) == 1200
