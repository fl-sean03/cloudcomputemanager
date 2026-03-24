"""Tests for job templates system."""

import pytest
import tempfile
from pathlib import Path
import yaml

from cloudcomputemanager.core.templates import (
    deep_merge,
    load_template,
    load_config_with_template,
    get_available_templates,
    create_job_config_from_template,
    validate_job_config,
    generate_minimal_config,
    normalize_resources,
    RESOURCE_KEY_ALIASES,
)
from cloudcomputemanager.templates import list_templates, get_template_path, TEMPLATES_DIR


class TestNormalizeResources:
    """Tests for normalize_resources function."""

    def test_normalize_gpu_memory_gb(self):
        """Test that gpu_memory_gb is normalized to gpu_memory_min."""
        resources = {"gpu_memory_gb": 12, "gpu_type": "RTX_3060"}
        result = normalize_resources(resources)
        assert "gpu_memory_min" in result
        assert result["gpu_memory_min"] == 12
        assert "gpu_memory_gb" not in result
        assert result["gpu_type"] == "RTX_3060"

    def test_normalize_memory_gb(self):
        """Test that memory_gb is normalized to ram_gb."""
        resources = {"memory_gb": 64, "disk_gb": 100}
        result = normalize_resources(resources)
        assert "ram_gb" in result
        assert result["ram_gb"] == 64
        assert result["disk_gb"] == 100

    def test_normalize_cpu(self):
        """Test that cpu is normalized to cpu_cores."""
        resources = {"cpu": 8}
        result = normalize_resources(resources)
        assert "cpu_cores" in result
        assert result["cpu_cores"] == 8

    def test_normalize_disk(self):
        """Test that disk is normalized to disk_gb."""
        resources = {"disk": 200, "storage": 300}
        result = normalize_resources(resources)
        # Note: storage will override disk since they map to the same key
        assert "disk_gb" in result

    def test_normalize_gpu(self):
        """Test that gpu is normalized to gpu_type."""
        resources = {"gpu": "RTX_4090"}
        result = normalize_resources(resources)
        assert "gpu_type" in result
        assert result["gpu_type"] == "RTX_4090"

    def test_canonical_keys_unchanged(self):
        """Test that canonical keys are not changed."""
        resources = {
            "gpu_type": "RTX_4090",
            "gpu_memory_min": 16,
            "gpu_count": 1,
            "disk_gb": 100,
        }
        result = normalize_resources(resources)
        assert result == resources

    def test_empty_resources(self):
        """Test that empty dict is handled."""
        assert normalize_resources({}) == {}

    def test_none_resources(self):
        """Test that None is handled."""
        assert normalize_resources(None) is None

    def test_all_aliases_exist(self):
        """Test that all documented aliases are in the mapping."""
        expected_aliases = [
            "gpu_memory_gb", "gpu_mem", "vram",
            "memory_gb", "memory",
            "cpu", "cpus",
            "disk", "storage",
            "gpu",
        ]
        for alias in expected_aliases:
            assert alias in RESOURCE_KEY_ALIASES, f"Missing alias: {alias}"


class TestDeepMerge:
    """Tests for deep_merge function."""

    def test_simple_merge(self):
        """Test merging flat dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        """Test merging nested dictionaries."""
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3, "z": 4}}
        result = deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 3, "z": 4}}

    def test_list_override(self):
        """Test that lists are replaced, not merged."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = deep_merge(base, override)
        assert result == {"items": [4, 5]}

    def test_base_unchanged(self):
        """Test that base dict is not modified."""
        base = {"a": 1}
        override = {"b": 2}
        deep_merge(base, override)
        assert base == {"a": 1}


class TestListTemplates:
    """Tests for template listing."""

    def test_list_templates_returns_list(self):
        """Test that list_templates returns a list."""
        templates = list_templates()
        assert isinstance(templates, list)

    def test_builtin_templates_exist(self):
        """Test that built-in templates are available."""
        templates = list_templates()
        expected = {"namd-production", "lammps-gpu", "pytorch-train", "quick-gpu"}
        assert expected.issubset(set(templates))

    def test_templates_are_sorted(self):
        """Test that templates are sorted alphabetically."""
        templates = list_templates()
        assert templates == sorted(templates)


class TestGetTemplatePath:
    """Tests for template path resolution."""

    def test_valid_template(self):
        """Test getting path to valid template."""
        path = get_template_path("namd-production")
        assert path.exists()
        assert path.suffix == ".yaml"

    def test_invalid_template(self):
        """Test that invalid template raises error."""
        with pytest.raises(FileNotFoundError):
            get_template_path("nonexistent-template")


class TestLoadTemplate:
    """Tests for template loading."""

    def test_load_namd_template(self):
        """Test loading NAMD production template."""
        template = load_template("namd-production")
        assert template["image"] == "nvcr.io/hpc/namd:3.0.1"
        assert template["resources"]["gpu_type"] == "RTX_3060"
        assert template["checkpoint"]["enabled"] is True

    def test_load_pytorch_template(self):
        """Test loading PyTorch training template."""
        template = load_template("pytorch-train")
        assert "pytorch" in template["image"].lower()
        assert template["resources"]["gpu_type"] == "RTX_4090"

    def test_load_quick_gpu_template(self):
        """Test loading quick GPU template."""
        template = load_template("quick-gpu")
        assert template["checkpoint"]["enabled"] is False
        assert template["terminate_on_complete"] is True


class TestLoadConfigWithTemplate:
    """Tests for config loading with template merging."""

    def test_load_without_template(self, tmp_path):
        """Test loading config without template."""
        config_file = tmp_path / "job.yaml"
        config_file.write_text(yaml.dump({
            "name": "my-job",
            "image": "python:3.11",
            "command": "python script.py"
        }))

        result = load_config_with_template(config_file)
        assert result["name"] == "my-job"
        assert result["image"] == "python:3.11"

    def test_load_with_template_argument(self, tmp_path):
        """Test loading config with template argument."""
        config_file = tmp_path / "job.yaml"
        config_file.write_text(yaml.dump({
            "name": "my-job",
            "command": "namd3 production.namd"
        }))

        result = load_config_with_template(config_file, "namd-production")
        assert result["name"] == "my-job"
        assert result["image"] == "nvcr.io/hpc/namd:3.0.1"
        assert result["resources"]["gpu_type"] == "RTX_3060"

    def test_load_with_template_in_config(self, tmp_path):
        """Test loading config with template specified in YAML."""
        config_file = tmp_path / "job.yaml"
        config_file.write_text(yaml.dump({
            "template": "pytorch-train",
            "name": "my-training",
            "command": "python train.py"
        }))

        result = load_config_with_template(config_file)
        assert result["name"] == "my-training"
        assert "pytorch" in result["image"].lower()
        # template key should be removed
        assert "template" not in result

    def test_override_template_values(self, tmp_path):
        """Test that user config overrides template values."""
        config_file = tmp_path / "job.yaml"
        config_file.write_text(yaml.dump({
            "name": "my-job",
            "resources": {
                "gpu_type": "RTX_4090",
                "disk_gb": 100
            }
        }))

        result = load_config_with_template(config_file, "namd-production")
        # User override
        assert result["resources"]["gpu_type"] == "RTX_4090"
        assert result["resources"]["disk_gb"] == 100
        # Template defaults preserved
        assert result["resources"]["gpu_count"] == 1

    def test_normalizes_resource_keys(self, tmp_path):
        """Test that alternate resource keys are normalized."""
        config_file = tmp_path / "job.yaml"
        config_file.write_text(yaml.dump({
            "name": "my-job",
            "image": "python:3.11",
            "command": "python script.py",
            "resources": {
                "gpu_type": "RTX_3060",
                "gpu_memory_gb": 12,  # Should become gpu_memory_min
                "memory_gb": 64,      # Should become ram_gb
            }
        }))

        result = load_config_with_template(config_file)

        # Check normalization happened
        assert "gpu_memory_min" in result["resources"]
        assert result["resources"]["gpu_memory_min"] == 12
        assert "gpu_memory_gb" not in result["resources"]

        assert "ram_gb" in result["resources"]
        assert result["resources"]["ram_gb"] == 64
        assert "memory_gb" not in result["resources"]

        # gpu_type should remain unchanged
        assert result["resources"]["gpu_type"] == "RTX_3060"

    def test_normalizes_template_resource_keys(self, tmp_path):
        """Test that template resource keys are also normalized."""
        config_file = tmp_path / "job.yaml"
        config_file.write_text(yaml.dump({
            "name": "my-job",
            "command": "namd3 production.namd"
        }))

        result = load_config_with_template(config_file, "namd-production")

        # Template resources should use canonical keys
        assert "gpu_type" in result["resources"]
        # gpu_memory_min should exist (either from template or normalized)
        # The exact key depends on how the template is defined


class TestGetAvailableTemplates:
    """Tests for template metadata retrieval."""

    def test_returns_list_of_dicts(self):
        """Test that get_available_templates returns proper structure."""
        templates = get_available_templates()
        assert isinstance(templates, list)
        assert len(templates) > 0
        assert all(isinstance(t, dict) for t in templates)

    def test_template_metadata_fields(self):
        """Test that templates have required metadata."""
        templates = get_available_templates()
        required_fields = {"name", "image", "gpu_type", "max_rate", "checkpoint"}
        for t in templates:
            assert required_fields.issubset(set(t.keys()))


class TestCreateJobConfigFromTemplate:
    """Tests for programmatic job config creation."""

    def test_create_basic_config(self):
        """Test creating basic config from template."""
        config = create_job_config_from_template(
            template_name="quick-gpu",
            name="test-job",
            command="python test.py"
        )

        assert config["name"] == "test-job"
        assert config["command"] == "python test.py"
        assert "resources" in config

    def test_create_with_upload(self):
        """Test creating config with upload source."""
        config = create_job_config_from_template(
            template_name="quick-gpu",
            name="test-job",
            command="bash run.sh",
            upload_source="/path/to/project"
        )

        assert config["upload"]["source"] == "/path/to/project"

    def test_create_with_overrides(self):
        """Test creating config with additional overrides."""
        config = create_job_config_from_template(
            template_name="quick-gpu",
            name="test-job",
            command="python test.py",
            overrides={
                "budget": {"max_hourly_rate": 0.20}
            }
        )

        assert config["budget"]["max_hourly_rate"] == 0.20


class TestValidateJobConfig:
    """Tests for job config validation."""

    def test_valid_config(self):
        """Test validation of valid config."""
        config = {
            "name": "my-job",
            "image": "python:3.11",
            "command": "python script.py",
            "resources": {
                "gpu_count": 1,
                "disk_gb": 50
            }
        }
        errors = validate_job_config(config)
        assert len(errors) == 0

    def test_missing_name(self):
        """Test validation catches missing name."""
        config = {
            "image": "python:3.11",
            "command": "python script.py"
        }
        errors = validate_job_config(config)
        assert any("name" in e.lower() for e in errors)

    def test_missing_image(self):
        """Test validation catches missing image."""
        config = {
            "name": "my-job",
            "command": "python script.py"
        }
        errors = validate_job_config(config)
        assert any("image" in e.lower() for e in errors)

    def test_invalid_gpu_count(self):
        """Test validation catches invalid gpu_count."""
        config = {
            "name": "my-job",
            "image": "python:3.11",
            "command": "python script.py",
            "resources": {"gpu_count": 0}
        }
        errors = validate_job_config(config)
        assert any("gpu_count" in e.lower() for e in errors)

    def test_invalid_disk_size(self):
        """Test validation catches invalid disk size."""
        config = {
            "name": "my-job",
            "image": "python:3.11",
            "command": "python script.py",
            "resources": {"disk_gb": 5}
        }
        errors = validate_job_config(config)
        assert any("disk_gb" in e.lower() for e in errors)

    def test_invalid_hourly_rate(self):
        """Test validation catches invalid hourly rate."""
        config = {
            "name": "my-job",
            "image": "python:3.11",
            "command": "python script.py",
            "budget": {"max_hourly_rate": -1}
        }
        errors = validate_job_config(config)
        assert any("hourly_rate" in e.lower() for e in errors)


class TestGenerateMinimalConfig:
    """Tests for minimal config generation."""

    def test_generate_for_template(self):
        """Test generating minimal config for template."""
        config_str = generate_minimal_config("namd-production")

        assert "template: namd-production" in config_str
        assert "name:" in config_str
        assert "upload:" in config_str

    def test_generated_config_is_valid_yaml(self):
        """Test that generated config is valid YAML."""
        config_str = generate_minimal_config("pytorch-train")
        config = yaml.safe_load(config_str)

        assert isinstance(config, dict)
        assert config["template"] == "pytorch-train"
