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
)
from cloudcomputemanager.templates import list_templates, get_template_path, TEMPLATES_DIR


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
