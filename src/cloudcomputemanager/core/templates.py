"""Job template system for CCM.

Templates provide reusable configurations for common workloads.
User configs can inherit from templates and override specific values.
"""

from pathlib import Path
from typing import Any, Optional
import yaml

from cloudcomputemanager.templates import get_template_path, list_templates, TEMPLATES_DIR


def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries.

    Override values take precedence. Lists are replaced, not merged.
    Nested dicts are recursively merged.

    Args:
        base: Base dictionary (template)
        override: Override dictionary (user config)

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def load_template(name: str) -> dict:
    """
    Load a template by name.

    Args:
        name: Template name (e.g., "namd-production", "pytorch-train")

    Returns:
        Template configuration dictionary

    Raises:
        FileNotFoundError: If template doesn't exist
        yaml.YAMLError: If template is invalid YAML
    """
    template_path = get_template_path(name)

    with open(template_path) as f:
        template = yaml.safe_load(f)

    return template or {}


def load_config_with_template(config_path: Path, template_name: Optional[str] = None) -> dict:
    """
    Load a job config, optionally merging with a template.

    If the config has a "template" key, it will be used unless
    template_name is explicitly provided.

    Args:
        config_path: Path to user's job config YAML
        template_name: Optional template name to use (overrides config's template key)

    Returns:
        Merged configuration dictionary
    """
    with open(config_path) as f:
        user_config = yaml.safe_load(f) or {}

    # Determine template to use
    template_to_use = template_name or user_config.pop("template", None)

    if not template_to_use:
        return user_config

    # Load and merge template
    try:
        template = load_template(template_to_use)
    except FileNotFoundError:
        raise ValueError(f"Template not found: {template_to_use}. "
                        f"Available: {', '.join(list_templates())}")

    return deep_merge(template, user_config)


def get_available_templates() -> list[dict]:
    """
    Get list of available templates with metadata.

    Returns:
        List of dicts with template info (name, description, gpu_type, etc.)
    """
    templates = []

    for name in list_templates():
        try:
            template = load_template(name)
            templates.append({
                "name": name,
                "image": template.get("image", "N/A"),
                "gpu_type": template.get("resources", {}).get("gpu_type", "Any"),
                "max_rate": template.get("budget", {}).get("max_hourly_rate", "N/A"),
                "checkpoint": template.get("checkpoint", {}).get("enabled", False),
            })
        except Exception:
            templates.append({
                "name": name,
                "image": "Error loading",
                "gpu_type": "N/A",
                "max_rate": "N/A",
                "checkpoint": False,
            })

    return templates


def create_job_config_from_template(
    template_name: str,
    name: str,
    command: Optional[str] = None,
    upload_source: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> dict:
    """
    Create a complete job config from a template with common overrides.

    This is a convenience function for programmatic job creation.

    Args:
        template_name: Template to use
        name: Job name
        command: Command to run (optional, uses template default)
        upload_source: Local directory to upload
        overrides: Additional overrides to apply

    Returns:
        Complete job configuration dictionary
    """
    config = load_template(template_name)

    # Apply common overrides
    config["name"] = name

    if command:
        config["command"] = command

    if upload_source:
        if "upload" not in config:
            config["upload"] = {}
        config["upload"]["source"] = upload_source

    if overrides:
        config = deep_merge(config, overrides)

    return config


def validate_job_config(config: dict) -> list[str]:
    """
    Validate a job configuration.

    Args:
        config: Job configuration dictionary

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Required fields
    if not config.get("name"):
        errors.append("Missing required field: name")

    if not config.get("image"):
        errors.append("Missing required field: image")

    if not config.get("command"):
        errors.append("Missing required field: command")

    # Validate resources
    resources = config.get("resources", {})
    if resources:
        gpu_count = resources.get("gpu_count", 1)
        if not isinstance(gpu_count, int) or gpu_count < 1:
            errors.append("resources.gpu_count must be a positive integer")

        disk_gb = resources.get("disk_gb", 30)
        if not isinstance(disk_gb, (int, float)) or disk_gb < 10:
            errors.append("resources.disk_gb must be at least 10")

    # Validate budget
    budget = config.get("budget", {})
    if budget:
        max_rate = budget.get("max_hourly_rate")
        if max_rate is not None:
            if not isinstance(max_rate, (int, float)) or max_rate <= 0:
                errors.append("budget.max_hourly_rate must be a positive number")

    # Validate sync config
    sync = config.get("sync", {})
    if sync.get("enabled", False):
        interval = sync.get("interval", 600)
        if not isinstance(interval, int) or interval < 30:
            errors.append("sync.interval must be at least 30 seconds")

    return errors


def generate_minimal_config(template_name: str) -> str:
    """
    Generate a minimal YAML config that uses a template.

    This shows users what they need to provide when using a template.

    Args:
        template_name: Template name

    Returns:
        YAML string with minimal required config
    """
    template = load_template(template_name)

    minimal = {
        "template": template_name,
        "name": "my-job-name",
    }

    # Add command if template doesn't have a default
    if not template.get("command"):
        minimal["command"] = "# Add your command here"

    # Add upload source
    minimal["upload"] = {
        "source": "./my-project-dir/"
    }

    # Add any comments as YAML
    yaml_str = yaml.dump(minimal, default_flow_style=False, sort_keys=False)

    header = f"""# Job configuration using '{template_name}' template
# Template provides: image, resources, budget, sync, checkpoint settings
# You only need to specify job-specific values below

"""

    return header + yaml_str
