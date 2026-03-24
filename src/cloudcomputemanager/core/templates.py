"""Job template system for CCM.

Templates provide reusable configurations for common workloads.
User configs can inherit from templates and override specific values.
"""

import logging
import random
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any, Optional
import yaml

logger = logging.getLogger(__name__)

from cloudcomputemanager.templates import get_template_path, list_templates, TEMPLATES_DIR


# Mapping of common alternate key names to canonical names
RESOURCE_KEY_ALIASES = {
    "gpu_memory_gb": "gpu_memory_min",
    "gpu_mem": "gpu_memory_min",
    "vram": "gpu_memory_min",
    "memory_gb": "ram_gb",
    "memory": "ram_gb",
    "cpu": "cpu_cores",
    "cpus": "cpu_cores",
    "disk": "disk_gb",
    "storage": "disk_gb",
    "gpu": "gpu_type",
}


def normalize_resources(resources: dict) -> dict:
    """
    Normalize resource keys to canonical names.

    This ensures that alternate key names (like gpu_memory_gb) are
    converted to their canonical forms (gpu_memory_min) so that
    the rest of the codebase can use consistent key names.

    Args:
        resources: Resource dictionary with potentially non-canonical keys

    Returns:
        New dictionary with normalized keys
    """
    if not resources:
        return resources

    normalized = {}
    for key, value in resources.items():
        canonical_key = RESOURCE_KEY_ALIASES.get(key, key)
        normalized[canonical_key] = value

    return normalized


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


def substitute_variables(config_str: str, variables: Optional[dict[str, str]] = None) -> str:
    """
    Perform variable substitution on a YAML string.

    Supports ${VARIABLE} syntax. Built-in variables are always available.
    User-provided variables override built-ins.

    Args:
        config_str: Raw YAML string
        variables: User-provided variables from --set KEY=VALUE

    Returns:
        YAML string with variables substituted
    """
    # Built-in variables available in all job configs
    builtins = {
        "TIMESTAMP": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "DATE": datetime.now().strftime("%Y-%m-%d"),
        "RANDOM": str(random.randint(10000, 99999)),
    }

    # Merge: user vars override built-ins
    all_vars = {**builtins, **(variables or {})}

    # Use safe_substitute to leave unmatched ${VARS} as-is (don't crash)
    return Template(config_str).safe_substitute(all_vars)


def load_config_with_template(
    config_path: Path,
    template_name: Optional[str] = None,
    variables: Optional[dict[str, str]] = None,
) -> dict:
    """
    Load a job config, optionally merging with a template and substituting variables.

    If the config has a "template" key, it will be used unless
    template_name is explicitly provided.

    Resource keys are normalized (e.g., gpu_memory_gb -> gpu_memory_min)
    to ensure consistent handling throughout the codebase.

    Args:
        config_path: Path to user's job config YAML
        template_name: Optional template name to use (overrides config's template key)
        variables: Optional dict of variable substitutions (from --set KEY=VALUE)

    Returns:
        Merged configuration dictionary with normalized resource keys
    """
    raw_yaml = config_path.read_text()

    # Apply variable substitution before YAML parsing
    if variables:
        raw_yaml = substitute_variables(raw_yaml, variables)

    user_config = yaml.safe_load(raw_yaml) or {}

    # Normalize user resource keys before merging
    if "resources" in user_config:
        user_config["resources"] = normalize_resources(user_config["resources"])

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

    # Normalize template resources too
    if "resources" in template:
        template["resources"] = normalize_resources(template["resources"])

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
        except (yaml.YAMLError, FileNotFoundError, KeyError) as e:
            logger.warning("Failed to load template %s: %s", name, e)
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
