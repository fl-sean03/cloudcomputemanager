"""CCM Job Templates.

Built-in templates for common workloads.
"""

from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent


def get_template_path(name: str) -> Path:
    """Get path to a built-in template."""
    # Try with and without .yaml extension
    template_path = TEMPLATES_DIR / f"{name}.yaml"
    if template_path.exists():
        return template_path

    template_path = TEMPLATES_DIR / name
    if template_path.exists():
        return template_path

    raise FileNotFoundError(f"Template not found: {name}")


def list_templates() -> list[str]:
    """List available built-in templates."""
    templates = []
    for f in TEMPLATES_DIR.glob("*.yaml"):
        templates.append(f.stem)
    return sorted(templates)
