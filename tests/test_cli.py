"""CLI tests using Typer's CliRunner.

Tests the CLI layer in isolation -- no vast.ai API calls, no database needed.
All commands tested here are self-contained and produce output from local
configuration, templates, or status checks.
"""

import pytest
from typer.testing import CliRunner

from cloudcomputemanager.cli.main import app

runner = CliRunner()


def test_help():
    """ccm --help exits 0 and shows help text."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "GPU cloud management" in result.output


def test_version():
    """ccm --version exits 0 and shows version."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "CloudComputeManager" in result.output


def test_config_show():
    """ccm config show exits 0 and shows config table."""
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "Configuration" in result.output
    assert "Data Directory" in result.output


def test_config_init():
    """ccm config init exits 0 and shows directory creation messages."""
    result = runner.invoke(app, ["config", "init"])
    assert result.exit_code == 0
    assert "Created directories" in result.output


def test_templates_list():
    """ccm templates list exits 0 and shows template table."""
    result = runner.invoke(app, ["templates", "list"])
    assert result.exit_code == 0
    assert "Available Job Templates" in result.output
    assert "namd-production" in result.output


def test_templates_show():
    """ccm templates show namd-production exits 0 and shows YAML."""
    result = runner.invoke(app, ["templates", "show", "namd-production"])
    assert result.exit_code == 0
    assert "Template: namd-production" in result.output


def test_templates_show_nonexistent():
    """ccm templates show nonexistent exits 1."""
    result = runner.invoke(app, ["templates", "show", "nonexistent"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_templates_generate():
    """ccm templates generate quick-gpu exits 0 and outputs YAML."""
    result = runner.invoke(app, ["templates", "generate", "quick-gpu"])
    assert result.exit_code == 0
    assert "template" in result.output.lower()
    assert "quick-gpu" in result.output


def test_daemon_status_not_running():
    """ccm daemon status exits 0 and shows 'not running'."""
    result = runner.invoke(app, ["daemon", "status"])
    assert result.exit_code == 0
    assert "not running" in result.output.lower()


def test_packages_list():
    """ccm packages exits 0 and shows packages."""
    result = runner.invoke(app, ["packages"])
    assert result.exit_code == 0
    assert "Package" in result.output
