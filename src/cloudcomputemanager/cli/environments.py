"""CLI commands for environment management.

Provides commands for exporting, packing, and managing conda environments
for use with CCM cloud jobs.
"""

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()

env_app = typer.Typer(
    name="env",
    help="Manage environments for cloud jobs",
    no_args_is_help=True,
)


@env_app.command("export")
def env_export(
    name: Optional[str] = typer.Option(None, "-n", "--name", help="Conda environment name"),
    output: Path = typer.Option("environment.yml", "-o", "--output", help="Output file path"),
    no_builds: bool = typer.Option(True, "--no-builds/--with-builds", help="Exclude build strings for portability"),
):
    """Export a conda environment to a YAML file.

    The exported file can be used with `environment.conda_env` in a job YAML
    to recreate the environment on a cloud instance.

    Examples:
        ccm env export -n alchem -o alchem-env.yml
        ccm env export  # exports current active environment
    """
    cmd = ["conda", "env", "export"]
    if name:
        cmd.extend(["-n", name])
    if no_builds:
        cmd.append("--no-builds")

    console.print(f"Exporting conda environment{f' {name}' if name else ''}...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            console.print(f"[red]Error: {result.stderr.strip()}[/red]")
            raise typer.Exit(1)

        # Write output
        output.write_text(result.stdout)
        console.print(f"[green]Exported to {output}[/green]")

        # Show summary
        lines = result.stdout.strip().split("\n")
        pkg_count = sum(1 for line in lines if line.strip().startswith("- ") and "=" in line)
        console.print(f"  Packages: {pkg_count}")
        console.print(f"  Size: {output.stat().st_size / 1024:.1f} KB")
        console.print(f"\n  Use in job YAML:")
        console.print(f"    environment:")
        console.print(f"      conda_env: {output}")

    except FileNotFoundError:
        console.print("[red]Error: conda not found. Is conda installed?[/red]")
        raise typer.Exit(1)


@env_app.command("pack")
def env_pack(
    name: Optional[str] = typer.Option(None, "-n", "--name", help="Conda environment name"),
    output: Path = typer.Option("environment.tar.gz", "-o", "--output", help="Output tarball path"),
):
    """Create a conda-pack tarball from an environment.

    The tarball is a self-contained, relocatable copy of the environment
    that can be unpacked on any Linux machine without conda. This is the
    fastest way to deploy complex environments to cloud instances.

    Requires: conda-pack (install with `conda install conda-pack`)

    Examples:
        ccm env pack -n alchem -o alchem-env.tar.gz
        ccm env pack  # packs current active environment
    """
    # Check conda-pack is installed
    try:
        import conda_pack
    except ImportError:
        console.print("[red]conda-pack is not installed.[/red]")
        console.print("Install with: [bold]conda install conda-pack[/bold]")
        raise typer.Exit(1)

    env_name = name
    if not env_name:
        # Try to detect active environment
        import os
        env_name = os.environ.get("CONDA_DEFAULT_ENV")
        if not env_name or env_name == "base":
            console.print("[red]No environment specified and no active conda environment.[/red]")
            console.print("Use: ccm env pack -n <env_name>")
            raise typer.Exit(1)

    console.print(f"Packing conda environment: [bold]{env_name}[/bold]")
    console.print(f"Output: {output}")
    console.print("[dim]This may take a few minutes...[/dim]")

    try:
        conda_pack.pack(
            name=env_name,
            output=str(output),
            force=True,
        )

        size_mb = output.stat().st_size / (1024 * 1024)
        console.print(f"\n[green]Pack created: {output} ({size_mb:.1f} MB)[/green]")
        console.print(f"\n  Use in job YAML:")
        console.print(f"    environment:")
        console.print(f"      conda_pack: {output}")

    except Exception as e:
        console.print(f"[red]Error packing environment: {e}[/red]")
        raise typer.Exit(1)


@env_app.command("list")
def env_list():
    """List available local conda environments."""
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            console.print(f"[red]Error: {result.stderr.strip()}[/red]")
            raise typer.Exit(1)

        import json
        data = json.loads(result.stdout)
        envs = data.get("envs", [])

        table = Table(title="Conda Environments")
        table.add_column("Name", style="bold")
        table.add_column("Path")
        table.add_column("Packages", justify="right")

        for env_path in envs:
            env_path = Path(env_path)
            name = env_path.name if env_path.name != "miniconda3" else "base"

            # Count packages
            try:
                pkg_result = subprocess.run(
                    ["conda", "list", "-p", str(env_path), "--json"],
                    capture_output=True, text=True, timeout=10,
                )
                pkg_count = len(json.loads(pkg_result.stdout)) if pkg_result.returncode == 0 else "?"
            except Exception:
                pkg_count = "?"

            table.add_row(name, str(env_path), str(pkg_count))

        console.print(table)

    except FileNotFoundError:
        console.print("[red]Error: conda not found. Is conda installed?[/red]")
        raise typer.Exit(1)
