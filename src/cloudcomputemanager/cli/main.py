"""Main CLI entry point for CloudComputeManager.

Provides commands for:
- Job submission and management
- Instance operations
- Checkpoint and sync control
- Configuration
"""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from cloudcomputemanager import __version__
from cloudcomputemanager.core.config import get_settings

app = typer.Typer(
    name="cloudcomputemanager",
    help="GPU cloud management with automatic checkpointing and spot recovery",
    no_args_is_help=True,
)

console = Console()


def version_callback(value: bool):
    if value:
        rprint(f"[bold blue]CloudComputeManager[/bold blue] v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    """CloudComputeManager: GPU cloud management with automatic checkpointing."""
    pass


# ============================================================================
# Job Commands
# ============================================================================

jobs_app = typer.Typer(help="Manage jobs")
app.add_typer(jobs_app, name="jobs")


@jobs_app.command("submit")
def jobs_submit(
    config_file: Path = typer.Argument(
        ...,
        help="Path to job configuration YAML file",
        exists=True,
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Override job name"
    ),
    wait: bool = typer.Option(
        False, "--wait", "-w", help="Wait for job completion"
    ),
):
    """Submit a new job from a YAML configuration file."""
    from cloudcomputemanager.cli.jobs import submit_job

    asyncio.run(submit_job(config_file, name, wait))


@jobs_app.command("list")
def jobs_list(
    status: Optional[str] = typer.Option(
        None, "--status", "-s", help="Filter by status"
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project"
    ),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum jobs to show"),
):
    """List all jobs."""
    from cloudcomputemanager.cli.jobs import list_jobs

    asyncio.run(list_jobs(status, project, limit))


@jobs_app.command("status")
def jobs_status(
    job_id: str = typer.Argument(..., help="Job ID"),
    watch: bool = typer.Option(
        False, "--watch", "-w", help="Watch for updates"
    ),
):
    """Get job status and details."""
    from cloudcomputemanager.cli.jobs import show_job_status

    asyncio.run(show_job_status(job_id, watch))


@jobs_app.command("logs")
def jobs_logs(
    job_id: str = typer.Argument(..., help="Job ID"),
    tail: int = typer.Option(100, "--tail", "-t", help="Number of lines"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
):
    """Get job logs."""
    from cloudcomputemanager.cli.jobs import show_job_logs

    asyncio.run(show_job_logs(job_id, tail, follow))


@jobs_app.command("cancel")
def jobs_cancel(
    job_id: str = typer.Argument(..., help="Job ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Force cancellation"),
):
    """Cancel a running job."""
    from cloudcomputemanager.cli.jobs import cancel_job

    asyncio.run(cancel_job(job_id, force))


@jobs_app.command("sync")
def jobs_sync(
    job_id: str = typer.Argument(..., help="Job ID"),
):
    """Trigger immediate sync for a job."""
    from cloudcomputemanager.cli.jobs import sync_job

    asyncio.run(sync_job(job_id))


@jobs_app.command("checkpoint")
def jobs_checkpoint(
    job_id: str = typer.Argument(..., help="Job ID"),
):
    """Trigger immediate checkpoint for a job."""
    from cloudcomputemanager.cli.jobs import checkpoint_job

    asyncio.run(checkpoint_job(job_id))


# ============================================================================
# Instance Commands
# ============================================================================

instances_app = typer.Typer(help="Manage instances")
app.add_typer(instances_app, name="instances")


@instances_app.command("list")
def instances_list():
    """List all managed instances."""
    from cloudcomputemanager.cli.instances import list_instances

    asyncio.run(list_instances())


@instances_app.command("search")
def instances_search(
    gpu_type: Optional[str] = typer.Option(
        None, "--gpu", "-g", help="GPU type (e.g., RTX_4090)"
    ),
    gpu_count: int = typer.Option(1, "--count", "-c", help="Number of GPUs"),
    max_price: Optional[float] = typer.Option(
        None, "--max-price", "-p", help="Maximum hourly price"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum offers to show"),
):
    """Search available GPU offers."""
    from cloudcomputemanager.cli.instances import search_offers

    asyncio.run(search_offers(gpu_type, gpu_count, max_price, limit))


@instances_app.command("create")
def instances_create(
    offer_id: str = typer.Argument(..., help="Offer ID to use"),
    image: str = typer.Option(
        "vastai/pytorch:2.0.0", "--image", "-i", help="Docker image"
    ),
    disk: int = typer.Option(50, "--disk", "-d", help="Disk size in GB"),
):
    """Create a new instance."""
    from cloudcomputemanager.cli.instances import create_instance

    asyncio.run(create_instance(offer_id, image, disk))


@instances_app.command("ssh")
def instances_ssh(
    instance_id: str = typer.Argument(..., help="Instance ID"),
):
    """SSH into an instance."""
    from cloudcomputemanager.cli.instances import ssh_to_instance

    asyncio.run(ssh_to_instance(instance_id))


@instances_app.command("terminate")
def instances_terminate(
    instance_id: str = typer.Argument(..., help="Instance ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Force termination"),
):
    """Terminate an instance."""
    from cloudcomputemanager.cli.instances import terminate_instance

    asyncio.run(terminate_instance(instance_id, force))


# ============================================================================
# Sync Commands
# ============================================================================

sync_app = typer.Typer(help="Data synchronization")
app.add_typer(sync_app, name="sync")


@sync_app.command("status")
def sync_status(
    job_id: Optional[str] = typer.Argument(None, help="Job ID (optional)"),
):
    """Show sync status."""
    from cloudcomputemanager.cli.sync import show_sync_status

    asyncio.run(show_sync_status(job_id))


@sync_app.command("files")
def sync_files(
    job_id: str = typer.Argument(..., help="Job ID"),
):
    """List synced files for a job."""
    from cloudcomputemanager.cli.sync import list_synced_files

    asyncio.run(list_synced_files(job_id))


# ============================================================================
# Config Commands
# ============================================================================

config_app = typer.Typer(help="Configuration management")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show():
    """Show current configuration."""
    settings = get_settings()

    table = Table(title="CloudComputeManager Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Data Directory", str(settings.data_dir))
    table.add_row("Database", settings.database_url)
    table.add_row("Checkpoint Storage", settings.checkpoint_storage)
    table.add_row("Checkpoint Path", str(settings.checkpoint_local_path))
    table.add_row("Sync Path", str(settings.sync_local_path))
    table.add_row("API Host", f"{settings.api_host}:{settings.api_port}")
    table.add_row("SSH Key", str(settings.ssh_key_path))
    table.add_row("Debug", str(settings.debug))

    # Check Vast.ai API key
    try:
        settings.get_vast_api_key()
        table.add_row("Vast.ai API Key", "[green]Configured[/green]")
    except ValueError:
        table.add_row("Vast.ai API Key", "[red]Not configured[/red]")

    console.print(table)


@config_app.command("init")
def config_init():
    """Initialize CloudComputeManager configuration and directories."""
    settings = get_settings()
    settings.ensure_directories()

    console.print("[green]Created directories:[/green]")
    console.print(f"  - {settings.data_dir}")
    console.print(f"  - {settings.checkpoint_local_path}")
    console.print(f"  - {settings.sync_local_path}")

    # Check for Vast.ai API key
    try:
        settings.get_vast_api_key()
        console.print("\n[green]Vast.ai API key found.[/green]")
    except ValueError:
        console.print(
            f"\n[yellow]Warning: Vast.ai API key not found.[/yellow]\n"
            f"Create {settings.vast_api_key_file} or set CCM_VAST_API_KEY"
        )


# ============================================================================
# Quick Commands (top-level shortcuts)
# ============================================================================


@app.command("submit")
def quick_submit(
    config_file: Path = typer.Argument(..., help="Job configuration file"),
):
    """Quick submit: Submit a job from YAML config."""
    from cloudcomputemanager.cli.jobs import submit_job

    asyncio.run(submit_job(config_file, None, False))


@app.command("status")
def quick_status(
    job_id: Optional[str] = typer.Argument(None, help="Job ID (optional)"),
):
    """Quick status: Show job or all jobs status."""
    if job_id:
        from cloudcomputemanager.cli.jobs import show_job_status

        asyncio.run(show_job_status(job_id, False))
    else:
        from cloudcomputemanager.cli.jobs import list_jobs

        asyncio.run(list_jobs(None, None, 10))


@app.command("search")
def quick_search(
    gpu_type: str = typer.Argument("RTX_4090", help="GPU type"),
):
    """Quick search: Search GPU offers."""
    from cloudcomputemanager.cli.instances import search_offers

    asyncio.run(search_offers(gpu_type, 1, None, 10))


# ============================================================================
# PackStore Commands
# ============================================================================

packstore_app = typer.Typer(help="Manage pre-built packages (LAMMPS, QE, GROMACS, etc.)")
app.add_typer(packstore_app, name="packstore")


@packstore_app.command("list")
def packstore_list(
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Filter by category"
    ),
    variants: bool = typer.Option(
        False, "--variants", "-v", help="Show variants"
    ),
):
    """List available packages."""
    from cloudcomputemanager.cli.packstore import list_packages

    asyncio.run(list_packages(category, variants))


@packstore_app.command("info")
def packstore_info(
    package: str = typer.Argument(..., help="Package name"),
):
    """Show package details."""
    from cloudcomputemanager.cli.packstore import show_package_info

    asyncio.run(show_package_info(package))


@packstore_app.command("search")
def packstore_search(
    query: str = typer.Argument(..., help="Search query"),
    cuda: Optional[str] = typer.Option(None, "--cuda", help="CUDA version"),
    arch: Optional[str] = typer.Option(None, "--arch", help="GPU architecture (e.g., sm_89)"),
):
    """Search for packages."""
    from cloudcomputemanager.cli.packstore import search_packages

    asyncio.run(search_packages(query, cuda, arch))


@packstore_app.command("deploy")
def packstore_deploy(
    instance_id: str = typer.Argument(..., help="Instance ID"),
    packages: list[str] = typer.Argument(..., help="Packages to deploy"),
    strategy: str = typer.Option("auto", "--strategy", "-s", help="Deployment strategy"),
    no_verify: bool = typer.Option(False, "--no-verify", help="Skip verification"),
):
    """Deploy packages to an instance."""
    from cloudcomputemanager.cli.packstore import deploy_packages

    asyncio.run(deploy_packages(instance_id, packages, strategy, not no_verify))


@packstore_app.command("verify")
def packstore_verify(
    instance_id: str = typer.Argument(..., help="Instance ID"),
    package: str = typer.Argument(..., help="Package to verify"),
):
    """Verify package installation."""
    from cloudcomputemanager.cli.packstore import verify_package

    asyncio.run(verify_package(instance_id, package))


# ============================================================================
# Quick PackStore Commands (top-level)
# ============================================================================


@app.command("packages")
def quick_packages():
    """Quick packages: List available packages."""
    from cloudcomputemanager.cli.packstore import list_packages

    asyncio.run(list_packages(None, False))


@app.command("deploy")
def quick_deploy(
    instance_id: str = typer.Argument(..., help="Instance ID"),
    packages: list[str] = typer.Argument(..., help="Packages to deploy"),
):
    """Quick deploy: Deploy packages to an instance."""
    from cloudcomputemanager.cli.packstore import deploy_packages

    asyncio.run(deploy_packages(instance_id, packages, "auto", True))


if __name__ == "__main__":
    app()
