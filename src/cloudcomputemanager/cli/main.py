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


@jobs_app.command("wait")
def jobs_wait(
    job_id: str = typer.Argument(..., help="Job ID"),
    timeout: int = typer.Option(
        86400, "--timeout", "-t", help="Max wait time in seconds (default: 24h)"
    ),
    auto_terminate: bool = typer.Option(
        True, "--terminate/--no-terminate", help="Terminate instance on completion"
    ),
):
    """Wait for a job to complete, then sync results and optionally terminate.

    Examples:
        ccm jobs wait job_abc123                    # Wait and auto-terminate
        ccm jobs wait job_abc123 --no-terminate    # Wait but keep instance
        ccm jobs wait job_abc123 --timeout 3600    # Wait max 1 hour
    """
    from cloudcomputemanager.cli.jobs import wait_for_existing_job

    asyncio.run(wait_for_existing_job(job_id, timeout, auto_terminate))


@jobs_app.command("complete")
def jobs_complete(
    job_id: str = typer.Argument(..., help="Job ID"),
    status: str = typer.Option(
        "completed", "--status", "-s", help="Final status (completed/failed)"
    ),
    terminate: bool = typer.Option(
        True, "--terminate/--no-terminate", help="Terminate instance"
    ),
):
    """Mark a job as complete, sync results, and terminate instance.

    Use this when a job has finished but CCM didn't detect it automatically.

    Examples:
        ccm jobs complete job_abc123                  # Mark as completed
        ccm jobs complete job_abc123 --status failed # Mark as failed
        ccm jobs complete job_abc123 --no-terminate  # Keep instance running
    """
    from cloudcomputemanager.cli.jobs import complete_job

    asyncio.run(complete_job(job_id, status, terminate))


# ============================================================================
# Batch Commands
# ============================================================================

batch_app = typer.Typer(help="Batch job operations")
app.add_typer(batch_app, name="batch")


@batch_app.command("submit")
def batch_submit_cmd(
    config_files: list[Path] = typer.Argument(
        ...,
        help="Job configuration YAML files (supports glob patterns)",
    ),
    parallel: int = typer.Option(
        3, "--parallel", "-p", help="Maximum concurrent jobs"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Preview without submitting"
    ),
    project: Optional[str] = typer.Option(
        None, "--project", help="Override project name for all jobs"
    ),
):
    """Submit multiple jobs from YAML configuration files.

    Examples:
        ccm batch submit job1.yaml job2.yaml job3.yaml
        ccm batch submit jobs/*.yaml --parallel 5
        ccm batch submit jobs/*.yaml --dry-run
        ccm batch submit jobs/*.yaml --project my-project
    """
    from cloudcomputemanager.cli.batch import batch_submit

    # Expand glob patterns
    expanded_files = []
    for path in config_files:
        if '*' in str(path):
            expanded_files.extend(Path('.').glob(str(path)))
        else:
            expanded_files.append(path)

    if not expanded_files:
        console.print("[red]No configuration files found[/red]")
        raise typer.Exit(1)

    asyncio.run(batch_submit(expanded_files, parallel, dry_run, project))


@batch_app.command("status")
def batch_status_cmd(
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project"
    ),
    limit: int = typer.Option(
        50, "--limit", "-l", help="Maximum jobs to show"
    ),
):
    """Show status summary of multiple jobs.

    Examples:
        ccm batch status
        ccm batch status --project ensemble-sampling
    """
    from cloudcomputemanager.cli.batch import batch_status

    asyncio.run(batch_status(project, limit))


@batch_app.command("wait")
def batch_wait_cmd(
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project"
    ),
    timeout: int = typer.Option(
        86400, "--timeout", "-t", help="Max wait time in seconds"
    ),
    poll_interval: int = typer.Option(
        60, "--poll", help="Seconds between status checks"
    ),
):
    """Wait for all running jobs to complete.

    Monitors running jobs, syncs results, and terminates instances
    as jobs complete.

    Examples:
        ccm batch wait
        ccm batch wait --project ensemble-sampling
        ccm batch wait --timeout 3600  # Max 1 hour
    """
    from cloudcomputemanager.cli.batch import batch_wait

    asyncio.run(batch_wait(project, timeout, poll_interval))


@batch_app.command("cancel")
def batch_cancel_cmd(
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project"
    ),
    status: Optional[str] = typer.Option(
        None, "--status", "-s", help="Filter by status"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Actually cancel (default: preview only)"
    ),
):
    """Cancel multiple jobs.

    By default, shows what would be cancelled. Use --force to execute.

    Examples:
        ccm batch cancel                           # Preview all active jobs
        ccm batch cancel --project my-proj         # Preview project jobs
        ccm batch cancel --project my-proj --force # Actually cancel
    """
    from cloudcomputemanager.cli.batch import batch_cancel

    asyncio.run(batch_cancel(project, status, force))


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
# Daemon Commands
# ============================================================================

daemon_app = typer.Typer(help="Background monitoring daemon")
app.add_typer(daemon_app, name="daemon")


@daemon_app.command("start")
def daemon_start(
    foreground: bool = typer.Option(
        False, "--foreground", "-f", help="Run in foreground (don't daemonize)"
    ),
    poll_interval: int = typer.Option(
        30, "--poll-interval", help="Seconds between status checks"
    ),
):
    """Start the CCM daemon for background job monitoring.

    The daemon monitors all running jobs and:
    - Detects job completion
    - Syncs results automatically
    - Terminates instances after completion
    - Handles preemption recovery

    Examples:
        ccm daemon start              # Start as background daemon
        ccm daemon start --foreground # Run in foreground (for debugging)
    """
    from cloudcomputemanager.daemon.service import DaemonService, run_daemon, daemonize
    from cloudcomputemanager.daemon.monitor import MonitorConfig

    if DaemonService.is_running():
        console.print("[yellow]Daemon is already running[/yellow]")
        pid = DaemonService.get_pid()
        console.print(f"  PID: {pid}")
        return

    config = MonitorConfig(poll_interval=poll_interval)

    if foreground:
        console.print("[bold]Starting daemon in foreground...[/bold]")
        console.print("Press Ctrl+C to stop\n")
        run_daemon(config)
    else:
        console.print("[bold]Starting daemon in background...[/bold]")
        import subprocess
        import sys

        # Start daemon as subprocess
        proc = subprocess.Popen(
            [sys.executable, "-m", "cloudcomputemanager.daemon.service"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait a moment and check if started
        import time
        time.sleep(2)

        if DaemonService.is_running():
            console.print(f"[green]Daemon started[/green]")
            console.print(f"  PID: {DaemonService.get_pid()}")
        else:
            console.print("[red]Failed to start daemon[/red]")


@daemon_app.command("stop")
def daemon_stop():
    """Stop the running CCM daemon."""
    from cloudcomputemanager.daemon.service import DaemonService

    if not DaemonService.is_running():
        console.print("[yellow]Daemon is not running[/yellow]")
        return

    pid = DaemonService.get_pid()
    console.print(f"Stopping daemon (PID: {pid})...")

    if DaemonService.stop():
        import time
        time.sleep(1)
        if not DaemonService.is_running():
            console.print("[green]Daemon stopped[/green]")
        else:
            console.print("[yellow]Daemon stopping...[/yellow]")
    else:
        console.print("[red]Failed to stop daemon[/red]")


@daemon_app.command("status")
def daemon_status():
    """Show daemon status."""
    from cloudcomputemanager.daemon.service import DaemonService

    status = DaemonService.get_status()
    pid = DaemonService.get_pid()

    if pid:
        console.print(f"[green]Daemon is running[/green]")
        console.print(f"  PID: {pid}")
        if status:
            console.print(f"  Monitored jobs: {status.get('monitored_jobs', 'unknown')}")
            console.print(f"  Poll interval: {status.get('poll_interval', 'unknown')}s")
            console.print(f"  Last update: {status.get('timestamp', 'unknown')}")
    else:
        console.print("[yellow]Daemon is not running[/yellow]")
        if status and status.get("stopped_at"):
            console.print(f"  Last stopped: {status['stopped_at']}")


@daemon_app.command("logs")
def daemon_logs(
    lines: int = typer.Option(50, "--lines", "-n", help="Number of log lines"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
):
    """Show daemon logs."""
    from cloudcomputemanager.daemon.service import DaemonService
    from cloudcomputemanager.core.config import get_settings

    settings = get_settings()
    log_file = settings.data_dir / "daemon.log"

    if not log_file.exists():
        console.print("[yellow]No daemon logs found[/yellow]")
        return

    if follow:
        # Follow mode - use tail -f
        import subprocess
        console.print(f"[dim]Following {log_file}... (Ctrl+C to stop)[/dim]\n")
        try:
            subprocess.run(["tail", "-f", str(log_file)])
        except KeyboardInterrupt:
            pass
    else:
        # Show recent logs
        logs = DaemonService.get_logs(lines)
        if not logs:
            console.print("[yellow]No recent logs[/yellow]")
            return

        for entry in logs:
            ts = entry.get("timestamp", "")[:19]
            event = entry.get("event", "unknown")
            job_id = entry.get("job_id", "")
            data = entry.get("data", {})

            # Color by event type
            if "completed" in event:
                color = "green"
            elif "failed" in event or "error" in event:
                color = "red"
            elif "preempted" in event:
                color = "yellow"
            else:
                color = "blue"

            console.print(f"[dim]{ts}[/dim] [{color}]{event}[/{color}] {job_id} {data}")


# ============================================================================
# Cleanup Command
# ============================================================================


@app.command("cleanup")
def cleanup(
    dry_run: bool = typer.Option(
        True, "--dry-run/--execute", help="Preview changes without executing"
    ),
    stale_jobs: bool = typer.Option(
        True, "--jobs/--no-jobs", help="Clean up stale jobs"
    ),
    orphan_instances: bool = typer.Option(
        False, "--orphans/--no-orphans", help="Terminate orphan instances"
    ),
):
    """Clean up stale jobs and orphan instances.

    By default, runs in dry-run mode showing what would be cleaned.
    Use --execute to actually perform cleanup.

    Examples:
        ccm cleanup                    # Preview what would be cleaned
        ccm cleanup --execute          # Actually clean up stale jobs
        ccm cleanup --orphans          # Also show orphan instances
        ccm cleanup --execute --orphans # Clean up jobs AND terminate orphans
    """
    from cloudcomputemanager.core.database import init_db
    from cloudcomputemanager.core.cleanup import (
        cleanup_stale_jobs,
        cleanup_orphan_instances,
        get_cleanup_summary,
    )
    from cloudcomputemanager.providers.vast import VastProvider

    async def run_cleanup():
        await init_db()
        provider = VastProvider()

        # Get summary first
        summary = await get_cleanup_summary(provider)

        if dry_run:
            console.print("\n[bold]Cleanup Preview[/bold] (dry-run mode)\n")
        else:
            console.print("\n[bold]Executing Cleanup[/bold]\n")

        # Stale jobs
        if stale_jobs:
            console.print(f"[cyan]Stale Jobs:[/cyan] {summary['stale_jobs']}")
            if summary['stale_job_reasons']:
                for reason, count in summary['stale_job_reasons'].items():
                    console.print(f"  - {reason}: {count}")

            if summary['stale_jobs'] > 0:
                cleaned = await cleanup_stale_jobs(provider, dry_run=dry_run)
                if dry_run:
                    console.print("\n[yellow]Would mark as FAILED:[/yellow]")
                else:
                    console.print("\n[green]Marked as FAILED:[/green]")
                for job_id, old_status, reason in cleaned:
                    console.print(f"  {job_id}: {old_status} -> FAILED ({reason})")

        # Orphan instances
        if orphan_instances:
            console.print(f"\n[cyan]Orphan Instances:[/cyan] {summary['orphan_instances']}")
            if summary['orphan_instances'] > 0:
                orphans = await cleanup_orphan_instances(provider, dry_run=dry_run)
                if dry_run:
                    console.print("\n[yellow]Would terminate:[/yellow]")
                else:
                    console.print("\n[green]Terminated:[/green]")
                for instance_id, label in orphans:
                    console.print(f"  {instance_id}: {label or '(no label)'}")

        if dry_run and (summary['stale_jobs'] > 0 or summary['orphan_instances'] > 0):
            console.print("\n[dim]Run with --execute to apply changes[/dim]")
        elif not dry_run:
            console.print("\n[green]Cleanup complete![/green]")
        else:
            console.print("\n[green]Nothing to clean up![/green]")

    asyncio.run(run_cleanup())


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
