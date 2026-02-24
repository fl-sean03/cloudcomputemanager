"""Job management CLI commands."""

import asyncio
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from cloudcomputemanager.core.config import get_settings
from cloudcomputemanager.core.database import init_db, get_session
from cloudcomputemanager.core.models import Job, JobStatus
from cloudcomputemanager.providers.vast import VastProvider

console = Console()


async def submit_job(
    config_file: Path,
    name_override: Optional[str],
    wait: bool,
) -> None:
    """Submit a job from a YAML configuration file."""
    # Load configuration
    with open(config_file) as f:
        config = yaml.safe_load(f)

    job_name = name_override or config.get("name", config_file.stem)

    console.print(f"\n[bold]Submitting job:[/bold] {job_name}")

    # Parse configuration
    image = config.get("image", "vastai/pytorch:latest")
    command = config.get("command", "")
    resources = config.get("resources", {})
    checkpoint_config = config.get("checkpoint", {})
    sync_config = config.get("sync", {})
    budget = config.get("budget", {})
    upload_config = config.get("upload", {})

    # Initialize database
    await init_db()

    # Search for offers
    provider = VastProvider()

    with console.status("Searching for GPU offers..."):
        offers = await provider.search_offers(
            gpu_type=resources.get("gpu_type", "RTX_4090"),
            gpu_count=resources.get("gpu_count", 1),
            gpu_memory_min=resources.get("gpu_memory_min", 16),
            disk_gb_min=resources.get("disk_gb", 50),
            max_hourly_rate=budget.get("max_hourly_rate"),
        )

    if not offers:
        console.print("[red]No suitable offers found![/red]")
        return

    # Show best offer
    best = offers[0]
    console.print(f"\n[green]Found {len(offers)} offers. Best:[/green]")
    console.print(f"  GPU: {best.gpu_type} x{best.gpu_count}")
    console.print(f"  Price: ${best.hourly_rate:.3f}/hr")
    console.print(f"  Location: {best.location}")

    # Create instance
    with console.status("Creating instance..."):
        instance = await provider.create_instance(
            offer_id=best.offer_id,
            image=image,
            disk_gb=resources.get("disk_gb", 50),
        )

    console.print(f"\n[green]Instance created:[/green] {instance.instance_id}")
    console.print(f"  SSH: ssh -p {instance.ssh_port} {instance.ssh_user}@{instance.ssh_host}")

    # Wait for instance to be ready
    timeout = resources.get("startup_timeout", 300)
    with console.status(f"Waiting for instance to be ready (timeout: {timeout}s)..."):
        ready = await provider.wait_for_ready(instance.instance_id, timeout=timeout)

    if not ready:
        console.print(f"[red]Instance {instance.instance_id} failed to start within {timeout}s![/red]")
        console.print("[yellow]Destroying instance to prevent charges...[/yellow]")
        try:
            await provider.terminate_instance(instance.instance_id)
            console.print("[green]Instance destroyed successfully.[/green]")
        except Exception as e:
            console.print(f"[red]Failed to destroy instance: {e}[/red]")
            console.print(f"[yellow]Manually destroy with: vastai destroy instance {instance.instance_id}[/yellow]")
        return

    console.print("[green]Instance is ready![/green]")

    # Upload files if configured
    if upload_config.get("source"):
        source_path = Path(upload_config["source"]).expanduser()
        dest_path = upload_config.get("destination", "/workspace")

        if not source_path.exists():
            console.print(f"[red]Upload source not found: {source_path}[/red]")
            console.print("[yellow]Destroying instance...[/yellow]")
            await provider.terminate_instance(instance.instance_id)
            return

        console.print(f"\n[bold]Uploading files:[/bold] {source_path} -> {dest_path}")
        with console.status("Uploading files..."):
            upload_success = await provider.rsync_upload(
                instance.instance_id,
                str(source_path) + "/",
                dest_path + "/",
                exclude=upload_config.get("exclude", []),
            )

        if not upload_success:
            console.print("[red]File upload failed![/red]")
            console.print("[yellow]Destroying instance...[/yellow]")
            await provider.terminate_instance(instance.instance_id)
            return

        console.print("[green]Files uploaded successfully![/green]")

    # Create job record
    import json

    job = Job(
        name=job_name,
        project=config.get("project"),
        status=JobStatus.RUNNING,
        image=image,
        command=command,
        resources_json=json.dumps(resources),
        checkpoint_json=json.dumps(checkpoint_config),
        sync_json=json.dumps(sync_config),
        budget_json=json.dumps(budget),
        instance_id=instance.instance_id,
    )

    # Save to database
    async with get_session() as session:
        session.add(job)

    console.print(f"\n[bold green]Job submitted successfully![/bold green]")
    console.print(f"  Job ID: {job.job_id}")
    console.print(f"  Instance: {instance.instance_id}")

    # Start the job
    if command:
        console.print("\n[bold]Starting job...[/bold]")

        # For multi-line commands, write to a script file and execute it
        import base64
        script_content = f"#!/bin/bash\nset -e\n{command}"
        script_b64 = base64.b64encode(script_content.encode()).decode()

        # Write script, make executable, and run in background
        setup_cmd = f"echo {script_b64} | base64 -d > /workspace/run_job.sh && chmod +x /workspace/run_job.sh"
        exit_code, stdout, stderr = await provider.execute_command(
            instance.instance_id, setup_cmd
        )

        if exit_code != 0:
            console.print(f"[red]Failed to create job script: {stderr}[/red]")
        else:
            # Run the script in background
            run_cmd = "cd /workspace && nohup bash /workspace/run_job.sh > /workspace/job.log 2>&1 &"
            exit_code, stdout, stderr = await provider.execute_command(
                instance.instance_id, run_cmd
            )

            if exit_code == 0:
                console.print("[green]Job started in background[/green]")
            else:
                console.print(f"[red]Failed to start job: {stderr}[/red]")

    # Show sync info
    if sync_config.get("enabled", True):
        settings = get_settings()
        sync_dir = settings.sync_local_path / job.job_id
        console.print(f"\n[bold]Sync directory:[/bold] {sync_dir}")

    if wait:
        console.print("\n[dim]Waiting for job completion... (Ctrl+C to detach)[/dim]")
        # TODO: Implement job watching
        await asyncio.sleep(5)


async def list_jobs(
    status: Optional[str],
    project: Optional[str],
    limit: int,
) -> None:
    """List all jobs."""
    await init_db()

    from sqlmodel import select

    async with get_session() as session:
        stmt = select(Job).order_by(Job.created_at.desc()).limit(limit)

        if status:
            stmt = stmt.where(Job.status == JobStatus(status))
        if project:
            stmt = stmt.where(Job.project == project)

        result = await session.execute(stmt)
        jobs = result.scalars().all()

    if not jobs:
        console.print("[yellow]No jobs found.[/yellow]")
        return

    table = Table(title="Jobs")
    table.add_column("Job ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Status", style="bold")
    table.add_column("Instance", style="dim")
    table.add_column("Created", style="dim")

    status_colors = {
        JobStatus.PENDING: "yellow",
        JobStatus.PROVISIONING: "yellow",
        JobStatus.RUNNING: "green",
        JobStatus.CHECKPOINTING: "blue",
        JobStatus.RECOVERING: "yellow",
        JobStatus.COMPLETED: "green",
        JobStatus.FAILED: "red",
        JobStatus.CANCELLED: "dim",
    }

    for job in jobs:
        color = status_colors.get(job.status, "white")
        table.add_row(
            job.job_id,
            job.name,
            f"[{color}]{job.status.value}[/{color}]",
            job.instance_id or "-",
            job.created_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)


async def show_job_status(job_id: str, watch: bool) -> None:
    """Show detailed job status."""
    await init_db()

    from sqlmodel import select

    async with get_session() as session:
        stmt = select(Job).where(Job.job_id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

    if not job:
        console.print(f"[red]Job not found: {job_id}[/red]")
        return

    # Build status panel
    status_color = {
        JobStatus.RUNNING: "green",
        JobStatus.COMPLETED: "green",
        JobStatus.FAILED: "red",
    }.get(job.status, "yellow")

    panel_content = f"""
[bold]Name:[/bold] {job.name}
[bold]Status:[/bold] [{status_color}]{job.status.value}[/{status_color}]
[bold]Instance:[/bold] {job.instance_id or 'None'}
[bold]Attempt:[/bold] {job.attempt_number + 1}
[bold]Created:[/bold] {job.created_at.strftime('%Y-%m-%d %H:%M:%S')}
[bold]Started:[/bold] {job.started_at.strftime('%Y-%m-%d %H:%M:%S') if job.started_at else 'Not started'}
[bold]Cost:[/bold] ${job.total_cost_usd:.2f}
"""

    if job.error_message:
        panel_content += f"\n[bold red]Error:[/bold red] {job.error_message}"

    console.print(Panel(panel_content, title=f"Job: {job.job_id}"))

    # Show instance details if running
    if job.instance_id and job.status == JobStatus.RUNNING:
        provider = VastProvider()
        instance = await provider.get_instance(job.instance_id)

        if instance:
            console.print(f"\n[bold]Instance Details:[/bold]")
            console.print(f"  SSH: ssh -p {instance.ssh_port} {instance.ssh_user}@{instance.ssh_host}")
            console.print(f"  GPU: {instance.gpu_type} x{instance.gpu_count}")
            console.print(f"  Rate: ${instance.hourly_rate:.3f}/hr")


async def show_job_logs(job_id: str, tail: int, follow: bool) -> None:
    """Show job logs."""
    await init_db()

    from sqlmodel import select

    async with get_session() as session:
        stmt = select(Job).where(Job.job_id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

    if not job or not job.instance_id:
        console.print(f"[red]Job not found or no instance: {job_id}[/red]")
        return

    provider = VastProvider()

    cmd = f"tail -n {tail} /workspace/job.log 2>/dev/null || cat /workspace/log.lammps 2>/dev/null || echo 'No logs found'"
    exit_code, stdout, stderr = await provider.execute_command(job.instance_id, cmd)

    if stdout:
        console.print(Panel(stdout, title=f"Logs: {job_id}"))
    else:
        console.print("[yellow]No logs available.[/yellow]")


async def cancel_job(job_id: str, force: bool) -> None:
    """Cancel a running job."""
    await init_db()

    from sqlmodel import select

    async with get_session() as session:
        stmt = select(Job).where(Job.job_id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if not job:
            console.print(f"[red]Job not found: {job_id}[/red]")
            return

        if job.status not in [JobStatus.RUNNING, JobStatus.PROVISIONING, JobStatus.RECOVERING]:
            console.print(f"[yellow]Job is not running: {job.status.value}[/yellow]")
            return

        # Terminate instance if exists
        if job.instance_id:
            provider = VastProvider()
            with console.status("Terminating instance..."):
                await provider.terminate_instance(job.instance_id)

        # Update job status
        job.status = JobStatus.CANCELLED
        session.add(job)

    console.print(f"[green]Job cancelled: {job_id}[/green]")


async def sync_job(job_id: str) -> None:
    """Trigger immediate sync for a job."""
    await init_db()

    from sqlmodel import select

    async with get_session() as session:
        stmt = select(Job).where(Job.job_id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

    if not job or not job.instance_id:
        console.print(f"[red]Job not found or no instance: {job_id}[/red]")
        return

    provider = VastProvider()
    sync_config = job.sync_config

    if not sync_config:
        console.print("[yellow]No sync configuration for this job.[/yellow]")
        return

    settings = get_settings()
    local_dest = settings.sync_local_path / job_id
    local_dest.mkdir(parents=True, exist_ok=True)

    with console.status("Syncing..."):
        success = await provider.rsync_download(
            job.instance_id,
            sync_config.source + "/",
            str(local_dest) + "/",
            exclude=sync_config.exclude_patterns,
        )

    if success:
        console.print(f"[green]Sync completed![/green]")
        console.print(f"  Local path: {local_dest}")
    else:
        console.print("[red]Sync failed![/red]")


async def checkpoint_job(job_id: str) -> None:
    """Trigger immediate checkpoint for a job."""
    await init_db()

    from sqlmodel import select

    async with get_session() as session:
        stmt = select(Job).where(Job.job_id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

    if not job or not job.instance_id:
        console.print(f"[red]Job not found or no instance: {job_id}[/red]")
        return

    from cloudcomputemanager.checkpoint import CheckpointOrchestrator

    provider = VastProvider()
    orchestrator = CheckpointOrchestrator(provider)

    with console.status("Creating checkpoint..."):
        checkpoint = await orchestrator.save_checkpoint(
            job.job_id,
            job.instance_id,
            job.checkpoint_config,
        )

    if checkpoint:
        console.print(f"[green]Checkpoint created![/green]")
        console.print(f"  ID: {checkpoint.checkpoint_id}")
        console.print(f"  Location: {checkpoint.location}")
        console.print(f"  Size: {checkpoint.size_bytes} bytes")
        console.print(f"  Verified: {checkpoint.verified}")
    else:
        console.print("[red]Failed to create checkpoint![/red]")
