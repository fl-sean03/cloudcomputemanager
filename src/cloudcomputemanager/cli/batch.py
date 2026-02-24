"""Batch job operations for CloudComputeManager."""

import asyncio
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import print as rprint

from cloudcomputemanager.core.config import get_settings
from cloudcomputemanager.core.database import init_db, get_session
from cloudcomputemanager.core.models import Job, JobStatus
from cloudcomputemanager.providers.vast import VastProvider
from cloudcomputemanager.cli.jobs import check_job_completion

console = Console()


async def batch_submit(
    config_files: List[Path],
    max_parallel: int = 3,
    dry_run: bool = False,
    project: Optional[str] = None,
) -> None:
    """Submit multiple jobs from YAML configuration files.

    Args:
        config_files: List of job YAML configuration files
        max_parallel: Maximum concurrent jobs
        dry_run: If True, only preview what would be submitted
        project: Override project name for all jobs
    """
    from cloudcomputemanager.cli.jobs import submit_job

    console.print(f"\n[bold]Batch Submit[/bold]: {len(config_files)} jobs")
    console.print(f"  Max parallel: {max_parallel}")
    console.print(f"  Dry run: {dry_run}\n")

    if dry_run:
        console.print("[bold]Would submit:[/bold]")
        for i, config_file in enumerate(config_files):
            try:
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                name = config.get("name", config_file.stem)
                gpu = config.get("resources", {}).get("gpu_type", "unknown")
                console.print(f"  {i+1}. {name} ({gpu}) - {config_file}")
            except Exception as e:
                console.print(f"  {i+1}. [red]ERROR[/red]: {config_file} - {e}")
        return

    # Initialize
    await init_db()

    # Track submitted jobs
    submitted = []
    failed = []
    pending = list(config_files)

    # Submit with controlled parallelism
    running_tasks = []

    async def submit_one(config_file: Path):
        """Submit a single job and return (config_file, success, error)."""
        try:
            # Load config to get job name
            with open(config_file) as f:
                config = yaml.safe_load(f)

            if project:
                config["project"] = project

            job_name = config.get("name", config_file.stem)
            console.print(f"[blue]Submitting:[/blue] {job_name}")

            # Call submit_job but don't wait
            await submit_job(config_file, None, wait=False)
            return (config_file, True, None)
        except Exception as e:
            return (config_file, False, str(e))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Submitting jobs...", total=len(config_files))

        while pending or running_tasks:
            # Start new tasks up to max_parallel
            while pending and len(running_tasks) < max_parallel:
                config_file = pending.pop(0)
                task_coro = asyncio.create_task(submit_one(config_file))
                running_tasks.append(task_coro)

            if running_tasks:
                # Wait for at least one to complete
                done, running_tasks_set = await asyncio.wait(
                    running_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                running_tasks = list(running_tasks_set)

                for completed_task in done:
                    config_file, success, error = await completed_task
                    if success:
                        submitted.append(config_file)
                    else:
                        failed.append((config_file, error))
                    progress.update(task, advance=1)

            # Small delay to avoid hammering the API
            await asyncio.sleep(1)

    # Summary
    console.print(f"\n[bold]Batch Submit Complete[/bold]")
    console.print(f"  [green]Submitted:[/green] {len(submitted)}")
    console.print(f"  [red]Failed:[/red] {len(failed)}")

    if failed:
        console.print("\n[red]Failed jobs:[/red]")
        for config_file, error in failed:
            console.print(f"  - {config_file}: {error}")


async def batch_status(project: Optional[str] = None, limit: int = 50) -> None:
    """Show status of batch/multiple jobs.

    Args:
        project: Filter by project name
        limit: Maximum jobs to show
    """
    await init_db()

    from sqlmodel import select

    async with get_session() as session:
        stmt = select(Job).order_by(Job.created_at.desc()).limit(limit)
        if project:
            stmt = stmt.where(Job.project == project)
        result = await session.execute(stmt)
        jobs = result.scalars().all()

    if not jobs:
        console.print("[yellow]No jobs found.[/yellow]")
        return

    # Count by status
    status_counts = {}
    total_cost = 0.0
    for job in jobs:
        status_counts[job.status] = status_counts.get(job.status, 0) + 1
        total_cost += job.total_cost_usd or 0

    # Display summary
    console.print(f"\n[bold]Batch Status[/bold]")
    if project:
        console.print(f"  Project: {project}")
    console.print(f"  Total jobs: {len(jobs)}")
    console.print(f"  Total cost: ${total_cost:.2f}\n")

    # Status breakdown
    status_colors = {
        JobStatus.PENDING: ("yellow", "PENDING"),
        JobStatus.PROVISIONING: ("yellow", "PROVISIONING"),
        JobStatus.RUNNING: ("green", "RUNNING"),
        JobStatus.CHECKPOINTING: ("blue", "CHECKPOINTING"),
        JobStatus.RECOVERING: ("yellow", "RECOVERING"),
        JobStatus.COMPLETED: ("green", "COMPLETED"),
        JobStatus.FAILED: ("red", "FAILED"),
        JobStatus.CANCELLED: ("dim", "CANCELLED"),
    }

    for status, count in status_counts.items():
        color, name = status_colors.get(status, ("white", status.value))
        console.print(f"  [{color}]{name}[/{color}]: {count}")

    # Show running jobs
    running_jobs = [j for j in jobs if j.status == JobStatus.RUNNING]
    if running_jobs:
        console.print(f"\n[bold]Running Jobs ({len(running_jobs)}):[/bold]")
        for job in running_jobs[:10]:  # Show max 10
            console.print(f"  - {job.name} ({job.job_id}) on {job.instance_id}")

    # Show recent failures
    failed_jobs = [j for j in jobs if j.status == JobStatus.FAILED]
    if failed_jobs:
        console.print(f"\n[bold red]Recent Failures ({len(failed_jobs)}):[/bold red]")
        for job in failed_jobs[:5]:  # Show max 5
            error = job.error_message[:50] + "..." if job.error_message and len(job.error_message) > 50 else job.error_message
            console.print(f"  - {job.name}: {error or 'No error message'}")


async def batch_wait(
    project: Optional[str] = None,
    timeout: int = 86400,
    poll_interval: int = 60,
) -> None:
    """Wait for all running jobs to complete.

    Args:
        project: Filter by project name
        timeout: Maximum wait time in seconds
        poll_interval: Time between status checks
    """
    await init_db()
    provider = VastProvider()
    settings = get_settings()

    start_time = asyncio.get_event_loop().time()

    console.print(f"\n[bold]Waiting for jobs to complete...[/bold]")
    if project:
        console.print(f"  Project: {project}")
    console.print(f"  Timeout: {timeout}s")
    console.print(f"  Poll interval: {poll_interval}s\n")

    try:
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                console.print("[yellow]Timeout reached.[/yellow]")
                break

            # Get running jobs
            from sqlmodel import select
            async with get_session() as session:
                stmt = select(Job).where(Job.status == JobStatus.RUNNING)
                if project:
                    stmt = stmt.where(Job.project == project)
                result = await session.execute(stmt)
                running_jobs = result.scalars().all()

            if not running_jobs:
                console.print("[green]All jobs completed![/green]")
                break

            # Check each running job
            completed_this_round = []
            for job in running_jobs:
                if job.instance_id:
                    completed, exit_code = await check_job_completion(provider, job.instance_id)
                    if completed:
                        completed_this_round.append((job, exit_code))

            # Update completed jobs
            if completed_this_round:
                async with get_session() as session:
                    for job, exit_code in completed_this_round:
                        stmt = select(Job).where(Job.job_id == job.job_id)
                        result = await session.execute(stmt)
                        db_job = result.scalar_one_or_none()
                        if db_job:
                            db_job.exit_code = exit_code
                            db_job.status = JobStatus.COMPLETED if exit_code == 0 else JobStatus.FAILED
                            db_job.completed_at = datetime.utcnow()
                            session.add(db_job)

                            status = "COMPLETED" if exit_code == 0 else f"FAILED ({exit_code})"
                            console.print(f"[{'green' if exit_code == 0 else 'red'}]{job.name}: {status}[/]")

                            # Sync results
                            sync_config = job.sync_config or {}
                            if sync_config and job.instance_id:
                                sync_dir = settings.sync_local_path / job.job_id
                                sync_dir.mkdir(parents=True, exist_ok=True)
                                try:
                                    await provider.rsync_download(
                                        job.instance_id,
                                        sync_config.get("source", "/workspace") + "/",
                                        str(sync_dir) + "/",
                                        exclude=sync_config.get("exclude_patterns", []),
                                    )
                                except Exception:
                                    pass

                            # Terminate instance
                            if job.instance_id:
                                try:
                                    await provider.terminate_instance(job.instance_id)
                                except Exception:
                                    pass

            # Status update
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            console.print(f"[dim]Elapsed: {hours}h {minutes}m | Running: {len(running_jobs) - len(completed_this_round)}[/dim]")

            await asyncio.sleep(poll_interval)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Jobs continue running.[/yellow]")


async def batch_cancel(
    project: Optional[str] = None,
    status_filter: Optional[str] = None,
    force: bool = False,
) -> None:
    """Cancel multiple jobs.

    Args:
        project: Filter by project name
        status_filter: Filter by status (e.g., "running", "pending")
        force: Skip confirmation
    """
    await init_db()
    provider = VastProvider()

    from sqlmodel import select

    async with get_session() as session:
        stmt = select(Job).where(
            Job.status.in_([JobStatus.RUNNING, JobStatus.PROVISIONING, JobStatus.PENDING])
        )
        if project:
            stmt = stmt.where(Job.project == project)
        if status_filter:
            stmt = stmt.where(Job.status == JobStatus(status_filter))

        result = await session.execute(stmt)
        jobs = result.scalars().all()

    if not jobs:
        console.print("[yellow]No active jobs found to cancel.[/yellow]")
        return

    console.print(f"\n[bold]Jobs to cancel: {len(jobs)}[/bold]")
    for job in jobs[:10]:
        console.print(f"  - {job.name} ({job.job_id})")
    if len(jobs) > 10:
        console.print(f"  ... and {len(jobs) - 10} more")

    if not force:
        console.print("\n[yellow]Use --force to actually cancel these jobs[/yellow]")
        return

    # Cancel jobs
    cancelled = 0
    async with get_session() as session:
        for job in jobs:
            try:
                # Terminate instance if exists
                if job.instance_id:
                    try:
                        await provider.terminate_instance(job.instance_id)
                    except Exception:
                        pass

                # Update status
                stmt = select(Job).where(Job.job_id == job.job_id)
                result = await session.execute(stmt)
                db_job = result.scalar_one_or_none()
                if db_job:
                    db_job.status = JobStatus.CANCELLED
                    db_job.completed_at = datetime.utcnow()
                    session.add(db_job)
                    cancelled += 1

            except Exception as e:
                console.print(f"[red]Failed to cancel {job.job_id}: {e}[/red]")

    console.print(f"\n[green]Cancelled {cancelled} jobs[/green]")
