"""Job execution helpers: exec, upload, SSH by job_id."""

import os
import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from cloudcomputemanager.core.database import init_db, get_session
from cloudcomputemanager.core.models import Job, JobStatus
from cloudcomputemanager.providers.vast import VastProvider

console = Console()


async def _resolve_job_instance(target: str) -> tuple[Optional[Job], Optional[str]]:
    """Resolve a job_id or instance_id to get the instance_id.

    Returns (job, instance_id) — job may be None if target is a raw instance_id.
    """
    await init_db()

    from sqlmodel import select

    # Try as job_id first
    async with get_session() as session:
        stmt = select(Job).where(Job.job_id == target)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

    if job:
        if not job.instance_id:
            console.print(f"[red]Job {target} has no instance assigned[/red]")
            return job, None
        if job.status not in (JobStatus.RUNNING, JobStatus.CHECKPOINTING, JobStatus.RECOVERING):
            console.print(f"[yellow]Warning: Job {target} is {job.status.value}, instance may not be accessible[/yellow]")
        return job, job.instance_id

    # Treat as raw instance_id
    return None, target


async def exec_on_job(job_id: str, command: str) -> None:
    """Execute a command on a running job's instance."""
    _, instance_id = await _resolve_job_instance(job_id)
    if not instance_id:
        return

    provider = VastProvider()

    console.print(f"[dim]Executing on instance {instance_id}...[/dim]")
    exit_code, stdout, stderr = await provider.execute_command(instance_id, command, timeout=120)

    if stdout.strip():
        console.print(stdout.strip())
    if stderr.strip():
        console.print(f"[yellow]{stderr.strip()}[/yellow]")

    if exit_code != 0:
        console.print(f"[red]Exit code: {exit_code}[/red]")


async def upload_to_job(job_id: str, local_path: Path, remote_path: str) -> None:
    """Upload files to a running job's instance."""
    _, instance_id = await _resolve_job_instance(job_id)
    if not instance_id:
        return

    if not local_path.exists():
        console.print(f"[red]Local path not found: {local_path}[/red]")
        return

    provider = VastProvider()

    local_str = str(local_path)
    if local_path.is_dir():
        local_str += "/"
        remote_path = remote_path.rstrip("/") + "/"

    console.print(f"[bold]Uploading:[/bold] {local_path} -> {remote_path}")
    success = await provider.rsync_upload(instance_id, local_str, remote_path)

    if success:
        console.print("[green]Upload complete[/green]")
    else:
        console.print("[red]Upload failed[/red]")


async def ssh_to_job(target: str) -> None:
    """SSH into a job's instance (accepts job_id or instance_id)."""
    _, instance_id = await _resolve_job_instance(target)
    if not instance_id:
        return

    provider = VastProvider()
    instance = await provider.get_instance(instance_id)

    if not instance:
        console.print(f"[red]Instance {instance_id} not found[/red]")
        return

    from cloudcomputemanager.core.config import get_settings
    settings = get_settings()

    ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]

    if settings.ssh_key_path.exists():
        ssh_cmd.extend(["-i", str(settings.ssh_key_path)])

    ssh_cmd.extend(["-p", str(instance.ssh_port), f"{instance.ssh_user}@{instance.ssh_host}"])

    console.print(f"[bold]Connecting to:[/bold] {instance.ssh_user}@{instance.ssh_host}:{instance.ssh_port}")
    console.print(f"[dim]{' '.join(ssh_cmd)}[/dim]\n")

    # Replace current process with SSH (interactive)
    os.execvp("ssh", ssh_cmd)


async def get_ssh_credentials(job_id: str) -> Optional[dict]:
    """Get SSH credentials for a job's instance.

    Returns dict with host, port, user, key_path — or None if not available.
    Used by the Agent SDK to give agents direct SSH access.
    """
    _, instance_id = await _resolve_job_instance(job_id)
    if not instance_id:
        return None

    provider = VastProvider()
    instance = await provider.get_instance(instance_id)

    if not instance:
        return None

    from cloudcomputemanager.core.config import get_settings
    settings = get_settings()

    return {
        "host": instance.ssh_host,
        "port": instance.ssh_port,
        "user": instance.ssh_user,
        "key_path": str(settings.ssh_key_path) if settings.ssh_key_path.exists() else None,
        "instance_id": instance_id,
    }
