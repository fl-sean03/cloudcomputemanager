"""CLI commands for job metrics and performance monitoring.

Supports multiple log formats:
- NAMD (molecular dynamics)
- LAMMPS (molecular dynamics)
- PyTorch/ML training (loss, epoch, step)
- Generic progress patterns
"""

import asyncio
import re
from datetime import datetime
from typing import Optional, Callable

from rich.console import Console
from rich.table import Table

from cloudcomputemanager.core.database import init_db, get_session
from cloudcomputemanager.core.models import Job, JobMetrics
from cloudcomputemanager.providers.vast import VastProvider

console = Console()


# ============================================================================
# Log Parsers - Add new formats here
# ============================================================================


def parse_namd_timing(line: str, metrics: JobMetrics) -> bool:
    """Parse NAMD timing line.

    Format: TIMING: 1000000 CPU: 1234.5, 0.00123/step Wall: 5678.9, 0.00567/step, 10.5 hours remaining
    """
    if "TIMING:" not in line:
        return False

    try:
        # Extract step number
        step_match = re.search(r"TIMING:\s+(\d+)", line)
        if step_match:
            metrics.current_step = int(step_match.group(1))

        # Extract Wall time per step (the second X/step value after "Wall:")
        wall_match = re.search(r"Wall:\s+[\d.]+,\s+([\d.]+)/step", line)
        if wall_match:
            sec_per_step = float(wall_match.group(1))
            if sec_per_step > 0:
                metrics.seconds_per_step = sec_per_step
                metrics.steps_per_second = 1.0 / sec_per_step

        # Extract hours remaining
        hours_match = re.search(r"([\d.]+)\s+hours?\s+remaining", line)
        if hours_match:
            metrics.estimated_hours_remaining = float(hours_match.group(1))

        return True
    except (ValueError, IndexError):
        return False


def parse_lammps_timing(line: str, metrics: JobMetrics) -> bool:
    """Parse LAMMPS timing line.

    Formats:
    - Step 1000000 CPU = 1234.5, 0.00123 sec/step
    - Loop time of 1234.5 on 8 procs for 1000000 steps
    - Performance: 12.34 ns/day, 1.94 hours/ns
    """
    try:
        # Step line
        if line.startswith("Step "):
            match = re.search(r"Step\s+(\d+)", line)
            if match:
                metrics.current_step = int(match.group(1))

            match = re.search(r"(\d+\.?\d*)\s*sec/step", line)
            if match:
                metrics.seconds_per_step = float(match.group(1))
                metrics.steps_per_second = 1.0 / metrics.seconds_per_step
            return True

        # Performance line
        if "Performance:" in line:
            match = re.search(r"(\d+\.?\d*)\s*ns/day", line)
            if match:
                ns_per_day = float(match.group(1))
                # Store as custom metric via notes
                metrics.notes = f"Performance: {ns_per_day:.2f} ns/day"
            return True

        # Loop time summary
        if "Loop time" in line:
            match = re.search(r"for\s+(\d+)\s+steps", line)
            if match:
                metrics.total_steps = int(match.group(1))
            return True

    except (ValueError, IndexError):
        pass
    return False


def parse_pytorch_training(line: str, metrics: JobMetrics) -> bool:
    """Parse PyTorch/ML training output.

    Common formats:
    - Epoch 10/100, Step 500/1000, Loss: 0.0234
    - [Epoch 10] train_loss: 0.0234, val_loss: 0.0345
    - Step 1000: loss=0.0234, lr=0.001
    - Progress: 50.0% (500/1000)
    """
    try:
        # Epoch pattern
        epoch_match = re.search(r"[Ee]poch\s*[:\s]?\s*(\d+)(?:\s*/\s*(\d+))?", line)
        if epoch_match:
            current = int(epoch_match.group(1))
            total = int(epoch_match.group(2)) if epoch_match.group(2) else None
            metrics.current_step = current
            if total:
                metrics.total_steps = total
                metrics.progress_percent = (current / total) * 100

        # Step pattern
        step_match = re.search(r"[Ss]tep\s*[:\s]?\s*(\d+)(?:\s*/\s*(\d+))?", line)
        if step_match:
            current = int(step_match.group(1))
            total = int(step_match.group(2)) if step_match.group(2) else None
            metrics.current_step = current
            if total:
                metrics.total_steps = total
                metrics.progress_percent = (current / total) * 100

        # Loss pattern
        loss_match = re.search(r"[Ll]oss[:\s=]+(\d+\.?\d*(?:e[+-]?\d+)?)", line)
        if loss_match:
            loss = float(loss_match.group(1))
            existing_notes = metrics.notes or ""
            metrics.notes = f"{existing_notes} loss={loss:.6f}".strip()

        # Progress percentage
        progress_match = re.search(r"[Pp]rogress[:\s]+(\d+\.?\d*)%", line)
        if progress_match:
            metrics.progress_percent = float(progress_match.group(1))

        # Iteration/sec or samples/sec
        speed_match = re.search(r"(\d+\.?\d*)\s*(?:it|samples|steps)/s(?:ec)?", line)
        if speed_match:
            metrics.steps_per_second = float(speed_match.group(1))
            if metrics.steps_per_second > 0:
                metrics.seconds_per_step = 1.0 / metrics.steps_per_second

        return bool(epoch_match or step_match or loss_match or progress_match or speed_match)

    except (ValueError, IndexError):
        pass
    return False


def parse_generic_progress(line: str, metrics: JobMetrics) -> bool:
    """Parse generic progress indicators.

    Handles common patterns:
    - Progress: 50% or 50.5%
    - [50/100] or (50/100)
    - ETA: 2h 30m or 2.5 hours
    - Speed: 100 items/sec
    - Completed 500 of 1000
    """
    try:
        # Percentage progress
        pct_match = re.search(r"(\d+\.?\d*)%", line)
        if pct_match and metrics.progress_percent is None:
            metrics.progress_percent = float(pct_match.group(1))

        # Fraction progress [50/100] or (50/100)
        frac_match = re.search(r"[\[(](\d+)\s*/\s*(\d+)[\])]", line)
        if frac_match:
            current = int(frac_match.group(1))
            total = int(frac_match.group(2))
            metrics.current_step = current
            metrics.total_steps = total
            if total > 0:
                metrics.progress_percent = (current / total) * 100

        # "Completed X of Y" pattern
        completed_match = re.search(r"[Cc]ompleted?\s+(\d+)\s+of\s+(\d+)", line)
        if completed_match:
            current = int(completed_match.group(1))
            total = int(completed_match.group(2))
            metrics.current_step = current
            metrics.total_steps = total
            if total > 0:
                metrics.progress_percent = (current / total) * 100

        # ETA patterns
        eta_match = re.search(r"ETA[:\s]+(\d+\.?\d*)\s*(h|hour|hr|m|min|minute|s|sec)", line, re.I)
        if eta_match:
            value = float(eta_match.group(1))
            unit = eta_match.group(2).lower()
            if unit.startswith('h'):
                metrics.estimated_hours_remaining = value
            elif unit.startswith('m'):
                metrics.estimated_hours_remaining = value / 60
            elif unit.startswith('s'):
                metrics.estimated_hours_remaining = value / 3600

        # Time remaining patterns: "10.5 hours remaining"
        remaining_match = re.search(r"(\d+\.?\d*)\s*(hours?|hrs?|minutes?|mins?)\s+remaining", line, re.I)
        if remaining_match:
            value = float(remaining_match.group(1))
            unit = remaining_match.group(2).lower()
            if unit.startswith('h'):
                metrics.estimated_hours_remaining = value
            elif unit.startswith('m'):
                metrics.estimated_hours_remaining = value / 60

        # Speed patterns: "100 items/sec", "1.5 steps/s"
        speed_match = re.search(r"(\d+\.?\d*)\s*\w+/s(?:ec)?", line)
        if speed_match and metrics.steps_per_second is None:
            metrics.steps_per_second = float(speed_match.group(1))

        return bool(pct_match or frac_match or completed_match or eta_match or remaining_match)

    except (ValueError, IndexError):
        pass
    return False


# List of parsers to try in order (most specific first)
LOG_PARSERS: list[Callable[[str, JobMetrics], bool]] = [
    parse_namd_timing,
    parse_lammps_timing,
    parse_pytorch_training,
    parse_generic_progress,
]


def parse_log_line(line: str, metrics: JobMetrics) -> bool:
    """Try all parsers on a log line, return True if any matched."""
    for parser in LOG_PARSERS:
        if parser(line, metrics):
            return True
    return False


# ============================================================================
# CLI Commands
# ============================================================================


async def show_metrics(job_id: str) -> None:
    """Show performance metrics for a job.

    Args:
        job_id: Job ID to show metrics for
    """
    await init_db()

    async with get_session() as session:
        from sqlmodel import select

        result = await session.execute(select(Job).where(Job.job_id == job_id))
        job = result.scalar_one_or_none()

        if not job:
            console.print(f"[red]Job not found: {job_id}[/red]")
            return

        metrics = job.get_metrics()

        console.print(f"\n[bold]Metrics for job: {job.name}[/bold]")
        console.print(f"Job ID: {job.job_id}")
        console.print(f"Status: {job.status}")
        console.print()

        table = Table(title="Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        # Timing metrics
        if metrics.steps_per_second is not None:
            table.add_row("Steps/second", f"{metrics.steps_per_second:.2f}")
        if metrics.seconds_per_step is not None:
            table.add_row("Seconds/step", f"{metrics.seconds_per_step:.6f}")

        # Progress metrics
        if metrics.current_step is not None:
            table.add_row("Current step", f"{metrics.current_step:,}")
        if metrics.total_steps is not None:
            table.add_row("Total steps", f"{metrics.total_steps:,}")
        if metrics.progress_percent is not None:
            table.add_row("Progress", f"{metrics.progress_percent:.1f}%")
        if metrics.estimated_hours_remaining is not None:
            table.add_row("Est. remaining", f"{metrics.estimated_hours_remaining:.1f} hours")

        # Output metrics
        if metrics.output_size_mb is not None:
            table.add_row("Output size", f"{metrics.output_size_mb:.1f} MB")

        # Resource metrics
        if metrics.gpu_utilization is not None:
            table.add_row("GPU utilization", f"{metrics.gpu_utilization:.1f}%")
        if metrics.cpu_utilization is not None:
            table.add_row("CPU utilization", f"{metrics.cpu_utilization:.1f}%")
        if metrics.memory_usage_gb is not None:
            table.add_row("Memory usage", f"{metrics.memory_usage_gb:.1f} GB")

        # Status
        if metrics.last_updated is not None:
            table.add_row("Last updated", metrics.last_updated.isoformat())
        table.add_row("Healthy", "[green]Yes[/green]" if metrics.is_healthy else "[red]No[/red]")

        if metrics.notes:
            table.add_row("Notes", metrics.notes)

        console.print(table)


async def update_metrics_from_log(
    job_id: str,
    instance_id: Optional[str] = None,
    log_path: str = "/workspace/job.log",
    output_pattern: str = "*",
    lines: int = 100,
) -> None:
    """Update job metrics by parsing log file on instance.

    Supports multiple log formats automatically:
    - NAMD (molecular dynamics)
    - LAMMPS (molecular dynamics)
    - PyTorch/ML training
    - Generic progress patterns

    Args:
        job_id: Job ID to update
        instance_id: Instance ID (if not stored in job)
        log_path: Path to log file on instance
        output_pattern: Glob pattern for output files to measure size
        lines: Number of log lines to parse (from end of file)
    """
    await init_db()

    async with get_session() as session:
        from sqlmodel import select

        result = await session.execute(select(Job).where(Job.job_id == job_id))
        job = result.scalar_one_or_none()

        if not job:
            console.print(f"[red]Job not found: {job_id}[/red]")
            return

        target_instance = instance_id or job.instance_id
        if not target_instance:
            console.print("[red]No instance ID available[/red]")
            return

        # Get instance SSH details
        provider = VastProvider()
        instance = await provider.get_instance(target_instance)

        if not instance:
            console.print(f"[red]Instance not found: {target_instance}[/red]")
            return

        console.print(f"Parsing log from {instance.ssh_host}:{instance.ssh_port}...")

        metrics = job.get_metrics()
        parsed_any = False

        # Get last N lines of log
        exit_code, stdout, stderr = await provider.execute_command(
            target_instance,
            f"tail -n {lines} {log_path} 2>/dev/null"
        )

        if exit_code == 0 and stdout.strip():
            for line in stdout.strip().split('\n'):
                if parse_log_line(line.strip(), metrics):
                    parsed_any = True

        if parsed_any:
            console.print("[green]Parsed log successfully[/green]")
        else:
            console.print("[yellow]No recognized patterns found in log[/yellow]")

        # Get output file size (try multiple patterns)
        for pattern in [output_pattern, "*.dcd", "*.pt", "*.bin", "*.h5", "*.out"]:
            exit_code, stdout, stderr = await provider.execute_command(
                target_instance,
                f"du -sb /workspace/{pattern} 2>/dev/null | awk '{{sum += $1}} END {{print sum/1024/1024}}'"
            )

            if exit_code == 0 and stdout.strip():
                try:
                    size = float(stdout.strip())
                    if size > 0:
                        metrics.output_size_mb = size
                        break
                except ValueError:
                    pass

        # Get GPU utilization if nvidia-smi available
        exit_code, stdout, stderr = await provider.execute_command(
            target_instance,
            "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1"
        )

        if exit_code == 0 and stdout.strip():
            try:
                metrics.gpu_utilization = float(stdout.strip())
            except ValueError:
                pass

        # Get memory usage
        exit_code, stdout, stderr = await provider.execute_command(
            target_instance,
            "free -g | awk '/^Mem:/ {print $3}'"
        )

        if exit_code == 0 and stdout.strip():
            try:
                metrics.memory_usage_gb = float(stdout.strip())
            except ValueError:
                pass

        # Check if main process is running
        exit_code, stdout, stderr = await provider.execute_command(
            target_instance,
            "pgrep -c -f 'python|namd|lammps|julia' 2>/dev/null || echo 0"
        )

        if exit_code == 0:
            try:
                proc_count = int(stdout.strip())
                metrics.is_healthy = proc_count > 0
            except ValueError:
                pass

        # Update timestamp
        metrics.last_updated = datetime.utcnow()

        # Save metrics
        job.set_metrics(metrics)
        session.add(job)
        await session.commit()

        console.print("[green]Metrics updated[/green]")
        await show_metrics(job_id)


async def collect_metrics_live(
    instance_id: str,
    log_path: str = "/workspace/job.log",
    lines: int = 50,
) -> JobMetrics:
    """Collect metrics from a running instance without database.

    Useful for quick checks without job tracking.

    Args:
        instance_id: Vast.ai instance ID
        log_path: Path to log file
        lines: Number of lines to parse

    Returns:
        JobMetrics with collected data
    """
    provider = VastProvider()
    instance = await provider.get_instance(instance_id)

    if not instance:
        raise ValueError(f"Instance not found: {instance_id}")

    metrics = JobMetrics()

    # Parse log
    exit_code, stdout, stderr = await provider.execute_command(
        instance_id,
        f"tail -n {lines} {log_path} 2>/dev/null"
    )

    if exit_code == 0 and stdout.strip():
        for line in stdout.strip().split('\n'):
            parse_log_line(line.strip(), metrics)

    # Get GPU utilization
    exit_code, stdout, stderr = await provider.execute_command(
        instance_id,
        "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1"
    )

    if exit_code == 0 and stdout.strip():
        try:
            metrics.gpu_utilization = float(stdout.strip())
        except ValueError:
            pass

    metrics.last_updated = datetime.utcnow()
    return metrics


async def list_metrics() -> None:
    """List performance metrics for all running jobs."""
    await init_db()

    async with get_session() as session:
        from sqlmodel import select
        from cloudcomputemanager.core.models import JobStatus

        result = await session.execute(
            select(Job).where(Job.status == JobStatus.RUNNING)
        )
        jobs = result.scalars().all()

        if not jobs:
            console.print("[yellow]No running jobs found[/yellow]")
            return

        table = Table(title="Running Job Metrics")
        table.add_column("Job ID", style="cyan")
        table.add_column("Name")
        table.add_column("Progress", justify="right")
        table.add_column("Speed", justify="right")
        table.add_column("ETA", justify="right")
        table.add_column("Output", justify="right")
        table.add_column("GPU%", justify="right")
        table.add_column("Health")

        for job in jobs:
            metrics = job.get_metrics()
            progress = f"{metrics.progress_percent:.1f}%" if metrics.progress_percent else "-"
            speed = f"{metrics.steps_per_second:.1f}/s" if metrics.steps_per_second else "-"
            eta = f"{metrics.estimated_hours_remaining:.1f}h" if metrics.estimated_hours_remaining else "-"
            output = f"{metrics.output_size_mb:.0f}MB" if metrics.output_size_mb else "-"
            gpu = f"{metrics.gpu_utilization:.0f}%" if metrics.gpu_utilization else "-"
            health = "[green]OK[/green]" if metrics.is_healthy else "[red]WARN[/red]"

            table.add_row(
                job.job_id[:12],
                job.name[:20],
                progress,
                speed,
                eta,
                output,
                gpu,
                health,
            )

        console.print(table)
