"""Sync management CLI commands."""

from typing import Optional

from rich.console import Console
from rich.table import Table

from cloudcomputemanager.core.config import get_settings
from cloudcomputemanager.sync.monitor import SyncMonitor, format_size

console = Console()


async def show_sync_status(job_id: Optional[str]) -> None:
    """Show sync status for jobs."""
    settings = get_settings()
    sync_dir = settings.sync_local_path

    if not sync_dir.exists():
        console.print("[yellow]No sync directory found.[/yellow]")
        return

    monitor = SyncMonitor()

    if job_id:
        # Show specific job
        stats = monitor.get_sync_stats(job_id)
        if not stats["exists"]:
            console.print(f"[yellow]No synced data for job: {job_id}[/yellow]")
            return

        console.print(f"\n[bold]Sync Status: {job_id}[/bold]")
        console.print(f"  Files: {stats['file_count']}")
        console.print(f"  Size: {format_size(stats['total_size_bytes'])}")
        console.print(f"  Latest: {stats['latest_file']}")
        console.print(
            f"  Modified: {stats['latest_modified'].strftime('%Y-%m-%d %H:%M:%S') if stats['latest_modified'] else 'N/A'}"
        )
        console.print(f"  Path: {settings.sync_local_path / job_id}")

    else:
        # Show all jobs
        job_dirs = [d for d in sync_dir.iterdir() if d.is_dir()]

        if not job_dirs:
            console.print("[yellow]No synced jobs found.[/yellow]")
            return

        table = Table(title="Synced Jobs")
        table.add_column("Job ID", style="cyan")
        table.add_column("Files", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("Latest File", style="dim")
        table.add_column("Modified", style="dim")

        for job_dir in sorted(job_dirs, key=lambda d: d.stat().st_mtime, reverse=True):
            stats = monitor.get_sync_stats(job_dir.name)
            table.add_row(
                job_dir.name,
                str(stats["file_count"]),
                format_size(stats["total_size_bytes"]),
                (stats["latest_file"] or "-")[:30],
                stats["latest_modified"].strftime("%m-%d %H:%M") if stats["latest_modified"] else "-",
            )

        console.print(table)


async def list_synced_files(job_id: str) -> None:
    """List synced files for a job."""
    settings = get_settings()
    job_dir = settings.sync_local_path / job_id

    if not job_dir.exists():
        console.print(f"[yellow]No synced data for job: {job_id}[/yellow]")
        return

    files = sorted(
        [f for f in job_dir.rglob("*") if f.is_file()],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    if not files:
        console.print("[yellow]No files synced yet.[/yellow]")
        return

    table = Table(title=f"Synced Files: {job_id}")
    table.add_column("File", style="white")
    table.add_column("Size", justify="right")
    table.add_column("Modified", style="dim")

    for f in files[:50]:  # Limit to 50 files
        rel_path = f.relative_to(job_dir)
        table.add_row(
            str(rel_path),
            format_size(f.stat().st_size),
            f.stat().st_mtime.__str__()[:19],
        )

    console.print(table)

    if len(files) > 50:
        console.print(f"[dim]... and {len(files) - 50} more files[/dim]")

    console.print(f"\n[dim]Path: {job_dir}[/dim]")
