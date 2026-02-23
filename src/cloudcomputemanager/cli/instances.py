"""Instance management CLI commands."""

import os
import subprocess
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich import print as rprint

from cloudcomputemanager.core.config import get_settings
from cloudcomputemanager.providers.vast import VastProvider

console = Console()


async def list_instances() -> None:
    """List all managed instances."""
    provider = VastProvider()

    with console.status("Fetching instances..."):
        instances = await provider.list_instances()

    if not instances:
        console.print("[yellow]No instances found.[/yellow]")
        return

    table = Table(title="Instances")
    table.add_column("ID", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("GPU", style="white")
    table.add_column("$/hr", justify="right")
    table.add_column("SSH", style="dim")

    status_colors = {
        "running": "green",
        "starting": "yellow",
        "stopped": "red",
        "error": "red",
    }

    for inst in instances:
        color = status_colors.get(inst.status.value, "white")
        table.add_row(
            inst.instance_id,
            f"[{color}]{inst.status.value}[/{color}]",
            f"{inst.gpu_type} x{inst.gpu_count}",
            f"${inst.hourly_rate:.3f}",
            f"ssh -p {inst.ssh_port} {inst.ssh_user}@{inst.ssh_host}",
        )

    console.print(table)


async def search_offers(
    gpu_type: Optional[str],
    gpu_count: int,
    max_price: Optional[float],
    limit: int,
) -> None:
    """Search available GPU offers."""
    provider = VastProvider()

    with console.status("Searching offers..."):
        offers = await provider.search_offers(
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            max_hourly_rate=max_price,
        )

    if not offers:
        console.print("[yellow]No offers found matching criteria.[/yellow]")
        return

    offers = offers[:limit]

    table = Table(title=f"GPU Offers ({len(offers)} shown)")
    table.add_column("ID", style="cyan")
    table.add_column("GPU", style="white")
    table.add_column("VRAM", justify="right")
    table.add_column("CPU", justify="right")
    table.add_column("RAM", justify="right")
    table.add_column("Disk", justify="right")
    table.add_column("$/hr", justify="right", style="green")
    table.add_column("Location", style="dim")

    for offer in offers:
        table.add_row(
            offer.offer_id,
            f"{offer.gpu_type} x{offer.gpu_count}",
            f"{offer.gpu_memory_gb}GB",
            str(offer.cpu_cores),
            f"{offer.memory_gb}GB",
            f"{offer.disk_gb}GB",
            f"${offer.hourly_rate:.3f}",
            offer.location[:20],
        )

    console.print(table)

    if offers:
        console.print(
            f"\n[dim]To create an instance: vm instances create {offers[0].offer_id}[/dim]"
        )


async def create_instance(offer_id: str, image: str, disk: int) -> None:
    """Create a new instance."""
    provider = VastProvider()

    console.print(f"\n[bold]Creating instance from offer {offer_id}[/bold]")
    console.print(f"  Image: {image}")
    console.print(f"  Disk: {disk}GB")

    with console.status("Creating instance..."):
        instance = await provider.create_instance(
            offer_id=offer_id,
            image=image,
            disk_gb=disk,
        )

    console.print(f"\n[green]Instance created![/green]")
    console.print(f"  ID: {instance.instance_id}")
    console.print(f"  GPU: {instance.gpu_type} x{instance.gpu_count}")

    with console.status("Waiting for instance to be ready..."):
        ready = await provider.wait_for_ready(instance.instance_id, timeout=300)

    if ready:
        console.print(f"\n[green]Instance is ready![/green]")
        console.print(
            f"  SSH: ssh -p {instance.ssh_port} {instance.ssh_user}@{instance.ssh_host}"
        )
    else:
        console.print("[yellow]Instance is starting but not yet ready.[/yellow]")
        console.print("  Check status with: vm instances list")


async def ssh_to_instance(instance_id: str) -> None:
    """SSH into an instance."""
    provider = VastProvider()

    instance = await provider.get_instance(instance_id)
    if not instance:
        console.print(f"[red]Instance not found: {instance_id}[/red]")
        return

    if instance.status.value != "running":
        console.print(f"[red]Instance is not running: {instance.status.value}[/red]")
        return

    settings = get_settings()

    # Build SSH command
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-p", str(instance.ssh_port),
    ]

    if settings.ssh_key_path.exists():
        ssh_cmd.extend(["-i", str(settings.ssh_key_path)])

    ssh_cmd.append(f"{instance.ssh_user}@{instance.ssh_host}")

    console.print(f"[dim]Connecting to {instance.ssh_host}:{instance.ssh_port}...[/dim]\n")

    # Execute SSH interactively
    os.execvp("ssh", ssh_cmd)


async def terminate_instance(instance_id: str, force: bool) -> None:
    """Terminate an instance."""
    provider = VastProvider()

    instance = await provider.get_instance(instance_id)
    if not instance:
        console.print(f"[red]Instance not found: {instance_id}[/red]")
        return

    if not force:
        from rich.prompt import Confirm

        if not Confirm.ask(f"Terminate instance {instance_id}?"):
            console.print("[dim]Cancelled.[/dim]")
            return

    with console.status("Terminating instance..."):
        success = await provider.terminate_instance(instance_id)

    if success:
        console.print(f"[green]Instance terminated: {instance_id}[/green]")
    else:
        console.print(f"[red]Failed to terminate instance: {instance_id}[/red]")
