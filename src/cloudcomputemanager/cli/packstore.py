"""PackStore CLI commands."""

from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from cloudcomputemanager.packstore import (
    PackageRegistry,
    PackageDeployer,
    DeploymentStrategy,
)
from cloudcomputemanager.packstore.registry import PackageCategory
from cloudcomputemanager.providers.vast import VastProvider

console = Console()


async def list_packages(
    category: Optional[str] = None,
    show_variants: bool = False,
) -> None:
    """List available packages."""
    registry = PackageRegistry()

    cat_filter = None
    if category:
        try:
            cat_filter = PackageCategory(category)
        except ValueError:
            console.print(f"[red]Unknown category: {category}[/red]")
            console.print(f"[dim]Valid: {[c.value for c in PackageCategory]}[/dim]")
            return

    packages = registry.list_packages(category=cat_filter)

    if not packages:
        console.print("[yellow]No packages found.[/yellow]")
        return

    table = Table(title="Available Packages")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="dim")
    table.add_column("Variants", justify="right")
    table.add_column("Description", style="white")

    for pkg in sorted(packages, key=lambda p: p.name):
        table.add_row(
            pkg.display_name,
            pkg.category.value,
            str(len(pkg.variants)),
            pkg.description[:50] + "..." if len(pkg.description) > 50 else pkg.description,
        )

        if show_variants:
            for variant in pkg.variants:
                table.add_row(
                    f"  └─ {variant.id}",
                    "",
                    "",
                    variant.description[:40],
                )

    console.print(table)


async def show_package_info(package_name: str) -> None:
    """Show detailed package information."""
    registry = PackageRegistry()

    # Try exact match first
    package = registry.get(package_name)

    # Try search
    if not package:
        results = registry.search(package_name)
        if len(results) == 1:
            package = results[0]
        elif len(results) > 1:
            console.print(f"[yellow]Multiple matches found:[/yellow]")
            for p in results:
                console.print(f"  - {p.name}")
            return

    if not package:
        console.print(f"[red]Package not found: {package_name}[/red]")
        return

    # Build info panel
    info = f"""
[bold]Name:[/bold] {package.display_name}
[bold]Category:[/bold] {package.category.value}
[bold]Homepage:[/bold] {package.homepage or 'N/A'}
[bold]License:[/bold] {package.license or 'N/A'}

[bold]Description:[/bold]
{package.description}
"""
    console.print(Panel(info, title=f"Package: {package.name}"))

    # Show variants
    console.print("\n[bold]Variants:[/bold]")

    for variant in package.variants:
        variant_info = f"""
[cyan]{variant.id}[/cyan] (v{variant.version})
  {variant.description}

  [bold]Compatibility:[/bold]
    CUDA: {', '.join(variant.compatibility.cuda_versions) or 'Any'}
    GPU Arch: {', '.join(variant.compatibility.gpu_architectures) or 'Any'}
    Min Driver: {variant.compatibility.min_driver}

  [bold]Sources:[/bold]
"""
        for source in variant.sources:
            if source.image:
                variant_info += f"    - {source.type.value}: {source.full_image}\n"
            elif source.spec:
                variant_info += f"    - {source.type.value}: {source.spec}\n"

        if variant.packages_included:
            variant_info += f"\n  [bold]Packages:[/bold] {', '.join(variant.packages_included)}"

        console.print(variant_info)


async def search_packages(
    query: str,
    cuda_version: Optional[str] = None,
    gpu_arch: Optional[str] = None,
) -> None:
    """Search for packages."""
    registry = PackageRegistry()

    results = registry.search(
        query=query,
        cuda_version=cuda_version,
        gpu_arch=gpu_arch,
    )

    if not results:
        console.print(f"[yellow]No packages found matching '{query}'[/yellow]")
        return

    table = Table(title=f"Search Results for '{query}'")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="dim")
    table.add_column("Best Variant", style="white")
    table.add_column("GPU Arch", style="dim")

    for pkg in results:
        best_variant = pkg.variants[0] if pkg.variants else None
        if best_variant:
            archs = ", ".join(best_variant.compatibility.gpu_architectures[:3])
            if len(best_variant.compatibility.gpu_architectures) > 3:
                archs += "..."
        else:
            archs = "N/A"

        table.add_row(
            pkg.name,
            pkg.category.value,
            best_variant.id if best_variant else "N/A",
            archs,
        )

    console.print(table)


async def deploy_packages(
    instance_id: str,
    packages: list[str],
    strategy: str = "auto",
    verify: bool = True,
) -> None:
    """Deploy packages to an instance."""
    provider = VastProvider()
    deployer = PackageDeployer(provider)

    try:
        deploy_strategy = DeploymentStrategy(strategy)
    except ValueError:
        console.print(f"[red]Unknown strategy: {strategy}[/red]")
        console.print(f"[dim]Valid: {[s.value for s in DeploymentStrategy]}[/dim]")
        return

    console.print(f"\n[bold]Deploying to instance {instance_id}[/bold]")
    console.print(f"  Packages: {', '.join(packages)}")
    console.print(f"  Strategy: {strategy}")
    console.print()

    with console.status("Deploying packages..."):
        result = await deployer.deploy(
            instance_id=instance_id,
            packages=packages,
            strategy=deploy_strategy,
            verify=verify,
        )

    # Show environment info
    console.print(f"[bold]Instance Environment:[/bold]")
    console.print(f"  GPU: {result.environment.gpu_name} x{result.environment.gpu_count}")
    console.print(f"  Architecture: {result.environment.gpu_arch}")
    console.print(f"  CUDA: {result.environment.cuda_version}")
    console.print(f"  Driver: {result.environment.driver_version}")
    console.print()

    # Show deployment results
    table = Table(title="Deployment Results")
    table.add_column("Package", style="cyan")
    table.add_column("Variant", style="white")
    table.add_column("Status", style="bold")
    table.add_column("Strategy", style="dim")
    table.add_column("Verified", style="dim")
    table.add_column("Time", justify="right")

    for dep in result.deployments:
        status_color = "green" if dep.status.value == "completed" else "red"
        table.add_row(
            dep.package_name,
            dep.variant_id,
            f"[{status_color}]{dep.status.value}[/{status_color}]",
            dep.strategy_used.value,
            "Yes" if dep.verified else "No",
            f"{dep.duration_seconds}s",
        )

        if dep.error_message:
            console.print(f"  [red]Error: {dep.error_message}[/red]")

    console.print(table)

    # Show combined environment
    if result.all_environment:
        console.print("\n[bold]Environment Variables:[/bold]")
        for key, value in result.all_environment.items():
            console.print(f"  export {key}={value}")

    console.print(f"\n[dim]Total time: {result.total_duration_seconds}s[/dim]")


async def verify_package(
    instance_id: str,
    package_name: str,
) -> None:
    """Verify a package installation on an instance."""
    registry = PackageRegistry()
    provider = VastProvider()

    package = registry.get(package_name)
    if not package:
        console.print(f"[red]Package not found: {package_name}[/red]")
        return

    console.print(f"\n[bold]Verifying {package.display_name} on {instance_id}[/bold]")

    for variant in package.variants:
        if not variant.test_command:
            continue

        console.print(f"\n  Testing variant: {variant.id}")

        # Set up environment
        env_setup = ""
        for key, value in variant.environment.items():
            env_setup += f"export {key}={value}; "

        cmd = f"{env_setup}{variant.test_command}"
        exit_code, stdout, stderr = await provider.execute_command(instance_id, cmd)

        if exit_code == 0:
            if variant.expected_output_contains:
                if variant.expected_output_contains in stdout:
                    console.print(f"    [green]PASS[/green] - Output contains expected text")
                else:
                    console.print(f"    [yellow]WARN[/yellow] - Output missing expected text")
                    console.print(f"    [dim]Expected: {variant.expected_output_contains}[/dim]")
            else:
                console.print(f"    [green]PASS[/green] - Command succeeded")
        else:
            console.print(f"    [red]FAIL[/red] - Exit code {exit_code}")
            if stderr:
                console.print(f"    [dim]{stderr[:100]}[/dim]")
