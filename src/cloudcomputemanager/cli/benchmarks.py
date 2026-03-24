"""Benchmark CLI commands."""

import asyncio
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

console = Console()


async def run_benchmark(config_path: Path, gpu_filter: Optional[str] = None) -> None:
    """Run a benchmark suite from YAML config."""
    from cloudcomputemanager.benchmarks.engine import BenchmarkEngine

    engine = BenchmarkEngine()
    config = engine.load_config(config_path)

    console.print(f"\n[bold]Running benchmark:[/bold] {config['name']}")
    if config.get("description"):
        console.print(f"  {config['description']}")

    matrix = config.get("matrix", {})
    gpu_types = matrix.get("gpu_type", ["RTX_3060"])
    if isinstance(gpu_types, str):
        gpu_types = [gpu_types]
    if gpu_filter:
        gpu_types = [g for g in gpu_types if g == gpu_filter]

    reps = config.get("repetitions", 1)
    console.print(f"  GPU types: {', '.join(gpu_types)}")
    console.print(f"  Repetitions: {reps}")
    console.print(f"  Total runs: {len(gpu_types) * reps}\n")

    suite = await engine.run_suite(config, gpu_filter=gpu_filter)

    console.print(f"\n[bold green]Benchmark complete![/bold green]")
    console.print(f"  Suite ID: {suite.suite_id}")
    console.print(f"  Completed: {suite.completed_runs}/{suite.total_runs}")
    console.print(f"  Total cost: ${suite.total_cost_usd:.4f}")

    # Show results
    results = await engine.get_results(suite.suite_id)
    if results:
        console.print()
        _display_results_table(results)


async def show_results(
    suite_id: Optional[str] = None,
    sort_by: str = "cost",
) -> None:
    """Show benchmark results."""
    from cloudcomputemanager.benchmarks.engine import BenchmarkEngine

    results = await BenchmarkEngine.get_results(suite_id)
    if not results:
        console.print("[yellow]No benchmark results found.[/yellow]")
        return

    if sort_by == "performance":
        results.sort(key=lambda x: x["avg_value"], reverse=True)

    _display_results_table(results)


def _display_results_table(results: list[dict]) -> None:
    """Display benchmark results as a Rich table."""
    table = Table(title="Benchmark Results (sorted by cost efficiency)")
    table.add_column("GPU", style="cyan")
    table.add_column("Metric", style="white")
    table.add_column("Avg Value", style="green", justify="right")
    table.add_column("Min-Max", style="dim", justify="right")
    table.add_column("$/hr", style="yellow", justify="right")
    table.add_column("$/M units", style="bold", justify="right")
    table.add_column("Runs", style="dim", justify="right")

    for r in results:
        cost_str = f"${r['avg_cost_per_million']:.4f}" if r.get("avg_cost_per_million") else "N/A"
        unit = f" {r['metric_unit']}" if r.get("metric_unit") else ""
        table.add_row(
            r["gpu_type"],
            r["metric_name"],
            f"{r['avg_value']:.1f}{unit}",
            f"{r['min_value']:.1f}-{r['max_value']:.1f}",
            f"${r['hourly_rate']:.4f}",
            cost_str,
            str(r["runs"]),
        )

    console.print(table)

    # Highlight best value
    if results and results[0].get("avg_cost_per_million"):
        best = results[0]
        console.print(
            f"\n[bold green]BEST VALUE:[/bold green] {best['gpu_type']} "
            f"— {best['avg_value']:.1f} {best.get('metric_unit', '')} "
            f"@ ${best['hourly_rate']:.4f}/hr "
            f"= ${best['avg_cost_per_million']:.4f}/M units"
        )
