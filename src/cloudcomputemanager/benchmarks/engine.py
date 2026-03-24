"""Benchmark engine: orchestrates benchmark runs across GPU tiers.

Workflow:
1. Parse benchmark YAML config
2. For each GPU type in matrix:
   a. Search for offer
   b. Create instance
   c. Wait for ready
   d. Run setup commands
   e. Upload benchmark files
   f. Run benchmark command (with timeout)
   g. Extract metrics via regex
   h. Record results
   i. Destroy instance
3. Aggregate and compare results
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog
import yaml

from cloudcomputemanager.core.database import init_db, get_session
from cloudcomputemanager.providers.vast import VastProvider
from cloudcomputemanager.benchmarks.models import BenchmarkSuite, BenchmarkRun, BenchmarkResult

logger = structlog.get_logger(__name__)


class BenchmarkEngine:
    """Orchestrates benchmark runs across GPU tiers."""

    def __init__(self, provider: Optional[VastProvider] = None):
        self.provider = provider or VastProvider()

    @staticmethod
    def load_config(config_path: Path) -> dict:
        """Load and validate a benchmark YAML config."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        required = ["name", "workload"]
        for key in required:
            if key not in config:
                raise ValueError(f"Benchmark config missing required key: {key}")

        workload = config["workload"]
        if "command" not in workload:
            raise ValueError("workload.command is required")

        return config

    async def run_suite(
        self,
        config: dict,
        gpu_filter: Optional[str] = None,
    ) -> BenchmarkSuite:
        """Run a full benchmark suite.

        Args:
            config: Parsed benchmark config dict
            gpu_filter: Optional GPU type to limit testing to

        Returns:
            BenchmarkSuite with results
        """
        await init_db()

        # Create suite record
        suite = BenchmarkSuite(
            name=config["name"],
            description=config.get("description"),
            config_json=json.dumps(config),
            status="running",
            started_at=datetime.utcnow(),
        )

        async with get_session() as session:
            session.add(suite)
            await session.commit()
            await session.refresh(suite)

        workload = config["workload"]
        matrix = config.get("matrix", {})
        gpu_types = matrix.get("gpu_type", ["RTX_3060"])
        if isinstance(gpu_types, str):
            gpu_types = [gpu_types]

        if gpu_filter:
            gpu_types = [g for g in gpu_types if g == gpu_filter]

        repetitions = config.get("repetitions", 1)
        timeout = config.get("timeout", 300)
        metrics_config = config.get("metrics", [])
        budget_per_instance = config.get("budget", {}).get("max_per_instance", 1.0)

        total_runs = len(gpu_types) * repetitions
        suite.total_runs = total_runs

        async with get_session() as session:
            from sqlmodel import select
            stmt = select(BenchmarkSuite).where(BenchmarkSuite.suite_id == suite.suite_id)
            result = await session.execute(stmt)
            db_suite = result.scalar_one_or_none()
            if db_suite:
                db_suite.total_runs = total_runs
                session.add(db_suite)
                await session.commit()

        # Run benchmarks for each GPU type
        for gpu_type in gpu_types:
            for rep in range(1, repetitions + 1):
                try:
                    run_result = await self._run_single_benchmark(
                        suite=suite,
                        gpu_type=gpu_type,
                        repetition=rep,
                        workload=workload,
                        timeout=timeout,
                        metrics_config=metrics_config,
                        budget_limit=budget_per_instance,
                    )
                    if run_result is None:
                        # Run failed internally (no offers, bad exit code, etc.)
                        async with get_session() as session:
                            from sqlmodel import select
                            stmt = select(BenchmarkSuite).where(BenchmarkSuite.suite_id == suite.suite_id)
                            result = await session.execute(stmt)
                            db_suite = result.scalar_one_or_none()
                            if db_suite:
                                db_suite.failed_runs += 1
                                session.add(db_suite)
                                await session.commit()
                except Exception as e:
                    logger.error(
                        "Benchmark run failed",
                        gpu_type=gpu_type,
                        rep=rep,
                        error=str(e),
                    )

        # Finalize suite
        async with get_session() as session:
            from sqlmodel import select
            stmt = select(BenchmarkSuite).where(BenchmarkSuite.suite_id == suite.suite_id)
            result = await session.execute(stmt)
            db_suite = result.scalar_one_or_none()
            if db_suite:
                if db_suite.completed_runs == 0:
                    db_suite.status = "failed"
                else:
                    db_suite.status = "completed"
                db_suite.completed_at = datetime.utcnow()
                session.add(db_suite)
                await session.commit()
                await session.refresh(db_suite)
                return db_suite

        return suite

    async def _run_single_benchmark(
        self,
        suite: BenchmarkSuite,
        gpu_type: str,
        repetition: int,
        workload: dict,
        timeout: int,
        metrics_config: list[dict],
        budget_limit: float,
    ) -> Optional[BenchmarkRun]:
        """Run a single benchmark on a single GPU type."""
        logger.info("Starting benchmark run", gpu_type=gpu_type, rep=repetition)

        # Create run record
        run = BenchmarkRun(
            suite_id=suite.suite_id,
            gpu_type=gpu_type,
            repetition=repetition,
            status="running",
            started_at=datetime.utcnow(),
        )

        async with get_session() as session:
            session.add(run)
            await session.commit()
            await session.refresh(run)

        instance_id = None
        try:
            # Search for offer
            offers = await self.provider.search_offers(
                gpu_type=gpu_type,
                max_hourly_rate=budget_limit,
            )

            if not offers:
                logger.warning("No offers found for GPU type", gpu_type=gpu_type)
                await self._update_run_status(run, "failed", error="No offers found")
                return None

            best = offers[0]
            run.hourly_rate = best.hourly_rate

            # Create instance
            image = workload.get("image", "ubuntu:22.04")
            setup = workload.get("setup")

            instance = await self.provider.create_instance(
                offer_id=best.offer_id,
                image=image,
                disk_gb=30,
                startup_script=setup,
            )
            instance_id = instance.instance_id
            run.instance_id = instance_id
            run.cpu_cores = best.cpu_cores

            # Wait for ready
            ready = await self.provider.wait_for_ready(instance_id, timeout=300)
            if not ready:
                raise RuntimeError("Instance failed to start")

            # If there's a setup script, wait for it to complete
            if setup:
                await asyncio.sleep(10)  # Give setup time
                # Check for setup sentinel
                for _ in range(30):
                    rc, out, _ = await self.provider.execute_command(
                        instance_id,
                        "test -f /tmp/.ccm_setup_done && echo 'ready' || echo 'waiting'",
                        timeout=10,
                    )
                    if "ready" in out:
                        break
                    await asyncio.sleep(5)

            # Upload benchmark files
            files = workload.get("files", [])
            if files:
                for file_path in files:
                    p = Path(file_path)
                    if p.exists():
                        await self.provider.rsync_upload(
                            instance_id,
                            str(p),
                            f"/workspace/{p.name}",
                        )

            # Run benchmark command
            command = workload["command"]
            # Substitute instance-specific variables
            command = command.replace("${NCPUS}", str(best.cpu_cores))

            logger.info("Running benchmark command", command=command[:80])
            exit_code, stdout, stderr = await self.provider.execute_command(
                instance_id,
                f"cd /workspace && {command}",
                timeout=timeout,
            )

            run.exit_code = exit_code
            run.command_output = stdout[:10000]  # Limit stored output

            if exit_code != 0:
                await self._update_run_status(run, "failed", error=f"Exit code {exit_code}: {stderr[:200]}")
                return None

            # Extract metrics
            for metric_def in metrics_config:
                metric_name = metric_def["name"]
                source = metric_def.get("source", "stdout")
                regex = metric_def.get("regex", r"([\d.]+)")
                unit = metric_def.get("unit")

                text = stdout if source == "stdout" else stderr
                match = re.search(regex, text)

                if match:
                    value = float(match.group(1))

                    # Calculate cost efficiency
                    cost_per_unit = None
                    if value > 0 and run.hourly_rate > 0:
                        # Cost per million units
                        cost_per_unit = (run.hourly_rate / value) * 1_000_000

                    result = BenchmarkResult(
                        run_id=run.run_id,
                        suite_id=suite.suite_id,
                        metric_name=metric_name,
                        metric_value=value,
                        metric_unit=unit,
                        cost_per_unit=cost_per_unit,
                        gpu_type=gpu_type,
                        hourly_rate=run.hourly_rate,
                    )

                    async with get_session() as session:
                        session.add(result)
                        await session.commit()

                    logger.info(
                        "Metric extracted",
                        metric=metric_name,
                        value=value,
                        unit=unit,
                        cost_per_unit=cost_per_unit,
                    )

            # Mark run as completed
            run.completed_at = datetime.utcnow()
            run.duration_seconds = int(
                (run.completed_at - run.started_at).total_seconds()
            )
            run.cost_usd = run.hourly_rate * run.duration_seconds / 3600
            await self._update_run_status(run, "completed")

            # Update suite totals
            async with get_session() as session:
                from sqlmodel import select
                stmt = select(BenchmarkSuite).where(BenchmarkSuite.suite_id == suite.suite_id)
                result = await session.execute(stmt)
                db_suite = result.scalar_one_or_none()
                if db_suite:
                    db_suite.completed_runs += 1
                    db_suite.total_cost_usd += run.cost_usd
                    session.add(db_suite)
                    await session.commit()

            return run

        except Exception as e:
            logger.error("Benchmark run error", error=str(e))
            await self._update_run_status(run, "failed", error=str(e))
            return None

        finally:
            # Always destroy the instance
            if instance_id:
                try:
                    await self.provider.terminate_instance(instance_id)
                    logger.info("Benchmark instance terminated", instance_id=instance_id)
                except Exception as e:
                    logger.error("Failed to terminate benchmark instance", instance_id=instance_id, error=str(e))

    async def _update_run_status(
        self, run: BenchmarkRun, status: str, error: Optional[str] = None
    ) -> None:
        """Update a benchmark run's status in the database."""
        async with get_session() as session:
            from sqlmodel import select
            stmt = select(BenchmarkRun).where(BenchmarkRun.run_id == run.run_id)
            result = await session.execute(stmt)
            db_run = result.scalar_one_or_none()
            if db_run:
                db_run.status = status
                if error:
                    db_run.error_message = error
                if status in ("completed", "failed"):
                    db_run.completed_at = datetime.utcnow()
                session.add(db_run)
                await session.commit()

    @staticmethod
    async def get_results(suite_id: Optional[str] = None) -> list[dict]:
        """Get benchmark results, optionally filtered by suite.

        Returns aggregated results grouped by GPU type with averages across repetitions.
        """
        await init_db()

        from sqlmodel import select

        async with get_session() as session:
            stmt = select(BenchmarkResult)
            if suite_id:
                stmt = stmt.where(BenchmarkResult.suite_id == suite_id)
            stmt = stmt.order_by(BenchmarkResult.gpu_type)

            result = await session.execute(stmt)
            results = result.scalars().all()

        if not results:
            return []

        # Group by (gpu_type, metric_name) and average
        grouped: dict[tuple[str, str], list[BenchmarkResult]] = {}
        for r in results:
            key = (r.gpu_type, r.metric_name)
            grouped.setdefault(key, []).append(r)

        summary = []
        for (gpu_type, metric_name), entries in grouped.items():
            values = [e.metric_value for e in entries]
            costs = [e.cost_per_unit for e in entries if e.cost_per_unit is not None]

            avg_value = sum(values) / len(values)
            avg_cost = sum(costs) / len(costs) if costs else None

            summary.append({
                "gpu_type": gpu_type,
                "metric_name": metric_name,
                "metric_unit": entries[0].metric_unit,
                "avg_value": round(avg_value, 2),
                "min_value": round(min(values), 2),
                "max_value": round(max(values), 2),
                "runs": len(entries),
                "hourly_rate": entries[0].hourly_rate,
                "avg_cost_per_million": round(avg_cost, 4) if avg_cost else None,
            })

        # Sort by cost efficiency (best first)
        summary.sort(key=lambda x: x.get("avg_cost_per_million") or float("inf"))
        return summary
