"""Benchmark framework for cost-performance analysis across GPU tiers.

Provides workload-agnostic benchmarking:
- Define benchmarks via YAML (any workload, any metric)
- Run across multiple GPU types automatically
- Collect and compare results
- Persist historical data for future reference
"""

from cloudcomputemanager.benchmarks.engine import BenchmarkEngine
from cloudcomputemanager.benchmarks.models import BenchmarkSuite, BenchmarkRun, BenchmarkResult

__all__ = ["BenchmarkEngine", "BenchmarkSuite", "BenchmarkRun", "BenchmarkResult"]
