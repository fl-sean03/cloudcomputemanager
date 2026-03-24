"""Data models for the benchmark framework."""

import json as _json
from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field

from cloudcomputemanager.core.models import generate_id


class BenchmarkSuite(SQLModel, table=True):
    """A benchmark suite definition and its execution metadata."""

    __tablename__ = "benchmark_suites"

    id: int | None = Field(default=None, primary_key=True)
    suite_id: str = Field(
        default_factory=lambda: f"bench_{generate_id()}", unique=True, index=True
    )
    name: str = Field(index=True, description="Benchmark name")
    description: Optional[str] = Field(default=None)

    # Configuration (stored as JSON for flexibility)
    config_json: str = Field(default="{}", description="Full benchmark YAML config as JSON")

    # Execution metadata
    status: str = Field(default="pending", description="pending, running, completed, failed")
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    total_runs: int = Field(default=0)
    completed_runs: int = Field(default=0)
    failed_runs: int = Field(default=0)
    total_cost_usd: float = Field(default=0.0)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    def get_config(self) -> dict:
        return _json.loads(self.config_json)


class BenchmarkRun(SQLModel, table=True):
    """A single benchmark run (one GPU type, one repetition)."""

    __tablename__ = "benchmark_runs"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(
        default_factory=lambda: f"brun_{generate_id()}", unique=True, index=True
    )
    suite_id: str = Field(index=True, description="Parent benchmark suite")

    # Instance details
    gpu_type: str = Field(description="GPU model tested")
    gpu_count: int = Field(default=1)
    cpu_cores: int = Field(default=0)
    instance_id: Optional[str] = Field(default=None)
    hourly_rate: float = Field(default=0.0)

    # Execution
    status: str = Field(default="pending", description="pending, running, completed, failed")
    repetition: int = Field(default=1, description="Which repetition this is")
    command_output: Optional[str] = Field(default=None, description="stdout from benchmark command")
    exit_code: Optional[int] = Field(default=None)
    error_message: Optional[str] = Field(default=None)

    # Timing
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    duration_seconds: int = Field(default=0)
    cost_usd: float = Field(default=0.0)

    created_at: datetime = Field(default_factory=datetime.utcnow)


class BenchmarkResult(SQLModel, table=True):
    """An extracted metric from a benchmark run."""

    __tablename__ = "benchmark_results"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True, description="Parent benchmark run")
    suite_id: str = Field(index=True, description="Parent benchmark suite")

    # Metric details
    metric_name: str = Field(description="Name of the metric")
    metric_value: float = Field(description="Extracted numeric value")
    metric_unit: Optional[str] = Field(default=None, description="Unit label")

    # Cost efficiency
    cost_per_unit: Optional[float] = Field(
        default=None, description="Cost per million units of this metric"
    )

    # Context
    gpu_type: str = Field(description="GPU model")
    hourly_rate: float = Field(default=0.0)

    created_at: datetime = Field(default_factory=datetime.utcnow)
