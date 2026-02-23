"""Pytest configuration and fixtures for CloudComputeManager tests."""

import asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio

from cloudcomputemanager.core.config import Settings
from cloudcomputemanager.core.database import get_engine, init_db


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Create test settings with temporary directory."""
    return Settings(
        data_dir=tmp_path / "cloudcomputemanager",
        database_url=f"sqlite+aiosqlite:///{tmp_path}/test.db",
        checkpoint_local_path=tmp_path / "checkpoints",
        sync_local_path=tmp_path / "sync",
    )


@pytest_asyncio.fixture
async def test_db(test_settings: Settings) -> AsyncGenerator[None, None]:
    """Initialize test database."""
    test_settings.ensure_directories()
    await init_db()
    yield
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def sample_job_config() -> dict:
    """Sample job configuration for testing."""
    return {
        "name": "test-job",
        "project": "test-project",
        "image": "python:3.11",
        "command": "python -c 'print(hello)'",
        "resources": {
            "gpu_type": "RTX_4090",
            "gpu_count": 1,
            "gpu_memory_min": 16,
            "disk_gb": 50,
        },
        "checkpoint": {
            "strategy": "filesystem",
            "interval_minutes": 30,
            "path": "/workspace/checkpoints",
        },
        "sync": {
            "enabled": True,
            "source": "/workspace/results",
            "destination": "/tmp/results",
            "interval_minutes": 15,
        },
        "budget": {
            "max_cost_usd": 10.0,
            "max_hours": 2,
        },
    }
