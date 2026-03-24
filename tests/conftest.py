"""Pytest configuration and fixtures for CloudComputeManager tests."""

import asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio

from cloudcomputemanager.core.config import Settings
from cloudcomputemanager.core.database import get_engine, init_db


def pytest_addoption(parser):
    """Add command line options for integration tests."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that hit real APIs (may cost money)",
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires --run-integration)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless --run-integration is passed."""
    if config.getoption("--run-integration"):
        return

    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


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
    from cloudcomputemanager.core import database as db_module
    # Reset global engine so tests get a fresh DB with current schema
    db_module._engine = None
    db_module._async_session_factory = None
    test_settings.ensure_directories()
    await init_db()
    yield
    # Cleanup: close and reset engine
    await db_module.close_db()
    db_module._async_session_factory = None


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
