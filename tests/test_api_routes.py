"""Tests for CloudComputeManager API routes.

Uses httpx.AsyncClient with ASGITransport for modern async FastAPI testing.
Database-dependent endpoints are tested with mocked get_session using
in-memory SQLite. Package endpoints use the real in-memory PackageRegistry.
"""

import pytest
import pytest_asyncio
from contextlib import asynccontextmanager
from unittest.mock import patch, AsyncMock

from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

from cloudcomputemanager import __version__


# ---------------------------------------------------------------------------
# In-memory database fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def async_engine():
    """Create an in-memory async SQLite engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
    )
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(async_engine):
    """Create a session factory bound to the in-memory engine."""
    factory = sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    return factory


@pytest_asyncio.fixture
async def client(session_factory):
    """Create an httpx AsyncClient backed by the app with mocked DB.

    We patch:
      - init_db / close_db so the lifespan doesn't touch the real DB.
      - get_session so route handlers use our in-memory DB.
    """

    @asynccontextmanager
    async def _mock_get_session():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    with (
        patch("cloudcomputemanager.api.app.init_db", new_callable=AsyncMock),
        patch("cloudcomputemanager.api.app.close_db", new_callable=AsyncMock),
        patch(
            "cloudcomputemanager.api.routes.get_session",
            new=_mock_get_session,
        ),
    ):
        from cloudcomputemanager.api.app import create_app

        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            yield ac


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Tests for GET /health."""

    @pytest.mark.asyncio
    async def test_health_check(self, client: AsyncClient):
        """GET /health returns 200 with status 'healthy' and version."""
        resp = await client.get("/health")
        assert resp.status_code == 200

        data = resp.json()
        assert data["status"] == "healthy"
        assert data["version"] == __version__
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# Jobs endpoints
# ---------------------------------------------------------------------------


class TestJobs:
    """Tests for /v1/jobs routes."""

    @pytest.mark.asyncio
    async def test_list_jobs_empty(self, client: AsyncClient):
        """GET /v1/jobs returns 200 with empty list when no jobs exist."""
        resp = await client.get("/v1/jobs")
        assert resp.status_code == 200

        data = resp.json()
        assert data["jobs"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_submit_job(self, client: AsyncClient):
        """POST /v1/jobs with valid config returns 200 with a job_id."""
        payload = {
            "name": "test-lammps-run",
            "project": "unit-tests",
            "image": "nvcr.io/hpc/lammps:29Aug2024",
            "command": "lmp -in input.in",
            "resources": {
                "gpu_type": "RTX_4090",
                "gpu_count": 1,
                "gpu_memory_min": 16,
                "disk_gb": 50,
            },
            "checkpoint": {
                "strategy": "application",
                "interval_minutes": 30,
                "path": "/workspace/checkpoints",
            },
            "budget": {
                "max_cost_usd": 10.0,
                "max_hours": 2,
            },
        }

        resp = await client.post("/v1/jobs", json=payload)
        assert resp.status_code == 200

        data = resp.json()
        assert "job_id" in data
        assert data["job_id"].startswith("job_")
        assert data["name"] == "test-lammps-run"
        assert data["project"] == "unit-tests"
        assert data["status"] == "PENDING"
        assert data["total_cost_usd"] == 0.0

    @pytest.mark.asyncio
    async def test_submit_and_retrieve_job(self, client: AsyncClient):
        """Submit a job then GET it by ID."""
        payload = {
            "name": "retrieve-me",
            "image": "python:3.11",
            "command": "python train.py",
        }
        create_resp = await client.post("/v1/jobs", json=payload)
        assert create_resp.status_code == 200
        job_id = create_resp.json()["job_id"]

        get_resp = await client.get(f"/v1/jobs/{job_id}")
        assert get_resp.status_code == 200

        data = get_resp.json()
        assert data["job_id"] == job_id
        assert data["name"] == "retrieve-me"
        assert data["status"] == "PENDING"

    @pytest.mark.asyncio
    async def test_get_job_not_found(self, client: AsyncClient):
        """GET /v1/jobs/<nonexistent> returns 404."""
        resp = await client.get("/v1/jobs/nonexistent-id-999")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_list_jobs_after_submit(self, client: AsyncClient):
        """List jobs returns submitted jobs."""
        for i in range(3):
            payload = {
                "name": f"batch-job-{i}",
                "image": "python:3.11",
                "command": f"echo {i}",
            }
            resp = await client.post("/v1/jobs", json=payload)
            assert resp.status_code == 200

        list_resp = await client.get("/v1/jobs")
        assert list_resp.status_code == 200

        data = list_resp.json()
        assert data["total"] == 3
        assert len(data["jobs"]) == 3

    @pytest.mark.asyncio
    async def test_list_jobs_pagination(self, client: AsyncClient):
        """List jobs respects limit and offset parameters."""
        for i in range(5):
            payload = {
                "name": f"paginated-{i}",
                "image": "python:3.11",
                "command": f"echo {i}",
            }
            await client.post("/v1/jobs", json=payload)

        resp = await client.get("/v1/jobs", params={"limit": 2, "offset": 0})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["jobs"]) == 2
        assert data["total"] == 5


# ---------------------------------------------------------------------------
# Packages endpoints (no DB needed — uses in-memory PackageRegistry)
# ---------------------------------------------------------------------------


class TestPackages:
    """Tests for /v1/packages routes."""

    @pytest.mark.asyncio
    async def test_list_packages(self, client: AsyncClient):
        """GET /v1/packages returns 200 with the builtin package list."""
        resp = await client.get("/v1/packages")
        assert resp.status_code == 200

        data = resp.json()
        assert data["total"] > 0
        assert len(data["packages"]) == data["total"]

        # Verify known builtin packages are present
        names = [p["name"] for p in data["packages"]]
        assert "lammps" in names
        assert "pytorch" in names

    @pytest.mark.asyncio
    async def test_list_packages_by_category(self, client: AsyncClient):
        """GET /v1/packages?category=molecular_dynamics filters correctly."""
        resp = await client.get(
            "/v1/packages", params={"category": "molecular_dynamics"}
        )
        assert resp.status_code == 200

        data = resp.json()
        assert data["total"] > 0
        for pkg in data["packages"]:
            assert pkg["category"] == "molecular_dynamics"

    @pytest.mark.asyncio
    async def test_list_packages_invalid_category(self, client: AsyncClient):
        """GET /v1/packages?category=bogus returns 400."""
        resp = await client.get("/v1/packages", params={"category": "bogus"})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_search_packages(self, client: AsyncClient):
        """GET /v1/packages/search?query=lammps returns LAMMPS."""
        resp = await client.get(
            "/v1/packages/search", params={"query": "lammps"}
        )
        assert resp.status_code == 200

        data = resp.json()
        assert data["total"] >= 1
        names = [p["name"] for p in data["packages"]]
        assert "lammps" in names

    @pytest.mark.asyncio
    async def test_search_packages_no_results(self, client: AsyncClient):
        """Search for a nonsense query returns empty list."""
        resp = await client.get(
            "/v1/packages/search", params={"query": "zzz-nonexistent-xyz"}
        )
        assert resp.status_code == 200

        data = resp.json()
        assert data["total"] == 0
        assert data["packages"] == []

    @pytest.mark.asyncio
    async def test_get_package_by_name(self, client: AsyncClient):
        """GET /v1/packages/lammps returns the LAMMPS package with variants."""
        resp = await client.get("/v1/packages/lammps")
        assert resp.status_code == 200

        data = resp.json()
        assert data["name"] == "lammps"
        assert data["display_name"] == "LAMMPS"
        assert data["category"] == "molecular_dynamics"
        assert len(data["variants"]) >= 1

        # Verify variant structure
        variant = data["variants"][0]
        assert "id" in variant
        assert "version" in variant
        assert "cuda_versions" in variant
        assert "gpu_architectures" in variant

    @pytest.mark.asyncio
    async def test_get_package_not_found(self, client: AsyncClient):
        """GET /v1/packages/nonexistent returns 404."""
        resp = await client.get("/v1/packages/nonexistent-package")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()
