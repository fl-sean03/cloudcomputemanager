"""Tests for the CCM dashboard."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from cloudcomputemanager.core.models import Job, JobStatus, Instance, CostRecord


# ============================================================================
# Data Layer Tests
# ============================================================================


class TestDashboardSummary:
    """Test get_dashboard_summary()."""

    @pytest.mark.asyncio
    async def test_summary_empty_db(self):
        """Summary should return zeros on empty DB."""
        from cloudcomputemanager.core.database import init_db, get_session
        from cloudcomputemanager.core import database as db_module

        with tempfile.TemporaryDirectory() as tmpdir:
            from cloudcomputemanager.core.config import Settings
            settings = Settings(
                data_dir=Path(tmpdir) / "ccm",
                database_url=f"sqlite+aiosqlite:///{tmpdir}/test.db",
            )
            settings.ensure_directories()

            with patch("cloudcomputemanager.core.database.get_settings", return_value=settings):
                db_module._engine = None
                db_module._async_session_factory = None
                await init_db()

                from cloudcomputemanager.dashboard.data import get_dashboard_summary
                summary = await get_dashboard_summary()

                # Check structure — keys may vary but core data must be present
                assert summary["today_spend"] == 0
                assert summary["burn_rate"] == 0
                assert summary["week_spend"] == 0

            await db_module.close_db()
            db_module._engine = None
            db_module._async_session_factory = None

    @pytest.mark.asyncio
    async def test_summary_with_running_jobs(self):
        """Summary should reflect running jobs in spend."""
        from cloudcomputemanager.core.database import init_db, get_session
        from cloudcomputemanager.core import database as db_module

        with tempfile.TemporaryDirectory() as tmpdir:
            from cloudcomputemanager.core.config import Settings
            settings = Settings(
                data_dir=Path(tmpdir) / "ccm",
                database_url=f"sqlite+aiosqlite:///{tmpdir}/test.db",
            )
            settings.ensure_directories()

            with patch("cloudcomputemanager.core.database.get_settings", return_value=settings):
                db_module._engine = None
                db_module._async_session_factory = None
                await init_db()

                # Seed some jobs
                async with get_session() as session:
                    session.add(Job(
                        name="running-1", image="u", command="e",
                        status=JobStatus.RUNNING,
                        started_at=datetime.utcnow(),
                        total_cost_usd=2.50,
                    ))
                    session.add(Job(
                        name="running-2", image="u", command="e",
                        status=JobStatus.RUNNING,
                        started_at=datetime.utcnow(),
                        total_cost_usd=1.00,
                    ))

                from cloudcomputemanager.dashboard.data import get_dashboard_summary
                summary = await get_dashboard_summary()

                # Should have status breakdown with running jobs
                assert "jobs_by_status" in summary or "active_jobs" in summary
                assert summary["today_spend"] >= 3.50

            await db_module.close_db()
            db_module._engine = None
            db_module._async_session_factory = None


class TestDashboardAlerts:
    """Test get_alerts()."""

    @pytest.mark.asyncio
    async def test_alerts_detects_failed_jobs(self):
        """Alerts should include recently failed jobs."""
        from cloudcomputemanager.core.database import init_db, get_session
        from cloudcomputemanager.core import database as db_module

        with tempfile.TemporaryDirectory() as tmpdir:
            from cloudcomputemanager.core.config import Settings
            settings = Settings(
                data_dir=Path(tmpdir) / "ccm",
                database_url=f"sqlite+aiosqlite:///{tmpdir}/test.db",
            )
            settings.ensure_directories()

            with patch("cloudcomputemanager.core.database.get_settings", return_value=settings):
                db_module._engine = None
                db_module._async_session_factory = None
                await init_db()

                async with get_session() as session:
                    session.add(Job(
                        name="failed-job", image="u", command="e",
                        status=JobStatus.FAILED,
                        exit_code=1,
                        completed_at=datetime.utcnow(),
                        error_message="Segmentation fault",
                    ))

                from cloudcomputemanager.dashboard.data import get_alerts
                alerts = await get_alerts()

                red_alerts = [a for a in alerts if a["severity"] == "red"]
                assert len(red_alerts) >= 1
                assert any("failed-job" in a["message"] for a in red_alerts)

            await db_module.close_db()
            db_module._engine = None
            db_module._async_session_factory = None


class TestDashboardActiveJobs:
    """Test get_active_jobs()."""

    @pytest.mark.asyncio
    async def test_active_jobs_empty(self):
        """Active jobs should return empty list on empty DB."""
        from cloudcomputemanager.core.database import init_db
        from cloudcomputemanager.core import database as db_module

        with tempfile.TemporaryDirectory() as tmpdir:
            from cloudcomputemanager.core.config import Settings
            settings = Settings(
                data_dir=Path(tmpdir) / "ccm",
                database_url=f"sqlite+aiosqlite:///{tmpdir}/test.db",
            )
            settings.ensure_directories()

            with patch("cloudcomputemanager.core.database.get_settings", return_value=settings):
                db_module._engine = None
                db_module._async_session_factory = None
                await init_db()

                from cloudcomputemanager.dashboard.data import get_active_jobs
                jobs = await get_active_jobs()
                assert jobs == []

            await db_module.close_db()
            db_module._engine = None
            db_module._async_session_factory = None


class TestDashboardCosts:
    """Test get_cost_breakdown()."""

    @pytest.mark.asyncio
    async def test_cost_breakdown_empty(self):
        """Cost breakdown should handle empty DB."""
        from cloudcomputemanager.core.database import init_db
        from cloudcomputemanager.core import database as db_module

        with tempfile.TemporaryDirectory() as tmpdir:
            from cloudcomputemanager.core.config import Settings
            settings = Settings(
                data_dir=Path(tmpdir) / "ccm",
                database_url=f"sqlite+aiosqlite:///{tmpdir}/test.db",
            )
            settings.ensure_directories()

            with patch("cloudcomputemanager.core.database.get_settings", return_value=settings):
                db_module._engine = None
                db_module._async_session_factory = None
                await init_db()

                from cloudcomputemanager.dashboard.data import get_cost_breakdown
                costs = await get_cost_breakdown()
                assert costs["total_cost"] == 0
                assert costs["by_project"] == []

            await db_module.close_db()
            db_module._engine = None
            db_module._async_session_factory = None


# ============================================================================
# Route Tests
# ============================================================================


class TestDashboardRoutes:
    """Test dashboard HTTP endpoints."""

    def test_dashboard_routes_registered(self):
        """Dashboard routes should be registered on the app."""
        from cloudcomputemanager.api.app import create_app
        app = create_app()
        routes = [r.path for r in app.routes]
        assert "/dashboard" in routes
        assert "/dashboard/sse" in routes
        assert "/dashboard/partials/alerts" in routes
        assert "/dashboard/partials/jobs" in routes
        assert "/dashboard/actions/cancel/{job_id}" in routes

    def test_static_files_mounted(self):
        """Static files should be mounted at /dashboard/static."""
        from cloudcomputemanager.api.app import create_app
        app = create_app()
        routes = [r.path for r in app.routes]
        assert "/dashboard/static" in routes


class TestDashboardCLI:
    """Test dashboard CLI command."""

    def test_dashboard_command_registered(self):
        """ccm dashboard should be a registered command."""
        from cloudcomputemanager.cli import main
        import inspect
        source = inspect.getsource(main)
        assert "def dashboard(" in source
        assert "ccm dashboard" in source.lower() or "Open the CCM web dashboard" in source


# ============================================================================
# Template Tests
# ============================================================================


class TestDashboardTemplates:
    """Test that templates exist and are valid Jinja2."""

    def test_all_templates_exist(self):
        """All required template files should exist."""
        from cloudcomputemanager.dashboard.routes import TEMPLATES_DIR
        assert (TEMPLATES_DIR / "base.html").exists()
        assert (TEMPLATES_DIR / "dashboard.html").exists()
        assert (TEMPLATES_DIR / "partials" / "alerts.html").exists()
        assert (TEMPLATES_DIR / "partials" / "stats.html").exists()
        assert (TEMPLATES_DIR / "partials" / "jobs_table.html").exists()
        assert (TEMPLATES_DIR / "partials" / "job_detail.html").exists()
        assert (TEMPLATES_DIR / "partials" / "events.html").exists()
        assert (TEMPLATES_DIR / "partials" / "costs.html").exists()
        assert (TEMPLATES_DIR / "partials" / "finished.html").exists()

    def test_static_files_exist(self):
        """Static asset files should exist."""
        from cloudcomputemanager.dashboard import STATIC_DIR
        assert (STATIC_DIR / "htmx.min.js").exists()
        assert (STATIC_DIR / "pico.min.css").exists()
        assert (STATIC_DIR / "style.css").exists()

    def test_htmx_is_vendored(self):
        """htmx.min.js should be a real file, not empty."""
        from cloudcomputemanager.dashboard import STATIC_DIR
        htmx = STATIC_DIR / "htmx.min.js"
        assert htmx.stat().st_size > 10000  # HTMX is ~14-50KB

    def test_partials_render_with_empty_data(self):
        """Partials should render without errors given empty data."""
        from jinja2 import Environment, FileSystemLoader
        from cloudcomputemanager.dashboard.routes import TEMPLATES_DIR

        env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))

        # Each partial with its expected empty context
        partials = {
            "partials/alerts.html": {"alerts": []},
            "partials/stats.html": {"summary": {
                "active_jobs": 0, "active_breakdown": {},
                "today_spend": 0, "burn_rate": 0,
                "week_spend": 0, "week_projects": 0,
                "recoveries_24h": 0, "recoveries_ok": 0, "recoveries_failed": 0,
            }},
            "partials/jobs_table.html": {"jobs": []},
            "partials/events.html": {"events": []},
            "partials/costs.html": {"costs": {
                "by_project": [], "by_gpu": [],
                "total_cost": 0, "total_hours": 0,
            }},
            "partials/finished.html": {"finished": []},
        }

        for template_name, context in partials.items():
            tmpl = env.get_template(template_name)
            html = tmpl.render(**context)
            assert isinstance(html, str)
            # Some partials render to empty string when data is empty (e.g., alerts)
            # That's correct — just verify no Jinja2 errors
