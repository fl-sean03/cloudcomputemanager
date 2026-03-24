"""CCM Dashboard — single-page web interface for monitoring jobs, costs, and events."""

from pathlib import Path

from fastapi import APIRouter
from starlette.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"


def create_dashboard_router() -> APIRouter:
    """Create the dashboard router with all routes.

    Returns an APIRouter that should be included in the main FastAPI app.
    """
    from cloudcomputemanager.dashboard.routes import router
    return router


def mount_dashboard_static(app):
    """Mount the dashboard static files directory onto the FastAPI app."""
    app.mount("/dashboard/static", StaticFiles(directory=str(STATIC_DIR)), name="dashboard-static")
