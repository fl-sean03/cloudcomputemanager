"""FastAPI application factory for CloudComputeManager API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cloudcomputemanager import __version__
from cloudcomputemanager.api import routes
from cloudcomputemanager.core.config import get_settings
from cloudcomputemanager.core.database import init_db, close_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    settings = get_settings()
    settings.ensure_directories()
    await init_db()

    yield

    # Shutdown
    await close_db()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application
    """
    settings = get_settings()

    app = FastAPI(
        title="CloudComputeManager API",
        description="GPU cloud management with automatic checkpointing and spot recovery",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(routes.health_router, tags=["Health"])
    app.include_router(routes.jobs_router, prefix="/v1/jobs", tags=["Jobs"])
    app.include_router(routes.instances_router, prefix="/v1/instances", tags=["Instances"])
    app.include_router(routes.offers_router, prefix="/v1/offers", tags=["Offers"])
    app.include_router(routes.checkpoints_router, prefix="/v1/checkpoints", tags=["Checkpoints"])
    app.include_router(routes.packages_router, prefix="/v1/packages", tags=["Packages"])

    return app


# Create app instance for uvicorn
app = create_app()
