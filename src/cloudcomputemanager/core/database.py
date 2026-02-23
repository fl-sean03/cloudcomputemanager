"""Database setup and session management for CloudComputeManager.

Uses SQLModel with async SQLite for lightweight, embedded persistence.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

from cloudcomputemanager.core.config import get_settings

# Global engine instance
_engine = None
_async_session_factory = None


def get_engine():
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        settings.ensure_directories()

        # Expand ~ in database URL
        db_url = settings.database_url.replace("~", str(settings.data_dir.parent.parent))

        _engine = create_async_engine(
            db_url,
            echo=settings.debug,
            future=True,
        )
    return _engine


def get_session_factory():
    """Get or create the session factory."""
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_factory


async def init_db() -> None:
    """Initialize the database, creating all tables."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def close_db() -> None:
    """Close the database connection."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session.

    Usage:
        async with get_session() as session:
            # Use session...

    Yields:
        AsyncSession: Database session
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
