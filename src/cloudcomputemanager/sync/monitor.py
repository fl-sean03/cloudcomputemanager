"""Sync monitoring for CloudComputeManager.

Provides real-time monitoring of sync operations and file changes.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import structlog
from watchfiles import awatch, Change

from cloudcomputemanager.core.config import get_settings

logger = structlog.get_logger(__name__)


class SyncMonitor:
    """Monitors local sync directories for changes.

    Useful for:
    - Notifying when new files are synced
    - Tracking sync progress
    - Detecting sync completion
    """

    def __init__(self):
        """Initialize the sync monitor."""
        self._settings = get_settings()
        self._watchers: dict[str, asyncio.Task] = {}

    async def watch_directory(
        self,
        job_id: str,
        on_change: Callable[[str, Path], None],
        patterns: Optional[list[str]] = None,
    ) -> None:
        """Watch a job's sync directory for changes.

        Args:
            job_id: Job ID to watch
            on_change: Callback on file change (change_type, path)
            patterns: File patterns to watch (glob-style)
        """
        sync_dir = self._settings.sync_local_path / job_id
        if not sync_dir.exists():
            sync_dir.mkdir(parents=True, exist_ok=True)

        async def watch_loop():
            logger.info("Starting file watch", job_id=job_id, path=str(sync_dir))

            try:
                async for changes in awatch(sync_dir):
                    for change_type, path_str in changes:
                        path = Path(path_str)

                        # Filter by patterns if specified
                        if patterns:
                            if not any(path.match(p) for p in patterns):
                                continue

                        change_name = {
                            Change.added: "added",
                            Change.modified: "modified",
                            Change.deleted: "deleted",
                        }.get(change_type, "unknown")

                        logger.debug(
                            "File change detected",
                            job_id=job_id,
                            change=change_name,
                            path=str(path),
                        )

                        try:
                            on_change(change_name, path)
                        except Exception as e:
                            logger.error(
                                "Error in change callback",
                                error=str(e),
                            )

            except asyncio.CancelledError:
                logger.info("File watch cancelled", job_id=job_id)
                raise

        task = asyncio.create_task(watch_loop())
        self._watchers[job_id] = task

    async def stop_watching(self, job_id: str) -> None:
        """Stop watching a job's sync directory.

        Args:
            job_id: Job ID to stop watching
        """
        if job_id in self._watchers:
            self._watchers[job_id].cancel()
            try:
                await self._watchers[job_id]
            except asyncio.CancelledError:
                pass
            del self._watchers[job_id]
            logger.info("Stopped file watch", job_id=job_id)

    def get_sync_stats(self, job_id: str) -> dict:
        """Get statistics for a job's synced data.

        Args:
            job_id: Job ID

        Returns:
            Dict with sync statistics
        """
        sync_dir = self._settings.sync_local_path / job_id

        if not sync_dir.exists():
            return {
                "exists": False,
                "file_count": 0,
                "total_size_bytes": 0,
                "latest_file": None,
                "latest_modified": None,
            }

        files = [f for f in sync_dir.rglob("*") if f.is_file()]
        total_size = sum(f.stat().st_size for f in files)

        latest_file = None
        latest_modified = None
        if files:
            latest = max(files, key=lambda f: f.stat().st_mtime)
            latest_file = str(latest.relative_to(sync_dir))
            latest_modified = datetime.fromtimestamp(latest.stat().st_mtime)

        return {
            "exists": True,
            "file_count": len(files),
            "total_size_bytes": total_size,
            "latest_file": latest_file,
            "latest_modified": latest_modified,
        }


def format_size(size_bytes: int) -> str:
    """Format size in human-readable form.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"
