"""CCM Daemon Service - Background process management."""

import asyncio
import os
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json
import structlog

from cloudcomputemanager.core.config import get_settings
from cloudcomputemanager.daemon.monitor import JobMonitor, MonitorConfig, MonitorEvent, EventType

logger = structlog.get_logger()


class DaemonService:
    """
    Daemon service for CCM background monitoring.

    Manages the lifecycle of the job monitor and handles:
    - Daemonization (background process)
    - Signal handling (graceful shutdown)
    - PID file management
    - Status reporting
    - Event logging
    """

    def __init__(
        self,
        monitor_config: Optional[MonitorConfig] = None,
    ):
        self.settings = get_settings()
        self.monitor_config = monitor_config or MonitorConfig()

        self._monitor: Optional[JobMonitor] = None
        self._shutdown_event = asyncio.Event()

        # Paths
        self._pid_file = self.settings.data_dir / "daemon.pid"
        self._log_file = self.settings.data_dir / "daemon.log"
        self._status_file = self.settings.data_dir / "daemon.status"

    def _write_pid(self) -> None:
        """Write PID file."""
        self._pid_file.parent.mkdir(parents=True, exist_ok=True)
        self._pid_file.write_text(str(os.getpid()))

    def _remove_pid(self) -> None:
        """Remove PID file."""
        if self._pid_file.exists():
            self._pid_file.unlink()

    def _write_status(self, status: dict) -> None:
        """Write status file."""
        status["timestamp"] = datetime.utcnow().isoformat()
        self._status_file.write_text(json.dumps(status, indent=2))

    def _handle_event(self, event: MonitorEvent) -> None:
        """Handle monitor events - log them."""
        log_entry = {
            "timestamp": event.timestamp.isoformat(),
            "event": event.event_type.value,
            "job_id": event.job_id,
            "instance_id": event.instance_id,
            "data": event.data,
        }

        # Append to log file
        with open(self._log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Log to structlog
        logger.info(
            "Monitor event",
            event=event.event_type.value,
            job_id=event.job_id,
        )

    async def _run_monitor(self) -> None:
        """Run the monitor until shutdown."""
        self._monitor = JobMonitor(config=self.monitor_config)
        self._monitor.on_event(self._handle_event)

        await self._monitor.start()

        # Update status periodically
        while not self._shutdown_event.is_set():
            self._write_status({
                "running": True,
                "pid": os.getpid(),
                "monitored_jobs": self._monitor.monitored_job_count,
                "poll_interval": self.monitor_config.poll_interval,
            })
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=10
                )
            except asyncio.TimeoutError:
                pass

        await self._monitor.stop()

    def _setup_signals(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._shutdown)

    def _shutdown(self) -> None:
        """Signal handler for graceful shutdown."""
        logger.info("Shutdown signal received")
        self._shutdown_event.set()

    async def run(self) -> None:
        """Run the daemon service (foreground)."""
        logger.info("Starting CCM daemon service")

        self._write_pid()
        self._setup_signals()

        try:
            await self._run_monitor()
        except Exception as e:
            logger.error("Daemon error", error=str(e))
            raise
        finally:
            self._remove_pid()
            self._write_status({"running": False, "stopped_at": datetime.utcnow().isoformat()})
            logger.info("CCM daemon service stopped")

    @classmethod
    def get_status(cls) -> Optional[dict]:
        """Get daemon status from status file."""
        settings = get_settings()
        status_file = settings.data_dir / "daemon.status"

        if not status_file.exists():
            return None

        try:
            return json.loads(status_file.read_text())
        except Exception:
            return None

    @classmethod
    def get_pid(cls) -> Optional[int]:
        """Get daemon PID if running."""
        settings = get_settings()
        pid_file = settings.data_dir / "daemon.pid"

        if not pid_file.exists():
            return None

        try:
            pid = int(pid_file.read_text().strip())
            # Check if process exists
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError):
            # Clean up stale PID file
            pid_file.unlink(missing_ok=True)
            return None

    @classmethod
    def is_running(cls) -> bool:
        """Check if daemon is currently running."""
        return cls.get_pid() is not None

    @classmethod
    def stop(cls) -> bool:
        """Stop the running daemon."""
        pid = cls.get_pid()
        if pid is None:
            return False

        try:
            os.kill(pid, signal.SIGTERM)
            return True
        except ProcessLookupError:
            return False

    @classmethod
    def get_logs(cls, lines: int = 50) -> list[dict]:
        """Get recent daemon log entries."""
        settings = get_settings()
        log_file = settings.data_dir / "daemon.log"

        if not log_file.exists():
            return []

        try:
            with open(log_file, "r") as f:
                all_lines = f.readlines()
                recent = all_lines[-lines:] if len(all_lines) > lines else all_lines
                return [json.loads(line) for line in recent if line.strip()]
        except Exception:
            return []


def run_daemon(config: Optional[MonitorConfig] = None) -> None:
    """Entry point for running the daemon."""
    service = DaemonService(monitor_config=config)
    asyncio.run(service.run())


def daemonize() -> None:
    """
    Daemonize the current process (Unix only).

    Double-fork technique to become a true daemon.
    """
    if sys.platform == "win32":
        raise RuntimeError("Daemonization not supported on Windows")

    # First fork
    pid = os.fork()
    if pid > 0:
        # Parent exits
        sys.exit(0)

    # Create new session
    os.setsid()

    # Second fork
    pid = os.fork()
    if pid > 0:
        # Parent exits
        sys.exit(0)

    # Redirect stdio to /dev/null
    sys.stdin.close()
    sys.stdout.close()
    sys.stderr.close()

    # Run the daemon
    run_daemon()


if __name__ == "__main__":
    # Entry point for running as module: python -m cloudcomputemanager.daemon.service
    run_daemon()
