"""Configuration management for CloudComputeManager.

Uses pydantic-settings for environment-based configuration with sensible defaults.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """CloudComputeManager configuration settings.

    All settings can be overridden via environment variables prefixed with CCM_.
    For example: CCM_DATABASE_URL, CCM_VAST_API_KEY, etc.
    """

    model_config = SettingsConfigDict(
        env_prefix="CCM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # =========================================================================
    # General Settings
    # =========================================================================

    app_name: str = Field(default="CloudComputeManager", description="Application name")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    data_dir: Path = Field(
        default=Path.home() / ".cloudcomputemanager",
        description="Directory for local data storage",
    )

    # =========================================================================
    # Database Settings
    # =========================================================================

    database_url: str = Field(
        default=f"sqlite+aiosqlite:///{Path.home()}/.cloudcomputemanager/cloudcomputemanager.db",
        description="Database connection URL",
    )

    # =========================================================================
    # Vast.ai Settings
    # =========================================================================

    vast_api_key: Optional[str] = Field(
        default=None,
        description="Vast.ai API key (auto-loaded from ~/.vast_api_key if not set)",
    )
    vast_api_key_file: Path = Field(
        default=Path.home() / ".vast_api_key",
        description="Path to file containing Vast.ai API key",
    )

    # =========================================================================
    # SkyPilot Settings
    # =========================================================================

    skypilot_config_path: Path = Field(
        default=Path.home() / ".sky" / "config.yaml",
        description="Path to SkyPilot configuration file",
    )
    use_skypilot: bool = Field(
        default=True,
        description="Use SkyPilot for cloud orchestration (recommended)",
    )

    # =========================================================================
    # Checkpoint Settings
    # =========================================================================

    checkpoint_storage: str = Field(
        default="local",
        description="Checkpoint storage backend (local, s3, gcs)",
    )
    checkpoint_local_path: Path = Field(
        default=Path.home() / ".cloudcomputemanager" / "checkpoints",
        description="Local path for checkpoint storage",
    )
    checkpoint_s3_bucket: Optional[str] = Field(
        default=None,
        description="S3 bucket for checkpoint storage",
    )
    checkpoint_s3_prefix: str = Field(
        default="cloudcomputemanager/checkpoints",
        description="S3 prefix for checkpoints",
    )
    checkpoint_verify: bool = Field(
        default=True,
        description="Verify checkpoints after creation",
    )

    # =========================================================================
    # Sync Settings
    # =========================================================================

    sync_local_path: Path = Field(
        default=Path.home() / ".cloudcomputemanager" / "sync",
        description="Local path for synced data",
    )
    sync_default_interval: int = Field(
        default=300,
        ge=60,
        description="Default sync interval in seconds",
    )
    sync_on_preemption: bool = Field(
        default=True,
        description="Force sync on preemption detection",
    )

    # =========================================================================
    # API Server Settings
    # =========================================================================

    api_host: str = Field(default="127.0.0.1", description="API server host")
    api_port: int = Field(default=8765, ge=1, le=65535, description="API server port")
    api_reload: bool = Field(default=False, description="Enable auto-reload for development")
    api_workers: int = Field(default=1, ge=1, description="Number of API workers")

    # =========================================================================
    # SSH Settings
    # =========================================================================

    ssh_key_path: Path = Field(
        default=Path.home() / ".ssh" / "id_rsa",
        description="Path to SSH private key",
    )
    ssh_timeout: int = Field(default=30, ge=5, description="SSH connection timeout in seconds")
    ssh_retries: int = Field(default=3, ge=1, description="SSH connection retries")

    # =========================================================================
    # Monitoring Settings
    # =========================================================================

    health_check_interval: int = Field(
        default=60, ge=10, description="Instance health check interval in seconds"
    )
    job_status_check_interval: int = Field(
        default=15, ge=5, description="Job status check interval in seconds"
    )
    preemption_warning_seconds: int = Field(
        default=120,
        ge=30,
        description="Seconds before preemption to trigger checkpoint",
    )

    # =========================================================================
    # Cost Settings
    # =========================================================================

    default_max_hourly_rate: float = Field(
        default=1.0, gt=0, description="Default maximum hourly rate in USD"
    )
    cost_alert_threshold: float = Field(
        default=100.0, gt=0, description="Cost threshold for alerts in USD"
    )

    def get_vast_api_key(self) -> str:
        """Get Vast.ai API key from settings or file."""
        if self.vast_api_key:
            return self.vast_api_key

        if self.vast_api_key_file.exists():
            return self.vast_api_key_file.read_text().strip()

        raise ValueError(
            f"Vast.ai API key not found. Set CCM_VAST_API_KEY env var "
            f"or create {self.vast_api_key_file}"
        )

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_local_path.mkdir(parents=True, exist_ok=True)
        self.sync_local_path.mkdir(parents=True, exist_ok=True)

    def get_database_path(self) -> Path:
        """Get the SQLite database path."""
        # Handle ~ expansion in the URL
        url = self.database_url
        if url.startswith("sqlite"):
            path_part = url.split("///")[-1]
            return Path(path_part).expanduser()
        return Path(self.data_dir / "cloudcomputemanager.db")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings: Application settings
    """
    return Settings()
