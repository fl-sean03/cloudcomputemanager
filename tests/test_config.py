"""Tests for configuration management."""

from pathlib import Path

import pytest

from cloudcomputemanager.core.config import Settings, get_settings


class TestSettings:
    """Tests for Settings configuration."""

    def test_default_settings(self, tmp_path: Path):
        """Test default settings values."""
        settings = Settings(data_dir=tmp_path / "cloudcomputemanager")

        assert settings.app_name == "CloudComputeManager"
        assert settings.debug is False
        assert settings.log_level == "INFO"

    def test_api_settings(self, tmp_path: Path):
        """Test API server settings."""
        settings = Settings(
            data_dir=tmp_path / "cloudcomputemanager",
            api_host="0.0.0.0",
            api_port=9000,
        )

        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 9000

    def test_checkpoint_settings(self, tmp_path: Path):
        """Test checkpoint storage settings."""
        settings = Settings(
            data_dir=tmp_path / "cloudcomputemanager",
            checkpoint_storage="s3",
            checkpoint_s3_bucket="my-bucket",
        )

        assert settings.checkpoint_storage == "s3"
        assert settings.checkpoint_s3_bucket == "my-bucket"

    def test_ensure_directories(self, tmp_path: Path):
        """Test directory creation."""
        settings = Settings(
            data_dir=tmp_path / "cloudcomputemanager",
            checkpoint_local_path=tmp_path / "checkpoints",
            sync_local_path=tmp_path / "sync",
        )

        settings.ensure_directories()

        assert settings.data_dir.exists()
        assert settings.checkpoint_local_path.exists()
        assert settings.sync_local_path.exists()

    def test_vast_api_key_from_file(self, tmp_path: Path):
        """Test loading Vast.ai API key from file."""
        api_key_file = tmp_path / ".vast_api_key"
        api_key_file.write_text("test-api-key-12345")

        settings = Settings(
            data_dir=tmp_path / "cloudcomputemanager",
            vast_api_key_file=api_key_file,
        )

        assert settings.get_vast_api_key() == "test-api-key-12345"

    def test_vast_api_key_from_env(self, tmp_path: Path):
        """Test Vast.ai API key from direct setting."""
        settings = Settings(
            data_dir=tmp_path / "cloudcomputemanager",
            vast_api_key="direct-api-key",
        )

        assert settings.get_vast_api_key() == "direct-api-key"

    def test_vast_api_key_missing(self, tmp_path: Path):
        """Test error when Vast.ai API key is missing."""
        settings = Settings(
            data_dir=tmp_path / "cloudcomputemanager",
            vast_api_key_file=tmp_path / "nonexistent",
        )

        with pytest.raises(ValueError, match="API key not found"):
            settings.get_vast_api_key()


def test_get_settings_caching():
    """Test that get_settings returns cached instance."""
    settings1 = get_settings()
    settings2 = get_settings()

    assert settings1 is settings2
