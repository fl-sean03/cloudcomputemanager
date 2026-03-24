"""Tests for VastProvider error handling.

Verifies that the VastProvider handles various failure scenarios correctly,
including CLI errors, malformed output, SSH timeouts, retries, and unexpected
API responses. All tests mock at the subprocess level.
"""

import asyncio
import json
import subprocess

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cloudcomputemanager.providers.vast import VastProvider, SSH_MAX_RETRIES
from cloudcomputemanager.providers.base import ProviderInstance, ProviderStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_process(returncode: int, stdout: bytes = b"", stderr: bytes = b""):
    """Create a mock asyncio.subprocess.Process with the given outputs."""
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.kill = MagicMock()
    return proc


def _running_instance(instance_id: str = "99999") -> ProviderInstance:
    """Return a minimal running ProviderInstance for use in execute/rsync tests."""
    return ProviderInstance(
        instance_id=instance_id,
        provider="vast",
        status=ProviderStatus.RUNNING,
        gpu_type="RTX 4090",
        gpu_count=1,
        gpu_memory_gb=24,
        cpu_cores=8,
        memory_gb=32,
        disk_gb=100,
        ssh_host="ssh5.vast.ai",
        ssh_port=12345,
        ssh_user="root",
        hourly_rate=0.50,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider():
    """Create a VastProvider with a dummy API key."""
    return VastProvider(api_key="test-key")


# ---------------------------------------------------------------------------
# 1. search_offers — CLI returns non-zero exit code
# ---------------------------------------------------------------------------

class TestSearchOffersApiError:
    @pytest.mark.asyncio
    async def test_search_offers_api_error(self, provider):
        """vastai CLI returns non-zero exit code -> should raise RuntimeError."""
        proc = _make_mock_process(
            returncode=1,
            stdout=b"",
            stderr=b"Error: invalid API key",
        )

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            with pytest.raises(RuntimeError, match="vastai command failed"):
                await provider.search_offers(gpu_type="RTX_4090")


# ---------------------------------------------------------------------------
# 2. search_offers — CLI returns malformed JSON
# ---------------------------------------------------------------------------

class TestSearchOffersMalformedJson:
    @pytest.mark.asyncio
    async def test_search_offers_malformed_json(self, provider):
        """vastai CLI returns garbage -> should return empty list gracefully."""
        proc = _make_mock_process(
            returncode=0,
            stdout=b"this is not json at all {{{",
            stderr=b"",
        )

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            offers = await provider.search_offers(gpu_type="RTX_4090")
            assert offers == []


# ---------------------------------------------------------------------------
# 3. execute_command — SSH command times out
# ---------------------------------------------------------------------------

class TestExecuteCommandSshTimeout:
    @pytest.mark.asyncio
    async def test_execute_command_ssh_timeout(self, provider):
        """SSH command times out -> should return (-1, '', 'Command timed out')."""
        instance = _running_instance()

        # Make communicate() raise TimeoutError
        proc = AsyncMock()
        proc.returncode = None
        proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        proc.kill = MagicMock()

        with (
            patch.object(provider, "get_instance", AsyncMock(return_value=instance)),
            patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
        ):
            exit_code, stdout, stderr = await provider.execute_command(
                "99999", "sleep 999", timeout=5
            )

            assert exit_code == -1
            assert stdout == ""
            assert stderr == "Command timed out"
            proc.kill.assert_called_once()


# ---------------------------------------------------------------------------
# 4. execute_command — SSH retry on exit code 255
# ---------------------------------------------------------------------------

class TestExecuteCommandSshRetry:
    @pytest.mark.asyncio
    async def test_execute_command_ssh_retry(self, provider):
        """First two SSH attempts return 255 (conn failure), third succeeds."""
        instance = _running_instance()

        fail_proc_1 = _make_mock_process(returncode=255, stderr=b"Connection refused")
        fail_proc_2 = _make_mock_process(returncode=255, stderr=b"Connection refused")
        success_proc = _make_mock_process(
            returncode=0, stdout=b"hello world", stderr=b""
        )

        mock_create = AsyncMock(side_effect=[fail_proc_1, fail_proc_2, success_proc])

        with (
            patch.object(provider, "get_instance", AsyncMock(return_value=instance)),
            patch("asyncio.create_subprocess_exec", mock_create),
            patch("asyncio.sleep", new_callable=AsyncMock),  # skip backoff delays
        ):
            exit_code, stdout, stderr = await provider.execute_command(
                "99999", "echo hello world"
            )

            assert exit_code == 0
            assert stdout == "hello world"
            assert mock_create.call_count == 3


# ---------------------------------------------------------------------------
# 5. execute_command — no retry on non-255 command failure
# ---------------------------------------------------------------------------

class TestExecuteCommandNoRetryOnCommandFailure:
    @pytest.mark.asyncio
    async def test_execute_command_no_retry_on_command_failure(self, provider):
        """SSH works but command returns non-zero -> should NOT retry."""
        instance = _running_instance()

        proc = _make_mock_process(
            returncode=1,
            stdout=b"",
            stderr=b"command not found",
        )

        mock_create = AsyncMock(return_value=proc)

        with (
            patch.object(provider, "get_instance", AsyncMock(return_value=instance)),
            patch("asyncio.create_subprocess_exec", mock_create),
        ):
            exit_code, stdout, stderr = await provider.execute_command(
                "99999", "nonexistent_command"
            )

            assert exit_code == 1
            assert stderr == "command not found"
            # Only one invocation — no retries
            assert mock_create.call_count == 1


# ---------------------------------------------------------------------------
# 6. rsync_download — first attempt fails, second succeeds
# ---------------------------------------------------------------------------

class TestRsyncDownloadRetry:
    @pytest.mark.asyncio
    async def test_rsync_download_retry(self, provider):
        """First rsync attempt fails, second succeeds -> should retry and return True."""
        instance = _running_instance()

        fail_proc = _make_mock_process(returncode=12, stderr=b"rsync error")
        success_proc = _make_mock_process(returncode=0, stdout=b"done", stderr=b"")

        mock_create = AsyncMock(side_effect=[fail_proc, success_proc])

        with (
            patch.object(provider, "get_instance", AsyncMock(return_value=instance)),
            patch("asyncio.create_subprocess_exec", mock_create),
            patch("asyncio.sleep", new_callable=AsyncMock),  # skip backoff delays
        ):
            result = await provider.rsync_download(
                "99999", "/remote/data", "/local/data"
            )

            assert result is True
            assert mock_create.call_count == 2


# ---------------------------------------------------------------------------
# 7. create_instance — unexpected response format
# ---------------------------------------------------------------------------

class TestCreateInstanceUnexpectedResponse:
    @pytest.mark.asyncio
    async def test_create_instance_unexpected_response(self, provider):
        """vastai returns neither dict with 'new_contract' nor 'Started' string
        -> should raise RuntimeError."""
        # _run_vastai_cmd returns something unexpected (e.g. a list or a random string)
        with patch.object(
            provider,
            "_run_vastai_cmd",
            AsyncMock(return_value="some random output"),
        ):
            with pytest.raises(RuntimeError, match="Unexpected create result"):
                await provider.create_instance(
                    offer_id="12345",
                    image="pytorch/pytorch:latest",
                    ssh_public_key="ssh-rsa AAAA...",
                )


# ---------------------------------------------------------------------------
# 8. get_instance — "not found" error -> return None
# ---------------------------------------------------------------------------

class TestGetInstanceNotFound:
    @pytest.mark.asyncio
    async def test_get_instance_not_found(self, provider):
        """vastai returns 'not found' error -> should return None, not raise."""
        proc = _make_mock_process(
            returncode=1,
            stdout=b"",
            stderr=b"Error: Instance 99999 not found",
        )

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await provider.get_instance("99999")
            assert result is None
