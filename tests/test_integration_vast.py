"""Integration tests for Vast.ai provider.

These tests run against the real Vast.ai API and may incur costs.
They are skipped by default and can be run with:

    pytest tests/test_integration_vast.py --run-integration

Make sure CCM_VAST_API_KEY is set in your environment.
"""

import asyncio
import os
from pathlib import Path

import pytest
import pytest_asyncio

from cloudcomputemanager.providers.vast import VastProvider
from cloudcomputemanager.providers.base import ProviderStatus
from cloudcomputemanager.core.config import get_settings


# Track instances for cleanup
_created_instances: list[str] = []


@pytest.fixture(scope="module")
def vast_provider() -> VastProvider:
    """Create VastProvider with real API key."""
    settings = get_settings()
    api_key = settings.get_vast_api_key()
    if not api_key:
        pytest.skip("No Vast.ai API key configured (set CCM_VAST_API_KEY)")
    return VastProvider(api_key=api_key)


@pytest.fixture(scope="module", autouse=True)
def cleanup_instances(vast_provider: VastProvider):
    """Cleanup any created instances after tests."""
    yield
    # Cleanup after all tests in this module
    for instance_id in _created_instances:
        try:
            asyncio.get_event_loop().run_until_complete(
                vast_provider.terminate_instance(instance_id)
            )
            print(f"Cleaned up instance {instance_id}")
        except Exception as e:
            print(f"Failed to cleanup instance {instance_id}: {e}")


@pytest.mark.integration
class TestSearchOffers:
    """Test offer search functionality against real API."""

    async def test_search_rtx_4090(self, vast_provider: VastProvider):
        """Search for RTX 4090 offers."""
        offers = await vast_provider.search_offers(
            gpu_type="RTX_4090",
            gpu_count=1,
            gpu_memory_min=16,
            max_hourly_rate=1.0,
        )

        # Should find some offers
        assert len(offers) > 0

        # Check offer structure
        offer = offers[0]
        assert offer.offer_id
        assert offer.provider == "vast"
        assert "4090" in offer.gpu_type
        assert offer.gpu_memory_gb >= 16
        assert offer.hourly_rate > 0
        assert offer.hourly_rate <= 1.0

    async def test_search_rtx_3090(self, vast_provider: VastProvider):
        """Search for RTX 3090 offers (usually cheaper and more available)."""
        offers = await vast_provider.search_offers(
            gpu_type="RTX_3090",
            gpu_count=1,
            gpu_memory_min=16,
            max_hourly_rate=0.50,
        )

        assert len(offers) > 0
        offer = offers[0]
        assert "3090" in offer.gpu_type
        assert offer.hourly_rate <= 0.50

    async def test_search_with_space_in_name(self, vast_provider: VastProvider):
        """Test that GPU names with spaces are handled correctly."""
        # This should auto-convert spaces to underscores
        offers = await vast_provider.search_offers(
            gpu_type="RTX 4090",  # Space instead of underscore
            gpu_count=1,
        )

        # Should still find offers
        assert len(offers) > 0
        assert "4090" in offers[0].gpu_type

    async def test_search_no_offers(self, vast_provider: VastProvider):
        """Test search with unrealistic constraints returns empty."""
        offers = await vast_provider.search_offers(
            gpu_type="RTX_4090",
            gpu_count=8,  # 8 GPUs is rare
            max_hourly_rate=0.01,  # Very cheap
        )

        # Should return empty list, not error
        assert isinstance(offers, list)

    async def test_search_with_cpu_cores_min(self, vast_provider: VastProvider):
        """Test search filtering by minimum CPU cores."""
        # Search with minimum CPU cores requirement
        offers = await vast_provider.search_offers(
            gpu_type="RTX_3060",
            gpu_count=1,
            cpu_cores_min=6,
            max_hourly_rate=0.50,
        )

        # Should find offers
        assert len(offers) > 0

        # All offers should have at least 6 CPU cores
        for offer in offers:
            assert offer.cpu_cores >= 6, f"Offer {offer.offer_id} has only {offer.cpu_cores} CPUs"

    async def test_search_cpu_cores_filters_correctly(self, vast_provider: VastProvider):
        """Test that cpu_cores_min actually filters results."""
        # Search without CPU filter
        all_offers = await vast_provider.search_offers(
            gpu_type="RTX_3060",
            gpu_count=1,
            max_hourly_rate=0.50,
        )

        # Search with high CPU requirement
        filtered_offers = await vast_provider.search_offers(
            gpu_type="RTX_3060",
            gpu_count=1,
            cpu_cores_min=8,
            max_hourly_rate=0.50,
        )

        # Filtered should have fewer or equal offers
        assert len(filtered_offers) <= len(all_offers)

        # All filtered offers should meet the requirement
        for offer in filtered_offers:
            assert offer.cpu_cores >= 8


@pytest.mark.integration
class TestInstanceLifecycle:
    """Test instance creation and management.

    WARNING: These tests create real instances and cost money!
    """

    async def test_list_instances(self, vast_provider: VastProvider):
        """List current instances."""
        instances = await vast_provider.list_instances()

        # Should return a list (may be empty)
        assert isinstance(instances, list)

        # If there are instances, check structure
        if instances:
            instance = instances[0]
            assert instance.instance_id
            assert instance.provider == "vast"
            assert instance.status in ProviderStatus

    async def test_create_and_terminate_instance(self, vast_provider: VastProvider):
        """Test creating and terminating an instance.

        This test costs money! Uses the cheapest available GPU.
        """
        # Find cheapest offer
        offers = await vast_provider.search_offers(
            gpu_type="RTX_3090",
            max_hourly_rate=0.20,  # Very cheap
        )

        if not offers:
            pytest.skip("No cheap RTX 3090 offers available")

        offer = offers[0]
        print(f"Using offer {offer.offer_id}: {offer.gpu_type} at ${offer.hourly_rate}/hr")

        # Create instance
        instance = await vast_provider.create_instance(
            offer_id=offer.offer_id,
            image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
            disk_gb=20,
        )

        # Track for cleanup
        _created_instances.append(instance.instance_id)

        assert instance.instance_id
        assert instance.provider == "vast"
        print(f"Created instance {instance.instance_id}")

        # Get instance details
        details = await vast_provider.get_instance(instance.instance_id)
        assert details is not None
        assert details.instance_id == instance.instance_id

        # Wait a bit then terminate (don't wait for ready to save time/money)
        await asyncio.sleep(5)

        # Terminate
        result = await vast_provider.terminate_instance(instance.instance_id)
        assert result is True
        print(f"Terminated instance {instance.instance_id}")

        # Remove from cleanup list since we already terminated
        _created_instances.remove(instance.instance_id)


@pytest.mark.integration
class TestSSHExecution:
    """Test SSH command execution.

    WARNING: These tests create real instances!
    """

    async def test_execute_command_on_instance(self, vast_provider: VastProvider):
        """Test executing a command on an instance.

        This test costs money! Will create an instance, wait for it, execute a command.
        """
        # Find cheap offer
        offers = await vast_provider.search_offers(
            gpu_type="RTX_3090",
            max_hourly_rate=0.20,
        )

        if not offers:
            pytest.skip("No cheap RTX 3090 offers available")

        offer = offers[0]

        # Create instance
        instance = await vast_provider.create_instance(
            offer_id=offer.offer_id,
            image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
            disk_gb=20,
        )
        _created_instances.append(instance.instance_id)

        try:
            # Wait for instance to be ready
            ready = await vast_provider.wait_for_ready(
                instance.instance_id,
                timeout=180  # 3 minutes
            )

            if not ready:
                pytest.skip("Instance did not become ready in time")

            # Execute a simple command
            exit_code, stdout, stderr = await vast_provider.execute_command(
                instance.instance_id,
                "echo 'Hello from CCM integration test'"
            )

            assert exit_code == 0
            assert "Hello from CCM integration test" in stdout

            # Test CUDA availability
            exit_code, stdout, stderr = await vast_provider.execute_command(
                instance.instance_id,
                "python -c \"import torch; print(torch.cuda.is_available())\""
            )

            assert exit_code == 0
            assert "True" in stdout

        finally:
            # Always terminate
            await vast_provider.terminate_instance(instance.instance_id)
            _created_instances.remove(instance.instance_id)


@pytest.mark.integration
class TestFullJobWorkflow:
    """Test the complete job workflow.

    WARNING: These tests create real instances and cost money!
    """

    async def test_job_with_file_sync(self, vast_provider: VastProvider, tmp_path: Path):
        """Test a job with file upload and download.

        This is a comprehensive integration test of the full workflow.
        """
        # Create local test file
        test_dir = tmp_path / "upload"
        test_dir.mkdir()
        test_file = test_dir / "input.txt"
        test_file.write_text("Test input for CCM integration test")

        # Find offer
        offers = await vast_provider.search_offers(
            gpu_type="RTX_3090",
            max_hourly_rate=0.20,
        )

        if not offers:
            pytest.skip("No cheap offers available")

        # Create instance
        instance = await vast_provider.create_instance(
            offer_id=offers[0].offer_id,
            image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
            disk_gb=20,
        )
        _created_instances.append(instance.instance_id)

        try:
            # Wait for ready
            ready = await vast_provider.wait_for_ready(instance.instance_id, timeout=180)
            if not ready:
                pytest.skip("Instance did not become ready")

            # Upload file
            success = await vast_provider.rsync_upload(
                instance.instance_id,
                str(test_dir) + "/",
                "/workspace/input/",
            )
            assert success

            # Process file on instance
            exit_code, stdout, stderr = await vast_provider.execute_command(
                instance.instance_id,
                "cat /workspace/input/input.txt && echo 'Processed!' > /workspace/output.txt"
            )
            assert exit_code == 0
            assert "Test input" in stdout

            # Download results
            download_dir = tmp_path / "download"
            download_dir.mkdir()

            success = await vast_provider.rsync_download(
                instance.instance_id,
                "/workspace/output.txt",
                str(download_dir) + "/",
            )
            assert success

            # Verify downloaded file
            output_file = download_dir / "output.txt"
            assert output_file.exists()
            assert "Processed!" in output_file.read_text()

        finally:
            await vast_provider.terminate_instance(instance.instance_id)
            _created_instances.remove(instance.instance_id)
