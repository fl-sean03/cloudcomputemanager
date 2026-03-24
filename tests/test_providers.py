"""Unit tests for cloud providers.

These tests use mocking and don't require real API access.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from cloudcomputemanager.providers.vast import VastProvider
from cloudcomputemanager.providers.base import ProviderOffer


class TestVastProviderSearchOffers:
    """Test VastProvider.search_offers() parameter handling."""

    @pytest.fixture
    def provider(self):
        """Create a VastProvider instance."""
        return VastProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_search_offers_includes_cpu_cores_filter(self, provider):
        """Test that cpu_cores_min is included in search query."""
        mock_results = [
            {
                "id": 12345,
                "gpu_name": "RTX 3060",
                "num_gpus": 1,
                "gpu_ram": 12288,
                "cpu_cores_effective": 8,
                "cpu_ram": 32768,
                "disk_space": 100,
                "dph_total": 0.10,
                "geolocation": "US",
                "reliability2": 0.95,
                "cuda_max_good": 12.0,
                "driver_version": "535.0",
            }
        ]

        with patch.object(provider, '_run_vastai_cmd', new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = mock_results

            offers = await provider.search_offers(
                gpu_type="RTX_3060",
                cpu_cores_min=6,
            )

            # Check that the command was called with cpu_cores_effective filter
            call_args = mock_cmd.call_args
            query = call_args[0][2]  # Third positional argument is the query
            assert "cpu_cores_effective>=6" in query

    @pytest.mark.asyncio
    async def test_search_offers_without_cpu_filter(self, provider):
        """Test that cpu_cores filter is omitted when not specified."""
        mock_results = []

        with patch.object(provider, '_run_vastai_cmd', new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = mock_results

            await provider.search_offers(gpu_type="RTX_3060")

            call_args = mock_cmd.call_args
            query = call_args[0][2]
            assert "cpu_cores_effective" not in query

    @pytest.mark.asyncio
    async def test_search_offers_normalizes_gpu_name(self, provider):
        """Test that GPU names with spaces are normalized to underscores."""
        mock_results = []

        with patch.object(provider, '_run_vastai_cmd', new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = mock_results

            await provider.search_offers(gpu_type="RTX 4090")  # Space

            call_args = mock_cmd.call_args
            query = call_args[0][2]
            assert "gpu_name=RTX_4090" in query  # Underscore

    @pytest.mark.asyncio
    async def test_search_offers_returns_cpu_cores(self, provider):
        """Test that returned offers include cpu_cores from results."""
        mock_results = [
            {
                "id": 12345,
                "gpu_name": "RTX 3060",
                "num_gpus": 1,
                "gpu_ram": 12288,
                "cpu_cores_effective": 8,
                "cpu_ram": 32768,
                "disk_space": 100,
                "dph_total": 0.10,
                "geolocation": "US",
                "reliability2": 0.95,
                "cuda_max_good": 12.0,
                "driver_version": "535.0",
            }
        ]

        with patch.object(provider, '_run_vastai_cmd', new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = mock_results

            offers = await provider.search_offers(gpu_type="RTX_3060")

            assert len(offers) == 1
            assert offers[0].cpu_cores == 8


class TestProviderOfferModel:
    """Test ProviderOffer model."""

    def test_offer_has_cpu_cores(self):
        """Test that ProviderOffer includes cpu_cores field."""
        offer = ProviderOffer(
            offer_id="123",
            provider="vast",
            gpu_type="RTX 3060",
            gpu_count=1,
            gpu_memory_gb=12,
            cpu_cores=8,
            memory_gb=32,
            disk_gb=100,
            hourly_rate=0.10,
            location="US",
            reliability_score=0.95,
            interruptible=True,
        )

        assert offer.cpu_cores == 8
