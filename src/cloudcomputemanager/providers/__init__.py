"""Cloud provider adapters for CloudComputeManager.

Currently supports:
- Vast.ai (via SkyPilot integration)
"""

from cloudcomputemanager.providers.base import CloudProvider, ProviderInstance, ProviderOffer
from cloudcomputemanager.providers.vast import VastProvider

__all__ = [
    "CloudProvider",
    "ProviderInstance",
    "ProviderOffer",
    "VastProvider",
]
