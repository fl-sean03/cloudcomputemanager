"""REST API for CloudComputeManager.

Provides a FastAPI-based REST API for programmatic access to all CloudComputeManager features.
"""

from cloudcomputemanager.api.app import create_app

__all__ = ["create_app"]
