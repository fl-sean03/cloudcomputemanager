"""PackStore: Pre-built scientific computing package management.

Provides validated, pre-built packages (LAMMPS, Quantum ESPRESSO, GROMACS, etc.)
that can be quickly deployed to GPU cloud instances.
"""

from cloudcomputemanager.packstore.registry import (
    PackageRegistry,
    Package,
    PackageVariant,
    PackageSource,
    SourceType,
    Compatibility,
)
from cloudcomputemanager.packstore.deployer import PackageDeployer, DeploymentStrategy
from cloudcomputemanager.packstore.detector import EnvironmentDetector, InstanceEnvironment

__all__ = [
    "PackageRegistry",
    "Package",
    "PackageVariant",
    "PackageSource",
    "SourceType",
    "Compatibility",
    "PackageDeployer",
    "DeploymentStrategy",
    "EnvironmentDetector",
    "InstanceEnvironment",
]
