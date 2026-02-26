"""Code generation utilities for Osprey Framework.

This package provides generators for creating Osprey components and
simulation backends.

Architecture:
- config_updater: Auto-update config files
- backend_protocol: Protocol for custom simulation backends
- ioc_backends: Runtime IOC backend implementations
- soft_ioc_template: Soft IOC code generation
"""

from . import config_updater
from .backend_protocol import SimulationBackend
from .ioc_backends import (
    ChainedBackend,
    MockStyleBackend,
    PassthroughBackend,
    load_backends_from_config,
)

__all__ = [
    # Simulation Backend Protocol
    "SimulationBackend",
    # IOC Backends
    "ChainedBackend",
    "MockStyleBackend",
    "PassthroughBackend",
    "load_backends_from_config",
    # Utilities
    "config_updater",
]
