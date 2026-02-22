"""Code generation utilities for Osprey Framework.

This package provides generators for creating Osprey components and
simulation backends.

Architecture:
- models: Shared Pydantic models for LLM analysis
- registry_updater: Auto-register generated capabilities
- config_updater: Auto-update config files
- backend_protocol: Protocol for custom simulation backends
- ioc_backends: Runtime IOC backend implementations
- soft_ioc_template: Soft IOC code generation
"""

from . import config_updater, registry_updater
from .backend_protocol import SimulationBackend
from .ioc_backends import (
    ChainedBackend,
    MockStyleBackend,
    PassthroughBackend,
    load_backends_from_config,
)
from .models import (
    CapabilityMetadata,
    ClassifierAnalysis,
    ClassifierExampleRaw,
    ExampleStepRaw,
    OrchestratorAnalysis,
    ToolPattern,
)

__all__ = [
    # Models
    "CapabilityMetadata",
    "ClassifierAnalysis",
    "ClassifierExampleRaw",
    "ExampleStepRaw",
    "OrchestratorAnalysis",
    "ToolPattern",
    # Simulation Backend Protocol
    "SimulationBackend",
    # IOC Backends
    "ChainedBackend",
    "MockStyleBackend",
    "PassthroughBackend",
    "load_backends_from_config",
    # Utilities
    "registry_updater",
    "config_updater",
]
