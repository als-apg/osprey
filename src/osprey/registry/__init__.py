"""Registry system for Osprey components.

Re-exports registration dataclasses, the configuration provider interface,
and the global registry singleton.

.. seealso:: :doc:`/developer-guides/registry-system`
"""

# Core registry system
# Framework components for application use - all shared definitions in base
from .base import (
    ArielEnhancementModuleRegistration,
    ArielIngestionAdapterRegistration,
    ArielPipelineRegistration,
    ArielSearchModuleRegistration,
    CapabilityRegistration,
    ConnectorRegistration,
    ContextClassRegistration,
    DataSourceRegistration,
    DomainAnalyzerRegistration,
    ExecutionPolicyAnalyzerRegistration,
    ExtendedRegistryConfig,
    FrameworkPromptProviderRegistration,
    NodeRegistration,
    ProviderRegistration,
    RegistryConfig,
    RegistryConfigProvider,
    ServiceRegistration,
)

# Helper functions for simplified registry creation
from .helpers import (
    extend_framework_registry,
    generate_explicit_registry_code,
    get_framework_defaults,
)
from .manager import RegistryManager, get_registry, initialize_registry, reset_registry

__all__ = [
    # Core registry system
    "RegistryManager",
    "get_registry",
    "initialize_registry",
    "reset_registry",
    # Configuration classes for applications
    "RegistryConfigProvider",
    "NodeRegistration",
    "CapabilityRegistration",
    "ContextClassRegistration",
    "DataSourceRegistration",
    "ServiceRegistration",
    "FrameworkPromptProviderRegistration",
    "ProviderRegistration",
    "ConnectorRegistration",
    "RegistryConfig",
    "ExtendedRegistryConfig",
    # ARIEL module registration types
    "ArielSearchModuleRegistration",
    "ArielEnhancementModuleRegistration",
    "ArielPipelineRegistration",
    "ArielIngestionAdapterRegistration",
    # Helper functions
    "extend_framework_registry",
    "get_framework_defaults",
    "generate_explicit_registry_code",
]
