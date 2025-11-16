"""Registry System for Osprey Components.

This module provides the complete registry system for the Osprey Agentic Framework,
enabling centralized management of all system components including capabilities, nodes,
context classes, data sources, and services. The registry system eliminates circular
imports through lazy loading and provides dependency-ordered initialization.

The registry follows a convention-based approach where applications define their
components through standardized registration classes, and the framework automatically
discovers and integrates them at runtime. This provides clean separation between
framework infrastructure and application-specific functionality.

Key Components:
    - **RegistryManager**: Central registry for all framework components
    - **Registration Classes**: Metadata definitions for component lazy loading
    - **RegistryConfigProvider**: Interface for application registry implementations
    - **Global Functions**: Singleton access and initialization utilities

Architecture Overview:
    The registry system uses a two-tier architecture:

    1. **Framework Registry**: Core infrastructure components (nodes, base capabilities)
    2. **Application Registries**: Domain-specific components (capabilities, context classes)

    Applications register components by implementing RegistryConfigProvider in their
    registry module (applications.{app_name}.registry), which the framework loads
    using convention-based patterns after applications are listed in configuration.

Initialization Order:
    Components are loaded in dependency order to handle inter-component relationships:

    1. Context classes (required by capabilities)
    2. Data sources (required by capabilities)
    3. Core nodes (infrastructure components)
    4. Services (internal LangGraph service graphs)
    5. Capabilities (domain-specific functionality)
    6. Framework prompt providers (application-specific prompts)

.. note::
   The registry uses lazy loading to prevent circular import issues. Components are
   imported and instantiated only during the initialization phase, not at module load time.

.. warning::
   Registry initialization must complete successfully before any components can be accessed.
   Failed initialization will raise RegistryError and prevent framework operation.

Examples:
    Basic registry usage::

        >>> from osprey.registry import initialize_registry, get_registry
        >>>
        >>> # Initialize the registry system
        >>> initialize_registry()
        >>>
        >>> # Access components
        >>> registry = get_registry()
        >>> capability = registry.get_capability("pv_address_finding")
        >>> context_class = registry.get_context_class("PV_ADDRESSES")

    Application registry implementation::

        >>> from osprey.registry import RegistryConfigProvider, RegistryConfig
        >>> from osprey.registry import CapabilityRegistration, ContextClassRegistration
        >>>
        >>> class MyAppRegistryProvider(RegistryConfigProvider):
        ...     def get_registry_config(self) -> RegistryConfig:
        ...         return RegistryConfig(
        ...             capabilities=[
        ...                 CapabilityRegistration(
        ...                     name="my_capability",
        ...                     module_path="applications.myapp.capabilities.my_capability",
        ...                     class_name="MyCapability",
        ...                     description="Custom application capability",
        ...                     provides=["MY_CONTEXT"],
        ...                     requires=[]
        ...                 )
        ...             ],
        ...             context_classes=[
        ...                 ContextClassRegistration(
        ...                     context_type="MY_CONTEXT",
        ...                     module_path="applications.myapp.context_classes",
        ...                     class_name="MyContext"
        ...                 )
        ...             ]
        ...         )

.. seealso::
   :class:`RegistryManager` : Central registry management and component access
   :class:`RegistryConfigProvider` : Interface for application registry implementations
   :func:`initialize_registry` : Registry system initialization
   :func:`get_registry` : Singleton registry access
   :doc:`/developer-guides/registry-system` : Complete registry system documentation
"""

# Core registry system
# Framework components for application use - all shared definitions in base
from .base import (
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
from .manager import RegistryManager, get_registry, initialize_registry, registry, reset_registry

__all__ = [
    # Core registry system
    'RegistryManager',
    'get_registry',
    'initialize_registry',
    'reset_registry',
    'registry',

    # Configuration classes for applications
    'RegistryConfigProvider',
    'NodeRegistration',
    'CapabilityRegistration',
    'ContextClassRegistration',
    'DataSourceRegistration',
    'ServiceRegistration',
    'FrameworkPromptProviderRegistration',
    'ProviderRegistration',
    'ConnectorRegistration',
    'RegistryConfig',
    'ExtendedRegistryConfig',

    # Helper functions (NEW - Phase 4.6)
    'extend_framework_registry',
    'get_framework_defaults',
    'generate_explicit_registry_code',
]
