"""Registry Configuration Interface and Shared Definitions.

This module defines the abstract interface that all applications must implement
to provide their registry configuration, plus all shared dataclass definitions
used by both framework and applications. It serves as the foundation for the
registry system's type-safe, convention-based component registration.

The module provides two main categories of functionality:

1. **Registration Dataclasses**: Structured metadata definitions for lazy loading
   of all component types (capabilities, nodes, context classes, data sources, services)
2. **Configuration Interface**: Abstract base class that applications implement
   to provide their registry configuration in a standardized way

This design eliminates naming ambiguity, provides complete type safety, and ensures
zero backwards compatibility issues when the registry system evolves. Applications
only need to implement the RegistryConfigProvider interface and define their
components using the provided registration dataclasses.

Architecture Benefits:
    - **Type Safety**: All component metadata is strongly typed
    - **Lazy Loading**: Components are loaded only when needed, preventing circular imports
    - **Convention-Based**: Standardized patterns reduce configuration complexity
    - **Extensible**: New component types can be added without breaking existing applications
    - **Validation**: Built-in validation ensures configuration consistency

The registration system supports sophisticated component relationships through
the provides/requires pattern, enabling automatic dependency resolution and
validation at runtime.

.. note::
   All registration dataclasses use module_path and class_name for lazy loading,
   ensuring components are imported only during registry initialization, not at
   module load time.

.. warning::
   Applications must implement exactly one RegistryConfigProvider class in their
   registry module. Multiple implementations or missing implementations will cause
   RegistryError during initialization.

Examples:
    Basic application registry implementation::

        >>> from osprey.registry.base import RegistryConfigProvider, RegistryConfig
        >>> from osprey.registry.base import CapabilityRegistration, ContextClassRegistration
        >>>
        >>> class MyAppRegistryProvider(RegistryConfigProvider):
        ...     def get_registry_config(self) -> RegistryConfig:
        ...         return RegistryConfig(
        ...             capabilities=[
        ...                 CapabilityRegistration(
        ...                     name="data_analysis",
        ...                     module_path="applications.myapp.capabilities.data_analysis",
        ...                     class_name="DataAnalysisCapability",
        ...                     description="Analyze scientific data sets",
        ...                     provides=["ANALYSIS_RESULTS"],
        ...                     requires=["DATA_INPUT"]
        ...                 )
        ...             ],
        ...             context_classes=[
        ...                 ContextClassRegistration(
        ...                     context_type="ANALYSIS_RESULTS",
        ...                     module_path="applications.myapp.context_classes",
        ...                     class_name="AnalysisResults"
        ...                 )
        ...             ]
        ...         )

    Service registration with internal nodes::

        >>> ServiceRegistration(
        ...     name="data_processor",
        ...     module_path="applications.myapp.services.data_processor",
        ...     class_name="DataProcessorService",
        ...     description="Multi-step data processing service",
        ...     provides=["PROCESSED_DATA"],
        ...     requires=["RAW_DATA"],
        ...     internal_nodes=["validator", "transformer", "aggregator"]
        ... )

.. seealso::
   :class:`RegistryManager` : Registry manager that uses these definitions
   :mod:`osprey.registry.registry` : Framework registry provider implementation
   :doc:`/developer-guides/registry-system` : Complete registry system guide
   :doc:`/developer-guides/03_core-framework-systems/03_registry-and-discovery` : Component registration patterns
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# =============================================================================
# SHARED DATACLASS DEFINITIONS - Used by both framework and applications
# =============================================================================

@dataclass
class NodeRegistration:
    """Registration metadata for infrastructure node functions.

    Defines the metadata required for lazy loading of functional nodes.

    :param name: Unique identifier for the node in the registry
    :type name: str
    :param module_path: Python module path for lazy import
    :type module_path: str
    :param function_name: Function name within the module (decorated with @infrastructure_node)
    :type function_name: str
    :param description: Human-readable description of node functionality
    :type description: str
    """
    name: str
    module_path: str
    function_name: str
    description: str

@dataclass
class CapabilityRegistration:
    """Registration metadata for capabilities.

    Defines the metadata required for lazy loading of capability classes that
    implement specific functionality for agent systems. Enhanced for LangGraph
    migration with support for convention-based decorators and advanced features.

    :param name: Unique capability name for registration
    :type name: str
    :param module_path: Python module path for lazy import
    :type module_path: str
    :param class_name: Class name within the module
    :type class_name: str
    :param description: Human-readable description of capability
    :type description: str
    :param provides: List of context types this capability produces
    :type provides: list[str]
    :param requires: List of context types this capability needs
    :type requires: list[str]
    :param always_active: Whether capability is always active (no classification needed), defaults to False
    :type always_active: bool
    :param functional_node: Name of the functional node for execution (from capability.node attribute)
    :type functional_node: str
    :param example_usage: Example of how this capability is used
    :type example_usage: str

    """
    name: str  # Unique name like "pv_address_finding"
    module_path: str  # Like "applications.als_assistant.capabilities.pv_address_finding"
    class_name: str  # Like "PVAddressFindingCapability"
    description: str  # Human-readable description
    provides: list[str]  # Context types this capability produces
    requires: list[str]  # Context types this capability needs as input
    always_active: bool = False  # Whether capability is always active (no classification needed)
    functional_node: str = None  # Name of functional node (from capability.node attribute)
    example_usage: str = ""  # Example of how this capability is used

@dataclass
class ContextClassRegistration:
    """Registration metadata for context data classes.

    Defines the metadata required for lazy loading of context classes that
    represent structured data passed between capabilities.

    :param context_type: String identifier for the context type (e.g., 'PV_ADDRESSES')
    :type context_type: str
    :param module_path: Python module path for lazy import
    :type module_path: str
    :param class_name: Class name within the module
    :type class_name: str
    """
    context_type: str  # String constant like "PV_ADDRESSES"
    module_path: str  # Like "applications.als_assistant.context_classes"
    class_name: str  # Like "PVAddresses"

@dataclass
class DataSourceRegistration:
    """Registration metadata for external data source providers.

    Defines the metadata required for lazy loading of data source provider classes
    that provide access to external systems and databases.

    :param name: Unique identifier for the data source in the registry
    :type name: str
    :param module_path: Python module path for lazy import
    :type module_path: str
    :param class_name: Class name within the module
    :type class_name: str
    :param description: Human-readable description of data source
    :type description: str
    :param health_check_required: Whether provider requires health checking
    :type health_check_required: bool
    """
    name: str
    module_path: str  # Module path for lazy loading
    class_name: str   # Class name for lazy loading
    description: str
    health_check_required: bool = True

@dataclass
class ExecutionPolicyAnalyzerRegistration:
    """Registration metadata for configurable execution policy analyzers.

    Defines the metadata required for lazy loading of execution policy analyzer classes
    that make execution mode and approval decisions based on code analysis.

    :param name: Unique identifier for the policy analyzer in the registry
    :type name: str
    :param module_path: Python module path for lazy import
    :type module_path: str
    :param class_name: Class name within the module
    :type class_name: str
    :param description: Human-readable description of policy analyzer
    :type description: str
    :param priority: Analysis priority (lower numbers = higher priority)
    :type priority: int
    """
    name: str
    module_path: str  # Module path for lazy loading
    class_name: str   # Class name for lazy loading
    description: str
    priority: int = 50  # Default priority

@dataclass
class DomainAnalyzerRegistration:
    """Registration metadata for configurable domain analyzers.

    Defines the metadata required for lazy loading of domain analyzer classes
    that analyze generated code for domain-specific patterns and operations.

    :param name: Unique identifier for the domain analyzer in the registry
    :type name: str
    :param module_path: Python module path for lazy import
    :type module_path: str
    :param class_name: Class name within the module
    :type class_name: str
    :param description: Human-readable description of domain analyzer
    :type description: str
    :param priority: Analysis priority (lower numbers = higher priority)
    :type priority: int
    """
    name: str
    module_path: str  # Module path for lazy loading
    class_name: str   # Class name for lazy loading
    description: str
    priority: int = 50  # Default priority

@dataclass
class FrameworkPromptProviderRegistration:
    """Registration metadata for prompt provider customization.

    Defines which prompt builders to override from framework defaults.
    Uses the "selective override" pattern - only specify what you want to customize,
    everything else uses framework defaults automatically.

    :param module_path: Python module path containing your custom builder classes
    :type module_path: str
    :param prompt_builders: Mapping of prompt types to your custom builder class names
    :type prompt_builders: dict[str, str]
    :param application_name: (Deprecated) Application identifier - no longer used
    :type application_name: str | None
    :param description: (Deprecated) Human-readable description - no longer used
    :type description: str | None

    Examples:
        Override task extraction only::

            FrameworkPromptProviderRegistration(
                module_path="weather_agent.framework_prompts",
                prompt_builders={
                    "task_extraction": "WeatherTaskExtractionPromptBuilder"
                }
            )

        Override multiple prompts::

            FrameworkPromptProviderRegistration(
                module_path="als_assistant.framework_prompts",
                prompt_builders={
                    "orchestrator": "ALSOrchestratorPromptBuilder",
                    "task_extraction": "ALSTaskExtractionPromptBuilder",
                    "memory_extraction": "ALSMemoryExtractionPromptBuilder"
                }
            )
    """
    module_path: str      # Module path containing custom builders
    prompt_builders: dict[str, str] = field(default_factory=dict)  # prompt_type -> class_name

    # Deprecated fields (kept for backward compatibility)
    application_name: str | None = None
    description: str | None = None

    def __post_init__(self):
        """Emit deprecation warnings for removed fields."""
        import warnings

        if self.application_name is not None:
            warnings.warn(
                "The 'application_name' parameter in FrameworkPromptProviderRegistration is deprecated "
                "and will be removed in v0.10. It is no longer used by the framework. "
                "Please remove it from your registry configuration.",
                DeprecationWarning,
                stacklevel=2
            )

        if self.description is not None:
            warnings.warn(
                "The 'description' parameter in FrameworkPromptProviderRegistration is deprecated "
                "and will be removed in v0.10. It is no longer used by the framework. "
                "Please remove it from your registry configuration.",
                DeprecationWarning,
                stacklevel=2
            )

@dataclass
class ServiceRegistration:
    """Registration metadata for internal service graphs.

    Services are separate LangGraph graphs that can be called by capabilities
    without interfering with the main graph routing. Each service manages its
    own internal node flow and returns control to the calling capability.

    :param name: Unique identifier for the service in the registry
    :type name: str
    :param module_path: Python module path for lazy import
    :type module_path: str
    :param class_name: Service class name within the module
    :type class_name: str
    :param description: Human-readable description of service functionality
    :type description: str
    :param provides: List of context types this service produces
    :type provides: list[str]
    :param requires: List of context types this service needs
    :type requires: list[str]
    :param internal_nodes: List of node names internal to this service
    :type internal_nodes: list[str]
    """
    name: str
    module_path: str
    class_name: str
    description: str
    provides: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    internal_nodes: list[str] = field(default_factory=list)

@dataclass
class ProviderRegistration:
    """Minimal registration for lazy loading AI model providers.

    Provider metadata (requires_api_key, supports_proxy, etc.) is defined as
    class attributes on the provider class itself. The registry introspects
    these attributes after loading the class, following the same pattern as
    capabilities and context classes.

    This avoids metadata duplication between registration and class definition,
    maintaining a single source of truth on the provider class.

    :param module_path: Python module path for lazy import (e.g., 'osprey.models.providers.anthropic')
    :type module_path: str
    :param class_name: Provider class name within the module (e.g., 'AnthropicProviderAdapter')
    :type class_name: str

    Example:
        >>> ProviderRegistration(
        ...     module_path="osprey.models.providers.anthropic",
        ...     class_name="AnthropicProviderAdapter"
        ... )

        The registry will load this class and introspect metadata like:
        - AnthropicProviderAdapter.name
        - AnthropicProviderAdapter.requires_api_key
        - AnthropicProviderAdapter.supports_proxy
        etc.
    """
    module_path: str
    class_name: str

@dataclass
class ConnectorRegistration:
    """Registration metadata for control system and archiver connectors.

    Defines the metadata required for lazy loading of connector classes.
    Connectors are registered with the ConnectorFactory during registry
    initialization, providing unified management of all framework components.

    :param name: Unique connector name (e.g., 'epics', 'tango', 'mock')
    :type name: str
    :param connector_type: Type of connector ('control_system' or 'archiver')
    :type connector_type: str
    :param module_path: Python module path for lazy import
    :type module_path: str
    :param class_name: Connector class name within the module
    :type class_name: str
    :param description: Human-readable description
    :type description: str

    Examples:
        Control system connector registration::

            >>> ConnectorRegistration(
            ...     name="labview",
            ...     connector_type="control_system",
            ...     module_path="my_app.connectors.labview_connector",
            ...     class_name="LabVIEWConnector",
            ...     description="LabVIEW Web Services connector for NI systems"
            ... )

        Archiver connector registration::

            >>> ConnectorRegistration(
            ...     name="custom_archiver",
            ...     connector_type="archiver",
            ...     module_path="my_app.connectors.custom_archiver",
            ...     class_name="CustomArchiverConnector",
            ...     description="Custom facility archiver connector"
            ... )

    .. note::
       The connector classes are registered with ConnectorFactory during
       registry initialization, enabling lazy loading while maintaining
       the factory pattern for runtime connector creation.

    .. seealso::
       :class:`osprey.connectors.factory.ConnectorFactory` : Runtime connector factory
       :class:`osprey.connectors.control_system.base.ControlSystemConnector` : Base class for control system connectors
       :class:`osprey.connectors.archiver.base.ArchiverConnector` : Base class for archiver connectors
    """
    name: str
    connector_type: str  # 'control_system' or 'archiver'
    module_path: str
    class_name: str
    description: str

@dataclass
class RegistryConfig:
    """Complete registry configuration with all component metadata.

    Contains the complete configuration for the framework registry including
    all component registrations and initialization ordering. Enhanced for
    LangGraph migration with support for decorators and advanced features.

    Most fields are optional with sensible defaults to improve UX for applications.
    Applications typically only need to define capabilities, context_classes, and
    optionally data_sources and framework_prompt_providers.

    Registry Modes:
        **Standalone Mode (RegistryConfig)**:
            Direct use of RegistryConfig indicates a complete, self-contained registry.
            Framework registry is NOT loaded. Application must provide ALL components
            including core framework nodes, capabilities, and services.

        **Extend Mode (ExtendedRegistryConfig)**:
            Use of ExtendedRegistryConfig (via extend_framework_registry() helper)
            indicates registry extends framework defaults. Framework components are
            loaded first, then application components are merged.

    :param capabilities: Registration entries for domain capabilities
    :type capabilities: list[CapabilityRegistration]
    :param context_classes: Registration entries for context data classes
    :type context_classes: list[ContextClassRegistration]
    :param core_nodes: Registration entries for infrastructure nodes (optional)
    :type core_nodes: list[NodeRegistration]
    :param data_sources: Registration entries for external data sources (optional)
    :type data_sources: list[DataSourceRegistration]
    :param services: Registration entries for internal service graphs (optional)
    :type services: list[ServiceRegistration]
    :param domain_analyzers: Registration entries for domain analyzers (optional)
    :type domain_analyzers: list[DomainAnalyzerRegistration]
    :param execution_policy_analyzers: Registration entries for execution policy analyzers (optional)
    :type execution_policy_analyzers: list[ExecutionPolicyAnalyzerRegistration]
    :param framework_prompt_providers: Registration entries for prompt providers (optional)
    :type framework_prompt_providers: list[FrameworkPromptProviderRegistration]
    :param providers: Registration entries for AI model providers (optional)
    :type providers: list[ProviderRegistration]
    :param connectors: Registration entries for control system and archiver connectors (optional)
    :type connectors: list[ConnectorRegistration]
    :param framework_exclusions: Framework component names to exclude by type (optional)
    :type framework_exclusions: dict[str, list[str]]
    :param initialization_order: Component type initialization sequence (optional)
    :type initialization_order: list[str]
    """
    # Required fields (what applications typically define)
    capabilities: list[CapabilityRegistration]
    context_classes: list[ContextClassRegistration]

    # Optional fields with sensible defaults (mostly for framework)
    core_nodes: list[NodeRegistration] = field(default_factory=list)
    data_sources: list[DataSourceRegistration] = field(default_factory=list)
    services: list[ServiceRegistration] = field(default_factory=list)
    domain_analyzers: list[DomainAnalyzerRegistration] = field(default_factory=list)
    execution_policy_analyzers: list[ExecutionPolicyAnalyzerRegistration] = field(default_factory=list)
    framework_prompt_providers: list[FrameworkPromptProviderRegistration] = field(default_factory=list)
    providers: list[ProviderRegistration] = field(default_factory=list)
    connectors: list[ConnectorRegistration] = field(default_factory=list)
    framework_exclusions: dict[str, list[str]] = field(default_factory=dict)
    initialization_order: list[str] = field(default_factory=lambda: [
        "context_classes",
        "data_sources",
        "domain_analyzers",
        "execution_policy_analyzers",
        "providers",
        "connectors",
        "capabilities",
        "framework_prompt_providers",
        "core_nodes",
        "services"
    ])


@dataclass
class ExtendedRegistryConfig(RegistryConfig):
    """Registry configuration that extends framework defaults (marker subclass).

    This is a marker subclass of RegistryConfig used to indicate that this
    registry should be merged with framework defaults rather than used standalone.

    The extend_framework_registry() helper returns this type to signal extend mode.
    The registry manager detects this type and merges with framework components.

    All fields and behavior are identical to RegistryConfig - this class exists
    purely to distinguish extend mode from standalone mode at the type level.

    Examples:
        Extend mode (returned by helper)::

            >>> config = extend_framework_registry(
            ...     capabilities=[...],
            ...     context_classes=[...]
            ... )
            >>> isinstance(config, ExtendedRegistryConfig)  # True
            >>> isinstance(config, RegistryConfig)  # Also True (inheritance)

        Standalone mode (direct construction)::

            >>> config = RegistryConfig(
            ...     capabilities=[...],
            ...     context_classes=[...],
            ...     core_nodes=[...],  # Must provide ALL framework components
            ...     ...
            ... )
            >>> isinstance(config, ExtendedRegistryConfig)  # False
            >>> isinstance(config, RegistryConfig)  # True

    .. seealso::
       :func:`extend_framework_registry` : Helper that returns this type
       :class:`RegistryConfig` : Base configuration class
    """
    pass  # Marker class - inherits all fields and behavior from RegistryConfig

# =============================================================================
# REGISTRY CONFIGURATION INTERFACE
# =============================================================================

class RegistryConfigProvider(ABC):
    """Abstract interface for application registry configuration.

    All applications must implement this interface to provide their registry
    configuration in a standardized, type-safe manner. This interface eliminates
    naming ambiguity and ensures consistent registry patterns across all applications
    in the framework ecosystem.

    The framework loads classes implementing this interface from each configured
    application's registry module using the convention-based path:
    applications.{app_name}.registry

    This design provides several key benefits:

    - **Type Safety**: All registry configurations are strongly typed
    - **Convention-Based Loading**: Standardized module path patterns
    - **Validation**: Automatic validation of registry structure
    - **Extensibility**: Easy to add new component types without breaking changes
    - **Documentation**: Self-documenting registry configurations

    Implementation Requirements:
        1. Exactly one class per application registry module must implement this interface
        2. The get_registry_config() method must return a complete RegistryConfig
        3. All component registrations must use valid module paths and class names
        4. Context type dependencies (provides/requires) must be consistent

    The registry system will call get_registry_config() once during initialization
    and merge the returned configuration with the framework's base registry.
    Applications can override framework components by using the same names.

    :raises RegistryError: If multiple implementations found in one module
    :raises RegistryError: If no implementation found in application registry module
    :raises NotImplementedError: If get_registry_config() is not implemented

    Examples:
        Basic application registry::

            >>> class MyAppRegistryProvider(RegistryConfigProvider):
            ...     def get_registry_config(self) -> RegistryConfig:
            ...         return RegistryConfig(
            ...             capabilities=[
            ...                 CapabilityRegistration(
            ...                     name="my_capability",
            ...                     module_path="applications.myapp.capabilities.my_capability",
            ...                     class_name="MyCapability",
            ...                     description="Application-specific functionality",
            ...                     provides=["MY_RESULTS"],
            ...                     requires=["INPUT_DATA"]
            ...                 )
            ...             ],
            ...             context_classes=[
            ...                 ContextClassRegistration(
            ...                     context_type="MY_RESULTS",
            ...                     module_path="applications.myapp.context_classes",
            ...                     class_name="MyResults"
            ...                 )
            ...             ]
            ...         )

        Advanced registry with services and data sources::

            >>> class AdvancedAppRegistryProvider(RegistryConfigProvider):
            ...     def get_registry_config(self) -> RegistryConfig:
            ...         return RegistryConfig(
            ...             capabilities=[...],
            ...             context_classes=[...],
            ...             data_sources=[
            ...                 DataSourceRegistration(
            ...                     name="app_database",
            ...                     module_path="applications.myapp.data_sources.database",
            ...                     class_name="AppDatabaseProvider",
            ...                     description="Application-specific database access",
            ...                     health_check_required=True
            ...                 )
            ...             ],
            ...             services=[
            ...                 ServiceRegistration(
            ...                     name="data_processor",
            ...                     module_path="applications.myapp.services.processor",
            ...                     class_name="DataProcessorService",
            ...                     description="Multi-step data processing workflow",
            ...                     provides=["PROCESSED_DATA"],
            ...                     requires=["RAW_DATA"],
            ...                     internal_nodes=["validator", "transformer", "aggregator"]
            ...                 )
            ...             ]
            ...         )

    .. note::
       The framework will merge application registries with the base framework registry,
       allowing applications to override framework components by using the same names.

    .. warning::
       Each application registry module must contain exactly one class implementing
       this interface. Multiple implementations or missing implementations will cause
       RegistryError during framework initialization.

    .. seealso::
       :class:`RegistryConfig` : Configuration structure returned by implementations
       :class:`RegistryManager` : Manager that discovers and uses these providers
       :doc:`/developer-guides/02_quick-start-patterns/01_building-your-first-capability` : Complete application development guide
    """

    @abstractmethod
    def get_registry_config(self) -> RegistryConfig:
        """Get complete application registry configuration.

        This method is called once during registry initialization to retrieve
        the complete component configuration for the application. The returned
        RegistryConfig will be merged with the framework's base registry,
        allowing applications to extend or override framework functionality.

        The configuration should include all components that the application
        provides: capabilities, context classes, data sources, services, and
        any other registered components. Components are loaded lazily using
        the module_path and class_name metadata provided in the registrations.

        Implementation Guidelines:
            - Return a complete RegistryConfig with all application components
            - Use descriptive names and clear documentation for all components
            - Ensure provides/requires relationships are consistent across components
            - Validate that all module paths and class names are correct
            - Consider component initialization order when defining dependencies

        :return: Complete registry configuration for this application including
            all capabilities, context classes, data sources, services, and other
            components that should be available in the framework
        :rtype: RegistryConfig
        :raises NotImplementedError: If not implemented by subclass (required by ABC)
        :raises ImportError: If any component module paths are invalid
        :raises AttributeError: If any component class names are not found

        .. note::
           This method is called exactly once during registry initialization.
           The framework caches the returned configuration, so dynamic changes
           after initialization are not supported.

        .. warning::
           All module paths must be importable and all class names must exist
           in their respective modules. Invalid paths will cause registry
           initialization to fail.

        Examples:
            Minimal application configuration::

                >>> def get_registry_config(self) -> RegistryConfig:
                ...     return RegistryConfig(
                ...         capabilities=[
                ...             CapabilityRegistration(
                ...                 name="hello_world",
                ...                 module_path="applications.myapp.capabilities.hello",
                ...                 class_name="HelloWorldCapability",
                ...                 description="Simple greeting capability",
                ...                 provides=["GREETING"],
                ...                 requires=[]
                ...             )
                ...         ],
                ...         context_classes=[
                ...             ContextClassRegistration(
                ...                 context_type="GREETING",
                ...                 module_path="applications.myapp.context_classes",
                ...                 class_name="GreetingContext"
                ...             )
                ...         ]
                ...     )

            Complex application with all component types::

                >>> def get_registry_config(self) -> RegistryConfig:
                ...     return RegistryConfig(
                ...         capabilities=[...],  # Domain-specific capabilities
                ...         context_classes=[...],  # Data structures
                ...         data_sources=[...],  # External data providers
                ...         services=[...],  # Internal service graphs
                ...         framework_prompt_providers=[...]  # Custom prompts
                ...     )

        .. seealso::
           :class:`RegistryConfig` : Structure of the returned configuration
           :class:`CapabilityRegistration` : Capability registration metadata
           :class:`ContextClassRegistration` : Context class registration metadata
           :doc:`/developer-guides/03_core-framework-systems/03_registry-and-discovery` : Component registration guide
        """
        # This should never be called due to @abstractmethod, but provide helpful error
        # in case someone bypasses the abstract enforcement or has implementation issues
        class_name = self.__class__.__name__

        raise NotImplementedError(
            f"{class_name} must implement get_registry_config() method. "
            f"This abstract method is required by RegistryConfigProvider interface.\n\n"
            f"Expected implementation:\n"
            f"class {class_name}(RegistryConfigProvider):\n"
            f"    def get_registry_config(self) -> RegistryConfig:\n"
            f"        return RegistryConfig(\n"
            f"            capabilities=[...],\n"
            f"            context_classes=[...]\n"
            f"        )"
        )
