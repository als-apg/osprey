"""Registration dataclasses and the ``RegistryConfigProvider`` ABC.

All component types (capabilities, nodes, context classes, data sources, services,
etc.) are represented as frozen metadata for lazy loading by the registry manager.

.. seealso:: :doc:`/developer-guides/registry-system`
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


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
    implement specific functionality for agent systems. Supports convention-based
    decorators and advanced features.

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

    name: str
    module_path: str
    class_name: str
    description: str
    provides: list[str]
    requires: list[str]
    always_active: bool = False
    functional_node: str = None
    example_usage: str = ""
    _is_explicit_override: bool = False


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

    context_type: str
    module_path: str
    class_name: str


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
    module_path: str
    class_name: str
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
    module_path: str
    class_name: str
    description: str
    priority: int = 50


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
    module_path: str
    class_name: str
    description: str
    priority: int = 50


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

    module_path: str
    prompt_builders: dict[str, str] = field(default_factory=dict)

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
                stacklevel=2,
            )

        if self.description is not None:
            warnings.warn(
                "The 'description' parameter in FrameworkPromptProviderRegistration is deprecated "
                "and will be removed in v0.10. It is no longer used by the framework. "
                "Please remove it from your registry configuration.",
                DeprecationWarning,
                stacklevel=2,
            )


@dataclass
class ServiceRegistration:
    """Registration metadata for internal service graphs.

    Services are separate graphs that can be called by capabilities
    without interfering with the main routing. Each service manages its
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
    name: str | None = None  # Provider name for config-driven filtering


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
    connector_type: str
    module_path: str
    class_name: str
    description: str


@dataclass
class CodeGeneratorRegistration:
    """Registration metadata for Python executor code generators.

    Defines the metadata required for lazy loading of code generator classes that
    implement the CodeGenerator protocol for Python code generation. This enables
    applications to register custom code generation strategies alongside framework
    defaults like the basic LLM generator and Claude Code SDK generator.

    :param name: Unique generator name (e.g., 'basic', 'claude_code', 'custom')
    :type name: str
    :param module_path: Python module path for lazy import
    :type module_path: str
    :param class_name: Generator class name within the module
    :type class_name: str
    :param description: Human-readable description of the generator
    :type description: str
    :param optional_dependencies: Optional list of package names required for this generator
    :type optional_dependencies: list[str]

    Examples:
        Framework code generator registration::

            >>> CodeGeneratorRegistration(
            ...     name="claude_code",
            ...     module_path="osprey.services.python_executor.claude_code_generator",
            ...     class_name="ClaudeCodeGenerator",
            ...     description="Claude Code SDK-based generator with multi-turn reasoning",
            ...     optional_dependencies=["claude-agent-sdk"]
            ... )

        Application custom generator registration::

            >>> CodeGeneratorRegistration(
            ...     name="domain_specific",
            ...     module_path="applications.myapp.generators.domain_generator",
            ...     class_name="DomainSpecificGenerator",
            ...     description="Domain-specific code generator with specialized templates"
            ... )

    .. note::
       Code generators must implement the CodeGenerator protocol with an async
       generate_code(request, error_chain) method. The factory will validate
       protocol compliance at runtime.

    .. note::
       Generators with optional_dependencies will be skipped if dependencies
       are not installed, allowing graceful degradation to fallback generators.

    .. seealso::
       :class:`osprey.services.python_executor.code_generator_interface.CodeGenerator` : Protocol all generators must implement
       :class:`osprey.services.python_executor.generator_factory.create_code_generator` : Factory that uses these registrations
       :class:`osprey.services.python_executor.basic_generator.BasicLLMCodeGenerator` : Built-in basic generator
       :class:`osprey.services.python_executor.claude_code_generator.ClaudeCodeGenerator` : Built-in Claude Code generator
    """

    name: str
    module_path: str
    class_name: str
    description: str
    optional_dependencies: list[str] = field(default_factory=list)


@dataclass
class ArielSearchModuleRegistration:
    """Registration metadata for ARIEL search modules.

    Defines the metadata required for lazy loading of search modules that provide
    tool descriptors for the ARIEL agent executor and capabilities API.

    :param name: Config key, e.g., "ariel_keyword"
    :type name: str
    :param module_path: Module exporting get_tool_descriptor()
    :type module_path: str
    :param description: Human-readable description
    :type description: str
    """

    name: str
    module_path: str
    description: str


@dataclass
class ArielEnhancementModuleRegistration:
    """Registration metadata for ARIEL enhancement modules.

    Defines the metadata required for lazy loading of enhancement module classes
    that process logbook entries (e.g., keyword extraction, embedding generation).

    :param name: Config key, e.g., "ariel_text_embedding"
    :type name: str
    :param module_path: Module containing the class
    :type module_path: str
    :param class_name: Enhancement module class name, e.g., "TextEmbeddingModule"
    :type class_name: str
    :param description: Human-readable description
    :type description: str
    :param execution_order: Lower = earlier execution (e.g., 10, 20)
    :type execution_order: int
    """

    name: str
    module_path: str
    class_name: str
    description: str
    execution_order: int = 50


@dataclass
class ArielIngestionAdapterRegistration:
    """Registration metadata for ARIEL ingestion adapters.

    Defines the metadata required for lazy loading of ingestion adapter classes
    that connect to facility-specific logbook data sources.

    :param name: Config key, e.g., "als_logbook"
    :type name: str
    :param module_path: Module containing the adapter class
    :type module_path: str
    :param class_name: Adapter class name, e.g., "ALSLogbookAdapter"
    :type class_name: str
    :param description: Human-readable description
    :type description: str
    """

    name: str
    module_path: str
    class_name: str
    description: str


@dataclass
class ArielPipelineRegistration:
    """Registration metadata for ARIEL pipeline descriptors.

    Defines the metadata required for lazy loading of pipeline descriptor modules
    that declare RAG or Agent execution strategies.

    :param name: Config key, e.g., "ariel_rag"
    :type name: str
    :param module_path: Module exporting get_pipeline_descriptor()
    :type module_path: str
    :param description: Human-readable description
    :type description: str
    :param category: Pipeline category ("llm" or "direct")
    :type category: str
    """

    name: str
    module_path: str
    description: str
    category: str = "llm"


@dataclass
class RegistryConfig:
    """Complete registry configuration with all component metadata.

    Contains the complete configuration for the framework registry including
    all component registrations and initialization ordering. Supports decorators
    and advanced features.

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
    :param code_generators: Registration entries for Python executor code generators (optional)
    :type code_generators: list[CodeGeneratorRegistration]
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
    execution_policy_analyzers: list[ExecutionPolicyAnalyzerRegistration] = field(
        default_factory=list
    )
    framework_prompt_providers: list[FrameworkPromptProviderRegistration] = field(
        default_factory=list
    )
    providers: list[ProviderRegistration] = field(default_factory=list)
    connectors: list[ConnectorRegistration] = field(default_factory=list)
    code_generators: list[CodeGeneratorRegistration] = field(default_factory=list)
    ariel_search_modules: list[ArielSearchModuleRegistration] = field(default_factory=list)
    ariel_enhancement_modules: list[ArielEnhancementModuleRegistration] = field(
        default_factory=list
    )
    ariel_pipelines: list[ArielPipelineRegistration] = field(default_factory=list)
    ariel_ingestion_adapters: list[ArielIngestionAdapterRegistration] = field(default_factory=list)
    framework_exclusions: dict[str, list[str]] = field(default_factory=dict)
    initialization_order: list[str] = field(
        default_factory=lambda: [
            "context_classes",
            "data_sources",
            "domain_analyzers",
            "execution_policy_analyzers",
            "providers",
            "connectors",
            "code_generators",
            "ariel_search_modules",
            "ariel_enhancement_modules",
            "ariel_pipelines",
            "ariel_ingestion_adapters",
            "capabilities",
            "framework_prompt_providers",
            "core_nodes",
            "services",
        ]
    )


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


class RegistryConfigProvider(ABC):
    """Abstract interface for application registry configuration.

    Each application registry module must contain exactly one subclass that
    implements :meth:`get_registry_config` and returns a :class:`RegistryConfig`
    (or :class:`ExtendedRegistryConfig` when extending framework defaults).

    The framework discovers implementations automatically from each configured
    application's registry module (``applications.{app_name}.registry``).

    .. seealso:: :class:`RegistryConfig`, :class:`RegistryManager`
    """

    @abstractmethod
    def get_registry_config(self) -> RegistryConfig:
        """Return the complete registry configuration for this application.

        Called once during registry initialization. The returned configuration
        is merged with the framework's base registry.

        :return: Registry configuration with all application components.
        :rtype: RegistryConfig
        """
        ...
