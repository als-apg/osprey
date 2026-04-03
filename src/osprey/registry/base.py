"""Registration dataclasses and the ``RegistryConfigProvider`` ABC.

All component types (services, providers, connectors, ARIEL modules, etc.)
are represented as frozen metadata for lazy loading by the registry manager.

.. seealso:: :doc:`/developer-guides/registry-system`
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


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
    """

    name: str
    module_path: str
    class_name: str
    description: str
    provides: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)


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
    all component registrations and initialization ordering.

    All fields are optional with sensible defaults.

    Registry Modes:
        **Standalone Mode (RegistryConfig)**:
            Direct use of RegistryConfig indicates a complete, self-contained registry.

        **Extend Mode (ExtendedRegistryConfig)**:
            Use of ExtendedRegistryConfig (via extend_framework_registry() helper)
            indicates registry extends framework defaults. Framework components are
            loaded first, then application components are merged.

    :param services: Registration entries for internal service graphs (optional)
    :type services: list[ServiceRegistration]
    :param providers: Registration entries for AI model providers (optional)
    :type providers: list[ProviderRegistration]
    :param connectors: Registration entries for control system and archiver connectors (optional)
    :type connectors: list[ConnectorRegistration]
    :param framework_exclusions: Framework component names to exclude by type (optional)
    :type framework_exclusions: dict[str, list[str]]
    :param initialization_order: Component type initialization sequence (optional)
    :type initialization_order: list[str]
    """

    services: list[ServiceRegistration] = field(default_factory=list)
    providers: list[ProviderRegistration] = field(default_factory=list)
    connectors: list[ConnectorRegistration] = field(default_factory=list)
    ariel_search_modules: list[ArielSearchModuleRegistration] = field(default_factory=list)
    ariel_enhancement_modules: list[ArielEnhancementModuleRegistration] = field(
        default_factory=list
    )
    ariel_pipelines: list[ArielPipelineRegistration] = field(default_factory=list)
    ariel_ingestion_adapters: list[ArielIngestionAdapterRegistration] = field(default_factory=list)
    framework_exclusions: dict[str, list[str]] = field(default_factory=dict)
    initialization_order: list[str] = field(
        default_factory=lambda: [
            "providers",
            "connectors",
            "ariel_search_modules",
            "ariel_enhancement_modules",
            "ariel_pipelines",
            "ariel_ingestion_adapters",
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

            >>> config = extend_framework_registry()
            >>> isinstance(config, ExtendedRegistryConfig)  # True
            >>> isinstance(config, RegistryConfig)  # Also True (inheritance)

        Standalone mode (direct construction)::

            >>> config = RegistryConfig()
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
