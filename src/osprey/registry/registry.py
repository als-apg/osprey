"""Framework Registry Provider.

This module contains the framework's registry provider implementation,
following the same RegistryConfigProvider interface pattern used by all
applications. It defines the core infrastructure components that form
the foundation of the Osprey Agentic Framework.

The framework registry provides essential components that all applications
can build upon:

Core Infrastructure:
    - **Infrastructure Nodes**: Core routing, classification, orchestration, and error handling
    - **Framework Capabilities**: Memory operations, time parsing, Python execution, communication
    - **Context Classes**: Standard data structures for framework operations
    - **Data Sources**: Core user memory and system data providers
    - **Services**: Internal LangGraph service graphs for complex operations
    - **Prompt Providers**: Default prompt implementations for all framework operations

This registry serves as the baseline configuration that gets merged with
application-specific registries. Applications can override framework components
by registering components with the same names, allowing for customization
while maintaining compatibility.

Architecture Benefits:
    - **Consistent Interface**: Uses the same RegistryConfigProvider pattern as applications
    - **Extensible Foundation**: Applications build upon these core components
    - **Override Support**: Applications can replace framework components as needed
    - **LangGraph Integration**: All components are designed for LangGraph execution
    - **Dependency Management**: Components are initialized in proper dependency order

The framework registry is loaded first during registry initialization,
ensuring core infrastructure is available before application components are loaded.

.. note::
   This registry provides the minimal set of components required for framework
   operation. Applications should not depend on implementation details of these
   components, only their public interfaces.

.. warning::
   Overriding framework components should be done carefully as it may affect
   the behavior of other framework subsystems that depend on these components.

Examples:
    Framework registry is loaded automatically::

        >>> from osprey.registry import initialize_registry, get_registry
        >>> initialize_registry()  # Loads framework registry first
        >>> registry = get_registry()
        >>>
        >>> # Access framework components
        >>> memory_capability = registry.get_capability("memory")
        >>> time_parsing = registry.get_capability("time_range_parsing")
        >>> python_service = registry.get_service("python_executor")

    Application override of framework component::

        >>> # In applications/myapp/registry.py
        >>> class MyAppRegistryProvider(RegistryConfigProvider):
        ...     def get_registry_config(self) -> RegistryConfig:
        ...         return RegistryConfig(
        ...             capabilities=[
        ...                 CapabilityRegistration(
        ...                     name="memory",  # Override framework memory capability
        ...                     module_path="applications.myapp.capabilities.custom_memory",
        ...                     class_name="CustomMemoryCapability",
        ...                     description="Application-specific memory handling",
        ...                     provides=["MEMORY_CONTEXT"],
        ...                     requires=[]
        ...                 )
        ...             ]
        ...         )

.. seealso::
   :class:`RegistryConfigProvider` : Interface implemented by this provider
   :class:`RegistryManager` : Manager that loads and merges this registry
   :doc:`/developer-guides/osprey-components` : Osprey component documentation
   :doc:`/developer-guides/component-override` : Guide to overriding Osprey components
"""

from .base import (
    CapabilityRegistration,
    ConnectorRegistration,
    ContextClassRegistration,
    DataSourceRegistration,
    FrameworkPromptProviderRegistration,
    NodeRegistration,
    ProviderRegistration,
    RegistryConfig,
    RegistryConfigProvider,
    ServiceRegistration,
)


class FrameworkRegistryProvider(RegistryConfigProvider):
    """Framework registry provider implementing the standard interface pattern.

    This provider generates the framework-only registry configuration containing
    all core infrastructure components required for framework operation. It follows
    the same RegistryConfigProvider interface pattern used by applications,
    ensuring consistency across the entire registry system.

    The framework registry provides the foundational components that applications
    build upon, including routing infrastructure, core capabilities, standard
    context classes, and essential services. Applications register their own
    components through separate registry modules that are discovered and merged
    with this framework configuration at runtime.

    Component Categories Provided:
        - **Core Nodes**: Router, classifier, orchestrator, error handler
        - **Framework Capabilities**: Memory, time parsing, Python execution, communication
        - **Context Classes**: Standard data structures for framework operations
        - **Data Sources**: User memory and core system data providers
        - **Services**: Python executor and other internal service graphs
        - **Prompt Providers**: Default prompt implementations for all framework operations

    The registry configuration returned by this provider serves as the baseline
    that gets merged with application registries. Applications can override any
    framework component by registering a component with the same name.

    Initialization Priority:
        This framework registry is always loaded first during registry initialization,
        ensuring core infrastructure is available before application components are
        processed. The initialization follows dependency order within the framework
        components themselves.

    .. note::
       This provider is used by the registry system during framework initialization.
       Manual instantiation is not required or recommended.

    .. warning::
       Changes to this registry affect all applications using the framework.
       New components should be added carefully with consideration for backward
       compatibility.

    Examples:
        The framework registry is used automatically::

            >>> # Framework registry is loaded automatically during initialization
            >>> from osprey.registry import initialize_registry, get_registry
            >>> initialize_registry()
            >>> registry = get_registry()
            >>>
            >>> # Framework components are available to all applications
            >>> memory_cap = registry.get_capability("memory")
            >>> router_node = registry.get_node("router")
            >>> time_context = registry.get_context_class("TIME_RANGE")

        Applications can override framework components::

            >>> # Applications can replace framework components by name
            >>> # This happens automatically during registry merging
            >>> custom_memory = registry.get_capability("memory")  # May be app override

    .. seealso::
       :class:`RegistryConfigProvider` : Interface implemented by this class
       :class:`RegistryManager` : Manager that uses this provider
       :func:`get_registry_config` : Method that returns the framework configuration
       :doc:`/developer-guides/osprey-architecture` : Osprey component architecture
    """

    def get_registry_config(self) -> RegistryConfig:
        """Create comprehensive framework registry configuration.

        Generates the complete registry configuration for all core framework
        infrastructure components. This configuration serves as the foundation
        that applications build upon and can selectively override.

        The framework registry provides essential components organized by category:

        Infrastructure Nodes:
            - Router: Central routing and decision authority
            - Task Extraction: Structured task parsing from user queries
            - Classifier: Query classification for capability selection
            - Orchestrator: Execution planning and coordination
            - Error Handler: Comprehensive error processing and recovery

        Framework Capabilities:
            - Memory Operations: User memory storage and retrieval
            - Time Range Parsing: Temporal query parsing and normalization
            - Python Execution: Code generation and execution
            - Communication: Response generation and clarification requests

        Standard Context Classes:
            - Memory Context: Memory operation results
            - Time Range Context: Parsed temporal information
            - Python Results Context: Code execution results

        Core Data Sources:
            - User Memory Provider: Personal memory and preferences

        Internal Services:
            - Python Executor Service: Multi-node code processing workflow

        Default Prompt Providers:
            - Complete set of default prompt builders for all framework operations

        All components are designed for LangGraph integration with proper
        decorator patterns and dependency management. The configuration
        follows dependency order to ensure proper initialization.

        :return: Complete framework registry configuration with all core
            infrastructure components, context classes, services, and prompt providers
        :rtype: RegistryConfig

        .. note::
           This method is called once during registry initialization. The returned
           configuration is merged with application registries, with applications
           able to override any framework component by name.

        .. warning::
           All components in this registry are considered part of the framework's
           public API. Changes should maintain backward compatibility.

        Examples:
            The configuration includes all framework components::

                >>> provider = FrameworkRegistryProvider()
                >>> config = provider.get_registry_config()
                >>>
                >>> # Framework provides core nodes
                >>> assert any(node.name == "router" for node in config.core_nodes)
                >>> assert any(node.name == "orchestrator" for node in config.core_nodes)
                >>>
                >>> # Framework provides base capabilities
                >>> assert any(cap.name == "memory" for cap in config.capabilities)
                >>> assert any(cap.name == "python" for cap in config.capabilities)
                >>>
                >>> # Framework provides standard context classes
                >>> assert any(ctx.context_type == "TIME_RANGE" for ctx in config.context_classes)

            Applications can override framework components::

                >>> # Application registry can override by using same name
                >>> app_config = RegistryConfig(
                ...     capabilities=[
                ...         CapabilityRegistration(
                ...             name="memory",  # Overrides framework memory capability
                ...             module_path="applications.myapp.capabilities.custom_memory",
                ...             class_name="CustomMemoryCapability",
                ...             description="Custom memory implementation",
                ...             provides=["MEMORY_CONTEXT"],
                ...             requires=[]
                ...         )
                ...     ]
                ... )

        .. seealso::
           :class:`RegistryConfig` : Structure of the returned configuration
           :class:`RegistryManager` : Manager that merges this with application configs
           :doc:`/developer-guides/osprey-components` : Osprey component details
           :doc:`/developer-guides/component-override` : Overriding Osprey components
        """
        return RegistryConfig(
            # Core infrastructure nodes - enhanced for LangGraph
            core_nodes=[
                NodeRegistration(
                    name="router",
                    module_path="osprey.infrastructure.router_node",
                    function_name="RouterNode",
                    description="Central routing decision authority"
                ),
                NodeRegistration(
                    name="task_extraction",
                    module_path="osprey.infrastructure.task_extraction_node",
                    function_name="TaskExtractionNode",
                    description="Extracts structured tasks from user queries"
                ),
                NodeRegistration(
                    name="classifier",
                    module_path="osprey.infrastructure.classification_node",
                    function_name="ClassificationNode",
                    description="Classifies user queries for capability selection"
                ),
                NodeRegistration(
                    name="orchestrator",
                    module_path="osprey.infrastructure.orchestration_node",
                    function_name="OrchestrationNode",
                    description="Execution planning and orchestration"
                ),
                NodeRegistration(
                    name="error",
                    module_path="osprey.infrastructure.error_node",
                    function_name="ErrorNode",
                    description="Error handling and recovery"
                ),
            ],

            # Framework-level capabilities (not application-specific)
            capabilities=[
                # Memory operations capability (framework-level)
                CapabilityRegistration(
                    name="memory",
                    module_path="osprey.capabilities.memory",
                    class_name="MemoryOperationsCapability",
                    description="Save content to and retrieve content from user memory files",
                    provides=["MEMORY_CONTEXT"],
                    requires=[]
                ),

                # Time range parsing capability (framework-level)
                CapabilityRegistration(
                    name="time_range_parsing",
                    module_path="osprey.capabilities.time_range_parsing",
                    class_name="TimeRangeParsingCapability",
                    description="Extract and parse time ranges from user queries into absolute datetime objects",
                    provides=["TIME_RANGE"],
                    requires=[],
                    functional_node="time_range_parsing_node"
                ),

                # Python capability (framework-level)
                CapabilityRegistration(
                    name="python",
                    module_path="osprey.capabilities.python",
                    class_name="PythonCapability",
                    description="Generate and execute simple Python code for computational tasks",
                    provides=["PYTHON_RESULTS"],
                    requires=[],
                    functional_node="python_node"
                ),

                # Communication capabilities (framework-level) - always active
                CapabilityRegistration(
                    name="respond",
                    module_path="osprey.infrastructure.respond_node",
                    class_name="RespondCapability",
                    description="Generate responses to user queries",
                    provides=["FINAL_RESPONSE"],
                    requires=[],
                    always_active=True,
                    functional_node="respond"
                ),
                CapabilityRegistration(
                    name="clarify",
                    module_path="osprey.infrastructure.clarify_node",
                    class_name="ClarifyCapability",
                    description="Ask clarifying questions for ambiguous queries",
                    provides=["CLARIFICATION_REQUEST"],
                    requires=[],
                    always_active=True,
                    functional_node="clarify"
                )
            ],

            # Framework-level context classes
            context_classes=[
                # Memory context (framework-level)
                ContextClassRegistration(
                    context_type="MEMORY_CONTEXT",
                    module_path="osprey.capabilities.memory",
                    class_name="MemoryContext"
                ),
                # Time range context (framework-level)
                ContextClassRegistration(
                    context_type="TIME_RANGE",
                    module_path="osprey.capabilities.time_range_parsing",
                    class_name="TimeRangeContext"
                ),
                # Python results context (framework-level)
                ContextClassRegistration(
                    context_type="PYTHON_RESULTS",
                    module_path="osprey.capabilities.python",
                    class_name="PythonResultsContext"
                )
            ],

            # Framework-level data sources
            data_sources=[
                # Core user memory (framework-level)
                DataSourceRegistration(
                    name="core_user_memory",
                    module_path="osprey.services.memory_storage.memory_provider",
                    class_name="UserMemoryProvider",
                    description="Provides user memory and preferences",
                    health_check_required=True
                ),
            ],

            # Framework-level services - internal LangGraph service agents
            services=[
                # Python executor service (framework-level)
                ServiceRegistration(
                    name="python_executor",
                    module_path="osprey.services.python_executor.service",
                    class_name="PythonExecutorService",
                    description="Python code generation, analysis, and execution service",
                    provides=["PYTHON_RESULTS"],
                    requires=[],
                    internal_nodes=["python_code_generator", "python_code_analyzer", "python_code_executor", "python_approval_node"],

                ),
            ],

            # Framework prompt providers (defaults - typically overridden by applications)
            framework_prompt_providers=[
                FrameworkPromptProviderRegistration(
                    module_path="osprey.prompts.defaults",
                    prompt_builders={
                        "orchestrator": "DefaultOrchestratorPromptBuilder",
                        "task_extraction": "DefaultTaskExtractionPromptBuilder",
                        "response_generation": "DefaultResponseGenerationPromptBuilder",
                        "classification": "DefaultClassificationPromptBuilder",
                        "error_analysis": "DefaultErrorAnalysisPromptBuilder",
                        "clarification": "DefaultClarificationPromptBuilder",
                        "memory_extraction": "DefaultMemoryExtractionPromptBuilder",
                        "time_range_parsing": "DefaultTimeRangeParsingPromptBuilder",
                        "python": "DefaultPythonPromptBuilder"
                    }
                )
            ],

            # Framework AI model providers (SIMPLIFIED - metadata introspected from class)
            providers=[
                ProviderRegistration(
                    module_path="osprey.models.providers.anthropic",
                    class_name="AnthropicProviderAdapter"
                ),
                ProviderRegistration(
                    module_path="osprey.models.providers.openai",
                    class_name="OpenAIProviderAdapter"
                ),
                ProviderRegistration(
                    module_path="osprey.models.providers.google",
                    class_name="GoogleProviderAdapter"
                ),
                ProviderRegistration(
                    module_path="osprey.models.providers.ollama",
                    class_name="OllamaProviderAdapter"
                ),
                ProviderRegistration(
                    module_path="osprey.models.providers.cborg",
                    class_name="CBorgProviderAdapter"
                ),
            ],

            # Framework connectors for control systems and archivers
            connectors=[
                # Control system connectors
                ConnectorRegistration(
                    name="mock",
                    connector_type="control_system",
                    module_path="osprey.connectors.control_system.mock_connector",
                    class_name="MockConnector",
                    description="Mock control system connector for development and testing"
                ),
                ConnectorRegistration(
                    name="epics",
                    connector_type="control_system",
                    module_path="osprey.connectors.control_system.epics_connector",
                    class_name="EPICSConnector",
                    description="EPICS Channel Access control system connector"
                ),
                # Archiver connectors
                ConnectorRegistration(
                    name="mock_archiver",
                    connector_type="archiver",
                    module_path="osprey.connectors.archiver.mock_archiver_connector",
                    class_name="MockArchiverConnector",
                    description="Mock archiver connector for development and testing"
                ),
                ConnectorRegistration(
                    name="epics_archiver",
                    connector_type="archiver",
                    module_path="osprey.connectors.archiver.epics_archiver_connector",
                    class_name="EPICSArchiverConnector",
                    description="EPICS Archiver Appliance connector"
                ),
            ],

            # Simplified initialization order - decorators and subgraphs are imported directly when needed
            initialization_order=[
                "context_classes",    # First - needed by capabilities
                "data_sources",       # Second - needed by capabilities
                "providers",          # Third - AI model providers early for use by capabilities
                "connectors",         # Fourth - control system/archiver connectors
                "core_nodes",         # Fifth - infrastructure nodes
                "services",           # Sixth - internal service graphs
                "capabilities",       # Seventh - depends on everything else including services
                "framework_prompt_providers"  # Last - imports applications that may need capabilities/context
            ]
        )
