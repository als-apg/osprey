"""Framework Registry Provider.

This module contains the framework's registry provider implementation,
following the same RegistryConfigProvider interface pattern used by all
applications. It registers the shared infrastructure components that
Osprey projects depend on.

The framework registry provides:

Shared Infrastructure:
    - **Connectors**: Control system (EPICS, mock) and archiver connectors
    - **Code Generators**: Python code generation backends (basic LLM, Claude Code SDK)
    - **ARIEL Modules**: Search, enhancement, pipeline, and ingestion components
    - **Providers**: AI model provider configuration

This registry serves as the baseline configuration that gets merged with
application-specific registries. Applications can override framework components
by registering components with the same names.

Architecture Benefits:
    - **Consistent Interface**: Uses the same RegistryConfigProvider pattern as applications
    - **Extensible Foundation**: Applications build upon these core components
    - **Override Support**: Applications can replace framework components as needed
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

.. seealso::
   :class:`RegistryConfigProvider` : Interface implemented by this provider
   :class:`RegistryManager` : Manager that loads and merges this registry
"""

from .base import (
    ArielEnhancementModuleRegistration,
    ArielIngestionAdapterRegistration,
    ArielPipelineRegistration,
    ArielSearchModuleRegistration,
    CodeGeneratorRegistration,
    ConnectorRegistration,
    ProviderRegistration,
    RegistryConfig,
    RegistryConfigProvider,
)


class FrameworkRegistryProvider(RegistryConfigProvider):
    """Framework registry provider implementing the standard interface pattern.

    This provider generates the framework-only registry configuration containing
    shared infrastructure components: connectors, code generators, ARIEL modules,
    and AI model providers. It follows the same RegistryConfigProvider interface
    pattern used by applications, ensuring consistency across the registry system.

    Component Categories Provided:
        - **Connectors**: Control system and archiver connectors (EPICS, mock)
        - **Code Generators**: Python code generation backends
        - **ARIEL Search Modules**: Keyword and semantic search
        - **ARIEL Enhancement Modules**: Semantic processing and text embedding
        - **ARIEL Pipelines**: RAG pipeline
        - **ARIEL Ingestion Adapters**: Facility-specific logbook adapters

    The registry configuration returned by this provider serves as the baseline
    that gets merged with application registries. Applications can override any
    framework component by registering a component with the same name.

    Initialization Priority:
        This framework registry is always loaded first during registry initialization,
        ensuring core infrastructure is available before application components are
        processed.

    .. note::
       This provider is used by the registry system during framework initialization.
       Manual instantiation is not required or recommended.

    .. warning::
       Changes to this registry affect all applications using the framework.
       New components should be added carefully with consideration for backward
       compatibility.

    Examples:
        The framework registry is used automatically::

            >>> from osprey.registry import initialize_registry, get_registry
            >>> initialize_registry()
            >>> registry = get_registry()

    .. seealso::
       :class:`RegistryConfigProvider` : Interface implemented by this class
       :class:`RegistryManager` : Manager that uses this provider
    """

    def get_registry_config(self) -> RegistryConfig:
        """Create framework registry configuration.

        Generates the registry configuration for shared infrastructure
        components: connectors, code generators, ARIEL modules, and
        providers. This configuration serves as the foundation that
        applications build upon and can selectively override.

        Components:
            - **Connectors**: EPICS and mock connectors for control systems and archivers
            - **Code Generators**: Basic LLM and Claude Code SDK generators
            - **ARIEL Search**: Keyword and semantic search modules
            - **ARIEL Enhancement**: Semantic processing and text embedding
            - **ARIEL Pipelines**: RAG pipeline
            - **ARIEL Ingestion**: Facility-specific logbook adapters (ALS, JLab, ORNL, generic)

        :return: Framework registry configuration with connectors, code generators,
            ARIEL modules, and providers
        :rtype: RegistryConfig

        .. note::
           This method is called once during registry initialization. The returned
           configuration is merged with application registries, with applications
           able to override any framework component by name.
        """
        return RegistryConfig(
            core_nodes=[],
            capabilities=[],
            context_classes=[],
            data_sources=[],
            services=[],
            framework_prompt_providers=[],
            # Framework AI model providers — built-in table now lives in
            # osprey.models.provider_registry (single source of truth).
            # RegistryManager._initialize_providers() delegates to it.
            providers=[],
            # Framework connectors for control systems and archivers
            connectors=[
                # Control system connectors
                ConnectorRegistration(
                    name="mock",
                    connector_type="control_system",
                    module_path="osprey.connectors.control_system.mock_connector",
                    class_name="MockConnector",
                    description="Mock control system connector for development and testing",
                ),
                ConnectorRegistration(
                    name="epics",
                    connector_type="control_system",
                    module_path="osprey.connectors.control_system.epics_connector",
                    class_name="EPICSConnector",
                    description="EPICS Channel Access control system connector",
                ),
                # Archiver connectors
                ConnectorRegistration(
                    name="mock_archiver",
                    connector_type="archiver",
                    module_path="osprey.connectors.archiver.mock_archiver_connector",
                    class_name="MockArchiverConnector",
                    description="Mock archiver connector for development and testing",
                ),
                ConnectorRegistration(
                    name="epics_archiver",
                    connector_type="archiver",
                    module_path="osprey.connectors.archiver.epics_archiver_connector",
                    class_name="EPICSArchiverConnector",
                    description="EPICS Archiver Appliance connector",
                ),
            ],
            # Framework code generators for Python executor
            code_generators=[
                # Basic LLM-based generator (always available)
                CodeGeneratorRegistration(
                    name="basic",
                    module_path="osprey.services.python_executor.generation.basic_generator",
                    class_name="BasicLLMCodeGenerator",
                    description="Simple single-pass LLM code generator",
                ),
                # Claude Code SDK generator (optional dependency)
                CodeGeneratorRegistration(
                    name="claude_code",
                    module_path="osprey.services.python_executor.generation.claude_code_generator",
                    class_name="ClaudeCodeGenerator",
                    description="Claude Code SDK-based generator with multi-turn reasoning and codebase awareness",
                    optional_dependencies=["claude-agent-sdk"],
                ),
            ],
            # ARIEL search modules
            ariel_search_modules=[
                ArielSearchModuleRegistration(
                    name="keyword",
                    module_path="osprey.services.ariel_search.search.keyword",
                    description="Full-text search with PostgreSQL FTS and fuzzy fallback",
                ),
                ArielSearchModuleRegistration(
                    name="semantic",
                    module_path="osprey.services.ariel_search.search.semantic",
                    description="Embedding similarity search using vector cosine distance",
                ),
            ],
            # ARIEL enhancement modules
            ariel_enhancement_modules=[
                ArielEnhancementModuleRegistration(
                    name="semantic_processor",
                    module_path="osprey.services.ariel_search.enhancement.semantic_processor.processor",
                    class_name="SemanticProcessorModule",
                    description="Extract keywords and summaries from logbook entries",
                    execution_order=10,
                ),
                ArielEnhancementModuleRegistration(
                    name="text_embedding",
                    module_path="osprey.services.ariel_search.enhancement.text_embedding.embedder",
                    class_name="TextEmbeddingModule",
                    description="Generate vector embeddings for logbook entries",
                    execution_order=20,
                ),
            ],
            # ARIEL pipelines
            ariel_pipelines=[
                ArielPipelineRegistration(
                    name="rag",
                    module_path="osprey.services.ariel_search.pipelines",
                    description="Retrieval-augmented generation with text embeddings, keyword search, and LLM summarization",
                ),
            ],
            # ARIEL ingestion adapters
            ariel_ingestion_adapters=[
                ArielIngestionAdapterRegistration(
                    name="als_logbook",
                    module_path="osprey.services.ariel_search.ingestion.adapters.als",
                    class_name="ALSLogbookAdapter",
                    description="ALS eLog adapter with JSONL streaming and HTTP API support",
                ),
                ArielIngestionAdapterRegistration(
                    name="jlab_logbook",
                    module_path="osprey.services.ariel_search.ingestion.adapters.jlab",
                    class_name="JLabLogbookAdapter",
                    description="Jefferson Lab logbook adapter",
                ),
                ArielIngestionAdapterRegistration(
                    name="ornl_logbook",
                    module_path="osprey.services.ariel_search.ingestion.adapters.ornl",
                    class_name="ORNLLogbookAdapter",
                    description="Oak Ridge National Laboratory logbook adapter",
                ),
                ArielIngestionAdapterRegistration(
                    name="generic_json",
                    module_path="osprey.services.ariel_search.ingestion.adapters.generic",
                    class_name="GenericJSONAdapter",
                    description="Generic JSON adapter for testing and facilities without custom APIs",
                ),
            ],
            initialization_order=[
                "providers",
                "connectors",
                "code_generators",
                "ariel_search_modules",
                "ariel_enhancement_modules",
                "ariel_pipelines",
                "ariel_ingestion_adapters",
            ],
        )
