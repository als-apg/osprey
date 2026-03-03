"""Framework registry provider.

Registers shared infrastructure (connectors, code generators, ARIEL modules)
that all Osprey applications build upon.

.. seealso:: :class:`RegistryConfigProvider`, :class:`RegistryManager`
"""

from .base import (
    ArielEnhancementModuleRegistration,
    ArielIngestionAdapterRegistration,
    ArielPipelineRegistration,
    ArielSearchModuleRegistration,
    ConnectorRegistration,
    RegistryConfig,
    RegistryConfigProvider,
    ServiceRegistration,
)


class FrameworkRegistryProvider(RegistryConfigProvider):
    """Provides the baseline framework registry configuration.

    Loaded automatically during registry initialization before any application
    registries. Applications can override framework components by name.
    """

    def get_registry_config(self) -> RegistryConfig:
        """Return framework registry configuration.

        :rtype: RegistryConfig
        """
        return RegistryConfig(
            services=[
                ServiceRegistration(
                    name="channel_finder",
                    module_path="osprey.services.channel_finder",
                    class_name="ChannelFinderService",
                    description="Natural-language channel address resolution service",
                    provides=["CHANNEL_ADDRESSES"],
                    requires=[],
                ),
            ],
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
                "ariel_search_modules",
                "ariel_enhancement_modules",
                "ariel_pipelines",
                "ariel_ingestion_adapters",
            ],
        )
