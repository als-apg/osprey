"""Framework registry provider.

Registers shared infrastructure (connectors, code generators, ARIEL modules)
that all Osprey applications build upon.

.. seealso:: :class:`RegistryConfigProvider`, :class:`RegistryManager`
"""

from osprey_connectors.factory import BUILTIN_CONNECTORS

from .base import (
    ArielEnhancementModuleRegistration,
    ArielIngestionAdapterRegistration,
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
            # Framework AI model providers — the built-in table lives in
            # osprey.models.provider_registry (single source of truth).
            # RegistryManager._initialize_providers() delegates to it.
            providers=[],
            # The connectors the connectors package ships, from its own table:
            # the module paths name implementation modules of that package,
            # which only it spells. Listed here and not only registered by the
            # factory because the two paths populate the registry independently:
            # ``initialize_registry()`` registers what this provider lists and
            # nothing else, and it is the setup step every python executor
            # sandbox runs.
            connectors=[
                ConnectorRegistration(
                    name=builtin.name,
                    connector_type=builtin.kind,
                    module_path=builtin.module_path,
                    class_name=builtin.class_name,
                    description=builtin.description,
                )
                for builtin in BUILTIN_CONNECTORS
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
                ArielSearchModuleRegistration(
                    name="hybrid",
                    module_path="osprey.services.ariel_search.search.qmd",
                    description="Hybrid keyword and semantic search via the qmd sidecar",
                ),
            ],
            # ARIEL enhancement modules
            ariel_enhancement_modules=[
                ArielEnhancementModuleRegistration(
                    name="image_caption",
                    module_path="osprey.services.ariel_search.enhancement.image_caption.module",
                    class_name="ImageCaptionModule",
                    description="Caption logbook pictures with a vision model (catch-up only)",
                    execution_order=5,
                ),
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
                ArielEnhancementModuleRegistration(
                    name="image_embedding",
                    module_path="osprey.services.ariel_search.enhancement.image_embedding.module",
                    class_name="ImageEmbeddingModule",
                    description="Embed logbook pictures for picture search (catch-up only)",
                    execution_order=25,
                ),
                ArielEnhancementModuleRegistration(
                    name="qmd_export",
                    module_path="osprey.services.ariel_search.enhancement.qmd_export.exporter",
                    class_name="QmdExportModule",
                    description="Mirror entries to the markdown tree the qmd sidecar indexes",
                    execution_order=30,
                ),
            ],
            # ARIEL ingestion adapters
            ariel_ingestion_adapters=[
                ArielIngestionAdapterRegistration(
                    name="als_logbook",
                    module_path="osprey.services.ariel_search.ingestion.adapters.als",
                    class_name="ALSLogbookAdapter",
                    description="ALS eLog reference format: JSONL streaming and HTTP API support",
                ),
                ArielIngestionAdapterRegistration(
                    name="jlab_logbook",
                    module_path="osprey.services.ariel_search.ingestion.adapters.jlab",
                    class_name="JLabLogbookAdapter",
                    description="JLab logbook reference format",
                ),
                ArielIngestionAdapterRegistration(
                    name="ornl_logbook",
                    module_path="osprey.services.ariel_search.ingestion.adapters.ornl",
                    class_name="ORNLLogbookAdapter",
                    description="ORNL logbook reference format",
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
                "ariel_ingestion_adapters",
            ],
        )
