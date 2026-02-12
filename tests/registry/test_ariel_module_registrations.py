"""Tests for ARIEL module registration dataclasses and RegistryConfig fields.

Validates:
- ArielSearchModuleRegistration, ArielEnhancementModuleRegistration, ArielPipelineRegistration
- New RegistryConfig fields (ariel_search_modules, ariel_enhancement_modules, ariel_pipelines)
- initialization_order includes the new types
"""

from osprey.registry.base import (
    ArielEnhancementModuleRegistration,
    ArielPipelineRegistration,
    ArielSearchModuleRegistration,
    RegistryConfig,
)


class TestArielSearchModuleRegistration:
    """Test ArielSearchModuleRegistration dataclass."""

    def test_basic_creation(self):
        reg = ArielSearchModuleRegistration(
            name="keyword",
            module_path="osprey.services.ariel_search.search.keyword",
            description="Full-text search",
        )
        assert reg.name == "keyword"
        assert reg.module_path == "osprey.services.ariel_search.search.keyword"
        assert reg.description == "Full-text search"

    def test_custom_module(self):
        reg = ArielSearchModuleRegistration(
            name="custom_search",
            module_path="my_app.search.custom",
            description="Custom search module",
        )
        assert reg.name == "custom_search"


class TestArielEnhancementModuleRegistration:
    """Test ArielEnhancementModuleRegistration dataclass."""

    def test_basic_creation(self):
        reg = ArielEnhancementModuleRegistration(
            name="text_embedding",
            module_path="osprey.services.ariel_search.enhancement.text_embedding.embedder",
            class_name="TextEmbeddingModule",
            description="Generate vector embeddings",
            execution_order=20,
        )
        assert reg.name == "text_embedding"
        assert reg.class_name == "TextEmbeddingModule"
        assert reg.execution_order == 20

    def test_default_execution_order(self):
        reg = ArielEnhancementModuleRegistration(
            name="custom",
            module_path="my_app.enhancement.custom",
            class_name="CustomModule",
            description="Custom enhancement",
        )
        assert reg.execution_order == 50

    def test_custom_execution_order(self):
        reg = ArielEnhancementModuleRegistration(
            name="early",
            module_path="my_app.enhancement.early",
            class_name="EarlyModule",
            description="Runs first",
            execution_order=5,
        )
        assert reg.execution_order == 5


class TestArielPipelineRegistration:
    """Test ArielPipelineRegistration dataclass."""

    def test_basic_creation(self):
        reg = ArielPipelineRegistration(
            name="rag",
            module_path="osprey.services.ariel_search.pipelines",
            description="RAG pipeline",
        )
        assert reg.name == "rag"
        assert reg.category == "llm"  # default

    def test_custom_category(self):
        reg = ArielPipelineRegistration(
            name="direct_pipeline",
            module_path="my_app.pipelines",
            description="Direct pipeline",
            category="direct",
        )
        assert reg.category == "direct"


class TestRegistryConfigArielFields:
    """Test RegistryConfig with new ARIEL fields."""

    def test_default_ariel_fields_empty(self):
        config = RegistryConfig(
            capabilities=[],
            context_classes=[],
        )
        assert config.ariel_search_modules == []
        assert config.ariel_enhancement_modules == []
        assert config.ariel_pipelines == []

    def test_ariel_fields_in_config(self):
        config = RegistryConfig(
            capabilities=[],
            context_classes=[],
            ariel_search_modules=[
                ArielSearchModuleRegistration(
                    name="keyword",
                    module_path="test.keyword",
                    description="Keyword search",
                ),
            ],
            ariel_enhancement_modules=[
                ArielEnhancementModuleRegistration(
                    name="embedder",
                    module_path="test.embedder",
                    class_name="Embedder",
                    description="Embedder",
                    execution_order=10,
                ),
            ],
            ariel_pipelines=[
                ArielPipelineRegistration(
                    name="rag",
                    module_path="test.pipelines",
                    description="RAG",
                ),
            ],
        )
        assert len(config.ariel_search_modules) == 1
        assert len(config.ariel_enhancement_modules) == 1
        assert len(config.ariel_pipelines) == 1

    def test_initialization_order_includes_ariel_types(self):
        config = RegistryConfig(
            capabilities=[],
            context_classes=[],
        )
        assert "ariel_search_modules" in config.initialization_order
        assert "ariel_enhancement_modules" in config.initialization_order
        assert "ariel_pipelines" in config.initialization_order

    def test_ariel_types_before_capabilities_in_init_order(self):
        config = RegistryConfig(
            capabilities=[],
            context_classes=[],
        )
        order = config.initialization_order
        cap_idx = order.index("capabilities")
        assert order.index("ariel_search_modules") < cap_idx
        assert order.index("ariel_enhancement_modules") < cap_idx
        assert order.index("ariel_pipelines") < cap_idx


class TestArielRegistrationImports:
    """Test that ARIEL registration types are properly exported."""

    def test_import_from_registry_package(self):
        from osprey.registry import (
            ArielEnhancementModuleRegistration,
            ArielPipelineRegistration,
            ArielSearchModuleRegistration,
        )

        assert ArielSearchModuleRegistration is not None
        assert ArielEnhancementModuleRegistration is not None
        assert ArielPipelineRegistration is not None
