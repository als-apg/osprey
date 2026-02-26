"""Tests for ARIEL module registry integration.

Validates:
- extend_framework_registry() with ARIEL module types (add, exclude, override)
- Framework defaults include ARIEL registrations
- RegistryConfig merge behavior for ARIEL fields
"""

from osprey.registry.base import (
    ArielEnhancementModuleRegistration,
    ArielPipelineRegistration,
    ArielSearchModuleRegistration,
    ExtendedRegistryConfig,
)
from osprey.registry.helpers import extend_framework_registry, get_framework_defaults


class TestFrameworkDefaultsIncludeAriel:
    """Test that framework defaults contain ARIEL registrations."""

    def test_framework_has_ariel_search_modules(self):
        defaults = get_framework_defaults()
        names = [m.name for m in defaults.ariel_search_modules]
        assert "keyword" in names
        assert "semantic" in names

    def test_framework_has_ariel_enhancement_modules(self):
        defaults = get_framework_defaults()
        names = [m.name for m in defaults.ariel_enhancement_modules]
        assert "semantic_processor" in names
        assert "text_embedding" in names

    def test_framework_enhancement_modules_have_order(self):
        defaults = get_framework_defaults()
        by_name = {m.name: m for m in defaults.ariel_enhancement_modules}
        assert (
            by_name["semantic_processor"].execution_order
            < by_name["text_embedding"].execution_order
        )

    def test_framework_has_ariel_pipelines(self):
        defaults = get_framework_defaults()
        names = [p.name for p in defaults.ariel_pipelines]
        assert "rag" in names
        assert "agent" in names

    def test_framework_init_order_includes_ariel(self):
        defaults = get_framework_defaults()
        assert "ariel_search_modules" in defaults.initialization_order
        assert "ariel_enhancement_modules" in defaults.initialization_order
        assert "ariel_pipelines" in defaults.initialization_order


class TestExtendFrameworkRegistryWithAriel:
    """Test extend_framework_registry() with ARIEL parameters."""

    def test_add_custom_search_module(self):
        config = extend_framework_registry(
            ariel_search_modules=[
                ArielSearchModuleRegistration(
                    name="custom_search",
                    module_path="my_app.search.custom",
                    description="Custom search module",
                ),
            ],
        )
        assert isinstance(config, ExtendedRegistryConfig)
        names = [m.name for m in config.ariel_search_modules]
        assert "custom_search" in names

    def test_add_custom_enhancement_module(self):
        config = extend_framework_registry(
            ariel_enhancement_modules=[
                ArielEnhancementModuleRegistration(
                    name="custom_enhancer",
                    module_path="my_app.enhancement.custom",
                    class_name="CustomEnhancer",
                    description="Custom enhancer",
                    execution_order=15,
                ),
            ],
        )
        names = [m.name for m in config.ariel_enhancement_modules]
        assert "custom_enhancer" in names

    def test_add_custom_pipeline(self):
        config = extend_framework_registry(
            ariel_pipelines=[
                ArielPipelineRegistration(
                    name="custom_pipeline",
                    module_path="my_app.pipelines",
                    description="Custom pipeline",
                    category="direct",
                ),
            ],
        )
        names = [p.name for p in config.ariel_pipelines]
        assert "custom_pipeline" in names

    def test_exclude_ariel_search_module(self):
        config = extend_framework_registry(
            exclude_ariel_search_modules=["semantic"],
        )
        assert config.framework_exclusions is not None
        assert "ariel_search_modules" in config.framework_exclusions
        assert "semantic" in config.framework_exclusions["ariel_search_modules"]

    def test_exclude_ariel_enhancement_module(self):
        config = extend_framework_registry(
            exclude_ariel_enhancement_modules=["text_embedding"],
        )
        assert "ariel_enhancement_modules" in config.framework_exclusions
        assert "text_embedding" in config.framework_exclusions["ariel_enhancement_modules"]

    def test_exclude_ariel_pipeline(self):
        config = extend_framework_registry(
            exclude_ariel_pipelines=["agent"],
        )
        assert "ariel_pipelines" in config.framework_exclusions
        assert "agent" in config.framework_exclusions["ariel_pipelines"]

    def test_empty_ariel_fields_default(self):
        """Calling extend_framework_registry without ARIEL params gives empty lists."""
        config = extend_framework_registry()
        assert config.ariel_search_modules == []
        assert config.ariel_enhancement_modules == []
        assert config.ariel_pipelines == []


class TestRegistryManagerArielMerge:
    """Test that RegistryManager merges ARIEL fields correctly.

    Uses the RegistryManager directly with mock configs to test merge behavior
    without requiring full initialization (which would import heavy modules).
    """

    def test_merge_adds_app_search_modules(self):
        """App search modules are appended to framework search modules."""
        # Simulate what _merge_application_with_override does
        from osprey.registry.base import RegistryConfig
        from osprey.registry.manager import RegistryManager

        merged = RegistryConfig(
            capabilities=[],
            context_classes=[],
            ariel_search_modules=[
                ArielSearchModuleRegistration(
                    name="keyword",
                    module_path="framework.keyword",
                    description="FW keyword",
                ),
            ],
        )

        app_config = RegistryConfig(
            capabilities=[],
            context_classes=[],
            ariel_search_modules=[
                ArielSearchModuleRegistration(
                    name="custom",
                    module_path="app.custom",
                    description="App custom",
                ),
            ],
        )

        manager = RegistryManager.__new__(RegistryManager)
        manager._excluded_provider_names = []
        manager._merge_application_with_override(merged, app_config, "test_app")

        names = [m.name for m in merged.ariel_search_modules]
        assert "keyword" in names
        assert "custom" in names

    def test_merge_overrides_same_name_search_module(self):
        """App search module with same name overrides framework module."""
        from osprey.registry.base import RegistryConfig
        from osprey.registry.manager import RegistryManager

        merged = RegistryConfig(
            capabilities=[],
            context_classes=[],
            ariel_search_modules=[
                ArielSearchModuleRegistration(
                    name="keyword",
                    module_path="framework.keyword",
                    description="FW keyword",
                ),
            ],
        )

        app_config = RegistryConfig(
            capabilities=[],
            context_classes=[],
            ariel_search_modules=[
                ArielSearchModuleRegistration(
                    name="keyword",
                    module_path="app.custom_keyword",
                    description="App keyword override",
                ),
            ],
        )

        manager = RegistryManager.__new__(RegistryManager)
        manager._excluded_provider_names = []
        manager._merge_application_with_override(merged, app_config, "test_app")

        names = [m.name for m in merged.ariel_search_modules]
        assert names.count("keyword") == 1
        assert merged.ariel_search_modules[0].module_path == "app.custom_keyword"
