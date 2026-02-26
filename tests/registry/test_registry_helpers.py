"""Tests for registry helper functions.

This test module validates the registry helper functions that simplify
application registry creation:
- extend_framework_registry() - Extend framework with app components
- get_framework_defaults() - Get framework registry configuration
- generate_explicit_registry_code() - Generate explicit registry code
"""

import pytest

from osprey.registry.base import (
    CapabilityRegistration,
    ContextClassRegistration,
    DataSourceRegistration,
    ExtendedRegistryConfig,
    RegistryConfig,
    ServiceRegistration,
)
from osprey.registry.helpers import (
    extend_framework_registry,
    generate_explicit_registry_code,
    get_framework_defaults,
)


class TestExtendFrameworkRegistry:
    """Test extend_framework_registry() helper function."""

    def test_simple_extension(self):
        """Test extending framework with simple capability."""
        config = extend_framework_registry(
            capabilities=[
                CapabilityRegistration(
                    name="my_capability",
                    module_path="my_app.capabilities.test",
                    class_name="MyCapability",
                    description="Test capability",
                    provides=[],
                    requires=[],
                )
            ]
        )

        # Returns ExtendedRegistryConfig (marker for extend mode)
        assert isinstance(config, ExtendedRegistryConfig)
        assert isinstance(config, RegistryConfig)

        # Returns application components only
        cap_names = [c.name for c in config.capabilities]
        assert "my_capability" in cap_names

        # Framework capabilities are not included (merged by RegistryManager)
        assert "memory" not in cap_names

        # Contains only specified capabilities
        assert len(config.capabilities) == 1

    def test_extension_with_context_classes(self):
        """Test extending with context classes."""
        config = extend_framework_registry(
            capabilities=[],
            context_classes=[
                ContextClassRegistration(
                    context_type="MY_CONTEXT",
                    module_path="my_app.context_classes",
                    class_name="MyContext",
                )
            ],
        )

        # Returns application components
        assert len(config.context_classes) == 1
        assert config.context_classes[0].context_type == "MY_CONTEXT"

    def test_extension_with_data_sources(self):
        """Test extending with data sources."""
        config = extend_framework_registry(
            capabilities=[],
            context_classes=[],
            data_sources=[
                DataSourceRegistration(
                    name="my_database",
                    module_path="my_app.data_sources.db",
                    class_name="MyDatabaseProvider",
                    description="Custom database",
                    health_check_required=True,
                )
            ],
        )

        assert len(config.data_sources) == 1
        assert config.data_sources[0].name == "my_database"

    def test_extension_with_services(self):
        """Test extending with services."""
        config = extend_framework_registry(
            capabilities=[],
            context_classes=[],
            services=[
                ServiceRegistration(
                    name="my_service",
                    module_path="my_app.services.processor",
                    class_name="ProcessorService",
                    description="Data processor",
                    provides=["PROCESSED_DATA"],
                    requires=["RAW_DATA"],
                    internal_nodes=["validate", "transform"],
                )
            ],
        )

        assert len(config.services) == 1
        assert config.services[0].name == "my_service"
        assert "validate" in config.services[0].internal_nodes

    def test_extension_with_exclusions(self):
        """Test excluding framework components."""
        config = extend_framework_registry(
            capabilities=[
                CapabilityRegistration(
                    name="custom_python",
                    module_path="my_app.capabilities.python",
                    class_name="CustomPythonCapability",
                    description="Custom Python capability",
                    provides=["PYTHON_RESULTS"],
                    requires=[],
                )
            ],
            exclude_capabilities=["python"],
        )

        # Returns application components only
        cap_names = [c.name for c in config.capabilities]
        assert "custom_python" in cap_names

        # Framework capabilities not included
        assert "python" not in cap_names
        assert "memory" not in cap_names

        # Exclusions stored in framework_exclusions field
        assert config.framework_exclusions is not None
        assert "capabilities" in config.framework_exclusions
        assert "python" in config.framework_exclusions["capabilities"]

    def test_extension_excludes_native_control_capability(self):
        """Test excluding a native control capability via extend helper."""
        config = extend_framework_registry(
            capabilities=[],
            exclude_capabilities=["channel_finding"],
        )

        assert config.framework_exclusions is not None
        assert "capabilities" in config.framework_exclusions
        assert "channel_finding" in config.framework_exclusions["capabilities"]

    def test_extension_excludes_native_context_class(self):
        """Test excluding a native control context class via extend helper."""
        config = extend_framework_registry(
            capabilities=[],
            exclude_context_classes=["CHANNEL_ADDRESSES"],
        )

        assert config.framework_exclusions is not None
        assert "context_classes" in config.framework_exclusions
        assert "CHANNEL_ADDRESSES" in config.framework_exclusions["context_classes"]

    def test_extension_excludes_multiple_native_capabilities(self):
        """Test excluding several native control capabilities at once."""
        config = extend_framework_registry(
            capabilities=[],
            exclude_capabilities=["channel_read", "channel_write"],
        )

        assert config.framework_exclusions is not None
        excluded = config.framework_exclusions["capabilities"]
        assert "channel_read" in excluded
        assert "channel_write" in excluded

    def test_extension_with_multiple_exclusions(self):
        """Test excluding multiple types of framework components."""
        config = extend_framework_registry(
            capabilities=[],
            context_classes=[],
            exclude_capabilities=["python", "memory"],
            exclude_nodes=["error"],
            exclude_context_classes=["PYTHON_RESULTS"],
        )

        # All exclusions stored
        assert "capabilities" in config.framework_exclusions
        assert "python" in config.framework_exclusions["capabilities"]
        assert "memory" in config.framework_exclusions["capabilities"]

        assert "nodes" in config.framework_exclusions
        assert "error" in config.framework_exclusions["nodes"]

        assert "context_classes" in config.framework_exclusions
        assert "PYTHON_RESULTS" in config.framework_exclusions["context_classes"]

    def test_extension_with_overrides(self):
        """Test overriding framework components."""
        custom_memory = CapabilityRegistration(
            name="memory",  # Same name as framework
            module_path="my_app.capabilities.custom_memory",
            class_name="CustomMemoryCapability",
            description="Custom memory implementation",
            provides=["MEMORY_CONTEXT"],
            requires=[],
        )

        config = extend_framework_registry(override_capabilities=[custom_memory])

        # Returns application config with override capability
        cap_names = [c.name for c in config.capabilities]
        assert "memory" in cap_names

        # Verify it's the custom implementation
        memory_cap = next(c for c in config.capabilities if c.name == "memory")
        assert memory_cap.module_path == "my_app.capabilities.custom_memory"
        assert memory_cap.class_name == "CustomMemoryCapability"

        # Contains only the specified override
        assert len(config.capabilities) == 1

    def test_extension_combining_additions_and_overrides(self):
        """Test combining regular capabilities and overrides."""
        new_cap = CapabilityRegistration(
            name="new_capability",
            module_path="my_app.capabilities.new",
            class_name="NewCapability",
            description="New functionality",
            provides=["NEW_DATA"],
            requires=[],
        )

        override_cap = CapabilityRegistration(
            name="python",  # Override framework Python
            module_path="my_app.capabilities.custom_python",
            class_name="CustomPythonCapability",
            description="Custom Python",
            provides=["PYTHON_RESULTS"],
            requires=[],
        )

        config = extend_framework_registry(
            capabilities=[new_cap], override_capabilities=[override_cap]
        )

        # Both should be included
        assert len(config.capabilities) == 2
        cap_names = [c.name for c in config.capabilities]
        assert "new_capability" in cap_names
        assert "python" in cap_names

        # Verify override has custom implementation
        python_cap = next(c for c in config.capabilities if c.name == "python")
        assert python_cap.module_path == "my_app.capabilities.custom_python"

    def test_extension_with_all_parameters(self):
        """Test using multiple parameters together."""
        config = extend_framework_registry(
            capabilities=[
                CapabilityRegistration(
                    name="new_cap",
                    module_path="app.cap",
                    class_name="Cap",
                    description="New",
                    provides=[],
                    requires=[],
                )
            ],
            context_classes=[
                ContextClassRegistration(
                    context_type="NEW_CONTEXT", module_path="app.context", class_name="NewContext"
                )
            ],
            data_sources=[
                DataSourceRegistration(
                    name="app_db",
                    module_path="app.db",
                    class_name="AppDB",
                    description="App database",
                )
            ],
            exclude_capabilities=["python"],
            exclude_nodes=["error"],
        )

        # Contains application components
        assert any(c.name == "new_cap" for c in config.capabilities)
        assert any(c.context_type == "NEW_CONTEXT" for c in config.context_classes)
        assert any(d.name == "app_db" for d in config.data_sources)

        # Exclusions stored in framework_exclusions field
        assert config.framework_exclusions is not None
        assert "python" in config.framework_exclusions.get("capabilities", [])
        assert "error" in config.framework_exclusions.get("nodes", [])

        # Framework components not included in application config
        assert not any(c.name == "memory" for c in config.capabilities)
        assert not any(n.name == "router" for n in config.core_nodes)

    def test_include_capabilities_whitelist(self):
        """Test include_capabilities whitelists specific framework capabilities."""
        config = extend_framework_registry(
            capabilities=[],
            include_capabilities=["memory", "respond", "clarify"],
        )
        # Should generate exclusions for everything NOT in the include list
        assert config.framework_exclusions is not None
        excluded = config.framework_exclusions.get("capabilities", [])
        assert "python" in excluded
        assert "time_range_parsing" in excluded
        assert "channel_finding" in excluded
        assert "channel_read" in excluded
        assert "channel_write" in excluded
        assert "archiver_retrieval" in excluded
        assert "state_manager" in excluded
        # Included capabilities should NOT be excluded
        assert "memory" not in excluded
        assert "respond" not in excluded
        assert "clarify" not in excluded

    def test_include_context_classes_whitelist(self):
        """Test include_context_classes whitelists specific framework context classes."""
        config = extend_framework_registry(
            capabilities=[],
            include_context_classes=["MEMORY_CONTEXT", "TIME_RANGE"],
        )
        assert config.framework_exclusions is not None
        excluded = config.framework_exclusions.get("context_classes", [])
        assert "PYTHON_RESULTS" in excluded
        assert "CHANNEL_ADDRESSES" in excluded
        assert "CHANNEL_VALUES" in excluded
        assert "CHANNEL_WRITE_RESULTS" in excluded
        assert "ARCHIVER_DATA" in excluded
        # Included context classes should NOT be excluded
        assert "MEMORY_CONTEXT" not in excluded
        assert "TIME_RANGE" not in excluded

    def test_include_and_exclude_capabilities_raises(self):
        """Test that using both include and exclude raises ValueError."""
        with pytest.raises(ValueError, match="Cannot use both"):
            extend_framework_registry(
                capabilities=[],
                include_capabilities=["memory"],
                exclude_capabilities=["python"],
            )

    def test_include_and_exclude_context_classes_raises(self):
        """Test that using both include and exclude context classes raises ValueError."""
        with pytest.raises(ValueError, match="Cannot use both"):
            extend_framework_registry(
                capabilities=[],
                include_context_classes=["MEMORY_CONTEXT"],
                exclude_context_classes=["PYTHON_RESULTS"],
            )

    def test_include_capabilities_empty_list_excludes_all(self):
        """Test that an empty include list excludes all framework capabilities."""
        config = extend_framework_registry(
            capabilities=[],
            include_capabilities=[],
        )
        assert config.framework_exclusions is not None
        excluded = config.framework_exclusions.get("capabilities", [])
        # All framework capabilities should be excluded
        assert "memory" in excluded
        assert "python" in excluded
        assert "respond" in excluded
        assert "clarify" in excluded
        assert "time_range_parsing" in excluded
        assert "state_manager" in excluded
        assert "channel_finding" in excluded

    def test_include_capabilities_with_app_capabilities(self):
        """Test include_capabilities works alongside app-specific capabilities."""
        config = extend_framework_registry(
            include_capabilities=["respond", "clarify", "memory"],
            capabilities=[
                CapabilityRegistration(
                    name="current_weather",
                    module_path="app.capabilities.weather",
                    class_name="WeatherCapability",
                    description="Weather",
                    provides=["WEATHER"],
                    requires=[],
                )
            ],
        )
        # App capability is present
        cap_names = [c.name for c in config.capabilities]
        assert "current_weather" in cap_names
        # Framework exclusions are set
        assert config.framework_exclusions is not None
        excluded = config.framework_exclusions.get("capabilities", [])
        assert "python" in excluded
        assert "memory" not in excluded


class TestGetFrameworkDefaults:
    """Test get_framework_defaults() helper function."""

    def test_returns_framework_config(self):
        """Test that get_framework_defaults returns framework config."""
        framework = get_framework_defaults()

        # Should be a RegistryConfig (not Extended)
        assert isinstance(framework, RegistryConfig)
        assert not isinstance(framework, ExtendedRegistryConfig)

        # Should have framework components
        assert len(framework.core_nodes) > 0
        assert len(framework.capabilities) > 0
        assert len(framework.context_classes) > 0

        # Should have specific framework capabilities
        cap_names = [c.name for c in framework.capabilities]
        assert "memory" in cap_names
        assert "time_range_parsing" in cap_names
        assert "python" in cap_names
        assert "respond" in cap_names
        assert "clarify" in cap_names

    def test_framework_has_core_nodes(self):
        """Test that framework includes core infrastructure nodes."""
        framework = get_framework_defaults()

        node_names = [n.name for n in framework.core_nodes]
        assert "router" in node_names
        assert "orchestrator" in node_names
        assert "classifier" in node_names
        assert "error" in node_names

    def test_framework_has_context_classes(self):
        """Test that framework includes context classes."""
        framework = get_framework_defaults()

        ctx_types = [c.context_type for c in framework.context_classes]
        assert "MEMORY_CONTEXT" in ctx_types
        assert "PYTHON_RESULTS" in ctx_types
        assert "TIME_RANGE" in ctx_types

    def test_framework_has_providers(self):
        """Test that framework includes AI model providers."""
        framework = get_framework_defaults()

        # Should have provider registrations
        assert len(framework.providers) > 0

        provider_modules = [p.module_path for p in framework.providers]
        assert "osprey.models.providers.anthropic" in provider_modules
        assert "osprey.models.providers.openai" in provider_modules

    def test_framework_initialization_order(self):
        """Test that framework has proper initialization order."""
        framework = get_framework_defaults()

        # Check that initialization order has expected key components
        # Order may vary, but should include these core types
        expected_types = {
            "context_classes",
            "data_sources",
            "providers",
            "capabilities",
            "core_nodes",
            "services",
            "framework_prompt_providers",
        }

        actual_types = set(framework.initialization_order)
        assert expected_types.issubset(actual_types), (
            f"Missing types: {expected_types - actual_types}"
        )

        # Verify context_classes comes before capabilities (dependency)
        ctx_idx = framework.initialization_order.index("context_classes")
        cap_idx = framework.initialization_order.index("capabilities")
        assert ctx_idx < cap_idx, "context_classes must be initialized before capabilities"


class TestGenerateExplicitRegistryCode:
    """Test generate_explicit_registry_code() helper function."""

    def test_generates_valid_python_code(self):
        """Test that generated code is valid Python."""
        code = generate_explicit_registry_code(
            app_class_name="TestAppRegistryProvider",
            app_display_name="Test App",
            package_name="test_app",
            capabilities=[
                CapabilityRegistration(
                    name="test_cap",
                    module_path="test_app.capabilities.test",
                    class_name="TestCapability",
                    description="Test capability",
                    provides=["TEST_DATA"],
                    requires=[],
                )
            ],
            context_classes=[
                ContextClassRegistration(
                    context_type="TEST_DATA",
                    module_path="test_app.context_classes",
                    class_name="TestDataContext",
                )
            ],
        )

        # Should be valid Python code (can be compiled)
        compile(code, "<generated>", "exec")

        # Should contain expected elements
        assert "class TestAppRegistryProvider(RegistryConfigProvider):" in code
        assert "def get_registry_config(self):" in code
        assert "return RegistryConfig(" in code

    def test_includes_framework_components(self):
        """Test that generated code includes framework components."""
        code = generate_explicit_registry_code(
            app_class_name="TestProvider",
            app_display_name="Test",
            package_name="test",
            capabilities=[],
            context_classes=[],
        )

        # Should include framework capabilities
        assert "memory" in code.lower()
        assert "python" in code.lower()
        assert "respond" in code.lower()

        # Should include framework nodes
        assert "router" in code.lower()
        assert "orchestrator" in code.lower()

    def test_includes_application_components(self):
        """Test that generated code includes application components."""
        code = generate_explicit_registry_code(
            app_class_name="WeatherProvider",
            app_display_name="Weather Agent",
            package_name="weather_agent",
            capabilities=[
                CapabilityRegistration(
                    name="current_weather",
                    module_path="weather_agent.capabilities.current_weather",
                    class_name="CurrentWeatherCapability",
                    description="Get current weather",
                    provides=["WEATHER_DATA"],
                    requires=[],
                )
            ],
            context_classes=[
                ContextClassRegistration(
                    context_type="WEATHER_DATA",
                    module_path="weather_agent.context_classes",
                    class_name="WeatherDataContext",
                )
            ],
        )

        # Should include application capability
        assert "current_weather" in code
        assert "CurrentWeatherCapability" in code
        assert "weather_agent.capabilities.current_weather" in code

        # Should include application context
        assert "WEATHER_DATA" in code
        assert "WeatherDataContext" in code

    def test_includes_data_sources(self):
        """Test that data sources are included in generated code."""
        code = generate_explicit_registry_code(
            app_class_name="TestProvider",
            app_display_name="Test",
            package_name="test",
            capabilities=[],
            context_classes=[],
            data_sources=[
                DataSourceRegistration(
                    name="test_db",
                    module_path="test.data_sources.db",
                    class_name="TestDatabase",
                    description="Test database",
                )
            ],
        )

        assert "test_db" in code
        assert "TestDatabase" in code
        assert "test.data_sources.db" in code

    def test_includes_services(self):
        """Test that services are included in generated code."""
        code = generate_explicit_registry_code(
            app_class_name="TestProvider",
            app_display_name="Test",
            package_name="test",
            capabilities=[],
            context_classes=[],
            services=[
                ServiceRegistration(
                    name="processor",
                    module_path="test.services.processor",
                    class_name="ProcessorService",
                    description="Data processor",
                    provides=["PROCESSED"],
                    requires=["RAW"],
                    internal_nodes=["validate", "transform"],
                )
            ],
        )

        assert "processor" in code
        assert "ProcessorService" in code
        assert "test.services.processor" in code
        assert "validate" in code
        assert "transform" in code

    def test_includes_native_control_capabilities(self):
        """Test that generated code includes native control capabilities and context types."""
        code = generate_explicit_registry_code(
            app_class_name="TestProvider",
            app_display_name="Test",
            package_name="test",
            capabilities=[],
            context_classes=[],
        )

        # Should include all 4 native control capabilities
        assert "channel_finding" in code
        assert "channel_read" in code
        assert "channel_write" in code
        assert "archiver_retrieval" in code

        # Should include their context types
        assert "CHANNEL_ADDRESSES" in code
        assert "CHANNEL_VALUES" in code
        assert "CHANNEL_WRITE_RESULTS" in code
        assert "ARCHIVER_DATA" in code

    def test_generated_code_has_proper_structure(self):
        """Test that generated code has proper section structure."""
        code = generate_explicit_registry_code(
            app_class_name="TestProvider",
            app_display_name="Test",
            package_name="test",
            capabilities=[],
            context_classes=[],
        )

        # Should have section headers
        assert "FRAMEWORK CORE NODES" in code
        assert "ALL CAPABILITIES" in code
        assert "ALL CONTEXT CLASSES" in code
        assert "DATA SOURCES" in code
        assert "SERVICES" in code
        assert "AI MODEL PROVIDERS" in code

        # Should have imports
        assert "from osprey.registry import" in code
        assert "RegistryConfigProvider" in code
        assert "RegistryConfig" in code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
