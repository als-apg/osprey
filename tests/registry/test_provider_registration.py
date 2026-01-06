"""Tests for custom AI model provider registration.

This test module validates the provider registration feature that allows
applications to register custom AI model providers through the registry system.
Tests cover:
- Registering custom providers in application registries
- Provider merging during registry initialization
- Provider exclusions (excluding framework providers)
- Helper function support for providers
- Provider override scenarios
"""

import pytest

from osprey.registry.base import (
    ProviderRegistration,
    RegistryConfig,
)
from osprey.registry.helpers import extend_framework_registry
from osprey.registry.manager import RegistryManager


class TestProviderMerging:
    """Test that application providers are properly merged into registry."""

    def test_application_providers_merged_into_config(self, tmp_path):
        """Test that providers from application registry are added to merged config."""
        # Create application registry with custom providers
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, extend_framework_registry, ProviderRegistration

class AppProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            capabilities=[],
            context_classes=[],
            providers=[
                ProviderRegistration(
                    module_path="test_app.providers.custom",
                    class_name="CustomAIProvider"
                ),
                ProviderRegistration(
                    module_path="test_app.providers.institutional",
                    class_name="InstitutionalAIProvider"
                )
            ]
        )
"""
        )

        # Load registry and get initial framework provider count
        framework_manager = RegistryManager(registry_path=None)
        framework_provider_count = len(framework_manager.config.providers)

        manager = RegistryManager(registry_path=str(registry_file))
        provider_module_paths = [p.module_path for p in manager.config.providers]

        # Framework providers should still be present
        assert "osprey.models.providers.anthropic" in provider_module_paths
        assert "osprey.models.providers.openai" in provider_module_paths

        # Application providers should be added
        assert "test_app.providers.custom" in provider_module_paths
        assert "test_app.providers.institutional" in provider_module_paths

        # Total count should be framework + 2 application providers
        assert len(manager.config.providers) == framework_provider_count + 2

    def test_multiple_providers_in_single_application(self, tmp_path):
        """Test that single application can add multiple providers."""
        # Create application registry with multiple providers
        app_file = tmp_path / "app" / "registry.py"
        app_file.parent.mkdir(parents=True)
        app_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, extend_framework_registry, ProviderRegistration

class AppProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            capabilities=[],
            context_classes=[],
            providers=[
                ProviderRegistration(
                    module_path="app.providers.custom1",
                    class_name="Custom1Provider"
                ),
                ProviderRegistration(
                    module_path="app.providers.custom2",
                    class_name="Custom2Provider"
                )
            ]
        )
"""
        )

        # Load registry
        manager = RegistryManager(registry_path=str(app_file))

        # Both application providers should be in merged config
        provider_module_paths = [p.module_path for p in manager.config.providers]
        assert "app.providers.custom1" in provider_module_paths
        assert "app.providers.custom2" in provider_module_paths

    def test_empty_providers_list_handled_correctly(self, tmp_path):
        """Test that empty providers list doesn't cause errors."""
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class AppProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            capabilities=[],
            context_classes=[],
            providers=[]  # Empty list
        )
"""
        )

        # Should not raise
        manager = RegistryManager(registry_path=str(registry_file))

        # Framework providers should still be present
        assert len(manager.config.providers) > 0


class TestProviderExclusions:
    """Test excluding framework providers."""

    def test_exclude_framework_provider(self, tmp_path):
        """Test excluding specific framework providers."""
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class AppProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            capabilities=[],
            context_classes=[],
            exclude_providers=["anthropic", "google"]
        )
"""
        )

        # Load registry
        manager = RegistryManager(registry_path=str(registry_file))

        # Verify exclusions were recorded
        assert "anthropic" in manager._excluded_provider_names
        assert "google" in manager._excluded_provider_names

    def test_exclude_all_framework_providers(self, tmp_path):
        """Test excluding all framework providers (custom-only setup)."""
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, extend_framework_registry, ProviderRegistration

class AppProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            capabilities=[],
            context_classes=[],
            providers=[
                ProviderRegistration(
                    module_path="app.providers.only",
                    class_name="OnlyProvider"
                )
            ],
            exclude_providers=["anthropic", "openai", "google", "ollama", "cborg"]
        )
"""
        )

        manager = RegistryManager(registry_path=str(registry_file))

        # All five framework providers should be in exclusion list
        expected_exclusions = ["anthropic", "openai", "google", "ollama", "cborg"]
        assert len(manager._excluded_provider_names) == len(expected_exclusions)
        for provider in expected_exclusions:
            assert provider in manager._excluded_provider_names

        # Custom provider should still be in config
        provider_modules = [p.module_path for p in manager.config.providers]
        assert "app.providers.only" in provider_modules


class TestHelperFunctionProviderSupport:
    """Test extend_framework_registry helper with provider parameters."""

    def test_helper_with_providers_parameter(self):
        """Test that helper function accepts providers parameter."""
        config = extend_framework_registry(
            capabilities=[],
            providers=[
                ProviderRegistration(
                    module_path="my_app.providers.custom", class_name="CustomProvider"
                )
            ],
        )

        # Returns application config with providers
        assert len(config.providers) == 1
        assert config.providers[0].module_path == "my_app.providers.custom"
        assert config.providers[0].class_name == "CustomProvider"

    def test_helper_with_exclude_providers(self):
        """Test that helper function accepts exclude_providers parameter."""
        config = extend_framework_registry(
            capabilities=[], exclude_providers=["anthropic", "google"]
        )

        # Exclusions stored in framework_exclusions
        assert config.framework_exclusions is not None
        assert "providers" in config.framework_exclusions
        assert "anthropic" in config.framework_exclusions["providers"]
        assert "google" in config.framework_exclusions["providers"]

    def test_helper_with_override_providers(self):
        """Test that helper function accepts override_providers parameter."""
        override_provider = ProviderRegistration(
            module_path="my_app.providers.custom_anthropic", class_name="CustomAnthropicProvider"
        )

        config = extend_framework_registry(capabilities=[], override_providers=[override_provider])

        # Override provider included in config
        assert len(config.providers) == 1
        assert config.providers[0].module_path == "my_app.providers.custom_anthropic"

    def test_helper_with_providers_and_overrides(self):
        """Test combining regular providers and overrides."""
        regular_provider = ProviderRegistration(
            module_path="my_app.providers.new", class_name="NewProvider"
        )

        override_provider = ProviderRegistration(
            module_path="my_app.providers.custom_openai", class_name="CustomOpenAIProvider"
        )

        config = extend_framework_registry(
            capabilities=[], providers=[regular_provider], override_providers=[override_provider]
        )

        # Both should be included
        assert len(config.providers) == 2
        provider_modules = [p.module_path for p in config.providers]
        assert "my_app.providers.new" in provider_modules
        assert "my_app.providers.custom_openai" in provider_modules

    def test_helper_used_in_registry_provider(self, tmp_path):
        """Test that helper with providers works in actual registry provider file.

        This validates the most common use case: an application using the
        extend_framework_registry helper to add custom providers.
        """
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import (
    RegistryConfigProvider,
    RegistryConfig,
    extend_framework_registry,
    ProviderRegistration
)

class AppProvider(RegistryConfigProvider):
    def get_registry_config(self) -> RegistryConfig:
        return extend_framework_registry(
            capabilities=[],
            providers=[
                ProviderRegistration(
                    module_path="app.providers.custom",
                    class_name="CustomAIProvider"
                )
            ]
        )
"""
        )

        manager = RegistryManager(registry_path=str(registry_file))
        provider_modules = [p.module_path for p in manager.config.providers]

        # Custom provider should be added
        assert "app.providers.custom" in provider_modules

        # Framework providers should still be present
        assert "osprey.models.providers.anthropic" in provider_modules
        assert "osprey.models.providers.openai" in provider_modules

    def test_helper_with_exclusions_in_registry_provider(self, tmp_path):
        """Test helper with provider exclusions in actual registry."""
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import (
    RegistryConfigProvider,
    RegistryConfig,
    extend_framework_registry,
    ProviderRegistration
)

class AppProvider(RegistryConfigProvider):
    def get_registry_config(self) -> RegistryConfig:
        return extend_framework_registry(
            capabilities=[],
            providers=[
                ProviderRegistration(
                    module_path="app.providers.custom",
                    class_name="CustomProvider"
                )
            ],
            exclude_providers=["anthropic", "google"]
        )
"""
        )

        manager = RegistryManager(registry_path=str(registry_file))

        # Exclusions should be recorded
        assert "anthropic" in manager._excluded_provider_names
        assert "google" in manager._excluded_provider_names

        # Custom provider should be in config
        provider_modules = [p.module_path for p in manager.config.providers]
        assert "app.providers.custom" in provider_modules


class TestProviderRegistrationDataclass:
    """Test ProviderRegistration dataclass behavior."""

    def test_provider_registration_in_registry_config(self):
        """Test ProviderRegistration can be used in RegistryConfig."""
        provider_reg = ProviderRegistration(module_path="test.provider", class_name="TestProvider")

        config = RegistryConfig(capabilities=[], context_classes=[], providers=[provider_reg])

        assert len(config.providers) == 1
        assert config.providers[0] is provider_reg


class TestProviderIntegrationScenarios:
    """Test realistic provider registration scenarios.

    These tests demonstrate complete, production-ready patterns for
    customizing provider configuration in real applications.
    """

    def test_replace_framework_provider_scenario(self, tmp_path):
        """Test scenario where app replaces a framework provider with custom version."""
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import (
    RegistryConfigProvider,
    extend_framework_registry,
    ProviderRegistration
)

class AppProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            capabilities=[],
            override_providers=[
                ProviderRegistration(
                    module_path="my_app.providers.custom_openai",
                    class_name="CustomOpenAIProvider"
                )
            ],
            exclude_providers=["openai"]  # Exclude framework OpenAI
        )
"""
        )

        manager = RegistryManager(registry_path=str(registry_file))

        # Framework OpenAI should be excluded
        assert "openai" in manager._excluded_provider_names

        # Custom provider should be present
        provider_modules = [p.module_path for p in manager.config.providers]
        assert "my_app.providers.custom_openai" in provider_modules

        # Other framework providers should still be in config (not excluded)
        assert "osprey.models.providers.anthropic" in provider_modules


class TestProviderMergingEdgeCases:
    """Test edge cases in provider merging."""

    def test_no_providers_in_application_registry(self, tmp_path):
        """Test that applications without providers work fine."""
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class AppProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            capabilities=[],
            context_classes=[]
            # No providers field at all
        )
"""
        )

        # Should not raise
        manager = RegistryManager(registry_path=str(registry_file))

        # Framework providers should still be present
        framework_provider_modules = [
            "osprey.models.providers.anthropic",
            "osprey.models.providers.openai",
        ]
        provider_modules = [p.module_path for p in manager.config.providers]
        for framework_module in framework_provider_modules:
            assert framework_module in provider_modules

    def test_framework_only_registry_has_providers(self):
        """Test that framework-only registry includes default providers."""
        manager = RegistryManager(registry_path=None)

        # Framework should have providers
        assert len(manager.config.providers) > 0

        # Should include standard framework providers
        provider_modules = [p.module_path for p in manager.config.providers]
        assert "osprey.models.providers.anthropic" in provider_modules
        assert "osprey.models.providers.openai" in provider_modules


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
