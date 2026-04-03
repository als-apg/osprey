"""Tests for custom AI model provider registration.

This test module validates the provider registration feature that allows
applications to register custom AI model providers through the registry system.

Architecture note: Built-in providers (anthropic, openai, etc.) now live in
``osprey.models.provider_registry`` as the single source of truth.
``RegistryManager._initialize_providers()`` delegates to it. Tests that
previously checked ``manager.config.providers`` for built-in entries now check
``get_provider_registry()`` instead.

Tests cover:
- Registering custom providers in application registries
- Provider merging during registry initialization
- Provider exclusions (excluding framework providers)
- Helper function support for providers
- Provider override scenarios
"""

import pytest

from osprey.models.provider_registry import get_provider_registry, reset_provider_registry
from osprey.registry.base import (
    ProviderRegistration,
    RegistryConfig,
)
from osprey.registry.helpers import extend_framework_registry
from osprey.registry.manager import RegistryManager


@pytest.fixture(autouse=True)
def _clean_provider_registry():
    """Reset the provider registry singleton around each test."""
    reset_provider_registry()
    yield
    reset_provider_registry()


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

        manager = RegistryManager(registry_path=str(registry_file))

        # Custom providers should be in config.providers
        provider_module_paths = [p.module_path for p in manager.config.providers]
        assert "test_app.providers.custom" in provider_module_paths
        assert "test_app.providers.institutional" in provider_module_paths

        # Built-in providers are resolved via ProviderRegistry (not config.providers)
        pr = get_provider_registry()
        assert pr.get_provider("anthropic") is not None
        assert pr.get_provider("openai") is not None

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
            providers=[]  # Empty list
        )
"""
        )

        # Should not raise
        RegistryManager(registry_path=str(registry_file))

        # Built-in providers should still be resolvable via ProviderRegistry
        pr = get_provider_registry()
        assert len(pr.list_providers()) >= 9


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
            providers=[
                ProviderRegistration(
                    module_path="app.providers.only",
                    class_name="OnlyProvider"
                )
            ],
            exclude_providers=["anthropic", "openai", "google", "ollama", "cborg", "amsc"]
        )
"""
        )

        manager = RegistryManager(registry_path=str(registry_file))

        # All listed framework providers should be in exclusion list
        expected_exclusions = ["anthropic", "openai", "google", "ollama", "cborg", "amsc"]
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
        config = extend_framework_registry(exclude_providers=["anthropic", "google"])

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

        config = extend_framework_registry(override_providers=[override_provider])

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
            providers=[regular_provider], override_providers=[override_provider]
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

        # Custom provider should be in config.providers
        assert "app.providers.custom" in provider_modules

        # Built-in providers are resolved via ProviderRegistry
        pr = get_provider_registry()
        assert pr.get_provider("anthropic") is not None
        assert pr.get_provider("openai") is not None

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

        config = RegistryConfig(providers=[provider_reg])

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

        # Custom provider should be present in config.providers
        provider_modules = [p.module_path for p in manager.config.providers]
        assert "my_app.providers.custom_openai" in provider_modules

        # Other built-in providers still resolvable via ProviderRegistry
        pr = get_provider_registry()
        assert pr.get_provider("anthropic") is not None


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
        return extend_framework_registry()  # No providers field at all
"""
        )

        # Should not raise
        RegistryManager(registry_path=str(registry_file))

        # Built-in providers resolved via ProviderRegistry (not config.providers)
        pr = get_provider_registry()
        assert pr.get_provider("anthropic") is not None
        assert pr.get_provider("openai") is not None

    def test_framework_only_registry_has_providers(self):
        """Test that framework-only setup still resolves built-in providers."""
        RegistryManager(registry_path=None)

        # config.providers is now empty (built-ins live in ProviderRegistry)
        # but providers are still resolvable
        pr = get_provider_registry()
        assert len(pr.list_providers()) >= 9
        assert pr.get_provider("anthropic") is not None
        assert pr.get_provider("openai") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
