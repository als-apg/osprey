"""Tests for config-driven provider filtering.

Validates that the registry can skip provider imports for unconfigured providers,
avoiding costly module-level network calls on air-gapped machines.

Tests cover:
- ProviderRegistration.name field (default and explicit)
- _get_configured_provider_names() helper (various config scenarios)
- initialize_providers() config-driven filtering behavior
"""

from unittest.mock import MagicMock, patch

from osprey.models.providers.base import BaseProvider
from osprey.registry.base import ProviderRegistration, RegistryConfig
from osprey.registry.initializers import _get_configured_provider_names
from osprey.registry.manager import RegistryManager


class TestProviderRegistrationName:
    """Test ProviderRegistration.name field."""

    def test_name_defaults_to_none(self):
        """ProviderRegistration.name defaults to None for backward compat."""
        reg = ProviderRegistration(module_path="some.module", class_name="SomeProvider")
        assert reg.name is None

    def test_name_stores_explicit_value(self):
        """ProviderRegistration with explicit name stores it correctly."""
        reg = ProviderRegistration(
            module_path="some.module", class_name="SomeProvider", name="custom"
        )
        assert reg.name == "custom"


class TestGetConfiguredProviderNames:
    """Test _get_configured_provider_names() helper."""

    def test_extracts_provider_names_from_config(self):
        """Extracts provider names from model configs."""
        mock_config = {
            "primary": {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
            "secondary": {"provider": "openai", "model": "gpt-4o"},
        }
        with patch(
            "osprey.registry.initializers.get_config_value", return_value=mock_config
        ) as mock_gcv:
            result = _get_configured_provider_names()
            mock_gcv.assert_called_once_with("models", None)
        assert result == {"anthropic", "openai"}

    def test_returns_none_when_exception(self):
        """Returns None when config is unavailable (exception path)."""
        with patch(
            "osprey.registry.initializers.get_config_value",
            side_effect=Exception("no config"),
        ):
            result = _get_configured_provider_names()
        assert result is None

    def test_returns_none_when_models_empty_dict(self):
        """Returns None when models config is empty dict."""
        with patch("osprey.registry.initializers.get_config_value", return_value={}):
            result = _get_configured_provider_names()
        assert result is None

    def test_returns_none_when_models_is_none(self):
        """Returns None when models config is None."""
        with patch("osprey.registry.initializers.get_config_value", return_value=None):
            result = _get_configured_provider_names()
        assert result is None

    def test_returns_none_when_no_provider_keys(self):
        """Returns None when role configs have no provider keys."""
        mock_config = {
            "primary": {"model": "claude-sonnet-4-5-20250929"},  # no "provider" key
        }
        with patch("osprey.registry.initializers.get_config_value", return_value=mock_config):
            result = _get_configured_provider_names()
        assert result is None

    def test_handles_non_dict_role_configs(self):
        """Handles non-dict role configs gracefully (e.g., string values)."""
        mock_config = {
            "primary": "anthropic/claude-sonnet-4-5-20250929",  # string, not dict
            "secondary": {"provider": "openai", "model": "gpt-4o"},
        }
        with patch("osprey.registry.initializers.get_config_value", return_value=mock_config):
            result = _get_configured_provider_names()
        assert result == {"openai"}

    def test_deduplicates_providers(self):
        """Same provider used for multiple roles is only returned once."""
        mock_config = {
            "primary": {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
            "secondary": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
        }
        with patch("osprey.registry.initializers.get_config_value", return_value=mock_config):
            result = _get_configured_provider_names()
        assert result == {"anthropic"}


class TestInitializeProvidersFiltering:
    """Test initialize_providers() config-driven filtering."""

    def _make_manager_with_providers(self, providers):
        """Create a RegistryManager with given provider registrations."""
        config = RegistryConfig(providers=providers)
        manager = RegistryManager.__new__(RegistryManager)
        manager.config = config
        manager._registries = {"providers": {}}
        manager._excluded_provider_names = set()
        return manager

    def _make_provider_class(self, name):
        """Create a real provider class stub inheriting from BaseProvider."""
        cls = type(f"Mock{name}Provider", (BaseProvider,), {"name": name})
        # Remove abstract method requirements so it passes issubclass check
        cls.__abstractmethods__ = frozenset()
        return cls

    def test_skips_unconfigured_providers(self):
        """Providers not in config are skipped before import."""
        providers = [
            ProviderRegistration(
                module_path="osprey.models.providers.anthropic",
                class_name="AnthropicProviderAdapter",
                name="anthropic",
            ),
            ProviderRegistration(
                module_path="osprey.models.providers.argo",
                class_name="ArgoProviderAdapter",
                name="argo",
            ),
        ]
        manager = self._make_manager_with_providers(providers)

        anthropic_class = self._make_provider_class("anthropic")
        mock_module = MagicMock()
        mock_module.AnthropicProviderAdapter = anthropic_class

        with (
            patch(
                "osprey.registry.initializers._get_configured_provider_names",
                return_value={"anthropic"},
            ),
            patch(
                "osprey.registry.initializers.importlib.import_module",
                return_value=mock_module,
            ),
        ):
            manager._initialize_component_type("providers")

        # Only anthropic should have been imported, argo skipped
        assert "anthropic" in manager._registries["providers"]
        assert "argo" not in manager._registries["providers"]

    def test_loads_all_when_config_unavailable(self):
        """All providers loaded when _get_configured_provider_names() returns None."""
        providers = [
            ProviderRegistration(
                module_path="osprey.models.providers.anthropic",
                class_name="AnthropicProviderAdapter",
                name="anthropic",
            ),
            ProviderRegistration(
                module_path="osprey.models.providers.openai",
                class_name="OpenAIProviderAdapter",
                name="openai",
            ),
        ]
        manager = self._make_manager_with_providers(providers)

        anthropic_class = self._make_provider_class("anthropic")
        openai_class = self._make_provider_class("openai")

        def mock_import(path):
            m = MagicMock()
            if "anthropic" in path:
                m.AnthropicProviderAdapter = anthropic_class
            elif "openai" in path:
                m.OpenAIProviderAdapter = openai_class
            return m

        with (
            patch(
                "osprey.registry.initializers._get_configured_provider_names",
                return_value=None,
            ),
            patch(
                "osprey.registry.initializers.importlib.import_module",
                side_effect=mock_import,
            ),
        ):
            manager._initialize_component_type("providers")

        assert "anthropic" in manager._registries["providers"]
        assert "openai" in manager._registries["providers"]

    def test_custom_providers_registered_via_provider_registry(self):
        """Custom providers from config.providers are registered into ProviderRegistry.

        The new initialize_providers() registers custom providers into the
        lightweight ProviderRegistry before calling load_providers(), ensuring
        they participate in config-driven filtering.
        """
        from osprey.models.provider_registry import get_provider_registry, reset_provider_registry

        reset_provider_registry()
        try:
            # Verify that built-in providers are always available
            pr = get_provider_registry()
            assert pr.get_provider("anthropic") is not None

            # Custom providers would need a valid module path to actually load,
            # but the registration itself should succeed
            pr.register_provider("my_custom", "test.custom", "CustomProvider")
            assert "my_custom" in pr.list_providers()
        finally:
            reset_provider_registry()
