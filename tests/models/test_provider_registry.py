"""Tests for the lightweight ProviderRegistry."""

import pytest

from osprey.models.provider_registry import (
    ProviderRegistry,
    get_provider_registry,
    reset_provider_registry,
)


@pytest.fixture(autouse=True)
def _clean_singleton():
    """Reset the singleton before and after every test."""
    reset_provider_registry()
    yield
    reset_provider_registry()


class TestProviderRegistry:
    """Unit tests for ProviderRegistry."""

    @pytest.mark.unit
    def test_get_builtin_provider(self):
        """Built-in providers (e.g. anthropic) resolve to a class."""
        reg = ProviderRegistry()
        cls = reg.get_provider("anthropic")
        assert cls is not None
        assert cls.name == "anthropic"

    @pytest.mark.unit
    def test_get_unknown_returns_none(self):
        """Unknown provider name returns None, never raises."""
        reg = ProviderRegistry()
        assert reg.get_provider("does_not_exist") is None

    @pytest.mark.unit
    def test_register_custom_provider(self):
        """Custom providers registered at runtime are resolvable."""
        reg = ProviderRegistry()
        reg.register_provider(
            "anthropic_custom",
            "osprey.models.providers.anthropic",
            "AnthropicProviderAdapter",
        )
        cls = reg.get_provider("anthropic_custom")
        assert cls is not None
        assert cls.name == "anthropic"

    @pytest.mark.unit
    def test_list_providers_contains_all_builtins(self):
        """list_providers returns all 11 built-in names."""
        reg = ProviderRegistry()
        names = reg.list_providers()
        expected = {
            "anthropic",
            "openai",
            "google",
            "ollama",
            "cborg",
            "amsc",
            "als-apg",
            "stanford",
            "argo",
            "asksage",
            "vllm",
        }
        assert expected == set(names)
        assert len(names) == 11

    @pytest.mark.unit
    def test_singleton_identity(self):
        """get_provider_registry() returns the same instance."""
        a = get_provider_registry()
        b = get_provider_registry()
        assert a is b

    @pytest.mark.unit
    def test_load_providers_config_filtered(self):
        """load_providers with configured_names only loads those."""
        reg = ProviderRegistry()
        result = reg.load_providers(configured_names={"anthropic", "cborg"})
        assert set(result.keys()) == {"anthropic", "cborg"}

    @pytest.mark.unit
    def test_load_providers_exclusion_filtered(self):
        """load_providers with excluded_names skips those."""
        reg = ProviderRegistry()
        result = reg.load_providers(excluded_names={"anthropic", "openai"})
        assert "anthropic" not in result
        assert "openai" not in result
        # All others should be present (may fail if import fails on CI)
        assert "cborg" in result

    @pytest.mark.unit
    def test_lazy_load_caches(self):
        """Second get_provider call returns cached class (no re-import)."""
        reg = ProviderRegistry()
        first = reg.get_provider("anthropic")
        second = reg.get_provider("anthropic")
        assert first is second

    @pytest.mark.unit
    def test_register_override_evicts_cache(self):
        """Overwriting an existing entry clears the cache for that name."""
        reg = ProviderRegistry()
        # Load anthropic into cache
        reg.get_provider("anthropic")
        assert "anthropic" in reg._providers

        # Override with a different class
        reg.register_provider(
            "anthropic",
            "osprey.models.providers.openai",
            "OpenAIProviderAdapter",
        )
        # Cache should be evicted
        assert "anthropic" not in reg._providers
        cls = reg.get_provider("anthropic")
        assert cls is not None
        assert cls.name == "openai"
