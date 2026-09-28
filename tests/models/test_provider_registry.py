"""Tests for the lightweight ProviderRegistry."""

import subprocess
import sys

import pytest

from osprey.models.provider_registry import (
    _BUILTIN_PROVIDERS,
    PROVIDER_API_KEYS,
    ProviderRegistry,
    get_provider_registry,
    reset_provider_registry,
)
from osprey.models.providers.base import BaseProvider

#: The built-in providers in the order ``.env.example`` and the build's key
#: summary list them.
BUILTIN_ORDER = [
    "anthropic",
    "openai",
    "google",
    "cborg",
    "amsc-i2",
    "argo",
    "stanford",
    "als-apg",
    "ollama",
    "asksage",
    "vllm",
    "ds4",
]


class _SiteGatewayAdapter(BaseProvider):
    name = "site-gateway"
    description = "A gateway a site registers for itself"
    requires_api_key = True
    api_key_env_var = "SITE_GATEWAY_TOKEN"


class _SiteCborgAdapter(BaseProvider):
    name = "cborg"
    description = "A site's own class registered under a built-in name"
    requires_api_key = True
    api_key_env_var = "SITE_CBORG_TOKEN"
    api_protocol = "anthropic"


@pytest.fixture(autouse=True)
def _clean_singleton():
    """Reset the singleton before and after every test."""
    reset_provider_registry()
    yield
    reset_provider_registry()


class TestProviderRegistry:
    """Unit tests for ProviderRegistry."""

    def test_get_builtin_provider(self):
        """Built-in providers (e.g. anthropic) resolve to a class."""
        reg = ProviderRegistry()
        cls = reg.get_provider("anthropic")
        assert cls is not None
        assert cls.name == "anthropic"

    def test_get_unknown_returns_none(self):
        """Unknown provider name returns None, never raises."""
        reg = ProviderRegistry()
        assert reg.get_provider("does_not_exist") is None

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

    def test_list_providers_contains_all_builtins(self):
        """list_providers returns all 12 built-in names."""
        reg = ProviderRegistry()
        names = reg.list_providers()
        expected = {
            "anthropic",
            "openai",
            "google",
            "ollama",
            "cborg",
            "amsc-i2",
            "als-apg",
            "stanford",
            "argo",
            "asksage",
            "vllm",
            "ds4",
        }
        assert expected == set(names)
        assert len(names) == 12

    def test_singleton_identity(self):
        """get_provider_registry() returns the same instance."""
        a = get_provider_registry()
        b = get_provider_registry()
        assert a is b

    def test_load_providers_config_filtered(self):
        """load_providers with configured_names only loads those."""
        reg = ProviderRegistry()
        result = reg.load_providers(configured_names={"anthropic", "cborg"})
        assert set(result.keys()) == {"anthropic", "cborg"}

    def test_load_providers_exclusion_filtered(self):
        """load_providers with excluded_names skips those."""
        reg = ProviderRegistry()
        result = reg.load_providers(excluded_names={"anthropic", "openai"})
        assert "anthropic" not in result
        assert "openai" not in result
        # All others should be present (may fail if import fails on CI)
        assert "cborg" in result

    def test_load_providers_skips_failed_imports(self):
        """A configured provider whose module can't be imported is silently
        dropped from the result (not crashing the bulk load) while valid
        siblings still load. This is the air-gapped behavior the `if cls is not
        None` guard exists for — exercised here via a deliberately broken entry.
        """
        reg = ProviderRegistry()
        reg.register_provider(
            "broken_provider",
            "osprey.models.providers.does_not_exist_xyz",
            "NoSuchAdapter",
        )
        result = reg.load_providers(configured_names={"anthropic", "broken_provider"})
        assert "broken_provider" not in result
        assert "anthropic" in result

    def test_lazy_load_caches(self):
        """Second get_provider call returns cached class (no re-import)."""
        reg = ProviderRegistry()
        first = reg.get_provider("anthropic")
        second = reg.get_provider("anthropic")
        assert first is second

    def test_ds4_is_registered(self):
        """ds4 resolves to its adapter and is keyless."""
        from osprey.models.provider_registry import PROVIDER_API_KEYS

        reg = ProviderRegistry()
        cls = reg.get_provider("ds4")
        assert cls is not None
        assert cls.name == "ds4"
        assert PROVIDER_API_KEYS.get("ds4", "MISSING") is None

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

    def test_the_registry_answer_agrees_with_every_adapter(self):
        """A key variable is named exactly when the adapter requires a key.

        The credential gate and the launch both read the registry's answer.
        """
        reg = ProviderRegistry()
        for name, var in reg.api_key_env_vars().items():
            cls = reg.get_provider(name)
            assert cls is not None, name
            assert (var is not None) == cls.requires_api_key, name

    def test_the_key_view_is_read_from_the_builtin_entries(self):
        assert dict(PROVIDER_API_KEYS) == {
            name: entry.key_env_var for name, entry in _BUILTIN_PROVIDERS.items()
        }
        with pytest.raises(TypeError):
            PROVIDER_API_KEYS["openai"] = "X"  # type: ignore[index]

    def test_key_variables_keep_the_builtin_order(self):
        assert list(ProviderRegistry().api_key_env_vars()) == BUILTIN_ORDER

    def test_a_builtin_answers_without_importing_its_adapter(self):
        code = (
            "import sys\n"
            "from osprey.models.provider_registry import get_provider_registry\n"
            "reg = get_provider_registry()\n"
            f"for name in {BUILTIN_ORDER!r}:\n"
            "    reg.api_key_env_var(name)\n"
            "reg.api_key_env_vars()\n"
            "print('litellm' in sys.modules,"
            " [m for m in sys.modules if m.startswith('osprey.models.providers.')])\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "False []"

    def test_a_registered_provider_answers_from_its_class(self):
        reg = ProviderRegistry()
        reg.register_provider("site-gateway", __name__, "_SiteGatewayAdapter")
        assert reg.api_key_env_var("site-gateway") == "SITE_GATEWAY_TOKEN"
        answers = list(reg.api_key_env_vars().items())
        assert [name for name, _ in answers[:-1]] == BUILTIN_ORDER
        assert answers[-1] == ("site-gateway", "SITE_GATEWAY_TOKEN")

    def test_a_registration_that_replaces_a_builtin_answers_from_its_class(self):
        reg = ProviderRegistry()
        reg.register_provider("cborg", __name__, "_SiteCborgAdapter")
        assert reg.api_key_env_var("cborg") == "SITE_CBORG_TOKEN"
        assert reg.api_key_env_vars()["cborg"] == "SITE_CBORG_TOKEN"

    def test_unknown_and_unloadable_names_have_no_key_variable(self):
        reg = ProviderRegistry()
        assert reg.api_key_env_var("does_not_exist") is None
        reg.register_provider(
            "broken", "osprey.models.providers.does_not_exist_xyz", "NoSuchAdapter"
        )
        assert reg.api_key_env_var("broken") is None
        assert "broken" not in reg.api_key_env_vars()
