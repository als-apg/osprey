"""Tests for the lightweight ProviderRegistry."""

import subprocess
import sys

import pytest

from osprey.models.provider_registry import (
    _BUILTIN_PROVIDERS,
    PROVIDER_API_KEYS,
    ProviderRegistry,
    _ProviderEntry,
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


class _SiteEmbeddingOnlyAdapter(BaseProvider):
    name = "site-embedder"
    description = "A site's embedding endpoint with no chat route"
    requires_api_key = True
    api_key_env_var = "SITE_EMBEDDER_TOKEN"

    def execute_embedding(self, texts, *_args, **_kwargs):
        return [[0.0] for _ in texts]


#: A registered class need not subclass ``BaseProvider``; this one declares
#: nothing beyond a name and a protocol.
_DuckSiteGatewayAdapter = type(
    "SiteGatewayAdapter", (), {"name": "duck-gateway", "api_protocol": "openai"}
)


def _registry_with_a_non_chat_builtin() -> ProviderRegistry:
    """A registry whose table carries one synthetic built-in row that is not chat."""
    reg = ProviderRegistry()
    reg._entries["embed-only"] = _ProviderEntry(
        "osprey.models.providers.does_not_exist_xyz",
        "NoSuchAdapter",
        key_env_var="EMBED_ONLY_KEY",
        api_protocol="openai",
        chat=False,
    )
    return reg


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
        """list_providers returns all 13 built-in names."""
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
            "llama-cpp",
        }
        assert expected == set(names)
        assert len(names) == 13

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

    def test_exclude_removes_the_entry_and_the_cached_class(self):
        """An excluded name is gone from lookup, listing, the cache and the bulk load."""
        reg = ProviderRegistry()
        reg.get_provider("openai")
        assert "openai" in reg._providers

        reg.exclude("openai")

        assert reg.get_provider("openai") is None
        assert "openai" not in reg.list_providers()
        assert "openai" not in reg._providers
        assert "openai" not in reg.load_providers()
        assert reg.get_provider("anthropic") is not None

    def test_exclude_of_an_unknown_name_changes_nothing(self):
        """Excluding a name the table does not carry is a no-op."""
        reg = ProviderRegistry()
        before = reg.list_providers()

        reg.exclude("does_not_exist")

        assert reg.list_providers() == before

    def test_register_after_exclude_restores_the_name(self):
        """A registration under an excluded name adds it back."""
        reg = ProviderRegistry()
        reg.exclude("anthropic")

        reg.register_provider(
            "anthropic", "osprey.models.providers.openai", "OpenAIProviderAdapter"
        )

        provider = reg.get_provider("anthropic")
        assert provider is not None
        assert provider.name == "openai"

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
            name: entry.key_env_var for name, entry in _BUILTIN_PROVIDERS.items() if entry.chat
        }
        assert "llama-cpp" not in PROVIDER_API_KEYS
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


class TestTheChatFact:
    """The table says which providers serve chat; registrations answer from their class."""

    def test_an_entry_defaults_to_chat(self):
        assert _ProviderEntry("m", "C").chat is True

    def test_the_chat_rows_are_the_chat_builtins_and_llama_cpp_is_not_one(self):
        assert [name for name, entry in _BUILTIN_PROVIDERS.items() if entry.chat] == BUILTIN_ORDER
        assert _BUILTIN_PROVIDERS["llama-cpp"].chat is False
        assert _BUILTIN_PROVIDERS["llama-cpp"].key_env_var is None
        assert _BUILTIN_PROVIDERS["llama-cpp"].api_protocol == "openai"

    def test_the_key_view_lists_chat_rows_only(self):
        assert dict(PROVIDER_API_KEYS) == {
            name: entry.key_env_var for name, entry in _BUILTIN_PROVIDERS.items() if entry.chat
        }

    def test_a_non_chat_row_is_filtered_without_importing_any_adapter(self, monkeypatch):
        reg = _registry_with_a_non_chat_builtin()

        def _refuse(_self, name, _entry):
            raise AssertionError(f"adapter import attempted for {name}")

        monkeypatch.setattr(ProviderRegistry, "_load", _refuse)

        assert "embed-only" in reg.list_providers()
        assert reg.list_providers(chat_only=True) == sorted(BUILTIN_ORDER)
        assert reg.is_chat("embed-only") is False
        assert reg.is_chat("anthropic") is True

    def test_a_non_chat_row_has_no_key_variable_entry(self):
        reg = _registry_with_a_non_chat_builtin()
        assert list(reg.api_key_env_vars()) == BUILTIN_ORDER

    def test_is_chat_for_a_builtin_answers_from_the_table(self, monkeypatch):
        reg = ProviderRegistry()
        monkeypatch.setattr(
            ProviderRegistry,
            "_load",
            lambda self, name, entry: (_ for _ in ()).throw(AssertionError(name)),
        )
        assert reg.is_chat("openai") is True

    def test_a_registered_embedding_only_class_is_not_chat(self):
        reg = ProviderRegistry()
        reg.register_provider("site-embedder", __name__, "_SiteEmbeddingOnlyAdapter")

        assert reg.is_chat("site-embedder") is False
        assert "site-embedder" in reg.list_providers()
        assert "site-embedder" not in reg.list_providers(chat_only=True)
        assert "site-embedder" not in reg.api_key_env_vars()

    def test_a_registered_stub_that_overrides_nothing_stays_a_chat_row(self):
        reg = ProviderRegistry()
        reg.register_provider("site-gateway", __name__, "_SiteGatewayAdapter")
        reg.register_provider("cborg", __name__, "_SiteCborgAdapter")

        assert reg.is_chat("site-gateway") is True
        assert reg.is_chat("cborg") is True
        assert {"site-gateway", "cborg"} <= set(reg.list_providers(chat_only=True))
        assert reg.api_key_env_vars()["site-gateway"] == "SITE_GATEWAY_TOKEN"
        assert reg.api_key_env_vars()["cborg"] == "SITE_CBORG_TOKEN"

    def test_a_duck_typed_registration_reads_as_chat(self):
        reg = ProviderRegistry()
        reg.register_provider("duck-gateway", __name__, "_DuckSiteGatewayAdapter")

        assert reg.is_chat("duck-gateway") is True
        assert "duck-gateway" in reg.list_providers(chat_only=True)
        keys = reg.api_key_env_vars()
        assert keys["duck-gateway"] is None
        assert list(keys)[:-1] == BUILTIN_ORDER
        assert reg.api_protocol("duck-gateway") == "openai"

    def test_an_unknown_or_unloadable_name_reads_as_chat(self):
        """So a chat call still reports its own ``Unknown provider`` error."""
        reg = ProviderRegistry()
        assert reg.is_chat("nope") is True
        reg.register_provider(
            "broken", "osprey.models.providers.does_not_exist_xyz", "NoSuchAdapter"
        )
        assert reg.is_chat("broken") is True
