"""Tests for the EmbeddingProviderRegistry."""

import pytest

from osprey.models.embedding_registry import (
    EmbeddingProviderRegistry,
    get_embedding_registry,
    reset_embedding_registry,
)


@pytest.fixture(autouse=True)
def _clean_singleton():
    """Reset the singleton before and after every test."""
    reset_embedding_registry()
    yield
    reset_embedding_registry()


class TestEmbeddingProviderRegistry:
    """Unit tests for EmbeddingProviderRegistry."""

    @pytest.mark.unit
    def test_get_builtin_ollama(self):
        """Built-in 'ollama' resolves to OllamaEmbeddingProvider."""
        reg = EmbeddingProviderRegistry()
        cls = reg.get_provider("ollama")
        assert cls is not None
        assert cls.name == "ollama"

    @pytest.mark.unit
    def test_get_builtin_openai(self):
        """Built-in 'openai' resolves to OpenAIEmbeddingProvider."""
        reg = EmbeddingProviderRegistry()
        cls = reg.get_provider("openai")
        assert cls is not None
        assert cls.name == "openai"

    @pytest.mark.unit
    def test_get_unknown_returns_none(self):
        """Unknown provider name returns None, never raises."""
        reg = EmbeddingProviderRegistry()
        assert reg.get_provider("does_not_exist") is None

    @pytest.mark.unit
    def test_register_custom_provider(self):
        """Custom providers registered at runtime are resolvable."""
        reg = EmbeddingProviderRegistry()
        reg.register_provider(
            "ollama_custom",
            "osprey.models.embeddings.ollama",
            "OllamaEmbeddingProvider",
        )
        cls = reg.get_provider("ollama_custom")
        assert cls is not None
        assert cls.name == "ollama"

    @pytest.mark.unit
    def test_list_providers_contains_builtins(self):
        """list_providers returns all built-in embedding provider names."""
        reg = EmbeddingProviderRegistry()
        names = reg.list_providers()
        assert set(names) == {"ollama", "openai"}
        assert len(names) == 2

    @pytest.mark.unit
    def test_singleton_identity(self):
        """get_embedding_registry() returns the same instance."""
        a = get_embedding_registry()
        b = get_embedding_registry()
        assert a is b

    @pytest.mark.unit
    def test_lazy_load_caches(self):
        """Second get_provider call returns cached class (no re-import)."""
        reg = EmbeddingProviderRegistry()
        first = reg.get_provider("ollama")
        second = reg.get_provider("ollama")
        assert first is second

    @pytest.mark.unit
    def test_register_override_evicts_cache(self):
        """Overwriting an existing entry clears the cache for that name."""
        reg = EmbeddingProviderRegistry()
        reg.get_provider("ollama")
        assert "ollama" in reg._providers

        reg.register_provider(
            "ollama",
            "osprey.models.embeddings.openai",
            "OpenAIEmbeddingProvider",
        )
        assert "ollama" not in reg._providers
        cls = reg.get_provider("ollama")
        assert cls is not None
        assert cls.name == "openai"
