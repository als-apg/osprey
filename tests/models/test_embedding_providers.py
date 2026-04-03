"""Tests for embedding provider classes and the factory function."""

import pytest

from osprey.models.embedding_registry import reset_embedding_registry
from osprey.models.embeddings import get_embedding_provider
from osprey.models.embeddings.base import BaseEmbeddingProvider
from osprey.models.embeddings.ollama import OllamaEmbeddingProvider
from osprey.models.embeddings.openai import OpenAIEmbeddingProvider


@pytest.fixture(autouse=True)
def _clean_singleton():
    """Reset the embedding registry singleton between tests."""
    reset_embedding_registry()
    yield
    reset_embedding_registry()


class TestOllamaEmbeddingProviderAttributes:
    """Verify OllamaEmbeddingProvider class attributes."""

    @pytest.mark.unit
    def test_name(self):
        assert OllamaEmbeddingProvider.name == "ollama"

    @pytest.mark.unit
    def test_requires_api_key(self):
        assert OllamaEmbeddingProvider.requires_api_key is False

    @pytest.mark.unit
    def test_requires_base_url(self):
        assert OllamaEmbeddingProvider.requires_base_url is True

    @pytest.mark.unit
    def test_litellm_prefix(self):
        assert OllamaEmbeddingProvider.litellm_prefix == "ollama"

    @pytest.mark.unit
    def test_is_base_subclass(self):
        assert issubclass(OllamaEmbeddingProvider, BaseEmbeddingProvider)


class TestOpenAIEmbeddingProviderAttributes:
    """Verify OpenAIEmbeddingProvider class attributes."""

    @pytest.mark.unit
    def test_name(self):
        assert OpenAIEmbeddingProvider.name == "openai"

    @pytest.mark.unit
    def test_requires_api_key(self):
        assert OpenAIEmbeddingProvider.requires_api_key is True

    @pytest.mark.unit
    def test_requires_base_url(self):
        assert OpenAIEmbeddingProvider.requires_base_url is False

    @pytest.mark.unit
    def test_litellm_prefix(self):
        assert OpenAIEmbeddingProvider.litellm_prefix == ""

    @pytest.mark.unit
    def test_default_model_id(self):
        assert OpenAIEmbeddingProvider.default_model_id == "text-embedding-3-small"

    @pytest.mark.unit
    def test_is_openai_compatible(self):
        assert OpenAIEmbeddingProvider.is_openai_compatible is True

    @pytest.mark.unit
    def test_default_base_url_is_none(self):
        assert OpenAIEmbeddingProvider.default_base_url is None

    @pytest.mark.unit
    def test_is_base_subclass(self):
        assert issubclass(OpenAIEmbeddingProvider, BaseEmbeddingProvider)


class TestGetEmbeddingProvider:
    """Tests for the get_embedding_provider factory function."""

    @pytest.mark.unit
    def test_get_ollama(self):
        provider = get_embedding_provider("ollama")
        assert isinstance(provider, OllamaEmbeddingProvider)

    @pytest.mark.unit
    def test_get_openai(self):
        provider = get_embedding_provider("openai")
        assert isinstance(provider, OpenAIEmbeddingProvider)

    @pytest.mark.unit
    def test_unknown_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown embedding provider: 'nonexistent'"):
            get_embedding_provider("nonexistent")

    @pytest.mark.unit
    def test_returns_new_instance_each_call(self):
        """Each call returns a fresh provider instance."""
        a = get_embedding_provider("ollama")
        b = get_embedding_provider("ollama")
        assert a is not b
        assert type(a) is type(b)
