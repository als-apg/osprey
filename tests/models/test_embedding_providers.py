"""Tests for embedding provider classes and the factory function."""

from unittest.mock import MagicMock, patch

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


class TestOpenAIEmbeddingBehavior:
    """Exercise OpenAIEmbeddingProvider behavior, not just class attributes."""

    @pytest.mark.unit
    def test_empty_texts_short_circuits(self):
        """Empty input returns [] without invoking litellm."""
        with patch("litellm.embedding") as mock_embed:
            result = OpenAIEmbeddingProvider().execute_embedding(
                texts=[], model_id="text-embedding-3-small", api_key="k"
            )
        assert result == []
        mock_embed.assert_not_called()

    @pytest.mark.unit
    def test_execute_embedding_forwards_optional_kwargs(self):
        """api_key/api_base/dimensions are conditionally forwarded and the
        response vectors are extracted from response.data."""
        with patch("litellm.embedding") as mock_embed:
            mock_embed.return_value = MagicMock(data=[{"embedding": [0.1, 0.2]}])
            result = OpenAIEmbeddingProvider().execute_embedding(
                texts=["hello"],
                model_id="text-embedding-3-small",
                api_key="sk-test",
                base_url="https://azure.example/v1",
                dimensions=256,
            )
        assert result == [[0.1, 0.2]]
        kwargs = mock_embed.call_args[1]
        assert kwargs["model"] == "text-embedding-3-small"  # OpenAI uses no prefix
        assert kwargs["input"] == ["hello"]
        assert kwargs["api_key"] == "sk-test"
        assert kwargs["api_base"] == "https://azure.example/v1"
        assert kwargs["dimensions"] == 256

    @pytest.mark.unit
    def test_execute_embedding_omits_unset_kwargs(self):
        """Without api_key/base_url/dimensions, those keys are not forwarded."""
        with patch("litellm.embedding") as mock_embed:
            mock_embed.return_value = MagicMock(data=[{"embedding": [1.0]}])
            OpenAIEmbeddingProvider().execute_embedding(
                texts=["x"], model_id="text-embedding-3-small"
            )
        kwargs = mock_embed.call_args[1]
        assert "api_key" not in kwargs
        assert "api_base" not in kwargs
        assert "dimensions" not in kwargs

    @pytest.mark.unit
    def test_check_health_requires_api_key(self):
        """Missing api_key is reported without any network call."""
        assert OpenAIEmbeddingProvider().check_health(api_key=None, base_url=None) == (
            False,
            "OpenAI API key is required",
        )

    @pytest.mark.unit
    def test_check_health_healthy(self):
        """A successful embed call reports healthy with the model name."""
        provider = OpenAIEmbeddingProvider()
        with patch.object(provider, "execute_embedding", return_value=[[0.0]]):
            ok, msg = provider.check_health(api_key="sk-test", base_url=None)
        assert ok is True
        assert "text-embedding-3-small" in msg

    @pytest.mark.unit
    def test_check_health_failure_is_wrapped(self):
        """An embed failure is caught and reported, not propagated."""
        provider = OpenAIEmbeddingProvider()
        with patch.object(provider, "execute_embedding", side_effect=RuntimeError("boom")):
            ok, msg = provider.check_health(api_key="sk-test", base_url=None)
        assert ok is False
        assert msg.startswith("OpenAI embedding health check failed")


class TestOllamaEmbeddingBehavior:
    """Exercise OllamaEmbeddingProvider's fallback/resolve/health logic."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "base_url,expected",
        [
            (
                "http://localhost:11434",
                [
                    "http://host.docker.internal:11434",
                    "http://host.containers.internal:11434",
                ],
            ),
            (
                "http://host.containers.internal:11434",
                [
                    "http://host.docker.internal:11434",
                    "http://localhost:11434",
                    "http://localhost:11434",
                ],
            ),
            (
                "http://host.docker.internal:11434",
                [
                    "http://host.containers.internal:11434",
                    "http://localhost:11434",
                    "http://localhost:11434",
                ],
            ),
            (
                "http://remote.example:11434",
                [
                    "http://localhost:11434",
                    "http://host.docker.internal:11434",
                    "http://host.containers.internal:11434",
                ],
            ),
        ],
    )
    def test_get_fallback_urls_branches(self, base_url, expected):
        """Each host-shape branch yields its exact ordered fallback list."""
        assert OllamaEmbeddingProvider._get_fallback_urls(base_url) == expected

    @pytest.mark.unit
    def test_empty_texts_short_circuits(self):
        """Empty input returns [] before any connection attempt."""
        with patch.object(OllamaEmbeddingProvider, "_test_connection") as mock_conn:
            result = OllamaEmbeddingProvider().execute_embedding(
                texts=[], model_id="nomic-embed-text"
            )
        assert result == []
        mock_conn.assert_not_called()

    @pytest.mark.unit
    def test_resolve_base_url_uses_primary_when_reachable(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        with patch.object(OllamaEmbeddingProvider, "_test_connection", return_value=True):
            resolved = OllamaEmbeddingProvider()._resolve_base_url("http://localhost:11434")
        assert resolved == "http://localhost:11434"

    @pytest.mark.unit
    def test_resolve_base_url_falls_back(self, monkeypatch):
        """Primary unreachable -> first reachable fallback is returned."""
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        # primary fails, first fallback (host.docker.internal) succeeds
        with patch.object(OllamaEmbeddingProvider, "_test_connection", side_effect=[False, True]):
            resolved = OllamaEmbeddingProvider()._resolve_base_url("http://localhost:11434")
        assert resolved == "http://host.docker.internal:11434"

    @pytest.mark.unit
    def test_resolve_base_url_raises_when_all_fail(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        with patch.object(OllamaEmbeddingProvider, "_test_connection", return_value=False):
            with pytest.raises(RuntimeError, match="Failed to connect to Ollama"):
                OllamaEmbeddingProvider()._resolve_base_url("http://localhost:11434")

    @pytest.mark.unit
    def test_execute_embedding_builds_litellm_model(self, monkeypatch):
        """Model string is 'ollama/<model>', resolved url is forwarded as
        api_base, and vectors are extracted from response.data."""
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        with (
            patch.object(OllamaEmbeddingProvider, "_test_connection", return_value=True),
            patch("litellm.embedding") as mock_embed,
        ):
            mock_embed.return_value = MagicMock(data=[{"embedding": [0.5, 0.6]}])
            result = OllamaEmbeddingProvider().execute_embedding(
                texts=["hi"], model_id="nomic-embed-text", base_url="http://localhost:11434"
            )
        assert result == [[0.5, 0.6]]
        kwargs = mock_embed.call_args[1]
        assert kwargs["model"] == "ollama/nomic-embed-text"
        assert kwargs["api_base"] == "http://localhost:11434"

    @pytest.mark.unit
    def test_check_health_cannot_connect(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        with patch.object(OllamaEmbeddingProvider, "_test_connection", return_value=False):
            ok, msg = OllamaEmbeddingProvider().check_health(
                api_key=None, base_url="http://localhost:11434"
            )
        assert ok is False
        assert msg == "Cannot connect to Ollama at http://localhost:11434"

    @pytest.mark.unit
    def test_check_health_model_not_found(self, monkeypatch):
        """Connected but the requested model isn't pulled -> not healthy."""
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"models": [{"name": "other-model:latest"}]}
        with (
            patch.object(OllamaEmbeddingProvider, "_test_connection", return_value=True),
            patch("requests.get", return_value=resp),
        ):
            ok, msg = OllamaEmbeddingProvider().check_health(
                api_key=None, base_url="http://localhost:11434"
            )
        assert ok is False
        assert "not found" in msg

    @pytest.mark.unit
    def test_check_health_healthy(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"models": [{"name": "nomic-embed-text:latest"}]}
        with (
            patch.object(OllamaEmbeddingProvider, "_test_connection", return_value=True),
            patch("requests.get", return_value=resp),
        ):
            ok, msg = OllamaEmbeddingProvider().check_health(
                api_key=None, base_url="http://localhost:11434"
            )
        assert ok is True
        assert "connected" in msg


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
