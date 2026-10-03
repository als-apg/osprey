"""Tests for the OpenAI text-embedding endpoint served by :class:`OpenAIProviderAdapter`.

The adapter forwards one LiteLLM ``embedding`` call; these tests patch that entry
point and check what it receives, and that the health check answers with a typed
:class:`HealthResult` instead of raising.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import litellm
import pytest

from osprey.models.providers.health import HealthResult
from osprey.models.providers.openai import OpenAIProviderAdapter

_EMBEDDING = "litellm.embedding"


def _response(*vectors: list[float]) -> SimpleNamespace:
    """A LiteLLM embedding response carrying *vectors* in order."""
    return SimpleNamespace(data=[{"embedding": list(v)} for v in vectors])


def _auth_error() -> litellm.AuthenticationError:
    return litellm.AuthenticationError(
        message="Incorrect API key provided", llm_provider="openai", model="text-embedding-3-small"
    )


@pytest.fixture
def adapter() -> OpenAIProviderAdapter:
    return OpenAIProviderAdapter()


class TestDeclarations:
    def test_embedding_defaults_name_the_small_v3_model(self):
        """Both the default and the health-check embedding model are text-embedding-3-small."""
        assert OpenAIProviderAdapter.default_embedding_model_id == "text-embedding-3-small"
        assert OpenAIProviderAdapter.health_check_embedding_model_id == "text-embedding-3-small"

    def test_serves_text_embeddings_alongside_chat(self):
        """Overriding the embedding endpoint makes the adapter report it, chat unchanged."""
        assert OpenAIProviderAdapter.supports_embeddings() is True
        assert OpenAIProviderAdapter.supports_chat() is True
        assert OpenAIProviderAdapter.supports_image_embeddings() is False


class TestExecuteEmbedding:
    def test_forwards_model_key_base_url_timeout_and_dimensions(self, adapter):
        """Every caller-supplied setting reaches LiteLLM unchanged."""
        with patch(_EMBEDDING, return_value=_response([0.1, 0.2])) as embed:
            vectors = adapter.execute_embedding(
                ["alpha"],
                model_id="text-embedding-3-large",
                api_key="sk-test",
                base_url="https://proxy.example/v1",
                dimensions=256,
                timeout=42.0,
            )
        assert vectors == [[0.1, 0.2]]
        embed.assert_called_once_with(
            model="text-embedding-3-large",
            input=["alpha"],
            timeout=42.0,
            api_key="sk-test",
            api_base="https://proxy.example/v1",
            dimensions=256,
        )

    def test_unset_options_are_omitted_from_the_call(self, adapter):
        """No key, no base URL and dimensions=None send none of those keywords."""
        with patch(_EMBEDDING, return_value=_response([1.0])) as embed:
            adapter.execute_embedding(["alpha"], model_id="text-embedding-3-small")
        kwargs = embed.call_args.kwargs
        assert kwargs == {"model": "text-embedding-3-small", "input": ["alpha"], "timeout": 600.0}

    def test_returns_one_vector_per_text_in_input_order(self, adapter):
        """The vectors come back in the order of the response data."""
        with patch(_EMBEDDING, return_value=_response([1.0], [2.0], [3.0])):
            vectors = adapter.execute_embedding(["a", "b", "c"], model_id="m", api_key="k")
        assert vectors == [[1.0], [2.0], [3.0]]

    def test_empty_input_makes_no_call(self, adapter):
        """Nothing to embed answers an empty list without touching the network."""
        with patch(_EMBEDDING) as embed:
            assert adapter.execute_embedding([], model_id="m") == []
        embed.assert_not_called()

    def test_missing_model_falls_back_to_the_default_embedding_model(self, adapter):
        """model_id=None embeds with default_embedding_model_id."""
        with patch(_EMBEDDING, return_value=_response([1.0])) as embed:
            adapter.execute_embedding(["a"], model_id=None, api_key="k")
        assert embed.call_args.kwargs["model"] == "text-embedding-3-small"

    def test_library_failure_is_raised_as_runtime_error_keeping_the_cause(self, adapter):
        """A LiteLLM error surfaces as RuntimeError chained to the original exception."""
        error = _auth_error()
        with patch(_EMBEDDING, side_effect=error):
            with pytest.raises(RuntimeError) as raised:
                adapter.execute_embedding(["a"], model_id="m", api_key="bad")
        assert raised.value.__cause__ is error


class TestCheckEmbeddingHealth:
    def test_healthy_call_embeds_one_short_text_with_the_health_model(self, adapter):
        """model_id=None checks health_check_embedding_model_id with a single tiny input."""
        with patch(_EMBEDDING, return_value=_response([0.5])) as embed:
            result = adapter.check_embedding_health(
                "sk-test", "https://proxy.example/v1", model_id=None, timeout=3.0
            )
        assert isinstance(result, HealthResult)
        assert result.reachable is True
        assert result.reason is None
        kwargs = embed.call_args.kwargs
        assert kwargs["model"] == "text-embedding-3-small"
        assert kwargs["api_base"] == "https://proxy.example/v1"
        assert kwargs["api_key"] == "sk-test"
        assert kwargs["timeout"] == 3.0
        assert len(kwargs["input"]) == 1
        assert len(kwargs["input"][0].split()) == 1

    def test_configured_model_is_the_one_checked(self, adapter):
        """An explicit model_id is the model the probe embeds with."""
        with patch(_EMBEDDING, return_value=_response([0.5])) as embed:
            adapter.check_embedding_health("sk-test", None, model_id="text-embedding-3-large")
        assert embed.call_args.kwargs["model"] == "text-embedding-3-large"

    def test_401_is_an_auth_verdict_not_an_exception(self, adapter):
        """A rejected key answers HealthResult(False, ..., 'auth')."""
        with patch(_EMBEDDING, side_effect=_auth_error()):
            result = adapter.check_embedding_health("sk-bad", None)
        assert result.reachable is False
        assert result.reason == "auth"
        assert result.message

    def test_connection_failure_is_unreachable(self, adapter):
        """A connection error answers 'unreachable'."""
        error = litellm.APIConnectionError(
            message="connection refused", llm_provider="openai", model="text-embedding-3-small"
        )
        with patch(_EMBEDDING, side_effect=error):
            result = adapter.check_embedding_health("sk-test", None)
        assert result == HealthResult(False, result.message, "unreachable")

    def test_unknown_model_is_a_model_verdict(self, adapter):
        """LiteLLM's NotFoundError answers 'model'."""
        error = litellm.NotFoundError(
            message="model not found", llm_provider="openai", model="text-embedding-9"
        )
        with patch(_EMBEDDING, side_effect=error):
            result = adapter.check_embedding_health("sk-test", None, model_id="text-embedding-9")
        assert (result.reachable, result.reason) == (False, "model")

    def test_unclassified_failure_falls_back_to_unreachable(self, adapter):
        """An exception the table does not know still carries a reason."""
        with patch(_EMBEDDING, side_effect=TimeoutError("timed out")):
            result = adapter.check_embedding_health("sk-test", None)
        assert (result.reachable, result.reason) == (False, "unreachable")

    def test_missing_key_is_a_config_verdict_without_a_call(self, adapter):
        """No API key answers a config verdict and makes no request."""
        with patch(_EMBEDDING) as embed:
            result = adapter.check_embedding_health(None, None)
        embed.assert_not_called()
        assert (result.reachable, result.reason) == (False, "config")
