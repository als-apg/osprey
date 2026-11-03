"""Tests for the text-embedding endpoint of the CBorg and AmSC i2 proxy adapters.

Both adapters front an OpenAI-compatible LiteLLM proxy, so an embedding call is
one LiteLLM ``embedding`` call routed ``openai/<model>`` against the proxy's
base URL. These tests patch that entry point and check what it receives, and
that the health check answers with a typed :class:`HealthResult` instead of
raising.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import litellm
import pytest

from osprey.models.providers.amsc_i2 import AMSCI2ProviderAdapter
from osprey.models.providers.cborg import CBorgProviderAdapter
from osprey.models.providers.health import HealthResult

_EMBEDDING = "litellm.embedding"
_CBORG_URL = "https://gateway.example.org/v1"
_AMSC_URL = "https://api.i2-core.example/v1"


def _response(*vectors: list[float]) -> SimpleNamespace:
    """A LiteLLM embedding response carrying *vectors* in order."""
    return SimpleNamespace(data=[{"embedding": list(v)} for v in vectors])


@pytest.fixture
def cborg() -> CBorgProviderAdapter:
    return CBorgProviderAdapter()


@pytest.fixture
def amsc() -> AMSCI2ProviderAdapter:
    return AMSCI2ProviderAdapter()


class TestDeclarations:
    @pytest.mark.parametrize("cls", [CBorgProviderAdapter, AMSCI2ProviderAdapter])
    def test_proxy_adapters_serve_text_embeddings_alongside_chat(self, cls):
        assert cls.supports_embeddings() is True
        assert cls.supports_chat() is True

    @pytest.mark.parametrize("cls", [CBorgProviderAdapter, AMSCI2ProviderAdapter])
    def test_proxy_adapters_serve_no_image_embeddings(self, cls):
        assert cls.supports_image_embeddings() is False
        assert cls.truncates_to_dimensions is False

    @pytest.mark.parametrize(
        "name", ["anthropic", "google", "stanford", "als-apg", "vllm", "argo", "ds4"]
    )
    def test_other_providers_still_serve_no_embeddings(self, name):
        from osprey.models.provider_registry import get_provider_registry

        cls = get_provider_registry().get_provider(name)
        assert cls is not None, name
        assert cls.supports_embeddings() is False

    def test_cborg_defaults_name_nomic_embed_text(self):
        assert CBorgProviderAdapter.default_embedding_model_id == "lbl/nomic-embed-text"
        assert CBorgProviderAdapter.health_check_embedding_model_id == "lbl/nomic-embed-text"

    def test_amsc_i2_declares_no_embedding_defaults(self):
        assert AMSCI2ProviderAdapter.default_embedding_model_id is None
        assert AMSCI2ProviderAdapter.health_check_embedding_model_id is None


class TestExecuteEmbedding:
    def test_cborg_routes_openai_compatible_with_attribution(self, cborg, monkeypatch):
        """The proxy gets openai/<model>, its base URL, and the acting identity."""
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
        with patch(_EMBEDDING, return_value=_response([0.1, 0.2])) as embed:
            vectors = cborg.execute_embedding(
                ["alpha"],
                model_id="lbl/nomic-embed-text",
                api_key="sk-test",
                base_url=_CBORG_URL,
                dimensions=256,
                timeout=42.0,
            )
        assert vectors == [[0.1, 0.2]]
        embed.assert_called_once_with(
            model="openai/lbl/nomic-embed-text",
            input=["alpha"],
            timeout=42.0,
            api_key="sk-test",
            api_base=_CBORG_URL,
            dimensions=256,
            user="alice",
            extra_headers={"x-litellm-tags": "osprey,surface:terminal"},
        )

    def test_cborg_missing_model_falls_back_to_its_default(self, cborg):
        with patch(_EMBEDDING, return_value=_response([1.0])) as embed:
            cborg.execute_embedding(["a"], model_id=None, api_key="k", base_url=_CBORG_URL)
        assert embed.call_args.kwargs["model"] == "openai/lbl/nomic-embed-text"

    def test_amsc_i2_routes_openai_compatible_without_attribution(self, amsc):
        """AmSC i2 declares no LiteLLM gateway, so no identity is attached."""
        with patch(_EMBEDDING, return_value=_response([1.0], [2.0])) as embed:
            vectors = amsc.execute_embedding(
                ["a", "b"], model_id="nomic-embed", api_key="k", base_url=_AMSC_URL
            )
        assert vectors == [[1.0], [2.0]]
        assert embed.call_args.kwargs == {
            "model": "openai/nomic-embed",
            "input": ["a", "b"],
            "timeout": 600.0,
            "api_key": "k",
            "api_base": _AMSC_URL,
        }

    def test_amsc_i2_without_a_model_is_refused_before_any_call(self, amsc):
        with patch(_EMBEDDING) as embed:
            with pytest.raises(ValueError, match="model"):
                amsc.execute_embedding(["a"], model_id=None, api_key="k", base_url=_AMSC_URL)
        embed.assert_not_called()

    def test_missing_base_url_is_refused_before_any_call(self, cborg):
        with patch(_EMBEDDING) as embed:
            with pytest.raises(ValueError):
                cborg.execute_embedding(["a"], model_id="lbl/nomic-embed-text", api_key="k")
        embed.assert_not_called()

    def test_empty_input_makes_no_call(self, cborg):
        with patch(_EMBEDDING) as embed:
            assert cborg.execute_embedding([], model_id=None, base_url=_CBORG_URL) == []
        embed.assert_not_called()

    def test_library_failure_is_raised_as_runtime_error_keeping_the_cause(self, cborg):
        error = litellm.AuthenticationError(
            message="bad key", llm_provider="openai", model="lbl/nomic-embed-text"
        )
        with patch(_EMBEDDING, side_effect=error):
            with pytest.raises(RuntimeError) as raised:
                cborg.execute_embedding(["a"], model_id=None, api_key="bad", base_url=_CBORG_URL)
        assert raised.value.__cause__ is error

    def test_per_provider_patch_target_is_honoured(self, cborg):
        """The helper is looked up in the provider's own module, so tests can patch it there."""
        with patch(
            "osprey.models.providers.cborg.execute_litellm_embedding", return_value=[[9.0]]
        ) as helper:
            assert cborg.execute_embedding(["a"], None, base_url=_CBORG_URL) == [[9.0]]
        assert helper.call_args.kwargs["provider"] == "cborg"
        assert helper.call_args.kwargs["base_url"] == _CBORG_URL


class TestCheckEmbeddingHealth:
    def test_cborg_healthy_probe_uses_the_health_model(self, cborg):
        with patch(_EMBEDDING, return_value=_response([0.5])) as embed:
            result = cborg.check_embedding_health("sk-test", _CBORG_URL, timeout=3.0)
        assert result == HealthResult(True, result.message, None)
        kwargs = embed.call_args.kwargs
        assert kwargs["model"] == "openai/lbl/nomic-embed-text"
        assert kwargs["api_base"] == _CBORG_URL
        assert kwargs["timeout"] == 3.0
        assert len(kwargs["input"]) == 1

    def test_missing_base_url_is_a_config_verdict_without_a_call(self, cborg):
        with patch(_EMBEDDING) as embed:
            result = cborg.check_embedding_health("sk-test", None)
        embed.assert_not_called()
        assert (result.reachable, result.reason) == (False, "config")

    def test_amsc_i2_without_a_model_is_a_config_verdict_without_a_call(self, amsc):
        with patch(_EMBEDDING) as embed:
            result = amsc.check_embedding_health("sk-test", _AMSC_URL)
        embed.assert_not_called()
        assert (result.reachable, result.reason) == (False, "config")

    def test_missing_key_is_a_config_verdict_without_a_call(self, cborg):
        with patch(_EMBEDDING) as embed:
            result = cborg.check_embedding_health(None, _CBORG_URL)
        embed.assert_not_called()
        assert (result.reachable, result.reason) == (False, "config")

    def test_auth_failure_is_an_auth_verdict(self, cborg):
        error = litellm.AuthenticationError(
            message="bad key", llm_provider="openai", model="lbl/nomic-embed-text"
        )
        with patch(_EMBEDDING, side_effect=error):
            result = cborg.check_embedding_health("sk-bad", _CBORG_URL)
        assert (result.reachable, result.reason) == (False, "auth")

    def test_unknown_model_is_a_model_verdict(self, amsc):
        error = litellm.NotFoundError(message="no such model", llm_provider="openai", model="x")
        with patch(_EMBEDDING, side_effect=error):
            result = amsc.check_embedding_health("sk", _AMSC_URL, model_id="x")
        assert (result.reachable, result.reason) == (False, "model")

    def test_unclassified_failure_falls_back_to_unreachable(self, cborg):
        with patch(_EMBEDDING, side_effect=TimeoutError("timed out")):
            result = cborg.check_embedding_health("sk-test", _CBORG_URL)
        assert (result.reachable, result.reason) == (False, "unreachable")
