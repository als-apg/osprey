"""Tests for vLLM provider adapter.

vLLM delegates completion and the model-backed half of the health check to
litellm_adapter, so those helpers are patched at the vllm import site. The
adapter's own logic is the credential/base_url defaulting and the /v1/models
probe that runs when no model_id is supplied.
"""

from unittest.mock import Mock, patch

import httpx
import pytest

from osprey.models.providers.vllm import VLLMProviderAdapter

COMPLETION = "osprey.models.providers.vllm.execute_litellm_completion"
HEALTH = "osprey.models.providers.vllm.check_litellm_health"


class TestVLLMMetadata:
    """Test vLLM provider metadata."""

    def test_provider_name(self):
        """Test provider name is set correctly."""
        assert VLLMProviderAdapter.name == "vllm"

    def test_provider_description(self):
        """Test provider has description."""
        assert "vLLM" in VLLMProviderAdapter.description

    def test_does_not_require_api_key(self):
        """Test vLLM needs no API key by default."""
        assert VLLMProviderAdapter.requires_api_key is False

    def test_requires_base_url(self):
        """Test provider requires base URL."""
        assert VLLMProviderAdapter.requires_base_url is True

    def test_requires_model_id(self):
        """Test provider requires model ID."""
        assert VLLMProviderAdapter.requires_model_id is True

    def test_supports_proxy(self):
        """Test provider supports HTTP proxy."""
        assert VLLMProviderAdapter.supports_proxy is True

    def test_default_base_url_is_local_server(self):
        """Test the default base URL points at a local vLLM server."""
        assert VLLMProviderAdapter.default_base_url == "http://localhost:8000/v1"

    def test_has_no_default_model_id(self):
        """Test no default model -- what is served depends on the deployment."""
        assert VLLMProviderAdapter.default_model_id is None

    def test_has_no_health_check_model(self):
        """Test no static health-check model -- the server is queried instead."""
        assert VLLMProviderAdapter.health_check_model_id is None

    def test_has_available_models(self):
        """Test provider lists example model families."""
        assert len(VLLMProviderAdapter.available_models) > 0

    def test_is_openai_compatible(self):
        """Test provider is flagged OpenAI-compatible for LiteLLM routing."""
        assert VLLMProviderAdapter.is_openai_compatible is True

    def test_supports_native_structured_output(self):
        """Test provider advertises constrained-decoding structured output."""
        assert VLLMProviderAdapter.supports_native_structured_output is True

    def test_is_instantiable(self):
        """Test adapter instantiates despite inheriting from an ABC."""
        assert isinstance(VLLMProviderAdapter(), VLLMProviderAdapter)


class TestVLLMExecuteCompletion:
    """Test vLLM completion delegation to LiteLLM."""

    def test_returns_litellm_result(self):
        """Test the LiteLLM result is returned unchanged."""
        provider = VLLMProviderAdapter()
        with patch(COMPLETION, return_value="hello") as mock_exec:
            result = provider.execute_completion(
                message="hi", model_id="m", api_key="key", base_url="http://server/v1"
            )

        assert result == "hello"
        mock_exec.assert_called_once()

    def test_forwards_arguments(self):
        """Test caller arguments reach execute_litellm_completion."""
        provider = VLLMProviderAdapter()
        sentinel = object()
        with patch(COMPLETION, return_value="ok") as mock_exec:
            provider.execute_completion(
                message="hi",
                model_id="m",
                api_key="key",
                base_url="http://server/v1",
                max_tokens=64,
                temperature=0.5,
                output_format=sentinel,
            )

        kwargs = mock_exec.call_args.kwargs
        assert kwargs["provider"] == "vllm"
        assert kwargs["message"] == "hi"
        assert kwargs["model_id"] == "m"
        assert kwargs["api_key"] == "key"
        assert kwargs["base_url"] == "http://server/v1"
        assert kwargs["max_tokens"] == 64
        assert kwargs["temperature"] == 0.5
        assert kwargs["output_format"] is sentinel

    def test_forwards_extra_kwargs(self):
        """Test unrecognized kwargs pass through to LiteLLM."""
        provider = VLLMProviderAdapter()
        with patch(COMPLETION, return_value="ok") as mock_exec:
            provider.execute_completion(
                message="hi",
                model_id="m",
                api_key="key",
                base_url="http://server/v1",
                enable_thinking=True,
                budget_tokens=256,
            )

        kwargs = mock_exec.call_args.kwargs
        assert kwargs["enable_thinking"] is True
        assert kwargs["budget_tokens"] == 256

    @pytest.mark.parametrize("api_key", [None, ""], ids=["none", "empty"])
    def test_missing_api_key_becomes_placeholder(self, api_key):
        """Test a missing API key is replaced with the EMPTY placeholder."""
        provider = VLLMProviderAdapter()
        with patch(COMPLETION, return_value="ok") as mock_exec:
            provider.execute_completion(
                message="hi", model_id="m", api_key=api_key, base_url="http://server/v1"
            )

        assert mock_exec.call_args.kwargs["api_key"] == "EMPTY"

    def test_missing_base_url_uses_default(self):
        """Test a missing base URL falls back to the default endpoint."""
        provider = VLLMProviderAdapter()
        with patch(COMPLETION, return_value="ok") as mock_exec:
            provider.execute_completion(message="hi", model_id="m", api_key="key", base_url=None)

        assert mock_exec.call_args.kwargs["base_url"] == VLLMProviderAdapter.default_base_url


class TestVLLMCheckHealthWithModelId:
    """Test vLLM health check when a model_id is supplied (no server probe)."""

    def test_delegates_to_litellm(self):
        """Test an explicit model_id skips the probe and delegates to LiteLLM."""
        provider = VLLMProviderAdapter()
        with (
            patch(HEALTH, return_value=(True, "ok")) as mock_health,
            patch("httpx.get") as mock_get,
        ):
            result = provider.check_health(api_key="key", base_url="http://server/v1", model_id="m")

        assert result == (True, "ok")
        mock_get.assert_not_called()
        kwargs = mock_health.call_args.kwargs
        assert kwargs["provider"] == "vllm"
        assert kwargs["api_key"] == "key"
        assert kwargs["base_url"] == "http://server/v1"
        assert kwargs["model_id"] == "m"

    def test_forwards_timeout(self):
        """Test the timeout reaches the LiteLLM health helper."""
        provider = VLLMProviderAdapter()
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            provider.check_health(
                api_key="key", base_url="http://server/v1", model_id="m", timeout=12.0
            )

        assert mock_health.call_args.kwargs["timeout"] == 12.0

    def test_missing_api_key_becomes_placeholder(self):
        """Test a missing API key is replaced with the EMPTY placeholder."""
        provider = VLLMProviderAdapter()
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            provider.check_health(api_key=None, base_url="http://server/v1", model_id="m")

        assert mock_health.call_args.kwargs["api_key"] == "EMPTY"

    def test_missing_base_url_uses_default(self):
        """Test a missing base URL falls back to the default endpoint."""
        provider = VLLMProviderAdapter()
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            provider.check_health(api_key="key", base_url=None, model_id="m")

        assert mock_health.call_args.kwargs["base_url"] == VLLMProviderAdapter.default_base_url

    def test_propagates_failure(self):
        """Test a LiteLLM health failure is returned unchanged."""
        provider = VLLMProviderAdapter()
        with patch(HEALTH, return_value=(False, "down")):
            result = provider.check_health(api_key="key", base_url="http://server/v1", model_id="m")

        assert result == (False, "down")


class TestVLLMCheckHealthModelDiscovery:
    """Test the /v1/models probe that runs when no model_id is supplied."""

    def test_discovers_first_model_and_delegates(self):
        """Test the first served model is used for the LiteLLM health check."""
        provider = VLLMProviderAdapter()
        with (
            patch("httpx.get") as mock_get,
            patch(HEALTH, return_value=(True, "ok")) as mock_health,
        ):
            mock_get.return_value = Mock(
                status_code=200,
                json=Mock(return_value={"data": [{"id": "served-a"}, {"id": "served-b"}]}),
            )
            result = provider.check_health(api_key="key", base_url="http://server/v1")

        assert result == (True, "ok")
        assert mock_health.call_args.kwargs["model_id"] == "served-a"

    @pytest.mark.parametrize(
        "base_url",
        [
            "http://server/v1",
            "http://server/v1/",
            "http://server",
            "http://server/",
        ],
        ids=["with_v1", "with_v1_slash", "bare", "bare_slash"],
    )
    def test_probe_url_is_normalized(self, base_url):
        """Test the probe targets /v1/models regardless of base_url shape."""
        provider = VLLMProviderAdapter()
        with patch("httpx.get") as mock_get, patch(HEALTH, return_value=(True, "ok")):
            mock_get.return_value = Mock(
                status_code=200, json=Mock(return_value={"data": [{"id": "m"}]})
            )
            provider.check_health(api_key="key", base_url=base_url)

        assert mock_get.call_args[0][0] == "http://server/v1/models"

    def test_probe_forwards_timeout(self):
        """Test the probe honors the caller's timeout."""
        provider = VLLMProviderAdapter()
        with patch("httpx.get") as mock_get, patch(HEALTH, return_value=(True, "ok")):
            mock_get.return_value = Mock(
                status_code=200, json=Mock(return_value={"data": [{"id": "m"}]})
            )
            provider.check_health(api_key="key", base_url="http://server/v1", timeout=7.0)

        assert mock_get.call_args.kwargs["timeout"] == 7.0

    def test_no_models_loaded(self):
        """Test a running server with an empty model list is reported."""
        provider = VLLMProviderAdapter()
        with patch("httpx.get") as mock_get, patch(HEALTH) as mock_health:
            mock_get.return_value = Mock(status_code=200, json=Mock(return_value={"data": []}))
            ok, message = provider.check_health(api_key="key", base_url="http://server/v1")

        assert ok is False
        assert message == "vLLM server running but no models loaded"
        mock_health.assert_not_called()

    def test_missing_data_key_reports_no_models(self):
        """Test a response without a data key is treated as no models loaded."""
        provider = VLLMProviderAdapter()
        with patch("httpx.get") as mock_get:
            mock_get.return_value = Mock(status_code=200, json=Mock(return_value={}))
            ok, message = provider.check_health(api_key="key", base_url="http://server/v1")

        assert ok is False
        assert message == "vLLM server running but no models loaded"

    def test_model_entry_without_id_reports_no_model_available(self):
        """Test a served model lacking an id falls through to the no-model guard.

        This is the only path that reaches the second `if not model_id` check:
        the list is non-empty, so the "no models loaded" branch is skipped, but
        models[0].get("id") yields None.
        """
        provider = VLLMProviderAdapter()
        with patch("httpx.get") as mock_get, patch(HEALTH) as mock_health:
            mock_get.return_value = Mock(
                status_code=200, json=Mock(return_value={"data": [{"object": "model"}]})
            )
            ok, message = provider.check_health(api_key="key", base_url="http://server/v1")

        assert ok is False
        assert message == "No model available for health check"
        mock_health.assert_not_called()

    def test_non_200_reports_status(self):
        """Test a non-200 probe response reports the status code."""
        provider = VLLMProviderAdapter()
        with patch("httpx.get") as mock_get:
            mock_get.return_value = Mock(status_code=503)
            ok, message = provider.check_health(api_key="key", base_url="http://server/v1")

        assert ok is False
        assert message == "vLLM server returned 503"

    def test_connect_error_reports_endpoint(self):
        """Test a connection failure names the endpoint that was tried."""
        provider = VLLMProviderAdapter()
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            ok, message = provider.check_health(api_key="key", base_url="http://server/v1")

        assert ok is False
        assert message == "Cannot connect to vLLM server at http://server/v1"

    def test_connect_error_names_default_endpoint(self):
        """Test the connection failure names the default endpoint when unset."""
        provider = VLLMProviderAdapter()
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            ok, message = provider.check_health(api_key="key", base_url=None)

        assert ok is False
        assert VLLMProviderAdapter.default_base_url in message

    def test_unexpected_error_is_reported(self):
        """Test a non-connect error is reported with its reason."""
        provider = VLLMProviderAdapter()
        with patch("httpx.get", side_effect=ValueError("malformed")):
            ok, message = provider.check_health(api_key="key", base_url="http://server/v1")

        assert ok is False
        assert message == "Error querying vLLM: malformed"

    def test_unexpected_error_reason_is_truncated(self):
        """Test long error reasons are truncated to the shared snippet length."""
        provider = VLLMProviderAdapter()
        with patch("httpx.get", side_effect=ValueError("x" * 500)):
            ok, message = provider.check_health(api_key="key", base_url="http://server/v1")

        assert ok is False
        assert message == f"Error querying vLLM: {'x' * 200}"

    def test_discovery_missing_api_key_becomes_placeholder(self):
        """Test a missing API key is replaced before delegating after discovery."""
        provider = VLLMProviderAdapter()
        with (
            patch("httpx.get") as mock_get,
            patch(HEALTH, return_value=(True, "ok")) as mock_health,
        ):
            mock_get.return_value = Mock(
                status_code=200, json=Mock(return_value={"data": [{"id": "m"}]})
            )
            provider.check_health(api_key=None, base_url="http://server/v1")

        assert mock_health.call_args.kwargs["api_key"] == "EMPTY"
