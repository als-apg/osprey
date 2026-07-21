"""Tests for AskSage provider adapter.

The fallback tests in TestAskSageGetAvailableModels assert the argo-aligned
contract that ``get_available_models`` honors: every error path and missing
credentials fall back to the static defaults (issue #400).
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
from pydantic import BaseModel

from osprey.models.providers.asksage import AskSageProviderAdapter


class SampleOutput(BaseModel):
    """Sample output model for structured-output testing."""

    result: str
    value: int


@pytest.fixture(autouse=True)
def _clear_models_cache():
    """Reset the class-level model cache around every test.

    ``get_available_models`` writes ``type(self)._models_cache`` (issue #400),
    so a successful fetch in one test would otherwise leak into the next and
    silently short-circuit its request.
    """
    AskSageProviderAdapter._models_cache = None
    yield
    AskSageProviderAdapter._models_cache = None


class TestAskSageMetadata:
    """Test AskSage provider metadata."""

    def test_provider_name(self):
        """Test provider name is set correctly."""
        assert AskSageProviderAdapter.name == "asksage"

    def test_provider_description(self):
        """Test provider has description."""
        assert "AskSage" in AskSageProviderAdapter.description

    def test_requires_api_key(self):
        """Test provider requires API key."""
        assert AskSageProviderAdapter.requires_api_key is True

    def test_requires_base_url(self):
        """Test provider requires base URL."""
        assert AskSageProviderAdapter.requires_base_url is True

    def test_requires_model_id(self):
        """Test provider requires model ID."""
        assert AskSageProviderAdapter.requires_model_id is True

    def test_supports_proxy(self):
        """Test provider supports HTTP proxy."""
        assert AskSageProviderAdapter.supports_proxy is True

    def test_has_no_default_base_url(self):
        """Test provider has no default base URL (deployment-specific)."""
        assert AskSageProviderAdapter.default_base_url is None

    def test_has_default_model_id(self):
        """Test provider has default model."""
        assert AskSageProviderAdapter.default_model_id is not None

    def test_has_health_check_model(self):
        """Test provider has health check model."""
        assert AskSageProviderAdapter.health_check_model_id is not None

    def test_available_models_includes_defaults(self):
        """Test available models includes the default and health-check models."""
        models = AskSageProviderAdapter.available_models
        assert len(models) > 0
        assert AskSageProviderAdapter.default_model_id in models
        assert AskSageProviderAdapter.health_check_model_id in models

    def test_is_instantiable(self):
        """Test adapter instantiates despite inheriting from an ABC."""
        assert isinstance(AskSageProviderAdapter(), AskSageProviderAdapter)


class TestAskSageGetAvailableModels:
    """Test AskSage model list retrieval."""

    def test_returns_models_from_api(self):
        """Test parsing a valid /models response."""
        provider = AskSageProviderAdapter()
        with patch("requests.get") as mock_get:
            mock_get.return_value = Mock(
                status_code=200,
                json=Mock(return_value={"data": [{"id": "model-a"}, {"id": "model-b"}]}),
            )
            models = provider.get_available_models(api_key="key", base_url="https://test")

        assert models == ["model-a", "model-b"]

    def test_requests_models_endpoint_with_bearer_token(self):
        """Test the request targets /models with an Authorization header."""
        provider = AskSageProviderAdapter()
        with patch("requests.get") as mock_get:
            mock_get.return_value = Mock(status_code=200, json=Mock(return_value={"data": []}))
            provider.get_available_models(api_key="key", base_url="https://test/", timeout=9.0)

        args, kwargs = mock_get.call_args
        assert args[0] == "https://test/models"
        assert kwargs["headers"] == {"Authorization": "Bearer key"}
        assert kwargs["timeout"] == 9.0

    def test_populates_cache_within_one_instance(self):
        """Test a successful fetch is reused on the same adapter instance.

        Only pins same-instance reuse. Production builds a fresh adapter per
        completion (completion.py:179), which this does not exercise -- see
        test_cache_is_shared_across_instances below.
        """
        provider = AskSageProviderAdapter()
        with patch("requests.get") as mock_get:
            mock_get.return_value = Mock(
                status_code=200, json=Mock(return_value={"data": [{"id": "model-a"}]})
            )
            provider.get_available_models(api_key="key", base_url="https://test")
            provider.get_available_models(api_key="key", base_url="https://test")

        assert mock_get.call_count == 1

    def test_cache_is_shared_across_instances(self):
        """Test the model cache is shared across adapter instances, as argo's is.

        completion.py:179 constructs a fresh adapter for every completion, so a
        per-instance cache means each one pays another /models round-trip on a
        5s timeout. This is the only cache property production depends on.
        """
        with patch("requests.get") as mock_get:
            mock_get.return_value = Mock(
                status_code=200, json=Mock(return_value={"data": [{"id": "model-a"}]})
            )
            AskSageProviderAdapter().get_available_models(api_key="key", base_url="https://test")
            AskSageProviderAdapter().get_available_models(api_key="key", base_url="https://test")

        assert mock_get.call_count == 1

    def test_cache_hit_returns_cached_value(self):
        """Test a populated cache short-circuits the request."""
        provider = AskSageProviderAdapter()
        provider._models_cache = ["cached-model"]
        with patch("requests.get") as mock_get:
            models = provider.get_available_models(api_key="key", base_url="https://test")

        assert models == ["cached-model"]
        mock_get.assert_not_called()

    def test_force_refresh_bypasses_cache(self):
        """Test force_refresh re-fetches even with a populated cache."""
        provider = AskSageProviderAdapter()
        provider._models_cache = ["stale-model"]
        with patch("requests.get") as mock_get:
            mock_get.return_value = Mock(
                status_code=200, json=Mock(return_value={"data": [{"id": "fresh-model"}]})
            )
            models = provider.get_available_models(
                api_key="key", base_url="https://test", force_refresh=True
            )

        assert models == ["fresh-model"]
        mock_get.assert_called_once()

    # --- Fallback contract (issue #400) -------------------------------------
    #
    # Every error path and missing credentials fall back to the static defaults,
    # matching argo.get_available_models.

    def test_missing_api_key_falls_back_to_defaults(self):
        """Test missing API key falls back to static defaults."""
        provider = AskSageProviderAdapter()
        result = provider.get_available_models(api_key=None, base_url="https://test")
        assert result == provider.available_models

    def test_missing_base_url_falls_back_to_defaults(self):
        """Test missing base URL falls back to static defaults."""
        provider = AskSageProviderAdapter()
        result = provider.get_available_models(api_key="key", base_url=None)
        assert result == provider.available_models

    def test_auth_failure_falls_back_to_defaults(self):
        """Test a 401 falls back to static defaults."""
        provider = AskSageProviderAdapter()
        with patch("requests.get") as mock_get:
            mock_get.return_value = Mock(status_code=401)
            result = provider.get_available_models(api_key="bad", base_url="https://test")

        assert result == provider.available_models

    def test_server_error_falls_back_to_defaults(self):
        """Test a non-200/401 status falls back to static defaults."""
        provider = AskSageProviderAdapter()
        with patch("requests.get") as mock_get:
            mock_get.return_value = Mock(status_code=500)
            result = provider.get_available_models(api_key="key", base_url="https://test")

        assert result == provider.available_models

    def test_timeout_falls_back_to_defaults(self):
        """Test a timeout falls back to static defaults."""
        provider = AskSageProviderAdapter()
        with patch("requests.get", side_effect=requests.Timeout()):
            result = provider.get_available_models(api_key="key", base_url="https://test")

        assert result == provider.available_models

    def test_request_exception_falls_back_to_defaults(self):
        """Test a connection error falls back to static defaults."""
        provider = AskSageProviderAdapter()
        with patch("requests.get", side_effect=requests.RequestException("boom")):
            result = provider.get_available_models(api_key="key", base_url="https://test")

        assert result == provider.available_models

    def test_unexpected_error_falls_back_to_defaults(self):
        """Test an unexpected error falls back to static defaults."""
        provider = AskSageProviderAdapter()
        with patch("requests.get", side_effect=ValueError("malformed")):
            result = provider.get_available_models(api_key="key", base_url="https://test")

        assert result == provider.available_models


class TestAskSageExecuteCompletion:
    """Test AskSage chat completion."""

    @pytest.fixture(autouse=True)
    def _stub_model_refresh(self):
        """Stub the model-list refresh so completion tests stay hermetic.

        execute_completion refreshes the model list first (asksage.py:124).
        Patching only openai.OpenAI leaves requests.get live, and these tests
        then attempt a real DNS lookup for the base_url host.
        """
        with patch("requests.get") as mock_get:
            mock_get.return_value = Mock(status_code=200, json=Mock(return_value={"data": []}))
            yield mock_get

    @staticmethod
    def _mock_client(content: str):
        """Build an openai.OpenAI mock returning a single choice."""
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=content))]
        )
        return client

    def test_returns_plain_content(self):
        """Test unstructured completion returns the message content."""
        provider = AskSageProviderAdapter()
        client = self._mock_client("hello there")
        with patch("openai.OpenAI", return_value=client):
            result = provider.execute_completion(
                message="hi", model_id="m", api_key="key", base_url="https://test"
            )

        assert result == "hello there"

    def test_passes_asksage_extra_body(self):
        """Test AskSage-specific body args are forwarded on every call."""
        provider = AskSageProviderAdapter()
        client = self._mock_client("ok")
        with patch("openai.OpenAI", return_value=client):
            provider.execute_completion(
                message="hi",
                model_id="m",
                api_key="key",
                base_url="https://test",
                max_tokens=64,
                temperature=0.5,
            )

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "m"
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert kwargs["max_tokens"] == 64
        assert kwargs["temperature"] == 0.5
        assert kwargs["extra_body"] == {
            "system_prompt": "-",
            "dataset": "none",
            "live": 0,
            "limit_references": 0,
        }

    def test_forwards_credentials_and_http_client(self):
        """Test credentials and a caller-supplied http_client reach the constructor."""
        provider = AskSageProviderAdapter()
        sentinel = object()
        with patch("openai.OpenAI", return_value=self._mock_client("ok")) as mock_openai:
            provider.execute_completion(
                message="hi",
                model_id="m",
                api_key="key",
                base_url="https://test",
                http_client=sentinel,
            )

        kwargs = mock_openai.call_args.kwargs
        assert kwargs["api_key"] == "key"
        assert kwargs["base_url"] == "https://test"
        assert kwargs["http_client"] is sentinel

    def test_caller_system_prompt_is_dropped(self):
        """Test a caller-supplied system_prompt is silently discarded (#400).

        asksage accepts system_prompt but never reads it, hard-coding
        ``"system_prompt": "-"`` into extra_body instead. Pinned so the
        behavior is visible rather than surprising; not an endorsement.
        """
        provider = AskSageProviderAdapter()
        client = self._mock_client("ok")
        with patch("openai.OpenAI", return_value=client):
            provider.execute_completion(
                message="hi",
                model_id="m",
                api_key="key",
                base_url="https://test",
                system_prompt="You are a careful assistant.",
            )

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["system_prompt"] == "-"
        assert all("careful assistant" not in m["content"] for m in kwargs["messages"])

    def test_model_refresh_failure_does_not_block_completion(self):
        """Test a failing model-list refresh is swallowed before completing."""
        provider = AskSageProviderAdapter()
        with (
            patch.object(
                AskSageProviderAdapter, "get_available_models", side_effect=RuntimeError("down")
            ),
            patch("openai.OpenAI", return_value=self._mock_client("still works")),
        ):
            result = provider.execute_completion(
                message="hi", model_id="m", api_key="key", base_url="https://test"
            )

        assert result == "still works"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"enable_thinking": True},
            {"budget_tokens": 512},
        ],
        ids=["enable_thinking", "budget_tokens"],
    )
    def test_warns_thinking_params_unsupported(self, caplog, kwargs):
        """Test unsupported thinking params emit a warning but still complete."""
        provider = AskSageProviderAdapter()
        with patch("openai.OpenAI", return_value=self._mock_client("ok")):
            with caplog.at_level("WARNING"):
                result = provider.execute_completion(
                    message="hi",
                    model_id="m",
                    api_key="key",
                    base_url="https://test",
                    **kwargs,
                )

        assert result == "ok"
        assert "not used for AskSage" in caplog.text

    def test_no_warning_without_thinking_params(self, caplog):
        """Test the default path emits no thinking warning."""
        provider = AskSageProviderAdapter()
        with patch("openai.OpenAI", return_value=self._mock_client("ok")):
            with caplog.at_level("WARNING"):
                provider.execute_completion(
                    message="hi", model_id="m", api_key="key", base_url="https://test"
                )

        assert "not used for AskSage" not in caplog.text

    def test_structured_output_returns_model(self):
        """Test structured output is parsed into the pydantic model."""
        provider = AskSageProviderAdapter()
        payload = json.dumps({"result": "ok", "value": 7})
        with patch("openai.OpenAI", return_value=self._mock_client(payload)):
            result = provider.execute_completion(
                message="hi",
                model_id="m",
                api_key="key",
                base_url="https://test",
                output_format=SampleOutput,
            )

        assert result == SampleOutput(result="ok", value=7)

    def test_structured_output_embeds_schema_in_prompt(self):
        """Test the JSON schema is injected into the user message."""
        provider = AskSageProviderAdapter()
        payload = json.dumps({"result": "ok", "value": 7})
        client = self._mock_client(payload)
        with patch("openai.OpenAI", return_value=client):
            provider.execute_completion(
                message="describe it",
                model_id="m",
                api_key="key",
                base_url="https://test",
                output_format=SampleOutput,
            )

        content = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "describe it" in content
        assert "Respond ONLY with the JSON object" in content
        assert '"result"' in content and '"value"' in content

    def test_structured_output_strips_markdown_fence(self):
        """Test a fenced JSON response is cleaned before validation."""
        provider = AskSageProviderAdapter()
        fenced = "```json\n" + json.dumps({"result": "ok", "value": 1}) + "\n```"
        with patch("openai.OpenAI", return_value=self._mock_client(fenced)):
            result = provider.execute_completion(
                message="hi",
                model_id="m",
                api_key="key",
                base_url="https://test",
                output_format=SampleOutput,
            )

        assert result == SampleOutput(result="ok", value=1)

    def test_typed_dict_output_returns_dict(self):
        """Test is_typed_dict_output dumps the model to a plain dict."""
        provider = AskSageProviderAdapter()
        payload = json.dumps({"result": "ok", "value": 3})
        with patch("openai.OpenAI", return_value=self._mock_client(payload)):
            result = provider.execute_completion(
                message="hi",
                model_id="m",
                api_key="key",
                base_url="https://test",
                output_format=SampleOutput,
                is_typed_dict_output=True,
            )

        assert result == {"result": "ok", "value": 3}

    def test_structured_output_parse_failure_raises_value_error(self):
        """Test unparseable structured output raises ValueError with context."""
        provider = AskSageProviderAdapter()
        with patch("openai.OpenAI", return_value=self._mock_client("not json at all")):
            with pytest.raises(ValueError, match="Failed to parse structured output"):
                provider.execute_completion(
                    message="hi",
                    model_id="m",
                    api_key="key",
                    base_url="https://test",
                    output_format=SampleOutput,
                )


class TestAskSageCheckHealth:
    """Test AskSage health check."""

    def test_no_api_key(self):
        """Test health check fails without an API key."""
        provider = AskSageProviderAdapter()
        ok, message = provider.check_health(api_key=None, base_url="https://test")
        assert ok is False
        assert "API key not set" in message

    def test_no_base_url(self):
        """Test health check fails without a base URL."""
        provider = AskSageProviderAdapter()
        ok, message = provider.check_health(api_key="key", base_url=None)
        assert ok is False
        assert "Base URL not configured" in message

    def test_successful(self):
        """Test a 200 reports the API as accessible."""
        provider = AskSageProviderAdapter()
        with patch("requests.get") as mock_get:
            mock_get.return_value = Mock(status_code=200)
            ok, message = provider.check_health(api_key="key", base_url="https://test")

        assert ok is True
        assert "accessible" in message.lower()

    def test_requests_models_endpoint_with_bearer_token(self):
        """Test the health probe targets /models with an Authorization header."""
        provider = AskSageProviderAdapter()
        with patch("requests.get") as mock_get:
            mock_get.return_value = Mock(status_code=200)
            provider.check_health(api_key="key", base_url="https://test/", timeout=3.0)

        args, kwargs = mock_get.call_args
        assert args[0] == "https://test/models"
        assert kwargs["headers"] == {"Authorization": "Bearer key"}
        assert kwargs["timeout"] == 3.0

    def test_authentication_failure(self):
        """Test a 401 reports an authentication failure."""
        provider = AskSageProviderAdapter()
        with patch("requests.get") as mock_get:
            mock_get.return_value = Mock(status_code=401)
            ok, message = provider.check_health(api_key="bad", base_url="https://test")

        assert ok is False
        assert "authentication failed" in message.lower()

    def test_unexpected_status(self):
        """Test a non-200/401 status reports the status code."""
        provider = AskSageProviderAdapter()
        with patch("requests.get") as mock_get:
            mock_get.return_value = Mock(status_code=503)
            ok, message = provider.check_health(api_key="key", base_url="https://test")

        assert ok is False
        assert "503" in message

    def test_timeout(self):
        """Test a timeout is reported as a connection timeout."""
        provider = AskSageProviderAdapter()
        with patch("requests.get", side_effect=requests.Timeout()):
            ok, message = provider.check_health(api_key="key", base_url="https://test")

        assert ok is False
        assert "timeout" in message.lower()

    def test_request_exception(self):
        """Test a connection error is reported with a truncated reason."""
        provider = AskSageProviderAdapter()
        with patch("requests.get", side_effect=requests.RequestException("refused")):
            ok, message = provider.check_health(api_key="key", base_url="https://test")

        assert ok is False
        assert "connection failed" in message.lower()
        assert "refused" in message

    def test_unexpected_error(self):
        """Test a non-requests error is reported as a health check failure."""
        provider = AskSageProviderAdapter()
        with patch("requests.get", side_effect=ValueError("weird")):
            ok, message = provider.check_health(api_key="key", base_url="https://test")

        assert ok is False
        assert "health check failed" in message.lower()

    def test_error_reason_is_truncated(self):
        """Test long error reasons are truncated to 50 characters."""
        provider = AskSageProviderAdapter()
        with patch("requests.get", side_effect=requests.RequestException("x" * 200)):
            ok, message = provider.check_health(api_key="key", base_url="https://test")

        assert ok is False
        assert "x" * 50 in message
        assert "x" * 51 not in message
