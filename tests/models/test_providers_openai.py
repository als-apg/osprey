"""Tests for the OpenAI provider adapter.

OpenAI is a thin LiteLLM delegator: ``execute_completion`` and ``check_health``
forward to the shared ``litellm_adapter`` helpers. These tests pin the
behavioral contract -- registry name, LiteLLM routing metadata, and how
credentials / base_url / model reach the litellm layer -- rather than the
copy-paste structure, so they survive the planned dedup of the six
near-identical LiteLLM adapters into one data-driven class.

OpenAI is the other native adapter: it uses the empty litellm prefix (models
route bare, e.g. ``gpt-5``) and, notably, does NOT forward its default_base_url
even though one is declared -- see ``test_missing_base_url_is_forwarded_as_none``.
"""

from unittest.mock import patch

from pydantic import BaseModel

from osprey.models.providers.openai import OpenAIProviderAdapter

COMPLETION = "osprey.models.providers.openai.execute_litellm_completion"
HEALTH = "osprey.models.providers.openai.check_litellm_health"


class _Sample(BaseModel):
    result: str


class TestOpenAIMetadata:
    """Registry- and routing-facing metadata is the single source of truth."""

    def test_registry_name(self):
        assert OpenAIProviderAdapter.name == "openai"

    def test_description_mentions_gpt(self):
        assert "GPT" in OpenAIProviderAdapter.description

    def test_requirement_flags(self):
        """OpenAI needs a key and a model, but no base_url (native endpoint)."""
        assert OpenAIProviderAdapter.requires_api_key is True
        assert OpenAIProviderAdapter.requires_base_url is False
        assert OpenAIProviderAdapter.requires_model_id is True
        assert OpenAIProviderAdapter.supports_proxy is True

    def test_default_base_url(self):
        assert OpenAIProviderAdapter.default_base_url == "https://api.openai.com/v1"

    def test_default_and_health_models(self):
        assert OpenAIProviderAdapter.default_model_id == "gpt-5"
        assert OpenAIProviderAdapter.health_check_model_id == "gpt-5-nano"

    def test_available_models_include_default_and_health(self):
        models = OpenAIProviderAdapter.available_models
        assert len(models) > 0
        assert OpenAIProviderAdapter.default_model_id in models
        assert OpenAIProviderAdapter.health_check_model_id in models

    def test_litellm_routing_metadata(self):
        """Native OpenAI routing: empty prefix (bare model id), not
        OpenAI-compatible-proxy mode, structured output auto-detected."""
        assert OpenAIProviderAdapter.litellm_prefix == ""
        assert OpenAIProviderAdapter.is_openai_compatible is False
        assert OpenAIProviderAdapter.supports_native_structured_output is None

    def test_api_key_help_present(self):
        assert OpenAIProviderAdapter.api_key_url is not None
        assert len(OpenAIProviderAdapter.api_key_instructions) > 0
        assert OpenAIProviderAdapter.api_key_note is None

    def test_is_instantiable(self):
        assert isinstance(OpenAIProviderAdapter(), OpenAIProviderAdapter)


class TestOpenAIExecuteCompletion:
    """execute_completion forwards to the litellm layer under the provider name."""

    def test_returns_litellm_result_unchanged(self):
        with patch(COMPLETION, return_value="hello") as mock_exec:
            result = OpenAIProviderAdapter().execute_completion(
                message="hi", model_id="gpt-5", api_key="key", base_url=None
            )
        assert result == "hello"
        mock_exec.assert_called_once()

    def test_forwards_core_args_and_defaults(self):
        with patch(COMPLETION, return_value="ok") as mock_exec:
            OpenAIProviderAdapter().execute_completion(
                message="hi", model_id="gpt-5", api_key="key", base_url=None
            )
        kwargs = mock_exec.call_args.kwargs
        assert kwargs["provider"] == "openai"
        assert kwargs["message"] == "hi"
        assert kwargs["model_id"] == "gpt-5"
        assert kwargs["api_key"] == "key"
        assert kwargs["max_tokens"] == 1024
        assert kwargs["temperature"] == 0.0
        assert kwargs["output_format"] is None

    def test_forwards_overrides(self):
        with patch(COMPLETION, return_value="ok") as mock_exec:
            OpenAIProviderAdapter().execute_completion(
                message="hi",
                model_id="gpt-5-mini",
                api_key="key",
                base_url=None,
                max_tokens=64,
                temperature=0.7,
                output_format=_Sample,
            )
        kwargs = mock_exec.call_args.kwargs
        assert kwargs["max_tokens"] == 64
        assert kwargs["temperature"] == 0.7
        assert kwargs["output_format"] is _Sample

    def test_forwards_extra_kwargs(self):
        with patch(COMPLETION, return_value="ok") as mock_exec:
            OpenAIProviderAdapter().execute_completion(
                message="hi",
                model_id="gpt-5",
                api_key="key",
                base_url=None,
                reasoning_effort="high",
            )
        assert mock_exec.call_args.kwargs["reasoning_effort"] == "high"

    def test_missing_base_url_is_forwarded_as_none(self):
        """OpenAI declares a default_base_url but does NOT substitute it; None
        passes straight through (litellm uses its own OpenAI endpoint). A dedup
        applying ``base_url or default`` would silently change this to the
        declared URL -- a real behavioral regression this pins."""
        with patch(COMPLETION, return_value="ok") as mock_exec:
            OpenAIProviderAdapter().execute_completion(
                message="hi", model_id="gpt-5", api_key="key", base_url=None
            )
        assert mock_exec.call_args.kwargs["base_url"] is None

    def test_explicit_base_url_is_forwarded(self):
        with patch(COMPLETION, return_value="ok") as mock_exec:
            OpenAIProviderAdapter().execute_completion(
                message="hi", model_id="gpt-5", api_key="key", base_url="https://azure.example/v1"
            )
        assert mock_exec.call_args.kwargs["base_url"] == "https://azure.example/v1"


class TestOpenAICheckHealth:
    """check_health forwards to the litellm health helper with model fallback."""

    def test_returns_result_unchanged(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            result = OpenAIProviderAdapter().check_health(api_key="key", base_url=None)
        assert result == (True, "ok")
        mock_health.assert_called_once()

    def test_forwards_args_and_default_timeout(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            OpenAIProviderAdapter().check_health(api_key="key", base_url=None)
        kwargs = mock_health.call_args.kwargs
        assert kwargs["provider"] == "openai"
        assert kwargs["api_key"] == "key"
        assert kwargs["base_url"] is None
        assert kwargs["timeout"] == 5.0

    def test_model_id_defaults_to_health_check_model(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            OpenAIProviderAdapter().check_health(api_key="key", base_url=None)
        assert mock_health.call_args.kwargs["model_id"] == "gpt-5-nano"

    def test_explicit_model_id_overrides_default(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            OpenAIProviderAdapter().check_health(api_key="key", base_url=None, model_id="gpt-5")
        assert mock_health.call_args.kwargs["model_id"] == "gpt-5"

    def test_custom_timeout_is_forwarded(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            OpenAIProviderAdapter().check_health(api_key="key", base_url=None, timeout=12.0)
        assert mock_health.call_args.kwargs["timeout"] == 12.0

    def test_missing_base_url_is_forwarded_as_none(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            OpenAIProviderAdapter().check_health(api_key="key", base_url=None)
        assert mock_health.call_args.kwargs["base_url"] is None

    def test_propagates_failure(self):
        with patch(HEALTH, return_value=(False, "down")):
            result = OpenAIProviderAdapter().check_health(api_key="key", base_url=None)
        assert result == (False, "down")
