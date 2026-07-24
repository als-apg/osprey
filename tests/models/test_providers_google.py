"""Tests for the Google (Gemini) provider adapter.

Google is a thin LiteLLM delegator: ``execute_completion`` and ``check_health``
forward to the shared ``litellm_adapter`` helpers. These tests pin the
behavioral contract -- registry name, LiteLLM routing metadata, and how
credentials / base_url / model reach the litellm layer -- rather than the
copy-paste structure, so they survive the planned dedup of the six
near-identical LiteLLM adapters into one data-driven class.

Google is one of the two native (non-OpenAI-compatible) adapters: it routes
through the ``gemini`` litellm prefix and needs no base_url.
"""

from unittest.mock import patch

from pydantic import BaseModel

from osprey.models.providers.google import GoogleProviderAdapter

COMPLETION = "osprey.models.providers.google.execute_litellm_completion"
HEALTH = "osprey.models.providers.google.check_litellm_health"


class _Sample(BaseModel):
    result: str


class TestGoogleMetadata:
    """Registry- and routing-facing metadata is the single source of truth."""

    def test_registry_name(self):
        assert GoogleProviderAdapter.name == "google"

    def test_description_mentions_gemini(self):
        assert "Gemini" in GoogleProviderAdapter.description

    def test_requirement_flags(self):
        """Google needs a key and a model, but no base_url (native endpoint)."""
        assert GoogleProviderAdapter.requires_api_key is True
        assert GoogleProviderAdapter.requires_base_url is False
        assert GoogleProviderAdapter.requires_model_id is True
        assert GoogleProviderAdapter.supports_proxy is True

    def test_no_default_base_url(self):
        assert GoogleProviderAdapter.default_base_url is None

    def test_default_and_health_models(self):
        assert GoogleProviderAdapter.default_model_id == "gemini-2.5-flash"
        assert GoogleProviderAdapter.health_check_model_id == "gemini-2.5-flash-lite"

    def test_available_models_include_default_and_health(self):
        models = GoogleProviderAdapter.available_models
        assert len(models) > 0
        assert GoogleProviderAdapter.default_model_id in models
        assert GoogleProviderAdapter.health_check_model_id in models

    def test_litellm_routing_metadata(self):
        """Native gemini routing: gemini prefix, not OpenAI-compatible,
        structured-output support auto-detected (None default)."""
        assert GoogleProviderAdapter.litellm_prefix == "gemini"
        assert GoogleProviderAdapter.is_openai_compatible is False
        assert GoogleProviderAdapter.supports_native_structured_output is None

    def test_api_key_help_present(self):
        assert GoogleProviderAdapter.api_key_url is not None
        assert len(GoogleProviderAdapter.api_key_instructions) > 0
        assert GoogleProviderAdapter.api_key_note is None

    def test_is_instantiable(self):
        assert isinstance(GoogleProviderAdapter(), GoogleProviderAdapter)


class TestGoogleExecuteCompletion:
    """execute_completion forwards to the litellm layer under the provider name."""

    def test_returns_litellm_result_unchanged(self):
        with patch(COMPLETION, return_value="hello") as mock_exec:
            result = GoogleProviderAdapter().execute_completion(
                message="hi", model_id="gemini-2.5-flash", api_key="key", base_url=None
            )
        assert result == "hello"
        mock_exec.assert_called_once()

    def test_forwards_core_args_and_defaults(self):
        with patch(COMPLETION, return_value="ok") as mock_exec:
            GoogleProviderAdapter().execute_completion(
                message="hi", model_id="gemini-2.5-flash", api_key="key", base_url=None
            )
        kwargs = mock_exec.call_args.kwargs
        assert kwargs["provider"] == "google"
        assert kwargs["message"] == "hi"
        assert kwargs["model_id"] == "gemini-2.5-flash"
        assert kwargs["api_key"] == "key"
        assert kwargs["base_url"] is None
        assert kwargs["max_tokens"] == 1024
        assert kwargs["temperature"] == 0.0
        assert kwargs["output_format"] is None

    def test_forwards_overrides(self):
        with patch(COMPLETION, return_value="ok") as mock_exec:
            GoogleProviderAdapter().execute_completion(
                message="hi",
                model_id="gemini-2.5-pro",
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
        """Google extended thinking is passed through via kwargs (budget_tokens)."""
        with patch(COMPLETION, return_value="ok") as mock_exec:
            GoogleProviderAdapter().execute_completion(
                message="hi",
                model_id="gemini-2.5-flash",
                api_key="key",
                base_url=None,
                enable_thinking=True,
                budget_tokens=256,
            )
        kwargs = mock_exec.call_args.kwargs
        assert kwargs["enable_thinking"] is True
        assert kwargs["budget_tokens"] == 256


class TestGoogleCheckHealth:
    """check_health forwards to the litellm health helper with model fallback."""

    def test_returns_result_unchanged(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            result = GoogleProviderAdapter().check_health(api_key="key", base_url=None)
        assert result == (True, "ok")
        mock_health.assert_called_once()

    def test_forwards_args_and_default_timeout(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            GoogleProviderAdapter().check_health(api_key="key", base_url=None)
        kwargs = mock_health.call_args.kwargs
        assert kwargs["provider"] == "google"
        assert kwargs["api_key"] == "key"
        assert kwargs["base_url"] is None
        assert kwargs["timeout"] == 5.0

    def test_model_id_defaults_to_health_check_model(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            GoogleProviderAdapter().check_health(api_key="key", base_url=None)
        assert mock_health.call_args.kwargs["model_id"] == "gemini-2.5-flash-lite"

    def test_explicit_model_id_overrides_default(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            GoogleProviderAdapter().check_health(
                api_key="key", base_url=None, model_id="gemini-2.5-pro"
            )
        assert mock_health.call_args.kwargs["model_id"] == "gemini-2.5-pro"

    def test_custom_timeout_is_forwarded(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            GoogleProviderAdapter().check_health(api_key="key", base_url=None, timeout=12.0)
        assert mock_health.call_args.kwargs["timeout"] == 12.0

    def test_propagates_failure(self):
        with patch(HEALTH, return_value=(False, "down")):
            result = GoogleProviderAdapter().check_health(api_key="key", base_url=None)
        assert result == (False, "down")
