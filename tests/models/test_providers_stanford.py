"""Tests for the Stanford AI Playground provider adapter.

Stanford is a thin LiteLLM delegator: ``execute_completion`` and ``check_health``
forward to the shared ``litellm_adapter`` helpers. These tests pin the
behavioral contract -- registry name, LiteLLM routing metadata, and how
credentials / base_url / model reach the litellm layer -- rather than the
copy-paste structure, so they survive the planned dedup of the six
near-identical LiteLLM adapters into one data-driven class.

Stanford is the ONLY one of the six that applies a base_url fallback
(``base_url or self.default_base_url``) inside both methods. That quirk -- a
missing base_url resolves to the Stanford endpoint before delegation -- is the
central behavioral difference pinned here; the other five forward base_url
(including None) unchanged.
"""

from unittest.mock import patch

from pydantic import BaseModel

from osprey.models.providers.stanford import StanfordProviderAdapter

COMPLETION = "osprey.models.providers.stanford.execute_litellm_completion"
HEALTH = "osprey.models.providers.stanford.check_litellm_health"
DEFAULT_URL = "https://aiapi-prod.stanford.edu/v1"


class _Sample(BaseModel):
    result: str


class TestStanfordMetadata:
    """Registry- and routing-facing metadata is the single source of truth."""

    def test_registry_name(self):
        assert StanfordProviderAdapter.name == "stanford"

    def test_description_mentions_stanford(self):
        assert "Stanford" in StanfordProviderAdapter.description

    def test_requirement_flags(self):
        assert StanfordProviderAdapter.requires_api_key is True
        assert StanfordProviderAdapter.requires_base_url is True
        assert StanfordProviderAdapter.requires_model_id is True
        assert StanfordProviderAdapter.supports_proxy is True

    def test_default_base_url(self):
        assert StanfordProviderAdapter.default_base_url == DEFAULT_URL

    def test_default_and_health_models(self):
        assert StanfordProviderAdapter.default_model_id == "gpt-4o"
        assert StanfordProviderAdapter.health_check_model_id == "gpt-4o-mini"

    def test_available_models_include_default_and_health(self):
        models = StanfordProviderAdapter.available_models
        assert len(models) > 0
        assert StanfordProviderAdapter.default_model_id in models
        assert StanfordProviderAdapter.health_check_model_id in models

    def test_litellm_routing_metadata(self):
        """OpenAI-compatible proxy: routes via openai/<model>, native json_schema."""
        assert StanfordProviderAdapter.is_openai_compatible is True
        assert StanfordProviderAdapter.supports_native_structured_output is True
        assert StanfordProviderAdapter.litellm_prefix is None

    def test_api_key_help_present(self):
        assert StanfordProviderAdapter.api_key_url is not None
        assert len(StanfordProviderAdapter.api_key_instructions) > 0
        assert StanfordProviderAdapter.api_key_note is not None

    def test_is_instantiable(self):
        assert isinstance(StanfordProviderAdapter(), StanfordProviderAdapter)


class TestStanfordExecuteCompletion:
    """execute_completion forwards to the litellm layer under the provider name."""

    def test_returns_litellm_result_unchanged(self):
        with patch(COMPLETION, return_value="hello") as mock_exec:
            result = StanfordProviderAdapter().execute_completion(
                message="hi", model_id="m", api_key="key", base_url="https://custom"
            )
        assert result == "hello"
        mock_exec.assert_called_once()

    def test_forwards_core_args_and_defaults(self):
        with patch(COMPLETION, return_value="ok") as mock_exec:
            StanfordProviderAdapter().execute_completion(
                message="hi", model_id="m", api_key="key", base_url="https://custom"
            )
        kwargs = mock_exec.call_args.kwargs
        assert kwargs["provider"] == "stanford"
        assert kwargs["message"] == "hi"
        assert kwargs["model_id"] == "m"
        assert kwargs["api_key"] == "key"
        assert kwargs["max_tokens"] == 1024
        assert kwargs["temperature"] == 0.0
        assert kwargs["output_format"] is None

    def test_forwards_overrides(self):
        with patch(COMPLETION, return_value="ok") as mock_exec:
            StanfordProviderAdapter().execute_completion(
                message="hi",
                model_id="m",
                api_key="key",
                base_url="https://custom",
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
            StanfordProviderAdapter().execute_completion(
                message="hi",
                model_id="m",
                api_key="key",
                base_url="https://custom",
                enable_thinking=True,
                budget_tokens=256,
            )
        kwargs = mock_exec.call_args.kwargs
        assert kwargs["enable_thinking"] is True
        assert kwargs["budget_tokens"] == 256

    def test_explicit_base_url_passes_through(self):
        """A supplied base_url is used verbatim (the fallback only fills gaps)."""
        with patch(COMPLETION, return_value="ok") as mock_exec:
            StanfordProviderAdapter().execute_completion(
                message="hi", model_id="m", api_key="key", base_url="https://custom"
            )
        assert mock_exec.call_args.kwargs["base_url"] == "https://custom"

    def test_missing_base_url_falls_back_to_default(self):
        """The Stanford-only quirk: base_url None resolves to default_base_url
        (``base_url or self.default_base_url``) before delegation."""
        with patch(COMPLETION, return_value="ok") as mock_exec:
            StanfordProviderAdapter().execute_completion(
                message="hi", model_id="m", api_key="key", base_url=None
            )
        assert mock_exec.call_args.kwargs["base_url"] == DEFAULT_URL

    def test_empty_base_url_falls_back_to_default(self):
        """An empty string is falsy, so it also resolves to the default."""
        with patch(COMPLETION, return_value="ok") as mock_exec:
            StanfordProviderAdapter().execute_completion(
                message="hi", model_id="m", api_key="key", base_url=""
            )
        assert mock_exec.call_args.kwargs["base_url"] == DEFAULT_URL


class TestStanfordCheckHealth:
    """check_health forwards to the litellm health helper with model + url fallback."""

    def test_returns_result_unchanged(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            result = StanfordProviderAdapter().check_health(
                api_key="key", base_url="https://custom"
            )
        assert result == (True, "ok")
        mock_health.assert_called_once()

    def test_forwards_args_and_default_timeout(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            StanfordProviderAdapter().check_health(api_key="key", base_url="https://custom")
        kwargs = mock_health.call_args.kwargs
        assert kwargs["provider"] == "stanford"
        assert kwargs["api_key"] == "key"
        assert kwargs["base_url"] == "https://custom"
        assert kwargs["timeout"] == 5.0

    def test_model_id_defaults_to_health_check_model(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            StanfordProviderAdapter().check_health(api_key="key", base_url="https://custom")
        assert mock_health.call_args.kwargs["model_id"] == "gpt-4o-mini"

    def test_explicit_model_id_overrides_default(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            StanfordProviderAdapter().check_health(
                api_key="key", base_url="https://custom", model_id="o3-mini"
            )
        assert mock_health.call_args.kwargs["model_id"] == "o3-mini"

    def test_custom_timeout_is_forwarded(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            StanfordProviderAdapter().check_health(
                api_key="key", base_url="https://custom", timeout=12.0
            )
        assert mock_health.call_args.kwargs["timeout"] == 12.0

    def test_missing_base_url_falls_back_to_default(self):
        """check_health applies the same base_url fallback as execute_completion."""
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            StanfordProviderAdapter().check_health(api_key="key", base_url=None)
        assert mock_health.call_args.kwargs["base_url"] == DEFAULT_URL

    def test_propagates_failure(self):
        with patch(HEALTH, return_value=(False, "down")):
            result = StanfordProviderAdapter().check_health(
                api_key="key", base_url="https://custom"
            )
        assert result == (False, "down")
