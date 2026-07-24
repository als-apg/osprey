"""Tests for the AMSC i2 (American Science Cloud) provider adapter.

AMSC i2 is a thin LiteLLM delegator: ``execute_completion`` and ``check_health``
forward to the shared ``litellm_adapter`` helpers. These tests pin the
behavioral contract -- registry name, LiteLLM routing metadata, and how
credentials / base_url / model reach the litellm layer -- rather than the
copy-paste structure, so they survive the planned dedup of the six
near-identical LiteLLM adapters into one data-driven class.
"""

from unittest.mock import patch

from pydantic import BaseModel

from osprey.models.providers.amsc_i2 import AMSCI2ProviderAdapter

COMPLETION = "osprey.models.providers.amsc_i2.execute_litellm_completion"
HEALTH = "osprey.models.providers.amsc_i2.check_litellm_health"


class _Sample(BaseModel):
    result: str


class TestAMSCI2Metadata:
    """Registry- and routing-facing metadata is the single source of truth."""

    def test_registry_name(self):
        assert AMSCI2ProviderAdapter.name == "amsc-i2"

    def test_description_mentions_provider(self):
        assert "American Science Cloud" in AMSCI2ProviderAdapter.description

    def test_requirement_flags(self):
        assert AMSCI2ProviderAdapter.requires_api_key is True
        assert AMSCI2ProviderAdapter.requires_base_url is True
        assert AMSCI2ProviderAdapter.requires_model_id is True
        assert AMSCI2ProviderAdapter.supports_proxy is True

    def test_no_default_base_url(self):
        """AMSC i2 has no shipped endpoint -- it is deployment/config supplied."""
        assert AMSCI2ProviderAdapter.default_base_url is None

    def test_default_and_health_models(self):
        assert AMSCI2ProviderAdapter.default_model_id == "claude-haiku"
        assert AMSCI2ProviderAdapter.health_check_model_id == "claude-haiku"

    def test_available_models_include_default_and_health(self):
        models = AMSCI2ProviderAdapter.available_models
        assert len(models) > 0
        assert AMSCI2ProviderAdapter.default_model_id in models
        assert AMSCI2ProviderAdapter.health_check_model_id in models

    def test_litellm_routing_metadata(self):
        """OpenAI-compatible proxy: routes via openai/<model>, native json_schema."""
        assert AMSCI2ProviderAdapter.is_openai_compatible is True
        assert AMSCI2ProviderAdapter.supports_native_structured_output is True
        assert AMSCI2ProviderAdapter.litellm_prefix is None

    def test_api_key_help_present(self):
        assert AMSCI2ProviderAdapter.api_key_url is not None
        assert len(AMSCI2ProviderAdapter.api_key_instructions) > 0
        assert AMSCI2ProviderAdapter.api_key_note is not None

    def test_is_instantiable(self):
        assert isinstance(AMSCI2ProviderAdapter(), AMSCI2ProviderAdapter)


class TestAMSCI2ExecuteCompletion:
    """execute_completion forwards to the litellm layer under the provider name."""

    def test_returns_litellm_result_unchanged(self):
        with patch(COMPLETION, return_value="hello") as mock_exec:
            result = AMSCI2ProviderAdapter().execute_completion(
                message="hi", model_id="m", api_key="key", base_url="https://proxy"
            )
        assert result == "hello"
        mock_exec.assert_called_once()

    def test_forwards_core_args_and_defaults(self):
        with patch(COMPLETION, return_value="ok") as mock_exec:
            AMSCI2ProviderAdapter().execute_completion(
                message="hi", model_id="m", api_key="key", base_url="https://proxy"
            )
        kwargs = mock_exec.call_args.kwargs
        assert kwargs["provider"] == "amsc-i2"
        assert kwargs["message"] == "hi"
        assert kwargs["model_id"] == "m"
        assert kwargs["api_key"] == "key"
        assert kwargs["base_url"] == "https://proxy"
        assert kwargs["max_tokens"] == 1024
        assert kwargs["temperature"] == 0.0
        assert kwargs["output_format"] is None

    def test_forwards_overrides(self):
        with patch(COMPLETION, return_value="ok") as mock_exec:
            AMSCI2ProviderAdapter().execute_completion(
                message="hi",
                model_id="m",
                api_key="key",
                base_url="https://proxy",
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
            AMSCI2ProviderAdapter().execute_completion(
                message="hi",
                model_id="m",
                api_key="key",
                base_url="https://proxy",
                enable_thinking=True,
                budget_tokens=256,
            )
        kwargs = mock_exec.call_args.kwargs
        assert kwargs["enable_thinking"] is True
        assert kwargs["budget_tokens"] == 256

    def test_missing_base_url_is_forwarded_as_none(self):
        """AMSC i2 does NOT substitute a default base_url; None passes through
        (unlike stanford). A dedup applying ``base_url or default`` to every
        adapter would regress this."""
        with patch(COMPLETION, return_value="ok") as mock_exec:
            AMSCI2ProviderAdapter().execute_completion(
                message="hi", model_id="m", api_key="key", base_url=None
            )
        assert mock_exec.call_args.kwargs["base_url"] is None


class TestAMSCI2CheckHealth:
    """check_health forwards to the litellm health helper with model fallback."""

    def test_returns_result_unchanged(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            result = AMSCI2ProviderAdapter().check_health(api_key="key", base_url="https://proxy")
        assert result == (True, "ok")
        mock_health.assert_called_once()

    def test_forwards_args_and_default_timeout(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            AMSCI2ProviderAdapter().check_health(api_key="key", base_url="https://proxy")
        kwargs = mock_health.call_args.kwargs
        assert kwargs["provider"] == "amsc-i2"
        assert kwargs["api_key"] == "key"
        assert kwargs["base_url"] == "https://proxy"
        assert kwargs["timeout"] == 5.0

    def test_model_id_defaults_to_health_check_model(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            AMSCI2ProviderAdapter().check_health(api_key="key", base_url="https://proxy")
        assert mock_health.call_args.kwargs["model_id"] == "claude-haiku"

    def test_explicit_model_id_overrides_default(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            AMSCI2ProviderAdapter().check_health(
                api_key="key", base_url="https://proxy", model_id="claude-opus"
            )
        assert mock_health.call_args.kwargs["model_id"] == "claude-opus"

    def test_custom_timeout_is_forwarded(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            AMSCI2ProviderAdapter().check_health(
                api_key="key", base_url="https://proxy", timeout=12.0
            )
        assert mock_health.call_args.kwargs["timeout"] == 12.0

    def test_missing_base_url_is_forwarded_as_none(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            AMSCI2ProviderAdapter().check_health(api_key="key", base_url=None)
        assert mock_health.call_args.kwargs["base_url"] is None

    def test_propagates_failure(self):
        with patch(HEALTH, return_value=(False, "down")):
            result = AMSCI2ProviderAdapter().check_health(api_key="key", base_url="https://proxy")
        assert result == (False, "down")
