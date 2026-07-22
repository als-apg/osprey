"""Tests for the ALS-APG provider adapter.

ALS-APG is a thin LiteLLM delegator: ``execute_completion`` and ``check_health``
forward to the shared ``litellm_adapter`` helpers. These tests pin the
behavioral contract -- registry name, LiteLLM routing metadata, and how
credentials / base_url / model reach the litellm layer -- rather than the
copy-paste structure, so they survive the planned dedup of the six
near-identical LiteLLM adapters into one data-driven class.

ALS-APG differs from the other OpenAI-compatible proxies (amsc-i2, cborg,
stanford) in one deliberate way: it leaves ``supports_native_structured_output``
at the None default so structured-output support is auto-detected, rather than
asserting True. That default is pinned below.
"""

from unittest.mock import patch

from pydantic import BaseModel

from osprey.models.providers.als_apg import ALSAPGProviderAdapter

COMPLETION = "osprey.models.providers.als_apg.execute_litellm_completion"
HEALTH = "osprey.models.providers.als_apg.check_litellm_health"


class _Sample(BaseModel):
    result: str


class TestALSAPGMetadata:
    """Registry- and routing-facing metadata is the single source of truth."""

    def test_registry_name(self):
        assert ALSAPGProviderAdapter.name == "als-apg"

    def test_description_mentions_provider(self):
        assert "ALS" in ALSAPGProviderAdapter.description

    def test_requirement_flags(self):
        assert ALSAPGProviderAdapter.requires_api_key is True
        assert ALSAPGProviderAdapter.requires_base_url is True
        assert ALSAPGProviderAdapter.requires_model_id is True
        assert ALSAPGProviderAdapter.supports_proxy is True

    def test_default_base_url(self):
        assert ALSAPGProviderAdapter.default_base_url == "https://llm.gianlucamartino.com"

    def test_default_and_health_models(self):
        assert ALSAPGProviderAdapter.default_model_id == "claude-haiku-4-5-20251001"
        assert ALSAPGProviderAdapter.health_check_model_id == "claude-haiku-4-5-20251001"

    def test_available_models_include_default_and_health(self):
        models = ALSAPGProviderAdapter.available_models
        assert len(models) > 0
        assert ALSAPGProviderAdapter.default_model_id in models
        assert ALSAPGProviderAdapter.health_check_model_id in models

    def test_litellm_routing_metadata(self):
        """OpenAI-compatible proxy, but structured-output support is auto-detected.

        Unlike amsc-i2/cborg/stanford, als-apg intentionally leaves the flag at
        None (see the adapter's comment) so litellm.supports_response_schema()
        decides. Pinned so a dedup does not accidentally flip it to True/False.
        """
        assert ALSAPGProviderAdapter.is_openai_compatible is True
        assert ALSAPGProviderAdapter.supports_native_structured_output is None
        assert ALSAPGProviderAdapter.litellm_prefix is None

    def test_api_key_help_present(self):
        assert len(ALSAPGProviderAdapter.api_key_instructions) > 0
        assert ALSAPGProviderAdapter.api_key_note is not None

    def test_is_instantiable(self):
        assert isinstance(ALSAPGProviderAdapter(), ALSAPGProviderAdapter)


class TestALSAPGExecuteCompletion:
    """execute_completion forwards to the litellm layer under the provider name."""

    def test_returns_litellm_result_unchanged(self):
        with patch(COMPLETION, return_value="hello") as mock_exec:
            result = ALSAPGProviderAdapter().execute_completion(
                message="hi", model_id="m", api_key="key", base_url="https://proxy"
            )
        assert result == "hello"
        mock_exec.assert_called_once()

    def test_forwards_core_args_and_defaults(self):
        with patch(COMPLETION, return_value="ok") as mock_exec:
            ALSAPGProviderAdapter().execute_completion(
                message="hi", model_id="m", api_key="key", base_url="https://proxy"
            )
        kwargs = mock_exec.call_args.kwargs
        assert kwargs["provider"] == "als-apg"
        assert kwargs["message"] == "hi"
        assert kwargs["model_id"] == "m"
        assert kwargs["api_key"] == "key"
        assert kwargs["base_url"] == "https://proxy"
        assert kwargs["max_tokens"] == 1024
        assert kwargs["temperature"] == 0.0
        assert kwargs["output_format"] is None

    def test_forwards_overrides(self):
        with patch(COMPLETION, return_value="ok") as mock_exec:
            ALSAPGProviderAdapter().execute_completion(
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
            ALSAPGProviderAdapter().execute_completion(
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

    def test_missing_base_url_falls_back_to_default(self):
        """als-apg routes openai-compatible, so a missing base_url must resolve
        to its default endpoint rather than falling through to api.openai.com."""
        with patch(COMPLETION, return_value="ok") as mock_exec:
            ALSAPGProviderAdapter().execute_completion(
                message="hi", model_id="m", api_key="key", base_url=None
            )
        assert mock_exec.call_args.kwargs["base_url"] == "https://llm.gianlucamartino.com"


class TestALSAPGCheckHealth:
    """check_health forwards to the litellm health helper with model fallback."""

    def test_returns_result_unchanged(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            result = ALSAPGProviderAdapter().check_health(api_key="key", base_url="https://proxy")
        assert result == (True, "ok")
        mock_health.assert_called_once()

    def test_forwards_args_and_default_timeout(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            ALSAPGProviderAdapter().check_health(api_key="key", base_url="https://proxy")
        kwargs = mock_health.call_args.kwargs
        assert kwargs["provider"] == "als-apg"
        assert kwargs["api_key"] == "key"
        assert kwargs["base_url"] == "https://proxy"
        assert kwargs["timeout"] == 5.0

    def test_model_id_defaults_to_health_check_model(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            ALSAPGProviderAdapter().check_health(api_key="key", base_url="https://proxy")
        assert mock_health.call_args.kwargs["model_id"] == "claude-haiku-4-5-20251001"

    def test_explicit_model_id_overrides_default(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            ALSAPGProviderAdapter().check_health(
                api_key="key", base_url="https://proxy", model_id="claude-opus-4-6"
            )
        assert mock_health.call_args.kwargs["model_id"] == "claude-opus-4-6"

    def test_custom_timeout_is_forwarded(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            ALSAPGProviderAdapter().check_health(
                api_key="key", base_url="https://proxy", timeout=12.0
            )
        assert mock_health.call_args.kwargs["timeout"] == 12.0

    def test_missing_base_url_falls_back_to_default(self):
        with patch(HEALTH, return_value=(True, "ok")) as mock_health:
            ALSAPGProviderAdapter().check_health(api_key="key", base_url=None)
        assert mock_health.call_args.kwargs["base_url"] == "https://llm.gianlucamartino.com"

    def test_propagates_failure(self):
        with patch(HEALTH, return_value=(False, "down")):
            result = ALSAPGProviderAdapter().check_health(api_key="key", base_url="https://proxy")
        assert result == (False, "down")
