"""Tests for LiteLLM adapter module."""

import pytest

from osprey.models.providers.litellm_adapter import (
    _clean_json_response,
    _supports_native_structured_output,
    get_litellm_model_name,
)


class TestGetLiteLLMModelName:
    """Tests for model name mapping."""

    def test_anthropic_model(self):
        """Anthropic models get anthropic/ prefix."""
        result = get_litellm_model_name("anthropic", "claude-sonnet-4")
        assert result == "anthropic/claude-sonnet-4"

    def test_google_model(self):
        """Google models get gemini/ prefix."""
        result = get_litellm_model_name("google", "gemini-2.5-flash")
        assert result == "gemini/gemini-2.5-flash"

    def test_openai_model(self):
        """OpenAI models don't need prefix."""
        result = get_litellm_model_name("openai", "gpt-4o")
        assert result == "gpt-4o"

    def test_ollama_model(self):
        """Ollama models get ollama/ prefix."""
        result = get_litellm_model_name("ollama", "llama3.1:8b")
        assert result == "ollama/llama3.1:8b"

    def test_cborg_model(self):
        """CBORG uses openai/ prefix (OpenAI-compatible)."""
        result = get_litellm_model_name("cborg", "anthropic/claude-haiku")
        assert result == "openai/anthropic/claude-haiku"

    def test_stanford_model(self):
        """Stanford uses openai/ prefix (OpenAI-compatible)."""
        result = get_litellm_model_name("stanford", "gpt-4o")
        assert result == "openai/gpt-4o"

    def test_argo_model(self):
        """ARGO uses openai/ prefix (OpenAI-compatible)."""
        result = get_litellm_model_name("argo", "claudesonnet45")
        assert result == "openai/claudesonnet45"

    def test_amsc_model(self):
        """AMSC uses openai/ prefix (OpenAI-compatible)."""
        result = get_litellm_model_name("amsc", "anthropic/claude-haiku")
        assert result == "openai/anthropic/claude-haiku"

    def test_unknown_provider(self):
        """Unknown providers use provider/model format (LiteLLM's default routing)."""
        result = get_litellm_model_name("unknown_provider", "some-model")
        assert result == "unknown_provider/some-model"


class TestSupportsNativeStructuredOutput:
    """Tests for structured output support detection.

    Note: _supports_native_structured_output delegates to LiteLLM's
    supports_response_schema() function, with fallback for OpenAI-compatible providers.
    """

    def test_takes_litellm_model_string(self):
        """Function accepts LiteLLM-formatted model string and provider."""
        # Should not raise - function accepts string and returns bool
        result = _supports_native_structured_output("anthropic/claude-sonnet-4", "anthropic")
        assert isinstance(result, bool)

    def test_handles_unknown_model_gracefully(self):
        """Returns False for unknown models instead of raising."""
        # Unknown models should return False (use prompt-based fallback)
        result = _supports_native_structured_output("unknown/nonexistent-model-xyz", "unknown")
        assert result is False

    def test_openai_models_format(self):
        """OpenAI models use direct model name (no prefix)."""
        # OpenAI models don't need prefix in LiteLLM
        result = _supports_native_structured_output("gpt-4o", "openai")
        assert isinstance(result, bool)

    def test_ollama_models_format(self):
        """Ollama models use ollama/ prefix."""
        result = _supports_native_structured_output("ollama/llama3.1:8b", "ollama")
        assert isinstance(result, bool)

    def test_openai_compatible_providers_return_true(self):
        """OpenAI-compatible providers (CBORG, etc.) always support structured output."""
        # These providers proxy to models that support structured output.
        # NOTE: ds4 is the intentional counterexample — it is OpenAI-compatible
        # (is_openai_compatible=True) but declares supports_native_structured_output=False
        # because it accepts but ignores response_format json_schema. ds4 is therefore
        # excluded from this loop; its False override is tested in TestStructuredOutputCapabilityFlag.
        for provider in ("cborg", "stanford", "argo", "vllm", "amsc"):
            result = _supports_native_structured_output("openai/some-model", provider)
            assert result is True, f"Provider {provider} should support structured output"


class TestCleanJsonResponse:
    """Tests for JSON response cleaning."""

    def test_clean_json_no_markdown(self):
        """Clean JSON without markdown passes through."""
        result = _clean_json_response('{"key": "value"}')
        assert result == '{"key": "value"}'

    def test_clean_json_with_json_block(self):
        """Removes ```json markdown blocks."""
        result = _clean_json_response('```json\n{"key": "value"}\n```')
        assert result == '{"key": "value"}'

    def test_clean_json_with_generic_block(self):
        """Removes generic ``` markdown blocks."""
        result = _clean_json_response('```\n{"key": "value"}\n```')
        assert result == '{"key": "value"}'

    def test_clean_json_with_whitespace(self):
        """Handles whitespace around JSON."""
        result = _clean_json_response('  {"key": "value"}  ')
        assert result == '{"key": "value"}'

    def test_clean_json_only_trailing_block(self):
        """Handles only trailing markdown."""
        result = _clean_json_response('{"key": "value"}```')
        assert result == '{"key": "value"}'


class TestStructuredOutputCapabilityFlag:
    """The capability attribute drives the structured-output path."""

    # Routing wiring (litellm_adapter reading this flag) lands in the structured-output refactor task; this test pins the base default only.

    @pytest.mark.unit
    def test_base_default_is_none(self):
        from osprey.models.providers.base import BaseProvider
        assert BaseProvider.supports_native_structured_output is None

    @pytest.mark.unit
    def test_openai_compatible_providers_declare_true(self):
        from osprey.models.providers.cborg import CBorgProviderAdapter
        from osprey.models.providers.stanford import StanfordProviderAdapter
        from osprey.models.providers.argo import ArgoProviderAdapter
        from osprey.models.providers.vllm import VLLMProviderAdapter
        from osprey.models.providers.amsc import AMSCProviderAdapter
        for cls in (CBorgProviderAdapter, StanfordProviderAdapter,
                    ArgoProviderAdapter, VLLMProviderAdapter, AMSCProviderAdapter):
            assert cls.supports_native_structured_output is True, cls.name

    @pytest.mark.unit
    def test_flag_true_takes_native_path(self):
        from osprey.models.providers.litellm_adapter import _supports_native_structured_output
        assert _supports_native_structured_output("openai/anything", "vllm") is True

    @pytest.mark.unit
    def test_flag_none_defers_to_litellm(self):
        from osprey.models.providers.litellm_adapter import _supports_native_structured_output
        # openai provider has supports_native_structured_output = None, so defers to litellm.
        # Use a model string litellm knows natively (gpt-4o returns True).
        assert _supports_native_structured_output("gpt-4o", "openai") is True

    @pytest.mark.unit
    def test_unknown_provider_defers_and_is_safe(self):
        from osprey.models.providers.litellm_adapter import _supports_native_structured_output
        assert _supports_native_structured_output("unknown/nonexistent-xyz", "unknown") is False

    @pytest.mark.unit
    def test_ds4_declares_false_end_to_end(self):
        from osprey.models.providers.litellm_adapter import _supports_native_structured_output
        # gpt-4o is known to litellm as supporting response_schema (True for "openai").
        # ds4 overrides this to False because it ignores json_schema despite being
        # OpenAI-compatible — so a False result here can ONLY come from ds4's registered flag.
        assert _supports_native_structured_output("openai/gpt-4o", "ds4") is False
        # Sanity: same model string under "openai" returns True, proving the difference
        # is the ds4 registration + flag, not the model string.
        assert _supports_native_structured_output("openai/gpt-4o", "openai") is True
