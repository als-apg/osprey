"""Tests for LiteLLM adapter module."""

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

    def test_unknown_provider(self):
        """Unknown providers use provider/model format (LiteLLM's default routing)."""
        result = get_litellm_model_name("unknown_provider", "some-model")
        assert result == "unknown_provider/some-model"


class TestSupportsNativeStructuredOutput:
    """Tests for structured output support detection.

    Note: _supports_native_structured_output uses explicit provider/model checks
    to determine if native JSON schema support is available.
    """

    def test_takes_provider_and_model_id(self):
        """Function accepts provider name and model_id."""
        # Should not raise - function accepts strings and returns bool
        result = _supports_native_structured_output("anthropic", "claude-sonnet-4")
        assert isinstance(result, bool)

    def test_handles_unknown_provider_gracefully(self):
        """Returns False for unknown providers instead of raising."""
        # Unknown providers should return False (use prompt-based fallback)
        result = _supports_native_structured_output("unknown", "nonexistent-model-xyz")
        assert result is False

    def test_openai_gpt4o_supported(self):
        """OpenAI GPT-4o models support structured output."""
        result = _supports_native_structured_output("openai", "gpt-4o")
        assert result is True

    def test_ollama_not_supported(self):
        """Ollama models don't support native structured output."""
        result = _supports_native_structured_output("ollama", "llama3.1:8b")
        assert result is False

    def test_cborg_vllm_support_structured_output(self):
        """CBORG and vLLM providers support structured output."""
        for provider in ("cborg", "vllm"):
            result = _supports_native_structured_output(provider, "some-model")
            assert result is True, f"Provider {provider} should support structured output"

    def test_argo_does_not_support_structured_output(self):
        """Argo API ignores response_format - must use prompt-based fallback."""
        # Argo doesn't support json_schema parameter - it returns markdown
        result = _supports_native_structured_output("argo", "claude-sonnet-4")
        assert result is False


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
