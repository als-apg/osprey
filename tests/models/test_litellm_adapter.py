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
        """Unknown providers return model_id as-is."""
        result = get_litellm_model_name("unknown_provider", "some-model")
        assert result == "some-model"


class TestSupportsNativeStructuredOutput:
    """Tests for structured output support detection."""

    def test_anthropic_claude_4_supports(self):
        """Claude 4+ models support native structured outputs."""
        assert _supports_native_structured_output("anthropic", "claude-sonnet-4") is True
        assert _supports_native_structured_output("anthropic", "claude-opus-4") is True

    def test_anthropic_claude_3_not_supported(self):
        """Claude 3.x models don't support native structured outputs."""
        assert _supports_native_structured_output("anthropic", "claude-3-sonnet") is False

    def test_openai_gpt4o_supports(self):
        """GPT-4o supports native structured outputs."""
        assert _supports_native_structured_output("openai", "gpt-4o") is True
        assert _supports_native_structured_output("openai", "gpt-4-turbo") is True

    def test_openai_gpt35_not_supported(self):
        """GPT-3.5 doesn't support native structured outputs."""
        assert _supports_native_structured_output("openai", "gpt-3.5-turbo") is False

    def test_google_not_supported(self):
        """Google models use prompt-based approach."""
        assert _supports_native_structured_output("google", "gemini-2.5-flash") is False

    def test_ollama_not_supported(self):
        """Ollama models use prompt-based approach."""
        assert _supports_native_structured_output("ollama", "llama3.1:8b") is False


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
