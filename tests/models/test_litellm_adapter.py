"""Tests for LiteLLM adapter module."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from osprey.models.providers.litellm_adapter import (
    _clean_json_response,
    _handle_structured_output,
    _supports_native_structured_output,
    check_litellm_health,
    execute_litellm_completion,
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

    def test_amsc_i2_model(self):
        """AMSC i2 uses openai/ prefix (OpenAI-compatible)."""
        result = get_litellm_model_name("amsc-i2", "anthropic/claude-haiku")
        assert result == "openai/anthropic/claude-haiku"

    def test_vllm_model(self):
        """vLLM uses openai/ prefix (OpenAI-compatible)."""
        result = get_litellm_model_name("vllm", "some-model")
        assert result == "openai/some-model"

    def test_als_apg_model(self):
        """als-apg uses openai/ prefix (OpenAI-compatible)."""
        result = get_litellm_model_name("als-apg", "some-model")
        assert result == "openai/some-model"

    def test_ds4_model(self):
        """ds4 uses openai/ prefix (OpenAI-compatible)."""
        result = get_litellm_model_name("ds4", "some-model")
        assert result == "openai/some-model"

    def test_registry_class_attributes_drive_routing(self, monkeypatch):
        """A registry-resolved provider class drives routing even when the
        provider name is absent from the hardcoded fallback maps."""
        import osprey.models.provider_registry as registry_module

        class _StubProvider:
            is_openai_compatible = True

        class _StubRegistry:
            def get_provider(self, name):
                return _StubProvider if name == "synthetic_oai" else None

        monkeypatch.setattr(registry_module, "get_provider_registry", lambda: _StubRegistry())
        result = get_litellm_model_name("synthetic_oai", "m")
        assert result == "openai/m"

    def test_registry_prefix_attribute_drives_routing(self, monkeypatch):
        """A registry-resolved provider class's litellm_prefix drives routing
        even when the provider name is absent from the hardcoded fallback maps."""
        import osprey.models.provider_registry as registry_module

        class _StubProvider:
            litellm_prefix = "xprefix"
            is_openai_compatible = False

        class _StubRegistry:
            def get_provider(self, name):
                return _StubProvider if name == "synthetic_prefixed" else None

        monkeypatch.setattr(registry_module, "get_provider_registry", lambda: _StubRegistry())
        result = get_litellm_model_name("synthetic_prefixed", "m")
        assert result == "xprefix/m"

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
        """anthropic declares the flag as None, so the decision defers to litellm.

        litellm does not recognize the bare "claude-sonnet-4" model string, so
        ``supports_response_schema`` reports False — i.e. OSPREY uses its
        prompt-based fallback. Asserting the concrete decision (not just that a
        bool came back) catches an inverted or short-circuited routing bug.
        """
        result = _supports_native_structured_output("anthropic/claude-sonnet-4", "anthropic")
        assert result is False

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
        for provider in ("cborg", "stanford", "argo", "vllm", "amsc-i2"):
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

    def test_fixes_python_style_booleans(self):
        """Python-style True/False values are rewritten to JSON true/false.

        This exercises the boolean-fixing regex branch (lines 419-422), which is
        otherwise untested — covering both ': True' and ', False' positions.
        """
        result = _clean_json_response('{"a": True, "b": False}')
        assert result == '{"a": true, "b": false}'

    def test_does_not_corrupt_true_inside_string_value(self):
        """The boolean regex must only touch bare values, never text inside a
        quoted string (the leading quote breaks the ':\\s*True' match)."""
        result = _clean_json_response('{"x": "True story"}')
        assert result == '{"x": "True story"}'


class _FallbackModel(BaseModel):
    name: str
    count: int


class TestHandleStructuredOutputPromptFallback:
    """The prompt-based fallback path used by providers that declare
    supports_native_structured_output=False (e.g. ds4).

    provider='ds4' deterministically routes here because its registered flag is
    False; chat_request=None exercises the single-message construction.
    """

    def _call(self, mock_litellm, content, *, is_typed_dict_output=False):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content
        mock_litellm.completion.return_value = mock_response

        completion_kwargs = {"model": "openai/deepseek-v4-flash"}
        return _handle_structured_output(
            provider="ds4",
            model_id="deepseek-v4-flash",
            litellm_model="openai/deepseek-v4-flash",
            message="extract info",
            completion_kwargs=completion_kwargs,
            output_format=_FallbackModel,
            is_typed_dict_output=is_typed_dict_output,
        )

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_parses_valid_json_and_appends_schema_instruction(self, mock_litellm):
        result = self._call(mock_litellm, '{"name": "abc", "count": 3}')
        assert result == _FallbackModel(name="abc", count=3)
        # The single user message must carry the original prompt plus the
        # injected schema instruction (this is how a no-native-support model is
        # coaxed into returning JSON).
        sent = mock_litellm.completion.call_args.kwargs["messages"]
        assert len(sent) == 1
        assert sent[0]["role"] == "user"
        assert sent[0]["content"].startswith("extract info")
        assert "must respond with valid JSON" in sent[0]["content"]

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_typed_dict_output_returns_dict(self, mock_litellm):
        result = self._call(mock_litellm, '{"name": "abc", "count": 3}', is_typed_dict_output=True)
        assert result == {"name": "abc", "count": 3}
        assert isinstance(result, dict)

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_markdown_wrapped_json_is_cleaned_then_parsed(self, mock_litellm):
        result = self._call(mock_litellm, '```json\n{"name": "y", "count": 1}\n```')
        assert result == _FallbackModel(name="y", count=1)

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_unparseable_response_raises_valueerror(self, mock_litellm):
        with pytest.raises(ValueError, match="Failed to parse structured output from ds4"):
            self._call(mock_litellm, "this is not json")


class TestStructuredOutputCapabilityFlag:
    """The capability attribute drives the structured-output path."""

    @pytest.mark.unit
    def test_base_default_is_none(self):
        from osprey.models.providers.base import BaseProvider

        assert BaseProvider.supports_native_structured_output is None

    @pytest.mark.unit
    def test_openai_compatible_providers_declare_true(self):
        from osprey.models.providers.amsc_i2 import AMSCI2ProviderAdapter
        from osprey.models.providers.argo import ArgoProviderAdapter
        from osprey.models.providers.cborg import CBorgProviderAdapter
        from osprey.models.providers.stanford import StanfordProviderAdapter
        from osprey.models.providers.vllm import VLLMProviderAdapter

        for cls in (
            CBorgProviderAdapter,
            StanfordProviderAdapter,
            ArgoProviderAdapter,
            VLLMProviderAdapter,
            AMSCI2ProviderAdapter,
        ):
            assert cls.supports_native_structured_output is True, cls.name

    @pytest.mark.unit
    def test_flag_true_takes_native_path(self):
        assert _supports_native_structured_output("openai/anything", "vllm") is True

    @pytest.mark.unit
    def test_flag_none_defers_to_litellm(self):
        # openai provider has supports_native_structured_output = None, so defers to litellm.
        # Use a model string litellm knows natively (gpt-4o returns True).
        assert _supports_native_structured_output("gpt-4o", "openai") is True

    @pytest.mark.unit
    def test_unknown_provider_defers_and_is_safe(self):
        assert _supports_native_structured_output("unknown/nonexistent-xyz", "unknown") is False

    @pytest.mark.unit
    def test_ds4_declares_false_end_to_end(self):
        # gpt-4o is known to litellm as supporting response_schema (True for "openai").
        # ds4 overrides this to False because it ignores json_schema despite being
        # OpenAI-compatible — so a False result here can ONLY come from ds4's registered flag.
        assert _supports_native_structured_output("openai/gpt-4o", "ds4") is False
        # Sanity: same model string under "openai" returns True, proving the difference
        # is the ds4 registration + flag, not the model string.
        assert _supports_native_structured_output("openai/gpt-4o", "openai") is True


class TestExecuteLitellmCompletionResponses:
    """Response-shape handling in execute_litellm_completion (non-structured)."""

    @patch("litellm.completion")
    def test_returns_normalized_tool_calls(self, mock_completion):
        """A response carrying tool_calls is normalized to the OSPREY tool-call
        dict shape (id/type/function), not returned as raw text."""
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "get_weather"
        tc.function.arguments = '{"city": "SF"}'
        message = MagicMock()
        message.tool_calls = [tc]
        response = MagicMock()
        response.choices = [MagicMock(message=message)]
        mock_completion.return_value = response

        result = execute_litellm_completion(
            provider="openai",
            message="weather?",
            model_id="gpt-4o",
            api_key="k",
            base_url=None,
        )

        assert result == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
            }
        ]

    @patch("litellm.completion")
    def test_returns_text_when_no_tool_calls(self, mock_completion):
        """A plain text response (tool_calls falsy) yields the content string."""
        message = MagicMock()
        message.tool_calls = None
        message.content = "hello there"
        response = MagicMock()
        response.choices = [MagicMock(message=message)]
        mock_completion.return_value = response

        result = execute_litellm_completion(
            provider="openai",
            message="hi",
            model_id="gpt-4o",
            api_key="k",
            base_url=None,
        )

        assert result == "hello there"


class TestCheckLitellmHealth:
    """Guard rails and litellm-exception mapping in check_litellm_health."""

    @pytest.mark.unit
    def test_missing_api_key(self):
        assert check_litellm_health(
            provider="openai", api_key=None, base_url=None, model_id="gpt-4o"
        ) == (False, "API key not set")

    @pytest.mark.unit
    def test_placeholder_api_key_rejected(self):
        ok, msg = check_litellm_health(
            provider="openai", api_key="${OPENAI_API_KEY}", base_url=None, model_id="gpt-4o"
        )
        assert ok is False
        assert "placeholder" in msg

    @pytest.mark.unit
    def test_missing_model_id(self):
        assert check_litellm_health(
            provider="openai", api_key="sk-real", base_url=None, model_id=None
        ) == (False, "Model ID required for health check")

    @patch("litellm.completion")
    def test_healthy_when_call_succeeds(self, mock_completion):
        mock_completion.return_value = MagicMock()
        assert check_litellm_health(
            provider="openai", api_key="sk-real", base_url=None, model_id="gpt-4o"
        ) == (True, "API accessible and authenticated")

    @patch("litellm.completion")
    def test_authentication_error_is_unhealthy(self, mock_completion):
        import litellm

        mock_completion.side_effect = litellm.AuthenticationError(
            message="bad key", llm_provider="openai", model="gpt-4o"
        )
        assert check_litellm_health(
            provider="openai", api_key="sk-bad", base_url=None, model_id="gpt-4o"
        ) == (False, "Authentication failed (invalid API key)")

    @patch("litellm.completion")
    def test_rate_limit_is_treated_as_healthy(self, mock_completion):
        """A rate-limit response proves the key works, so it reports healthy."""
        import litellm

        mock_completion.side_effect = litellm.RateLimitError(
            message="slow down", llm_provider="openai", model="gpt-4o"
        )
        ok, msg = check_litellm_health(
            provider="openai", api_key="sk-real", base_url=None, model_id="gpt-4o"
        )
        assert ok is True
        assert "rate limited" in msg

    @patch("litellm.completion")
    def test_unexpected_error_is_wrapped(self, mock_completion):
        mock_completion.side_effect = ValueError("boom")
        ok, msg = check_litellm_health(
            provider="openai", api_key="sk-real", base_url=None, model_id="gpt-4o"
        )
        assert ok is False
        assert msg.startswith("Unexpected error")
