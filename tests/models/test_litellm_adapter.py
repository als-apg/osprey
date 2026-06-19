"""Tests for LiteLLM adapter module."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from osprey.models.providers.litellm_adapter import (
    _clean_json_response,
    _execute_ollama_structured_output,
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

    def test_explicit_provider_class_skips_registry(self, monkeypatch):
        """When provider_class is passed in, the registry must NOT be consulted
        (the lazy import / lookup is skipped entirely)."""
        import osprey.models.provider_registry as registry_module

        def _boom():
            raise AssertionError("registry must not be consulted when class is supplied")

        monkeypatch.setattr(registry_module, "get_provider_registry", _boom)

        class _StubOA:
            is_openai_compatible = True

        assert get_litellm_model_name("whatever", "m", provider_class=_StubOA) == "openai/m"

    def test_class_without_prefix_falls_through_to_fallback_maps(self):
        """A resolved class that is neither OpenAI-compatible nor declares a
        litellm_prefix falls through to the hardcoded fallback maps, which still
        route by provider name."""

        class _StubBare:
            is_openai_compatible = False
            # no litellm_prefix attribute -> getattr returns None

        assert get_litellm_model_name("anthropic", "claude-x", provider_class=_StubBare) == (
            "anthropic/claude-x"
        )

    def test_fallback_openai_compatible_set_when_registry_unresolved(self, monkeypatch):
        """If the registry can't resolve the provider, an OpenAI-compatible name
        still routes via the last-resort _openai_compatible set (openai/ prefix)."""
        import osprey.models.provider_registry as registry_module

        class _EmptyRegistry:
            def get_provider(self, name):
                return None

        monkeypatch.setattr(registry_module, "get_provider_registry", lambda: _EmptyRegistry())
        assert get_litellm_model_name("ds4", "deepseek-v4-flash") == "openai/deepseek-v4-flash"

    def test_fallback_prefix_maps_when_registry_unresolved(self, monkeypatch):
        """If the registry can't resolve the provider, the last-resort prefix map
        still applies: anthropic gets a prefix, openai gets none."""
        import osprey.models.provider_registry as registry_module

        class _EmptyRegistry:
            def get_provider(self, name):
                return None

        monkeypatch.setattr(registry_module, "get_provider_registry", lambda: _EmptyRegistry())
        assert get_litellm_model_name("anthropic", "claude-x") == "anthropic/claude-x"
        assert get_litellm_model_name("openai", "gpt-4o") == "gpt-4o"  # empty prefix


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

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_litellm_lookup_exception_falls_back_to_false(self, mock_litellm):
        """If the provider declares None (defer to litellm) and
        supports_response_schema raises, the decision degrades safely to False
        (prompt-based fallback) instead of propagating the error.

        openai declares the flag as None, so the decision reaches litellm here.
        """
        mock_litellm.supports_response_schema.side_effect = Exception("lookup blew up")
        assert _supports_native_structured_output("openai/gpt-4o", "openai") is False

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

    def _call_with_chat_request(self, mock_litellm, messages):
        """Drive the fallback path with chat_request set (multi-turn). The schema
        instruction is appended into completion_kwargs['messages'] in place, so the
        caller's `messages` list is what gets mutated and sent."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"name": "abc", "count": 3}'
        mock_litellm.completion.return_value = mock_response

        completion_kwargs = {"model": "openai/deepseek-v4-flash", "messages": messages}
        result = _handle_structured_output(
            provider="ds4",
            model_id="deepseek-v4-flash",
            litellm_model="openai/deepseek-v4-flash",
            message="ignored when chat_request is set",
            completion_kwargs=completion_kwargs,
            output_format=_FallbackModel,
            is_typed_dict_output=False,
            chat_request=object(),  # only checked for `is not None`
        )
        return result, mock_litellm.completion.call_args.kwargs["messages"]

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_chat_request_appends_schema_to_last_user_message(self, mock_litellm):
        """With chat_request set, the schema instruction is appended to the LAST
        user turn only — the system prompt and earlier turns are left intact, so
        multi-turn ds4 context is preserved."""
        result, sent = self._call_with_chat_request(
            mock_litellm,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "turn1"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "turn2"},
            ],
        )
        assert result == _FallbackModel(name="abc", count=3)
        assert sent[0]["content"] == "sys"
        assert sent[1]["content"] == "turn1"  # earlier user turn untouched
        assert sent[3]["content"].startswith("turn2")
        assert "must respond with valid JSON" in sent[3]["content"]

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_chat_request_appends_schema_to_last_list_content_block(self, mock_litellm):
        """When the last user message's content is a list (Anthropic-style cache
        blocks), the instruction is appended to the last block's 'text' field
        rather than overwriting the list."""
        result, sent = self._call_with_chat_request(
            mock_litellm,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "block1"},
                        {"type": "text", "text": "block2"},
                    ],
                }
            ],
        )
        assert result == _FallbackModel(name="abc", count=3)
        blocks = sent[0]["content"]
        assert blocks[0]["text"] == "block1"  # earlier block untouched
        assert blocks[1]["text"].startswith("block2")
        assert "must respond with valid JSON" in blocks[1]["text"]

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_chat_request_skips_trailing_non_user_message(self, mock_litellm):
        """The reverse search for the last user message must skip a trailing
        assistant turn and append the schema to the real last user turn, leaving
        the assistant message untouched."""
        result, sent = self._call_with_chat_request(
            mock_litellm,
            [
                {"role": "user", "content": "realquestion"},
                {"role": "assistant", "content": "partial"},
            ],
        )
        assert result == _FallbackModel(name="abc", count=3)
        assert sent[0]["content"].startswith("realquestion")
        assert "must respond with valid JSON" in sent[0]["content"]
        assert sent[1]["content"] == "partial"  # trailing assistant untouched

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_chat_request_with_no_user_message_sends_messages_unchanged(self, mock_litellm):
        """A pathological chat_request with no user turn finds nothing to append
        to, so the messages are sent as-is (the loop completes without a break)."""
        result, sent = self._call_with_chat_request(
            mock_litellm,
            [
                {"role": "system", "content": "sys"},
                {"role": "assistant", "content": "a"},
            ],
        )
        assert result == _FallbackModel(name="abc", count=3)
        assert [m["content"] for m in sent] == ["sys", "a"]  # untouched


class TestExecuteOllamaStructuredOutput:
    """The direct-API Ollama structured-output path (bypasses LiteLLM bug #15463)."""

    @patch("httpx.post")
    def test_typed_dict_output_returns_dict(self, mock_post):
        """is_typed_dict_output=True dumps the validated model to a plain dict."""
        response = MagicMock()
        response.json.return_value = {"message": {"content": '{"name": "z", "count": 7}'}}
        response.raise_for_status = MagicMock()
        mock_post.return_value = response

        result = _execute_ollama_structured_output(
            model_id="llama3.1:8b",
            message="extract",
            output_format=_FallbackModel,
            base_url="http://localhost:11434",
            max_tokens=256,
            is_typed_dict_output=True,
        )
        assert result == {"name": "z", "count": 7}
        assert isinstance(result, dict)
        # The direct path must request JSON-formatted output from Ollama.
        assert mock_post.call_args.kwargs["json"]["format"] == "json"

    @patch("httpx.post")
    def test_unparseable_response_raises_valueerror(self, mock_post):
        response = MagicMock()
        response.json.return_value = {"message": {"content": "not json"}}
        response.raise_for_status = MagicMock()
        mock_post.return_value = response

        with pytest.raises(ValueError, match="Failed to parse structured output from Ollama"):
            _execute_ollama_structured_output(
                model_id="llama3.1:8b",
                message="extract",
                output_format=_FallbackModel,
                base_url="http://localhost:11434",
                max_tokens=256,
            )


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

    @patch("litellm.completion")
    def test_no_api_key_is_omitted_from_completion_kwargs(self, mock_completion):
        """A falsy api_key must not be forwarded as api_key=None to litellm (it is
        simply omitted, letting litellm/env resolve credentials)."""
        message = MagicMock()
        message.tool_calls = None
        message.content = "ok"
        response = MagicMock()
        response.choices = [MagicMock(message=message)]
        mock_completion.return_value = response

        execute_litellm_completion(
            provider="openai", message="hi", model_id="gpt-4o", api_key=None, base_url=None
        )
        assert "api_key" not in mock_completion.call_args.kwargs

    @patch("litellm.completion")
    def test_empty_choices_returns_empty_string(self, mock_completion):
        """A response with no choices yields '' rather than raising an IndexError."""
        response = MagicMock()
        response.choices = []
        mock_completion.return_value = response

        result = execute_litellm_completion(
            provider="openai", message="hi", model_id="gpt-4o", api_key="k", base_url=None
        )
        assert result == ""


class TestExecuteLitellmCompletionThinking:
    """Extended-thinking handling in execute_litellm_completion."""

    def test_budget_tokens_not_less_than_max_raises(self):
        """budget_tokens >= max_tokens is rejected before any API call."""
        with pytest.raises(ValueError, match="budget_tokens must be less than max_tokens"):
            execute_litellm_completion(
                provider="anthropic",
                message="x",
                model_id="claude-sonnet-4",
                api_key="k",
                base_url=None,
                max_tokens=100,
                enable_thinking=True,
                budget_tokens=100,
            )

    @patch("litellm.completion")
    def test_google_thinking_sets_thinking_config(self, mock_completion):
        """Google extended thinking is forwarded as thinking_config, not the
        Anthropic-style 'thinking' kwarg."""
        message = MagicMock()
        message.tool_calls = None
        message.content = "ok"
        response = MagicMock()
        response.choices = [MagicMock(message=message)]
        mock_completion.return_value = response

        execute_litellm_completion(
            provider="google",
            message="x",
            model_id="gemini-2.5-flash",
            api_key="k",
            base_url=None,
            max_tokens=1024,
            enable_thinking=True,
            budget_tokens=500,
        )
        kwargs = mock_completion.call_args.kwargs
        assert kwargs["thinking_config"] == {"thinking_budget": 500}
        assert "thinking" not in kwargs

    @patch("litellm.completion")
    def test_anthropic_thinking_returns_content_blocks(self, mock_completion):
        """For Anthropic thinking, a list-typed message.content (thinking + text
        blocks) is returned verbatim rather than coerced to a string."""
        blocks = [
            {"type": "thinking", "thinking": "reasoning..."},
            {"type": "text", "text": "answer"},
        ]
        choice = MagicMock()
        choice.message.content = blocks
        response = MagicMock()
        response.choices = [choice]
        mock_completion.return_value = response

        result = execute_litellm_completion(
            provider="anthropic",
            message="x",
            model_id="claude-sonnet-4",
            api_key="k",
            base_url=None,
            max_tokens=1024,
            enable_thinking=True,
            budget_tokens=500,
        )
        assert result == blocks

    @patch("litellm.completion")
    def test_thinking_ignored_for_non_anthropic_google_provider(self, mock_completion):
        """enable_thinking is only wired for anthropic/google. For any other
        provider (e.g. ds4) the thinking request is silently dropped — no
        'thinking' or 'thinking_config' kwarg leaks into the completion call."""
        message = MagicMock()
        message.tool_calls = None
        message.content = "ok"
        response = MagicMock()
        response.choices = [MagicMock(message=message)]
        mock_completion.return_value = response

        result = execute_litellm_completion(
            provider="ds4",
            message="x",
            model_id="deepseek-v4-flash",
            api_key="EMPTY",
            base_url=None,
            max_tokens=1024,
            enable_thinking=True,
            budget_tokens=500,
        )
        assert result == "ok"
        kwargs = mock_completion.call_args.kwargs
        assert "thinking" not in kwargs
        assert "thinking_config" not in kwargs

    @patch("litellm.completion")
    def test_anthropic_thinking_empty_choices_falls_through(self, mock_completion):
        """When the thinking branch is taken but the response has no choices, it
        falls through the block without indexing and ends up returning ''."""
        response = MagicMock()
        response.choices = []
        mock_completion.return_value = response

        result = execute_litellm_completion(
            provider="anthropic",
            message="x",
            model_id="claude-sonnet-4",
            api_key="k",
            base_url=None,
            max_tokens=1024,
            enable_thinking=True,
            budget_tokens=500,
        )
        assert result == ""


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
    def test_ollama_healthy_without_api_key(self, mock_completion):
        """ollama is exempt from the api-key guard, so a keyless health check
        proceeds to the live call and the api_key is omitted from the kwargs."""
        mock_completion.return_value = MagicMock()
        ok, msg = check_litellm_health(
            provider="ollama", api_key=None, base_url="http://localhost:11434", model_id="llama3.1"
        )
        assert (ok, msg) == (True, "API accessible and authenticated")
        assert "api_key" not in mock_completion.call_args.kwargs

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

    @patch("litellm.completion")
    def test_not_found_error_reports_model(self, mock_completion):
        """A NotFoundError (wrong model id on the server) is unhealthy and names
        the offending model — the common ds4 case of pointing at a model the
        local server hasn't loaded."""
        import litellm

        mock_completion.side_effect = litellm.NotFoundError(
            message="no such model", model="deepseek-v4-pro", llm_provider="openai"
        )
        ok, msg = check_litellm_health(
            provider="ds4", api_key="EMPTY", base_url="http://h:8000/v1", model_id="deepseek-v4-pro"
        )
        assert ok is False
        assert "deepseek-v4-pro" in msg
        assert "not found" in msg

    @patch("litellm.completion")
    def test_bad_request_error_is_unhealthy(self, mock_completion):
        import litellm

        mock_completion.side_effect = litellm.BadRequestError(
            message="malformed", model="gpt-4o", llm_provider="openai"
        )
        ok, msg = check_litellm_health(
            provider="openai", api_key="sk-real", base_url=None, model_id="gpt-4o"
        )
        assert ok is False
        assert msg.startswith("Bad request")

    @patch("litellm.completion")
    def test_timeout_is_unhealthy(self, mock_completion):
        import litellm

        mock_completion.side_effect = litellm.Timeout(
            message="took too long", model="gpt-4o", llm_provider="openai"
        )
        assert check_litellm_health(
            provider="openai", api_key="sk-real", base_url=None, model_id="gpt-4o"
        ) == (False, "Request timeout")

    @patch("litellm.completion")
    def test_connection_error_is_unhealthy(self, mock_completion):
        """An APIConnectionError is the typical failure when the local ds4 server
        is down — it maps to a 'Connection failed' diagnostic, not a crash."""
        import litellm

        mock_completion.side_effect = litellm.APIConnectionError(
            message="refused", model="deepseek-v4-flash", llm_provider="openai"
        )
        ok, msg = check_litellm_health(
            provider="ds4",
            api_key="EMPTY",
            base_url="http://h:8000/v1",
            model_id="deepseek-v4-flash",
        )
        assert ok is False
        assert msg.startswith("Connection failed")

    @patch("litellm.completion")
    def test_api_error_is_unhealthy(self, mock_completion):
        import litellm

        mock_completion.side_effect = litellm.APIError(
            status_code=500, message="server blew up", model="gpt-4o", llm_provider="openai"
        )
        ok, msg = check_litellm_health(
            provider="openai", api_key="sk-real", base_url=None, model_id="gpt-4o"
        )
        assert ok is False
        assert msg.startswith("API error")
