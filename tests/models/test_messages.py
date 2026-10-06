"""Tests for ChatMessage and ChatCompletionRequest."""

import base64
import copy
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from osprey.models.messages import ChatCompletionRequest, ChatMessage, parse_data_url


class TestChatMessage:
    """Test ChatMessage dataclass."""

    def test_to_dict(self):
        msg = ChatMessage("user", "hello")
        assert msg.to_dict() == {"role": "user", "content": "hello"}

    def test_system_role(self):
        msg = ChatMessage("system", "You are a helpful assistant")
        assert msg.to_dict() == {"role": "system", "content": "You are a helpful assistant"}

    def test_assistant_role(self):
        msg = ChatMessage("assistant", "Sure, I can help")
        assert msg.to_dict() == {"role": "assistant", "content": "Sure, I can help"}


class TestChatCompletionRequestBasic:
    """Test basic ChatCompletionRequest behavior."""

    def test_to_litellm_messages_returns_list_of_dicts(self):
        req = ChatCompletionRequest(messages=[ChatMessage("user", "hello")])
        result = req.to_litellm_messages()
        assert isinstance(result, list)
        assert all(isinstance(m, dict) for m in result)

    def test_empty_request(self):
        req = ChatCompletionRequest(messages=[])
        assert req.to_litellm_messages() == []

    def test_message_order_preserved(self):
        req = ChatCompletionRequest(
            messages=[
                ChatMessage("system", "sys"),
                ChatMessage("user", "u1"),
                ChatMessage("assistant", "a1"),
                ChatMessage("user", "u2"),
            ]
        )
        result = req.to_litellm_messages()
        assert [m["role"] for m in result] == ["system", "user", "assistant", "user"]

    def test_all_roles_present(self):
        req = ChatCompletionRequest(
            messages=[
                ChatMessage("system", "sys"),
                ChatMessage("user", "hi"),
                ChatMessage("assistant", "hello"),
            ]
        )
        result = req.to_litellm_messages()
        roles = {m["role"] for m in result}
        assert roles == {"system", "user", "assistant"}


class TestAnthropicCacheControl:
    """Test Anthropic-specific cache_control markers."""

    def test_system_message_gets_cache_control(self):
        req = ChatCompletionRequest(
            messages=[
                ChatMessage("system", "You are helpful"),
                ChatMessage("user", "hello"),
            ]
        )
        result = req.to_litellm_messages(provider="anthropic")
        sys_msg = result[0]
        assert isinstance(sys_msg["content"], list)
        assert sys_msg["content"][0]["type"] == "text"
        assert sys_msg["content"][0]["text"] == "You are helpful"
        assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_second_to_last_user_message_gets_cache_control(self):
        """With 3+ user messages, the second-to-last gets cache_control."""
        req = ChatCompletionRequest(
            messages=[
                ChatMessage("system", "sys"),
                ChatMessage("user", "u1"),
                ChatMessage("assistant", "a1"),
                ChatMessage("user", "u2"),
                ChatMessage("assistant", "a2"),
                ChatMessage("user", "u3"),
            ]
        )
        result = req.to_litellm_messages(provider="anthropic")
        # u2 is the second-to-last user message (index 3)
        assert isinstance(result[3]["content"], list)
        assert result[3]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_two_user_messages_first_gets_cache(self):
        """With exactly 2 user messages, the first user msg gets cache_control."""
        req = ChatCompletionRequest(
            messages=[
                ChatMessage("system", "sys"),
                ChatMessage("user", "u1"),
                ChatMessage("assistant", "a1"),
                ChatMessage("user", "u2"),
            ]
        )
        result = req.to_litellm_messages(provider="anthropic")
        # u1 (index 1) is the second-to-last user message
        assert isinstance(result[1]["content"], list)
        assert result[1]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_single_user_message_no_user_cache(self):
        """With only 1 user message, no user-level cache marker."""
        req = ChatCompletionRequest(
            messages=[
                ChatMessage("system", "sys"),
                ChatMessage("user", "only user msg"),
            ]
        )
        result = req.to_litellm_messages(provider="anthropic")
        # System gets cache, user does not
        assert isinstance(result[0]["content"], list)  # system has cache
        assert isinstance(result[1]["content"], str)  # user stays string

    def test_last_user_message_never_cached(self):
        """The final user message never gets cache_control."""
        req = ChatCompletionRequest(
            messages=[
                ChatMessage("system", "sys"),
                ChatMessage("user", "u1"),
                ChatMessage("assistant", "a1"),
                ChatMessage("user", "u2 - last"),
            ]
        )
        result = req.to_litellm_messages(provider="anthropic")
        last_user = result[3]
        # Last user message should remain a string (not wrapped in content blocks)
        assert isinstance(last_user["content"], str)

    def test_assistant_messages_unchanged(self):
        """Assistant messages are never modified by cache logic."""
        req = ChatCompletionRequest(
            messages=[
                ChatMessage("system", "sys"),
                ChatMessage("user", "u1"),
                ChatMessage("assistant", "a1"),
                ChatMessage("user", "u2"),
            ]
        )
        result = req.to_litellm_messages(provider="anthropic")
        assistant_msg = result[2]
        assert assistant_msg["content"] == "a1"
        assert isinstance(assistant_msg["content"], str)


class TestNonAnthropicProviders:
    """Test that non-Anthropic providers get clean dicts."""

    def test_openai_no_cache_markers(self):
        req = ChatCompletionRequest(
            messages=[
                ChatMessage("system", "sys"),
                ChatMessage("user", "hello"),
            ]
        )
        result = req.to_litellm_messages(provider="openai")
        for msg in result:
            assert isinstance(msg["content"], str)
            assert "cache_control" not in msg

    def test_google_no_cache_markers(self):
        req = ChatCompletionRequest(
            messages=[
                ChatMessage("system", "sys"),
                ChatMessage("user", "hello"),
            ]
        )
        result = req.to_litellm_messages(provider="google")
        for msg in result:
            assert isinstance(msg["content"], str)

    def test_none_provider_no_cache_markers(self):
        req = ChatCompletionRequest(
            messages=[
                ChatMessage("system", "sys"),
                ChatMessage("user", "hello"),
            ]
        )
        result = req.to_litellm_messages(provider=None)
        for msg in result:
            assert isinstance(msg["content"], str)

    def test_non_anthropic_content_stays_string(self):
        req = ChatCompletionRequest(
            messages=[
                ChatMessage("system", "sys"),
                ChatMessage("user", "hello"),
                ChatMessage("assistant", "hi"),
            ]
        )
        result = req.to_litellm_messages(provider="openai")
        for msg in result:
            assert isinstance(msg["content"], str)


class TestToSingleString:
    """Test to_single_string() flattening."""

    def test_flattens_all_messages(self):
        req = ChatCompletionRequest(
            messages=[
                ChatMessage("user", "hello"),
                ChatMessage("assistant", "hi there"),
            ]
        )
        result = req.to_single_string()
        assert "hello" in result
        assert "hi there" in result

    def test_system_and_user_and_assistant(self):
        req = ChatCompletionRequest(
            messages=[
                ChatMessage("system", "You are helpful"),
                ChatMessage("user", "What is 2+2?"),
                ChatMessage("assistant", "4"),
            ]
        )
        result = req.to_single_string()
        assert result == "You are helpful\n\nWhat is 2+2?\n\n4"

    def test_empty_request_returns_empty_string(self):
        req = ChatCompletionRequest(messages=[])
        assert req.to_single_string() == ""


class TestChatMessageToolCalling:
    """Test ChatMessage tool-calling fields."""

    def test_chat_message_with_tool_calls_to_dict(self):
        """Assistant message with tool_calls serializes correctly."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_context", "arguments": '{"context_type": "PV"}'},
            }
        ]
        msg = ChatMessage(role="assistant", tool_calls=tool_calls)
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["tool_calls"] == tool_calls
        assert "content" not in d  # content is None → omitted

    def test_chat_message_tool_response_to_dict(self):
        """Tool response message serializes correctly."""
        msg = ChatMessage(
            role="tool", content="result data", tool_call_id="call_1", name="read_context"
        )
        d = msg.to_dict()
        assert d == {
            "role": "tool",
            "content": "result data",
            "tool_call_id": "call_1",
            "name": "read_context",
        }

    def test_chat_message_none_content_to_dict(self):
        """content=None omits content key from dict."""
        msg = ChatMessage(
            role="assistant",
            tool_calls=[
                {"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        )
        d = msg.to_dict()
        assert "content" not in d

    def test_chat_message_none_content_to_single_string(self):
        """to_single_string() skips messages with None content."""
        req = ChatCompletionRequest(
            messages=[
                ChatMessage(role="user", content="hello"),
                ChatMessage(role="assistant"),  # None content
                ChatMessage(role="tool", content="result", tool_call_id="c1", name="f"),
            ]
        )
        result = req.to_single_string()
        assert "hello" in result
        assert "result" in result

    def test_chat_message_plain_excludes_tool_fields(self):
        """Plain message without tool fields omits tool_calls, tool_call_id, name."""
        msg = ChatMessage(role="user", content="hello")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "hello"}
        assert "tool_calls" not in d
        assert "tool_call_id" not in d
        assert "name" not in d


class TestChatCompletionRequestEdgeCases:
    """Edge case tests."""

    def test_deeply_nested_conversation(self):
        """10+ messages work correctly."""
        msgs = []
        msgs.append(ChatMessage("system", "sys"))
        for i in range(10):
            msgs.append(ChatMessage("user", f"user msg {i}"))
            msgs.append(ChatMessage("assistant", f"assistant msg {i}"))
        req = ChatCompletionRequest(messages=msgs)
        result = req.to_litellm_messages()
        assert len(result) == 21  # 1 system + 10 user + 10 assistant

    def test_consecutive_same_role_messages(self):
        """Two user messages in a row are both included."""
        req = ChatCompletionRequest(
            messages=[
                ChatMessage("user", "first"),
                ChatMessage("user", "second"),
            ]
        )
        result = req.to_litellm_messages()
        assert len(result) == 2
        assert result[0]["content"] == "first"
        assert result[1]["content"] == "second"


# --- Content parts (text + image) ---

_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4  # 1032 bytes
_PNG_B64 = base64.b64encode(_PNG_BYTES).decode("ascii")
_PNG_URL = f"data:image/png;base64,{_PNG_B64}"


def _image_part(url: str = _PNG_URL) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def _mixed_message(role: str = "user", text: str = "What is in this plot?") -> ChatMessage:
    return ChatMessage(role, [{"type": "text", "text": text}, _image_part()])


class TestParseDataUrl:
    """parse_data_url splits a base64 data URL into (mime, n_bytes, b64)."""

    def test_png(self):
        mime, n_bytes, b64 = parse_data_url(_PNG_URL)
        assert mime == "image/png"
        assert n_bytes == len(_PNG_BYTES)
        assert b64 == _PNG_B64

    @pytest.mark.parametrize("payload", [b"a", b"ab", b"abc", b"abcd", b""])
    def test_byte_count_matches_decoded_length_for_every_padding(self, payload):
        b64 = base64.b64encode(payload).decode("ascii")
        _, n_bytes, _ = parse_data_url(f"data:image/jpeg;base64,{b64}")
        assert n_bytes == len(payload)

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.org/a.png",
            "data:image/png,rawnotbase64",
            "data:;base64,AAAA",
            "not a url",
        ],
    )
    def test_rejects_non_base64_data_urls(self, url):
        with pytest.raises(ValueError):
            parse_data_url(url)


class TestContentParts:
    """ChatMessage.content may be a list of LiteLLM content parts."""

    def test_to_dict_passes_parts_through(self):
        msg = _mixed_message()
        d = msg.to_dict()
        assert d["content"] == msg.content

    def test_to_dict_copies_list_and_parts(self):
        msg = _mixed_message()
        d = msg.to_dict()
        assert d["content"] is not msg.content
        assert all(a is not b for a, b in zip(d["content"], msg.content, strict=True))
        d["content"][0]["text"] += " mutated"
        d["content"].append({"type": "text", "text": "extra"})
        assert msg.content == _mixed_message().content

    def test_mixed_message_round_trips_to_litellm_image_url_parts(self):
        req = ChatCompletionRequest(messages=[_mixed_message()])
        result = req.to_litellm_messages(provider="openai")
        assert result == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this plot?"},
                    {"type": "image_url", "image_url": {"url": _PNG_URL}},
                ],
            }
        ]

    def test_anthropic_single_user_list_message_has_no_marker(self):
        req = ChatCompletionRequest(messages=[_mixed_message()])
        result = req.to_litellm_messages(provider="anthropic")
        assert all("cache_control" not in p for p in result[0]["content"])

    def test_anthropic_marker_on_last_text_part_of_second_to_last_user(self):
        req = ChatCompletionRequest(
            messages=[
                ChatMessage(
                    "user",
                    [
                        {"type": "text", "text": "first"},
                        _image_part(),
                        {"type": "text", "text": "caption"},
                        _image_part(),
                    ],
                ),
                ChatMessage("assistant", "ok"),
                ChatMessage("user", "follow-up"),
            ]
        )
        result = req.to_litellm_messages(provider="anthropic")
        parts = result[0]["content"]
        assert parts[2] == {
            "type": "text",
            "text": "caption",
            "cache_control": {"type": "ephemeral"},
        }
        assert "cache_control" not in parts[0]
        assert "cache_control" not in parts[1]
        assert "cache_control" not in parts[3]
        assert result[2]["content"] == "follow-up"

    def test_anthropic_marker_on_last_text_part_of_list_system_message(self):
        req = ChatCompletionRequest(
            messages=[
                ChatMessage("system", [{"type": "text", "text": "sys"}, _image_part()]),
                ChatMessage("user", "hi"),
            ]
        )
        result = req.to_litellm_messages(provider="anthropic")
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in result[0]["content"][1]

    def test_anthropic_image_only_message_gets_no_marker(self):
        req = ChatCompletionRequest(
            messages=[
                ChatMessage("user", [_image_part()]),
                ChatMessage("user", "describe it"),
            ]
        )
        result = req.to_litellm_messages(provider="anthropic")
        assert result[0]["content"] == [_image_part()]

    def test_anthropic_calling_twice_leaves_original_unchanged(self):
        msgs = [
            ChatMessage("system", [{"type": "text", "text": "sys"}]),
            _mixed_message(),
            ChatMessage("assistant", "ok"),
            ChatMessage("user", "follow-up"),
        ]
        snapshot = copy.deepcopy(msgs)
        req = ChatCompletionRequest(messages=msgs)
        first = req.to_litellm_messages(provider="anthropic")
        second = req.to_litellm_messages(provider="anthropic")
        assert first == second
        assert msgs == snapshot
        assert all("cache_control" not in p for p in msgs[0].content)
        assert all("cache_control" not in p for p in msgs[1].content)


class _Answer(BaseModel):
    answer: str


class TestStructuredOutputFallbackDoesNotMutateCaller:
    """The prompt-based structured-output fallback appends to the last user
    message in place; the caller's ChatMessage must never see that append."""

    def _run(self, req: ChatCompletionRequest):
        from osprey.models.providers import litellm_adapter

        reply = MagicMock()
        reply.choices[0].message.content = '{"answer": "a plot"}'
        with (
            patch.object(litellm_adapter, "_supports_native_structured_output", return_value=False),
            patch.object(litellm_adapter.litellm, "completion", return_value=reply) as completion,
        ):
            kwargs = {"model": "openai/gpt-x", "messages": req.to_litellm_messages("openai")}
            out = litellm_adapter._handle_structured_output(
                provider="openai",
                model_id="gpt-x",
                litellm_model="openai/gpt-x",
                message="",
                completion_kwargs=kwargs,
                output_format=_Answer,
                is_typed_dict_output=False,
                chat_request=req,
            )
        return out, completion.call_args.kwargs["messages"]

    def test_openai_fallback_twice_leaves_original_unchanged(self):
        msg = ChatMessage(
            "user", [_image_part(), {"type": "text", "text": "What is in this plot?"}]
        )
        snapshot = copy.deepcopy(msg)
        req = ChatCompletionRequest(messages=[msg])

        out1, sent1 = self._run(req)
        out2, sent2 = self._run(req)

        assert out1.answer == out2.answer == "a plot"
        assert "valid JSON" in sent1[0]["content"][-1]["text"]
        # Each call appends the schema exactly once — no accumulation.
        assert sent2[0]["content"][-1]["text"].count("valid JSON") == 1
        assert msg == snapshot


class TestToSingleStringContentParts:
    """to_single_string renders image parts as a size note, never base64."""

    def test_image_part_rendered_as_mime_and_size(self):
        req = ChatCompletionRequest(messages=[_mixed_message()])
        s = req.to_single_string()
        assert "What is in this plot?" in s
        assert f"[image image/png, {len(_PNG_BYTES)} bytes]" in s

    def test_log_string_never_contains_base64(self):
        req = ChatCompletionRequest(
            messages=[
                ChatMessage("system", "sys"),
                _mixed_message(),
                ChatMessage("assistant", "ok"),
                ChatMessage("user", [_image_part(), _image_part()]),
            ]
        )
        s = req.to_single_string()
        assert _PNG_B64 not in s
        assert _PNG_B64[:32] not in s
        assert "base64" not in s
        assert s.count("[image image/png,") == 3

    def test_text_parts_joined(self):
        req = ChatCompletionRequest(
            messages=[
                ChatMessage(
                    "user",
                    [{"type": "text", "text": "alpha"}, {"type": "text", "text": "beta"}],
                )
            ]
        )
        s = req.to_single_string()
        assert "alpha" in s and "beta" in s
        assert s.index("alpha") < s.index("beta")

    def test_string_and_list_messages_mix(self):
        req = ChatCompletionRequest(
            messages=[ChatMessage("system", "sys"), ChatMessage("user", [_image_part()])]
        )
        assert req.to_single_string() == (f"sys\n\n[image image/png, {len(_PNG_BYTES)} bytes]")

    def test_non_data_url_image_never_echoes_url_payload(self):
        req = ChatCompletionRequest(
            messages=[ChatMessage("user", [_image_part("https://example.org/a.png")])]
        )
        assert req.to_single_string().startswith("[image")
