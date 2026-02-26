"""Tests for ChatMessage and ChatCompletionRequest."""

from osprey.models.messages import ChatCompletionRequest, ChatMessage


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
