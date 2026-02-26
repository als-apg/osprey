"""Tests for chat_request and tools handling in litellm_adapter."""

from unittest.mock import MagicMock, patch

import pytest

from osprey.models.messages import ChatCompletionRequest, ChatMessage


class TestToolsPassthrough:
    """Test tools parameter passthrough to litellm."""

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_tools_passed_to_completion_kwargs(self, mock_litellm):
        """tools list is added to completion_kwargs when provided."""
        from osprey.models.providers.litellm_adapter import execute_litellm_completion

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response"
        mock_response.choices[0].message.tool_calls = None
        mock_litellm.completion.return_value = mock_response

        tools = [{"type": "function", "function": {"name": "test_fn", "parameters": {}}}]

        execute_litellm_completion(
            provider="openai",
            message="hello",
            model_id="gpt-4o",
            api_key="test",
            base_url=None,
            tools=tools,
        )

        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs["tools"] == tools

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_tool_choice_defaults_to_auto(self, mock_litellm):
        """tool_choice='auto' set when tools provided without explicit tool_choice."""
        from osprey.models.providers.litellm_adapter import execute_litellm_completion

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response"
        mock_response.choices[0].message.tool_calls = None
        mock_litellm.completion.return_value = mock_response

        tools = [{"type": "function", "function": {"name": "test_fn", "parameters": {}}}]

        execute_litellm_completion(
            provider="openai",
            message="hello",
            model_id="gpt-4o",
            api_key="test",
            base_url=None,
            tools=tools,
        )

        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs["tool_choice"] == "auto"

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_tool_calls_in_response_returned_as_list(self, mock_litellm):
        """Response with tool_calls returns list of dicts."""
        from osprey.models.providers.litellm_adapter import execute_litellm_completion

        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "read_context"
        tc.function.arguments = '{"context_type": "PV"}'

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = [tc]
        mock_response.choices[0].message.content = None
        mock_litellm.completion.return_value = mock_response

        tools = [{"type": "function", "function": {"name": "read_context", "parameters": {}}}]

        result = execute_litellm_completion(
            provider="openai",
            message="hello",
            model_id="gpt-4o",
            api_key="test",
            base_url=None,
            tools=tools,
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["id"] == "call_1"
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "read_context"

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_no_tool_calls_returns_text(self, mock_litellm):
        """Normal text response returns string even when tools were provided."""
        from osprey.models.providers.litellm_adapter import execute_litellm_completion

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "text response"
        mock_litellm.completion.return_value = mock_response

        tools = [{"type": "function", "function": {"name": "test_fn", "parameters": {}}}]

        result = execute_litellm_completion(
            provider="openai",
            message="hello",
            model_id="gpt-4o",
            api_key="test",
            base_url=None,
            tools=tools,
        )

        assert result == "text response"

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_tools_and_output_format_raises(self, mock_litellm):
        """Providing both tools and output_format raises ValueError."""
        from pydantic import BaseModel

        from osprey.models.providers.litellm_adapter import execute_litellm_completion

        class TestModel(BaseModel):
            name: str

        tools = [{"type": "function", "function": {"name": "test_fn", "parameters": {}}}]

        with pytest.raises(ValueError, match="Cannot use both"):
            execute_litellm_completion(
                provider="openai",
                message="hello",
                model_id="gpt-4o",
                api_key="test",
                base_url=None,
                tools=tools,
                output_format=TestModel,
            )


class TestExecuteLiteLLMWithChatRequest:
    """Test execute_litellm_completion with chat_request kwarg."""

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_chat_request_sets_messages(self, mock_litellm):
        """When chat_request is in kwargs, messages come from chat_request."""
        from osprey.models.providers.litellm_adapter import execute_litellm_completion

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response"
        mock_litellm.completion.return_value = mock_response

        req = ChatCompletionRequest(
            messages=[
                ChatMessage("system", "You are helpful"),
                ChatMessage("user", "hello"),
            ]
        )

        execute_litellm_completion(
            provider="openai",
            message="",
            model_id="gpt-4o",
            api_key="test",
            base_url=None,
            chat_request=req,
        )

        call_kwargs = mock_litellm.completion.call_args[1]
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_message_fallback_when_no_chat_request(self, mock_litellm):
        """When no chat_request, messages are built from message param."""
        from osprey.models.providers.litellm_adapter import execute_litellm_completion

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response"
        mock_litellm.completion.return_value = mock_response

        execute_litellm_completion(
            provider="openai",
            message="hello world",
            model_id="gpt-4o",
            api_key="test",
            base_url=None,
        )

        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "hello world"}]

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_anthropic_cache_markers_applied(self, mock_litellm):
        """When provider=anthropic and chat_request is provided, cache_control appears."""
        from osprey.models.providers.litellm_adapter import execute_litellm_completion

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response"
        mock_litellm.completion.return_value = mock_response

        req = ChatCompletionRequest(
            messages=[
                ChatMessage("system", "System prompt"),
                ChatMessage("user", "hello"),
            ]
        )

        execute_litellm_completion(
            provider="anthropic",
            message="",
            model_id="claude-sonnet-4",
            api_key="test",
            base_url=None,
            chat_request=req,
        )

        call_kwargs = mock_litellm.completion.call_args[1]
        sys_msg = call_kwargs["messages"][0]
        assert isinstance(sys_msg["content"], list)
        assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}


class TestHandleStructuredOutputWithChatRequest:
    """Test _handle_structured_output with chat_request."""

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_native_path_does_not_clobber_chat_messages(self, mock_litellm):
        """When chat_request is set AND native structured output, messages are NOT overwritten."""
        from osprey.models.providers.litellm_adapter import _handle_structured_output

        mock_litellm.supports_response_schema.return_value = True
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"name": "test"}'
        mock_litellm.completion.return_value = mock_response

        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        req = ChatCompletionRequest(
            messages=[
                ChatMessage("system", "sys"),
                ChatMessage("user", "find things"),
            ]
        )
        completion_kwargs = {
            "model": "openai/gpt-4o",
            "messages": req.to_litellm_messages(),
            "max_tokens": 1024,
            "temperature": 0.0,
        }

        _handle_structured_output(
            provider="openai",
            model_id="gpt-4o",
            litellm_model="openai/gpt-4o",
            message="",
            completion_kwargs=completion_kwargs,
            output_format=TestModel,
            is_typed_dict_output=False,
            chat_request=req,
        )

        call_kwargs = mock_litellm.completion.call_args[1]
        # Messages should still have system + user (not rebuilt to single user message)
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_native_path_adds_response_format(self, mock_litellm):
        """Verify response_format is still added correctly with chat_request."""
        from osprey.models.providers.litellm_adapter import _handle_structured_output

        mock_litellm.supports_response_schema.return_value = True
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"name": "test"}'
        mock_litellm.completion.return_value = mock_response

        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        req = ChatCompletionRequest(messages=[ChatMessage("user", "find things")])
        completion_kwargs = {
            "model": "openai/gpt-4o",
            "messages": req.to_litellm_messages(),
            "max_tokens": 1024,
            "temperature": 0.0,
        }

        _handle_structured_output(
            provider="openai",
            model_id="gpt-4o",
            litellm_model="openai/gpt-4o",
            message="",
            completion_kwargs=completion_kwargs,
            output_format=TestModel,
            is_typed_dict_output=False,
            chat_request=req,
        )

        call_kwargs = mock_litellm.completion.call_args[1]
        assert "response_format" in call_kwargs

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_prompt_fallback_appends_to_last_user_message(self, mock_litellm):
        """When chat_request + prompt fallback, schema appended to last user msg."""
        from osprey.models.providers.litellm_adapter import _handle_structured_output

        mock_litellm.supports_response_schema.return_value = False
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"name": "test"}'
        mock_litellm.completion.return_value = mock_response

        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        req = ChatCompletionRequest(
            messages=[
                ChatMessage("system", "sys"),
                ChatMessage("user", "find things"),
            ]
        )
        completion_kwargs = {
            "model": "test/model",
            "messages": req.to_litellm_messages(),
            "max_tokens": 1024,
            "temperature": 0.0,
        }

        _handle_structured_output(
            provider="test",
            model_id="model",
            litellm_model="test/model",
            message="",
            completion_kwargs=completion_kwargs,
            output_format=TestModel,
            is_typed_dict_output=False,
            chat_request=req,
        )

        call_kwargs = mock_litellm.completion.call_args[1]
        # Should still have 2 messages (system + user with appended schema)
        assert len(call_kwargs["messages"]) == 2
        assert "json" in call_kwargs["messages"][1]["content"].lower()
        assert "find things" in call_kwargs["messages"][1]["content"]

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_no_chat_request_preserves_existing_behavior(self, mock_litellm):
        """When chat_request is None, both native and prompt-based paths work as before."""
        from osprey.models.providers.litellm_adapter import _handle_structured_output

        mock_litellm.supports_response_schema.return_value = True
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"name": "test"}'
        mock_litellm.completion.return_value = mock_response

        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        completion_kwargs = {
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "old message"}],
            "max_tokens": 1024,
            "temperature": 0.0,
        }

        _handle_structured_output(
            provider="openai",
            model_id="gpt-4o",
            litellm_model="openai/gpt-4o",
            message="original message",
            completion_kwargs=completion_kwargs,
            output_format=TestModel,
            is_typed_dict_output=False,
            chat_request=None,
        )

        call_kwargs = mock_litellm.completion.call_args[1]
        # Should rebuild to single user message from `message` param
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["content"] == "original message"
