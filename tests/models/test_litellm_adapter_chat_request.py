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
    def test_tools_and_output_format_raises(self, _mock_litellm):
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
        # Verify the full native structured-output contract, not just key presence:
        # type, the schema name (output_format.__name__), and that the model's
        # field made it into the forwarded JSON schema.
        response_format = call_kwargs["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["name"] == "TestModel"
        assert "name" in response_format["json_schema"]["schema"]["properties"]

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
        """When chat_request is None, both native and prompt-based paths still work."""
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

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_a_second_ask_sends_the_schema_instruction_once(self, mock_litellm):
        """The schema instruction is added before the first ask, so a second ask
        sends the same messages rather than a second copy of the instruction."""
        from pydantic import BaseModel

        from osprey.models.providers.litellm_adapter import _handle_structured_output

        def reply(content):
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = content
            return response

        mock_litellm.completion.side_effect = [reply("not json"), reply('{"name": "test"}')]

        class TestModel(BaseModel):
            name: str

        req = ChatCompletionRequest(
            messages=[
                ChatMessage("system", "sys"),
                ChatMessage("user", "turn1"),
                ChatMessage("assistant", "a"),
                ChatMessage("user", "find things"),
            ]
        )
        completion_kwargs = {
            "model": "openai/deepseek-v4-flash",
            "messages": req.to_litellm_messages(),
            "max_tokens": 1024,
            "temperature": 0.0,
        }

        result = _handle_structured_output(
            provider="ds4",
            model_id="deepseek-v4-flash",
            litellm_model="openai/deepseek-v4-flash",
            message="",
            completion_kwargs=completion_kwargs,
            output_format=TestModel,
            is_typed_dict_output=False,
            chat_request=req,
        )

        assert result == TestModel(name="test")
        assert mock_litellm.completion.call_count == 2
        first, second = (c.kwargs["messages"][-1] for c in mock_litellm.completion.call_args_list)
        second_text = (
            second["content"][-1]["text"]
            if isinstance(second["content"], list)
            else second["content"]
        )
        assert second_text.count("must respond with valid JSON") == 1
        assert second == first


_PNG_B64 = "iVBORw0KGgo="
_IMAGE_PART = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PNG_B64}"}}


def _ollama_reply(content: str):
    """A fake ``/api/chat`` response carrying *content* as the assistant reply."""
    response = MagicMock()
    response.json.return_value = {"message": {"role": "assistant", "content": content}}
    response.raise_for_status = MagicMock()
    return response


class TestOllamaVisionCompletion:
    """The direct Ollama path sends image parts as ``images:`` and text as one string."""

    @staticmethod
    def _complete(chat_request, **kwargs):
        from osprey.models.providers.litellm_adapter import execute_litellm_completion

        return execute_litellm_completion(
            provider="ollama",
            message="",
            model_id="llava",
            api_key="ollama",
            base_url="http://ollama:11434",
            chat_request=chat_request,
            **kwargs,
        )

    @patch("httpx.post")
    def test_image_parts_become_base64_images_with_think_false(self, mock_post):
        """A [text, image] user message posts joined text, images:[b64], think:false."""
        mock_post.return_value = _ollama_reply("A beam-current plot.")
        req = ChatCompletionRequest(
            messages=[
                ChatMessage("system", "Describe images."),
                ChatMessage(
                    "user",
                    [
                        {"type": "text", "text": "What is"},
                        _IMAGE_PART,
                        {"type": "text", "text": "this?"},
                    ],
                ),
            ]
        )

        reply = self._complete(req, timeout=7.5)

        assert reply == "A beam-current plot."
        assert mock_post.call_args.args[0] == "http://ollama:11434/api/chat"
        body = mock_post.call_args.kwargs["json"]
        assert body["think"] is False
        assert body["stream"] is False
        system, user = body["messages"]
        assert system == {"role": "system", "content": "Describe images."}
        assert user["role"] == "user"
        assert user["content"] == "What is\nthis?"
        assert user["images"] == [_PNG_B64]
        assert mock_post.call_args.kwargs["timeout"] == 7.5

    @patch("httpx.post")
    def test_text_only_request_sends_no_think_and_no_images(self, mock_post):
        """Without an image part the body carries neither think nor images."""
        mock_post.return_value = _ollama_reply("hello")
        req = ChatCompletionRequest(messages=[ChatMessage("user", "hi")])

        assert self._complete(req) == "hello"

        body = mock_post.call_args.kwargs["json"]
        assert "think" not in body
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert mock_post.call_args.kwargs["timeout"] == 120.0

    @patch("httpx.post")
    def test_text_only_list_content_is_joined_without_images(self, mock_post):
        """List content holding only text parts becomes a string, no images key."""
        mock_post.return_value = _ollama_reply("ok")
        req = ChatCompletionRequest(
            messages=[
                ChatMessage("user", [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
            ]
        )

        self._complete(req)

        body = mock_post.call_args.kwargs["json"]
        assert "think" not in body
        assert body["messages"] == [{"role": "user", "content": "a\nb"}]

    @patch("httpx.post")
    def test_several_images_keep_their_order(self, mock_post):
        """Two image parts in one message become two entries of ``images`` in order."""
        mock_post.return_value = _ollama_reply("two")
        second = {"type": "image_url", "image_url": "data:image/jpeg;base64,/9j/AA=="}
        req = ChatCompletionRequest(messages=[ChatMessage("user", [_IMAGE_PART, second])])

        self._complete(req)

        (user,) = mock_post.call_args.kwargs["json"]["messages"]
        assert user["images"] == [_PNG_B64, "/9j/AA=="]
        assert user["content"] == ""

    @patch("httpx.post")
    def test_non_data_url_image_raises_before_posting(self, mock_post):
        """An image part that is not a base64 data URL cannot reach Ollama."""
        bad = {"type": "image_url", "image_url": {"url": "https://example.org/x.png"}}
        req = ChatCompletionRequest(messages=[ChatMessage("user", [bad])])

        with pytest.raises(ValueError):
            self._complete(req)
        mock_post.assert_not_called()

    @patch("httpx.post")
    def test_caller_message_objects_are_not_mutated(self, mock_post):
        """Conversion builds new message dicts; the caller's parts stay intact."""
        mock_post.return_value = _ollama_reply("ok")
        parts = [{"type": "text", "text": "x"}, dict(_IMAGE_PART)]
        req = ChatCompletionRequest(messages=[ChatMessage("user", parts)])

        self._complete(req)

        assert req.messages[0].content == parts
        assert parts[1]["type"] == "image_url"


class TestStructuredOutputImageParts:
    """Structured output with image-bearing chat requests."""

    def test_ollama_with_chat_request_raises(self):
        """The ollama structured path takes one string, so a chat_request is refused."""
        from pydantic import BaseModel

        from osprey.models.providers.litellm_adapter import _handle_structured_output

        class Out(BaseModel):
            name: str

        req = ChatCompletionRequest(
            messages=[ChatMessage("user", [{"type": "text", "text": "x"}, _IMAGE_PART])]
        )

        with patch("httpx.post") as mock_post, pytest.raises(ValueError, match="ollama"):
            _handle_structured_output(
                provider="ollama",
                model_id="llava",
                litellm_model="ollama/llava",
                message="",
                completion_kwargs={"messages": req.to_litellm_messages(), "max_tokens": 64},
                output_format=Out,
                is_typed_dict_output=False,
                chat_request=req,
            )
        mock_post.assert_not_called()

    @staticmethod
    def _fallback(mock_litellm, content):
        from pydantic import BaseModel

        from osprey.models.providers.litellm_adapter import _handle_structured_output

        class Out(BaseModel):
            name: str

        mock_litellm.supports_response_schema.return_value = False
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = '{"name": "plot"}'
        mock_litellm.completion.return_value = response

        req = ChatCompletionRequest(messages=[ChatMessage("user", content)])
        result = _handle_structured_output(
            provider="test",
            model_id="model",
            litellm_model="test/model",
            message="",
            completion_kwargs={
                "model": "test/model",
                "messages": req.to_litellm_messages(),
                "max_tokens": 64,
            },
            output_format=Out,
            is_typed_dict_output=False,
            chat_request=req,
        )
        assert result == Out(name="plot")
        return mock_litellm.completion.call_args.kwargs["messages"][-1]["content"]

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_prompt_fallback_appends_to_last_text_part_before_image(self, mock_litellm):
        """[text, image]: the schema lands on the text part; the image part is untouched."""
        sent = self._fallback(mock_litellm, [{"type": "text", "text": "Describe."}, _IMAGE_PART])

        assert len(sent) == 2
        assert sent[0]["type"] == "text"
        assert sent[0]["text"].startswith("Describe.")
        assert "valid JSON" in sent[0]["text"]
        assert sent[1] == _IMAGE_PART

    @patch("osprey.models.providers.litellm_adapter.litellm")
    def test_prompt_fallback_adds_text_part_when_only_images(self, mock_litellm):
        """[image]: a text part carrying the schema instruction is appended."""
        sent = self._fallback(mock_litellm, [_IMAGE_PART])

        assert len(sent) == 2
        assert sent[0] == _IMAGE_PART
        assert sent[1]["type"] == "text"
        assert "valid JSON" in sent[1]["text"]
