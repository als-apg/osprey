"""Tests for the chat_request parameter in get_chat_completion()."""

from unittest.mock import MagicMock, patch

import pytest

from osprey.models.messages import ChatCompletionRequest, ChatMessage


class TestChatRequestValidation:
    """Test validation of message vs chat_request exclusivity."""

    def test_both_message_and_chat_request_raises(self):
        from osprey.models.completion import get_chat_completion

        req = ChatCompletionRequest(messages=[ChatMessage("user", "hello")])
        with pytest.raises(ValueError, match="Cannot pass both"):
            get_chat_completion(message="hello", chat_request=req, provider="openai")

    def test_neither_message_nor_chat_request_raises(self):
        from osprey.models.completion import get_chat_completion

        with pytest.raises(ValueError, match="Must pass either"):
            get_chat_completion(provider="openai")

    @patch("osprey.models.completion.get_provider_config", return_value={"api_key": "test"})
    def test_message_only_works(self, mock_config):
        """Existing message= path continues to work (mocked provider)."""
        from osprey.models.completion import get_chat_completion

        mock_provider_cls = MagicMock()
        mock_provider_cls.requires_api_key = False
        mock_provider_cls.requires_base_url = False
        mock_provider_cls.requires_model_id = False
        mock_instance = MagicMock()
        mock_instance.execute_completion.return_value = "response"
        mock_provider_cls.return_value = mock_instance

        mock_registry = MagicMock()
        mock_registry.get_provider.return_value = mock_provider_cls

        with (
            patch("osprey.registry.get_registry", return_value=mock_registry),
            patch("osprey.models.logging.log_api_call"),
        ):
            result = get_chat_completion(message="hello", provider="test", model_id="test-model")
        assert result == "response"

    @patch("osprey.models.completion.get_provider_config", return_value={"api_key": "test"})
    def test_chat_request_only_works(self, mock_config):
        """chat_request= path works (mocked provider)."""
        from osprey.models.completion import get_chat_completion

        mock_provider_cls = MagicMock()
        mock_provider_cls.requires_api_key = False
        mock_provider_cls.requires_base_url = False
        mock_provider_cls.requires_model_id = False
        mock_instance = MagicMock()
        mock_instance.execute_completion.return_value = "response"
        mock_provider_cls.return_value = mock_instance

        mock_registry = MagicMock()
        mock_registry.get_provider.return_value = mock_provider_cls

        req = ChatCompletionRequest(messages=[ChatMessage("user", "hello")])

        with (
            patch("osprey.registry.get_registry", return_value=mock_registry),
            patch("osprey.models.logging.log_api_call"),
        ):
            result = get_chat_completion(chat_request=req, provider="test", model_id="test-model")
        assert result == "response"


class TestChatRequestFlowThrough:
    """Test that chat_request flows through to provider."""

    @patch("osprey.models.completion.get_provider_config", return_value={"api_key": "test"})
    def test_chat_request_in_completion_kwargs(self, mock_config):
        """Verify chat_request is included in kwargs passed to execute_completion."""
        from osprey.models.completion import get_chat_completion

        mock_provider_cls = MagicMock()
        mock_provider_cls.requires_api_key = False
        mock_provider_cls.requires_base_url = False
        mock_provider_cls.requires_model_id = False
        mock_instance = MagicMock()
        mock_instance.execute_completion.return_value = "ok"
        mock_provider_cls.return_value = mock_instance

        mock_registry = MagicMock()
        mock_registry.get_provider.return_value = mock_provider_cls

        req = ChatCompletionRequest(messages=[ChatMessage("user", "test")])

        with (
            patch("osprey.registry.get_registry", return_value=mock_registry),
            patch("osprey.models.logging.log_api_call"),
        ):
            get_chat_completion(chat_request=req, provider="test", model_id="m")

        call_kwargs = mock_instance.execute_completion.call_args[1]
        assert call_kwargs["chat_request"] is req

    @patch("osprey.models.completion.get_provider_config", return_value={"api_key": "test"})
    def test_chat_request_none_when_using_message(self, mock_config):
        """When using message=, chat_request is None in kwargs."""
        from osprey.models.completion import get_chat_completion

        mock_provider_cls = MagicMock()
        mock_provider_cls.requires_api_key = False
        mock_provider_cls.requires_base_url = False
        mock_provider_cls.requires_model_id = False
        mock_instance = MagicMock()
        mock_instance.execute_completion.return_value = "ok"
        mock_provider_cls.return_value = mock_instance

        mock_registry = MagicMock()
        mock_registry.get_provider.return_value = mock_provider_cls

        with (
            patch("osprey.registry.get_registry", return_value=mock_registry),
            patch("osprey.models.logging.log_api_call"),
        ):
            get_chat_completion(message="hello", provider="test", model_id="m")

        call_kwargs = mock_instance.execute_completion.call_args[1]
        assert call_kwargs["chat_request"] is None

    @patch("osprey.models.completion.get_provider_config", return_value={"api_key": "test"})
    def test_log_api_call_uses_to_single_string(self, mock_config):
        """When chat_request is provided, log_api_call receives the flattened string."""
        from osprey.models.completion import get_chat_completion

        mock_provider_cls = MagicMock()
        mock_provider_cls.requires_api_key = False
        mock_provider_cls.requires_base_url = False
        mock_provider_cls.requires_model_id = False
        mock_instance = MagicMock()
        mock_instance.execute_completion.return_value = "ok"
        mock_provider_cls.return_value = mock_instance

        mock_registry = MagicMock()
        mock_registry.get_provider.return_value = mock_provider_cls

        req = ChatCompletionRequest(
            messages=[
                ChatMessage("system", "You are helpful"),
                ChatMessage("user", "What is 2+2?"),
            ]
        )

        with (
            patch("osprey.registry.get_registry", return_value=mock_registry),
            patch("osprey.models.logging.log_api_call") as mock_log,
        ):
            get_chat_completion(chat_request=req, provider="test", model_id="m")

        log_kwargs = mock_log.call_args[1]
        assert log_kwargs["message"] == "You are helpful\n\nWhat is 2+2?"
