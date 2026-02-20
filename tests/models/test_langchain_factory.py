"""Tests for LangChain model factory."""

from unittest.mock import MagicMock, patch

import pytest


class TestLangChainFactory:
    """Test LangChain model factory functions."""

    def test_list_supported_providers(self):
        """Test listing supported providers."""
        from osprey.models import list_supported_providers

        providers = list_supported_providers()

        # Check all expected providers are present
        expected = [
            "anthropic",
            "openai",
            "google",
            "ollama",
            "cborg",
            "amsc",
            "vllm",
            "stanford",
            "argo",
        ]
        for provider in expected:
            assert provider in providers
            assert isinstance(providers[provider], str)

    def test_supported_providers_constant(self):
        """Test SUPPORTED_PROVIDERS constant."""
        from osprey.models import SUPPORTED_PROVIDERS

        expected = [
            "anthropic",
            "openai",
            "google",
            "ollama",
            "cborg",
            "amsc",
            "vllm",
            "stanford",
            "argo",
        ]
        for provider in expected:
            assert provider in SUPPORTED_PROVIDERS

    def test_get_langchain_model_requires_provider(self):
        """Test that get_langchain_model raises error without provider."""
        from osprey.models import get_langchain_model

        with pytest.raises(ValueError, match="Provider must be specified"):
            get_langchain_model()

    def test_get_langchain_model_invalid_provider(self):
        """Test that get_langchain_model raises error for invalid provider."""
        from osprey.models import get_langchain_model

        with pytest.raises(ValueError, match="not supported"):
            get_langchain_model(provider="invalid_provider", model_id="some-model")

    def test_get_langchain_model_requires_model_id(self):
        """Test that get_langchain_model raises error without model_id."""
        from osprey.models import get_langchain_model

        with pytest.raises(ValueError, match="Model ID must be specified"):
            get_langchain_model(
                provider="anthropic",
                provider_config={"api_key": "test-key"},  # No default_model_id
            )

    @patch("osprey.models.langchain._import_chat_class")
    def test_get_langchain_model_anthropic(self, mock_import):
        """Test creating Anthropic model."""
        from osprey.models import get_langchain_model

        mock_chat_class = MagicMock()
        mock_chat_class.__name__ = "MockChatAnthropic"
        mock_import.return_value = mock_chat_class

        get_langchain_model(
            provider="anthropic",
            model_id="claude-sonnet-4-5-20250929",
            provider_config={"api_key": "test-key"},
        )

        mock_chat_class.assert_called_once()
        call_kwargs = mock_chat_class.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-5-20250929"
        assert call_kwargs["anthropic_api_key"] == "test-key"
        assert call_kwargs["max_tokens"] == 4096

    @patch("osprey.models.langchain._import_chat_class")
    def test_get_langchain_model_openai_compatible(self, mock_import):
        """Test creating OpenAI-compatible model (CBORG)."""
        from osprey.models import get_langchain_model

        mock_chat_class = MagicMock()
        mock_chat_class.__name__ = "MockChatOpenAI"
        mock_import.return_value = mock_chat_class

        get_langchain_model(
            provider="cborg",
            model_id="anthropic/claude-sonnet",
            provider_config={"api_key": "test-key", "base_url": "https://api.cborg.lbl.gov"},
        )

        mock_chat_class.assert_called_once()
        call_kwargs = mock_chat_class.call_args[1]
        assert call_kwargs["model"] == "anthropic/claude-sonnet"
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["base_url"] == "https://api.cborg.lbl.gov"

    @patch("osprey.models.langchain._import_chat_class")
    def test_get_langchain_model_openai_compatible_amsc(self, mock_import):
        """Test creating OpenAI-compatible model (AMSC)."""
        from osprey.models import get_langchain_model

        mock_chat_class = MagicMock()
        mock_chat_class.__name__ = "MockChatOpenAI"
        mock_import.return_value = mock_chat_class

        get_langchain_model(
            provider="amsc",
            model_id="anthropic/claude-haiku",
            provider_config={
                "api_key": "test-key",
                "base_url": "https://api.i2-core.american-science-cloud.org/v1",
            },
        )

        mock_chat_class.assert_called_once()
        call_kwargs = mock_chat_class.call_args[1]
        assert call_kwargs["model"] == "anthropic/claude-haiku"
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["base_url"] == "https://api.i2-core.american-science-cloud.org/v1"

    @patch("osprey.models.langchain._import_chat_class")
    def test_get_langchain_model_vllm_default_key(self, mock_import):
        """Test vLLM uses default 'EMPTY' API key."""
        from osprey.models import get_langchain_model

        mock_chat_class = MagicMock()
        mock_chat_class.__name__ = "MockChatOpenAI"
        mock_import.return_value = mock_chat_class

        get_langchain_model(
            provider="vllm",
            model_id="meta-llama/Llama-3-8b",
            provider_config={},  # No API key
        )

        mock_chat_class.assert_called_once()
        call_kwargs = mock_chat_class.call_args[1]
        assert call_kwargs["api_key"] == "EMPTY"

    @patch("osprey.models.langchain._import_chat_class")
    @patch.dict("os.environ", {"USER": "testuser"})
    def test_get_langchain_model_argo_uses_user(self, mock_import):
        """Test Argo uses $USER as API key."""
        from osprey.models import get_langchain_model

        mock_chat_class = MagicMock()
        mock_chat_class.__name__ = "MockChatOpenAI"
        mock_import.return_value = mock_chat_class

        get_langchain_model(
            provider="argo",
            model_id="claudesonnet45",
            provider_config={},  # No API key
        )

        mock_chat_class.assert_called_once()
        call_kwargs = mock_chat_class.call_args[1]
        assert call_kwargs["api_key"] == "testuser"

    def test_get_langchain_model_from_model_config(self):
        """Test creating model from model_config dict."""
        from osprey.models import get_langchain_model

        with patch("osprey.models.langchain._import_chat_class") as mock_import:
            mock_chat_class = MagicMock()
            mock_chat_class.__name__ = "MockChatOpenAI"
            mock_import.return_value = mock_chat_class

            get_langchain_model(
                model_config={
                    "provider": "openai",
                    "model_id": "gpt-4o",
                    "max_tokens": 8192,
                },
                provider_config={"api_key": "test-key"},
            )

            mock_chat_class.assert_called_once()
            call_kwargs = mock_chat_class.call_args[1]
            assert call_kwargs["model"] == "gpt-4o"
            assert call_kwargs["max_tokens"] == 8192

    def test_import_error_helpful_message(self):
        """Test that import errors provide helpful installation instructions."""
        from osprey.models.langchain import _import_chat_class

        with pytest.raises(ImportError, match="pip install"):
            _import_chat_class("nonexistent_package", "SomeClass")
