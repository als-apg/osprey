"""Tests for Ollama provider adapter."""

from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from osprey.models.providers.ollama import OllamaProviderAdapter


class SampleOutput(BaseModel):
    """Sample output model for testing."""

    result: str
    value: int


class TestOllamaMetadata:
    """Test Ollama provider metadata."""

    def test_provider_name(self):
        """Test provider name is set correctly."""
        assert OllamaProviderAdapter.name == "ollama"

    def test_provider_description(self):
        """Test provider has description."""
        assert "ollama" in OllamaProviderAdapter.description.lower()

    def test_requires_api_key(self):
        """Test provider does not require API key."""
        assert OllamaProviderAdapter.requires_api_key is False

    def test_requires_base_url(self):
        """Test provider requires base URL."""
        assert OllamaProviderAdapter.requires_base_url is True

    def test_requires_model_id(self):
        """Test provider requires model ID."""
        assert OllamaProviderAdapter.requires_model_id is True

    def test_supports_proxy(self):
        """Test provider does not support HTTP proxy."""
        assert OllamaProviderAdapter.supports_proxy is False

    def test_has_default_base_url(self):
        """Test provider has localhost default."""
        assert OllamaProviderAdapter.default_base_url is not None
        assert "localhost" in OllamaProviderAdapter.default_base_url

    def test_has_default_model_id(self):
        """Test provider has default model."""
        assert OllamaProviderAdapter.default_model_id is not None

    def test_has_health_check_model(self):
        """Test provider has health check model."""
        assert OllamaProviderAdapter.health_check_model_id is not None

    def test_has_available_models(self):
        """Test provider lists available models."""
        assert len(OllamaProviderAdapter.available_models) > 0

    def test_api_key_note(self):
        """Test provider notes no API key needed."""
        assert OllamaProviderAdapter.api_key_note is not None
        assert "local" in OllamaProviderAdapter.api_key_note.lower()


class TestOllamaFallbackUrls:
    """Test Ollama URL fallback logic."""

    def test_get_fallback_urls_from_container(self):
        """Test fallback URLs when running in container."""
        urls = OllamaProviderAdapter._get_fallback_urls("http://host.containers.internal:11434")
        assert "localhost" in str(urls)
        assert "http://localhost:11434" in urls

    def test_get_fallback_urls_from_localhost(self):
        """Test fallback URLs when running on localhost."""
        urls = OllamaProviderAdapter._get_fallback_urls("http://localhost:11434")
        assert "host.containers.internal" in str(urls)

    def test_get_fallback_urls_generic(self):
        """Test fallback URLs for generic base URL."""
        urls = OllamaProviderAdapter._get_fallback_urls("http://custom.server:11434")
        assert len(urls) > 0
        assert "localhost" in str(urls) or "host.containers.internal" in str(urls)


class TestOllamaTestConnection:
    """Test Ollama connection testing."""

    def test_test_connection_success(self):
        """Test successful connection test."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            result = OllamaProviderAdapter._test_connection("http://localhost:11434")
            assert result is True

    def test_test_connection_failure(self):
        """Test failed connection test."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_get.return_value = mock_response

            result = OllamaProviderAdapter._test_connection("http://localhost:11434")
            assert result is False

    def test_test_connection_exception(self):
        """Test connection test handles exceptions."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection failed")

            result = OllamaProviderAdapter._test_connection("http://localhost:11434")
            assert result is False


class TestOllamaCreateModel:
    """Test Ollama model instance creation."""

    def test_create_model_success(self):
        """Test model creation with successful connection."""
        provider = OllamaProviderAdapter()

        with (
            patch.object(OllamaProviderAdapter, "_test_connection", return_value=True),
            patch("openai.AsyncOpenAI") as mock_openai,
        ):
            mock_client = Mock()
            mock_openai.return_value = mock_client

            model = provider.create_model(
                model_id="mistral:7b",
                api_key=None,
                base_url="http://localhost:11434",
                timeout=30.0,
                http_client=None,
            )

            # Verify model was created
            assert model is not None

    def test_create_model_with_fallback(self):
        """Test model creation falls back on connection failure."""
        provider = OllamaProviderAdapter()

        with (
            patch.object(
                OllamaProviderAdapter,
                "_test_connection",
                side_effect=[False, True],  # First fails, second succeeds
            ),
            patch("openai.AsyncOpenAI") as mock_openai,
        ):
            mock_client = Mock()
            mock_openai.return_value = mock_client

            model = provider.create_model(
                model_id="mistral:7b",
                api_key=None,
                base_url="http://localhost:11434",
                timeout=30.0,
                http_client=None,
            )

            assert model is not None

    def test_create_model_all_connections_fail(self):
        """Test model creation fails when all connections fail."""
        provider = OllamaProviderAdapter()

        with patch.object(OllamaProviderAdapter, "_test_connection", return_value=False):
            with pytest.raises(ValueError, match="Failed to connect"):
                provider.create_model(
                    model_id="mistral:7b",
                    api_key=None,
                    base_url="http://localhost:11434",
                    timeout=30.0,
                    http_client=None,
                )

    def test_create_model_adds_v1_path(self):
        """Test model creation adds /v1 to base URL."""
        provider = OllamaProviderAdapter()

        with (
            patch.object(OllamaProviderAdapter, "_test_connection", return_value=True),
            patch("openai.AsyncOpenAI") as mock_openai,
        ):
            mock_client = Mock()
            mock_openai.return_value = mock_client

            provider.create_model(
                model_id="mistral:7b",
                api_key=None,
                base_url="http://localhost:11434",
                timeout=30.0,
                http_client=None,
            )

            # Verify /v1 was added to base URL
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["base_url"].endswith("/v1")

    def test_create_model_with_http_client(self):
        """Test model creation with custom HTTP client."""
        provider = OllamaProviderAdapter()
        mock_http_client = Mock()

        with (
            patch.object(OllamaProviderAdapter, "_test_connection", return_value=True),
            patch("openai.AsyncOpenAI") as mock_openai,
        ):
            mock_client = Mock()
            mock_openai.return_value = mock_client

            provider.create_model(
                model_id="mistral:7b",
                api_key=None,
                base_url="http://localhost:11434",
                timeout=None,
                http_client=mock_http_client,
            )

            # Verify HTTP client was used
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["http_client"] is mock_http_client


class TestOllamaExecuteCompletion:
    """Test Ollama completion execution."""

    def test_execute_text_completion(self):
        """Test basic text completion."""
        provider = OllamaProviderAdapter()

        with patch("ollama.Client") as mock_client_class:
            mock_client = Mock()
            mock_client.list.return_value = []  # Connection test
            mock_client.chat.return_value = {"message": {"content": "Test response"}}
            mock_client_class.return_value = mock_client

            result = provider.execute_completion(
                message="Hello",
                model_id="mistral:7b",
                api_key=None,
                base_url="http://localhost:11434",
            )

            assert result == "Test response"

    def test_execute_completion_with_fallback(self):
        """Test completion execution with fallback."""
        provider = OllamaProviderAdapter()

        with patch("ollama.Client") as mock_client_class:
            # First client fails, second succeeds
            mock_client_fail = Mock()
            mock_client_fail.list.side_effect = Exception("Connection failed")

            mock_client_success = Mock()
            mock_client_success.list.return_value = []
            mock_client_success.chat.return_value = {"message": {"content": "Response"}}

            mock_client_class.side_effect = [mock_client_fail, mock_client_success]

            result = provider.execute_completion(
                message="Hello",
                model_id="mistral:7b",
                api_key=None,
                base_url="http://localhost:11434",
            )

            assert result == "Response"

    def test_execute_completion_with_max_tokens(self):
        """Test completion with max tokens parameter."""
        provider = OllamaProviderAdapter()

        with patch("ollama.Client") as mock_client_class:
            mock_client = Mock()
            mock_client.list.return_value = []
            mock_client.chat.return_value = {"message": {"content": "Response"}}
            mock_client_class.return_value = mock_client

            provider.execute_completion(
                message="Hello",
                model_id="mistral:7b",
                api_key=None,
                base_url="http://localhost:11434",
                max_tokens=100,
            )

            # Verify max_tokens was passed as num_predict
            call_kwargs = mock_client.chat.call_args[1]
            assert "options" in call_kwargs
            assert call_kwargs["options"]["num_predict"] == 100

    def test_execute_completion_with_output_format(self):
        """Test completion with structured output."""
        provider = OllamaProviderAdapter()

        with patch("ollama.Client") as mock_client_class:
            mock_client = Mock()
            mock_client.list.return_value = []
            mock_client.chat.return_value = {
                "message": {"content": '{"result": "test", "value": 42}'}
            }
            mock_client_class.return_value = mock_client

            result = provider.execute_completion(
                message="Hello",
                model_id="mistral:7b",
                api_key=None,
                base_url="http://localhost:11434",
                output_format=SampleOutput,
            )

            # Verify result is parsed as Pydantic model
            assert isinstance(result, SampleOutput)
            assert result.result == "test"
            assert result.value == 42

    def test_execute_completion_chat_error(self):
        """Test completion handles chat execution errors."""
        provider = OllamaProviderAdapter()

        with patch("ollama.Client") as mock_client_class:
            mock_client = Mock()
            mock_client.list.return_value = []  # Connection succeeds
            mock_client.chat.side_effect = Exception("Chat failed")
            mock_client_class.return_value = mock_client

            with pytest.raises(ValueError, match="chat request failed"):
                provider.execute_completion(
                    message="Hello",
                    model_id="mistral:7b",
                    api_key=None,
                    base_url="http://localhost:11434",
                )


class TestOllamaHealthCheck:
    """Test Ollama health check functionality."""

    def test_health_check_no_base_url(self):
        """Test health check fails without base URL."""
        provider = OllamaProviderAdapter()
        success, message = provider.check_health(api_key=None, base_url=None)
        assert success is False
        assert "url" in message.lower()

    def test_health_check_success(self):
        """Test successful health check."""
        provider = OllamaProviderAdapter()

        with patch.object(OllamaProviderAdapter, "_test_connection", return_value=True):
            success, message = provider.check_health(
                api_key=None, base_url="http://localhost:11434"
            )
            assert success is True
            assert "accessible" in message.lower()

    def test_health_check_failure(self):
        """Test failed health check."""
        provider = OllamaProviderAdapter()

        with patch.object(OllamaProviderAdapter, "_test_connection", return_value=False):
            success, message = provider.check_health(
                api_key=None, base_url="http://localhost:11434"
            )
            assert success is False
            assert "not accessible" in message.lower()

    def test_health_check_exception(self):
        """Test health check handles exceptions."""
        provider = OllamaProviderAdapter()

        with patch.object(
            OllamaProviderAdapter, "_test_connection", side_effect=Exception("Test error")
        ):
            success, message = provider.check_health(
                api_key=None, base_url="http://localhost:11434"
            )
            assert success is False
            assert "failed" in message.lower()
