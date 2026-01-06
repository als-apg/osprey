"""Tests for Ollama provider adapter."""

from unittest.mock import MagicMock, Mock, patch

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


class TestOllamaExecuteCompletion:
    """Test Ollama completion execution via direct API."""

    @patch("httpx.post")
    @patch.object(OllamaProviderAdapter, "_test_connection", return_value=True)
    def test_execute_text_completion(self, mock_test, mock_post):
        """Test basic text completion via direct Ollama API."""
        provider = OllamaProviderAdapter()

        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Test response"}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = provider.execute_completion(
            message="Hello",
            model_id="mistral:7b",
            api_key=None,
            base_url="http://localhost:11434",
        )

        assert result == "Test response"
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/chat"
        assert call_args[1]["json"]["model"] == "mistral:7b"

    @patch("httpx.post")
    @patch.object(
        OllamaProviderAdapter,
        "_test_connection",
        side_effect=[False, True],  # First fails, second succeeds
    )
    def test_execute_completion_with_fallback(self, mock_test, mock_post):
        """Test completion execution with fallback."""
        provider = OllamaProviderAdapter()

        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Response"}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = provider.execute_completion(
            message="Hello",
            model_id="mistral:7b",
            api_key=None,
            base_url="http://localhost:11434",
        )

        assert result == "Response"

    @patch.object(OllamaProviderAdapter, "_test_connection", return_value=False)
    def test_execute_completion_all_connections_fail(self, mock_test):
        """Test completion fails when all connections fail."""
        provider = OllamaProviderAdapter()

        with pytest.raises(ValueError, match="Failed to connect"):
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
