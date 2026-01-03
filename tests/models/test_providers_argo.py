"""Tests for ARGO provider adapter."""

from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from osprey.models.providers.argo import ArgoProviderAdapter


class SampleOutput(BaseModel):
    """Sample output model for testing."""

    result: str
    value: int


class TestArgoMetadata:
    """Test ARGO provider metadata."""

    def test_provider_name(self):
        """Test provider name is set correctly."""
        assert ArgoProviderAdapter.name == "argo"

    def test_provider_description(self):
        """Test provider has description."""
        assert "ANL" in ArgoProviderAdapter.description or "Argo" in ArgoProviderAdapter.description

    def test_requires_api_key(self):
        """Test provider requires API key."""
        assert ArgoProviderAdapter.requires_api_key is True

    def test_requires_base_url(self):
        """Test provider requires base URL."""
        assert ArgoProviderAdapter.requires_base_url is True

    def test_requires_model_id(self):
        """Test provider requires model ID."""
        assert ArgoProviderAdapter.requires_model_id is True

    def test_supports_proxy(self):
        """Test provider supports HTTP proxy."""
        assert ArgoProviderAdapter.supports_proxy is True

    def test_has_default_base_url(self):
        """Test provider has default base URL."""
        assert ArgoProviderAdapter.default_base_url is not None
        assert "argo" in ArgoProviderAdapter.default_base_url.lower()

    def test_has_default_model_id(self):
        """Test provider has default model."""
        assert ArgoProviderAdapter.default_model_id is not None

    def test_has_health_check_model(self):
        """Test provider has health check model."""
        assert ArgoProviderAdapter.health_check_model_id is not None

    def test_has_available_models(self):
        """Test provider lists available models."""
        assert len(ArgoProviderAdapter.available_models) > 0

    def test_available_models_includes_defaults(self):
        """Test available models includes default models."""
        models = ArgoProviderAdapter.available_models
        assert ArgoProviderAdapter.default_model_id in models
        assert ArgoProviderAdapter.health_check_model_id in models


class TestArgoGetAvailableModels:
    """Test ARGO model list retrieval."""

    def test_get_available_models_without_credentials(self):
        """Test model list fallback without credentials."""
        with patch.dict("os.environ", {}, clear=True):
            models = ArgoProviderAdapter.get_available_models()
            assert isinstance(models, list)
            assert len(models) > 0

    def test_get_available_models_uses_cache(self):
        """Test model list caching."""
        ArgoProviderAdapter._models_cache = ["cached_model"]
        models = ArgoProviderAdapter.get_available_models(force_refresh=False)
        assert models == ["cached_model"]
        # Clean up
        ArgoProviderAdapter._models_cache = None

    def test_get_available_models_force_refresh(self):
        """Test force refresh bypasses cache."""
        ArgoProviderAdapter._models_cache = ["old_model"]
        with patch("httpx.Client") as mock_client:
            mock_resp = Mock()
            mock_resp.json.return_value = {"data": []}
            mock_resp.raise_for_status = Mock()
            mock_client.return_value.__enter__.return_value.get.return_value = mock_resp

            models = ArgoProviderAdapter.get_available_models(
                api_key="test", base_url="https://test", force_refresh=True
            )
            # Should have attempted refresh (even if failed)
            assert isinstance(models, list)

        # Clean up
        ArgoProviderAdapter._models_cache = None

    def test_get_available_models_with_valid_response(self):
        """Test parsing valid API response."""
        with patch("httpx.Client") as mock_client:
            mock_resp = Mock()
            mock_resp.json.return_value = {
                "data": [
                    {"id": "model1"},
                    {"model": "model2"},
                    {"name": "model3"},
                ]
            }
            mock_resp.raise_for_status = Mock()
            mock_client.return_value.__enter__.return_value.get.return_value = mock_resp

            models = ArgoProviderAdapter.get_available_models(
                api_key="test", base_url="https://test", force_refresh=True
            )
            assert "model1" in models
            assert "model2" in models
            assert "model3" in models

        # Clean up
        ArgoProviderAdapter._models_cache = None


class TestArgoCreateModel:
    """Test ARGO model instance creation."""

    def test_create_model_without_http_client(self):
        """Test model creation without custom HTTP client."""
        provider = ArgoProviderAdapter()

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            provider.create_model(
                model_id="test-model",
                api_key="test-key",
                base_url="https://test.url",
                timeout=30.0,
                http_client=None,
            )

            # Verify OpenAI client was created with correct args
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["api_key"] == "test-key"
            assert call_kwargs["base_url"] == "https://test.url"
            assert call_kwargs["timeout"] == 30.0

    def test_create_model_with_http_client(self):
        """Test model creation with custom HTTP client."""
        provider = ArgoProviderAdapter()
        mock_http_client = Mock()

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            provider.create_model(
                model_id="test-model",
                api_key="test-key",
                base_url="https://test.url",
                timeout=None,
                http_client=mock_http_client,
            )

            # Verify HTTP client was used
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["http_client"] is mock_http_client

    def test_create_model_default_timeout(self):
        """Test model creation uses default timeout when none provided."""
        provider = ArgoProviderAdapter()

        with patch("openai.AsyncOpenAI") as mock_openai:
            provider.create_model(
                model_id="test-model",
                api_key="test-key",
                base_url="https://test.url",
                timeout=None,
                http_client=None,
            )

            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["timeout"] == 60.0


class TestArgoHealthCheck:
    """Test ARGO health check functionality."""

    def test_health_check_no_api_key(self):
        """Test health check fails without API key."""
        provider = ArgoProviderAdapter()
        success, message = provider.check_health(api_key=None, base_url="https://test.url")
        assert success is False
        assert "key" in message.lower()

    def test_health_check_placeholder_api_key(self):
        """Test health check detects placeholder API key."""
        provider = ArgoProviderAdapter()
        success, message = provider.check_health(
            api_key="${ARGO_API_KEY}", base_url="https://test.url"
        )
        assert success is False
        assert "placeholder" in message.lower()

    def test_health_check_no_base_url(self):
        """Test health check fails without base URL."""
        provider = ArgoProviderAdapter()
        success, message = provider.check_health(api_key="test-key", base_url=None)
        assert success is False
        assert "url" in message.lower()

    def test_health_check_successful_with_model(self):
        """Test successful health check with model."""
        provider = ArgoProviderAdapter()

        with patch("openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            success, message = provider.check_health(
                api_key="test-key", base_url="https://test.url", model_id="test-model"
            )

            assert success is True
            assert "accessible" in message.lower()

    def test_health_check_authentication_error(self):
        """Test health check handles authentication error."""
        provider = ArgoProviderAdapter()

        with patch("openai.OpenAI") as mock_openai:
            import openai as openai_module

            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = openai_module.AuthenticationError(
                message="Invalid API key",
                response=Mock(status_code=401),
                body=None,
            )
            mock_openai.return_value = mock_client

            success, message = provider.check_health(
                api_key="bad-key", base_url="https://test.url", model_id="test-model"
            )

            assert success is False
            assert "authentication" in message.lower() or "invalid" in message.lower()

    def test_health_check_rate_limit(self):
        """Test health check handles rate limit as success."""
        provider = ArgoProviderAdapter()

        with patch("openai.OpenAI") as mock_openai:
            import openai as openai_module

            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = openai_module.RateLimitError(
                message="Rate limited",
                response=Mock(status_code=429),
                body=None,
            )
            mock_openai.return_value = mock_client

            success, message = provider.check_health(
                api_key="test-key", base_url="https://test.url", model_id="test-model"
            )

            assert success is True
            assert "rate" in message.lower()


class TestArgoExecuteCompletion:
    """Test ARGO completion execution."""

    def test_execute_text_completion(self):
        """Test basic text completion."""
        provider = ArgoProviderAdapter()

        with patch("openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_choice = Mock()
            mock_choice.message.content = "Test response"
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            result = provider.execute_completion(
                message="Hello",
                model_id="test-model",
                api_key="test-key",
                base_url="https://test.url",
            )

            assert result == "Test response"

    def test_execute_completion_with_system_prompt(self):
        """Test completion with system prompt."""
        provider = ArgoProviderAdapter()

        with patch("openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_choice = Mock()
            mock_choice.message.content = "Response"
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider.execute_completion(
                message="Hello",
                model_id="test-model",
                api_key="test-key",
                base_url="https://test.url",
                system_prompt="You are helpful",
            )

            # Verify system prompt was included
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"

    def test_execute_completion_empty_choices(self):
        """Test completion handles empty choices."""
        provider = ArgoProviderAdapter()

        with patch("openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = []
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            with pytest.raises(ValueError, match="empty choices"):
                provider.execute_completion(
                    message="Hello",
                    model_id="test-model",
                    api_key="test-key",
                    base_url="https://test.url",
                )
