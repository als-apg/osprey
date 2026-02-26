"""Tests for ARGO provider adapter."""

from unittest.mock import MagicMock, Mock, patch

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


class TestArgoHealthCheck:
    """Test ARGO health check functionality via LiteLLM."""

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

    def test_health_check_uses_default_model(self):
        """Test health check uses default model when none provided."""
        provider = ArgoProviderAdapter()
        # When model_id=None, provider uses health_check_model_id
        assert provider.health_check_model_id is not None

    @patch("osprey.models.providers.litellm_adapter.litellm.completion")
    def test_health_check_successful_with_model(self, mock_completion):
        """Test successful health check with model via LiteLLM."""
        provider = ArgoProviderAdapter()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(tool_calls=None, content="Hello"))]
        mock_completion.return_value = mock_response

        success, message = provider.check_health(
            api_key="test-key", base_url="https://test.url", model_id="test-model"
        )

        assert success is True
        assert "accessible" in message.lower()

    @patch("osprey.models.providers.litellm_adapter.litellm.completion")
    def test_health_check_authentication_error(self, mock_completion):
        """Test health check handles authentication error."""
        import litellm

        provider = ArgoProviderAdapter()
        mock_completion.side_effect = litellm.AuthenticationError(
            message="Invalid API key",
            llm_provider="openai",
            model="test-model",
        )

        success, message = provider.check_health(
            api_key="bad-key", base_url="https://test.url", model_id="test-model"
        )

        assert success is False
        assert "authentication" in message.lower()

    @patch("osprey.models.providers.litellm_adapter.litellm.completion")
    def test_health_check_rate_limit(self, mock_completion):
        """Test health check handles rate limit as success."""
        import litellm

        provider = ArgoProviderAdapter()
        mock_completion.side_effect = litellm.RateLimitError(
            message="Rate limited",
            llm_provider="openai",
            model="test-model",
        )

        success, message = provider.check_health(
            api_key="test-key", base_url="https://test.url", model_id="test-model"
        )

        assert success is True
        assert "rate limit" in message.lower()


class TestArgoExecuteCompletion:
    """Test ARGO completion execution via LiteLLM."""

    @patch("osprey.models.providers.litellm_adapter.litellm.completion")
    def test_execute_text_completion(self, mock_completion):
        """Test basic text completion."""
        provider = ArgoProviderAdapter()

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(tool_calls=None, content="Test response"))
        ]
        mock_completion.return_value = mock_response

        result = provider.execute_completion(
            message="Hello",
            model_id="test-model",
            api_key="test-key",
            base_url="https://test.url",
        )

        assert result == "Test response"
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        # ARGO uses openai/ prefix for LiteLLM
        assert call_kwargs["model"] == "openai/test-model"
        assert call_kwargs["api_base"] == "https://test.url"
