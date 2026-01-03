"""Tests for Anthropic model provider adapter."""

from unittest.mock import MagicMock, patch

import httpx
from pydantic_ai.models.anthropic import AnthropicModel

from osprey.models.providers.anthropic import AnthropicProviderAdapter

# =============================================================================
# Test Provider Metadata
# =============================================================================


class TestAnthropicProviderMetadata:
    """Test Anthropic provider metadata attributes."""

    def test_provider_name(self):
        """Test provider name is correctly set."""
        provider = AnthropicProviderAdapter()
        assert provider.name == "anthropic"

    def test_provider_description(self):
        """Test provider description is set."""
        provider = AnthropicProviderAdapter()
        assert provider.description == "Anthropic (Claude models)"
        assert "Claude" in provider.description

    def test_requires_api_key(self):
        """Test provider requires API key."""
        provider = AnthropicProviderAdapter()
        assert provider.requires_api_key is True

    def test_does_not_require_base_url(self):
        """Test provider does not require base URL."""
        provider = AnthropicProviderAdapter()
        assert provider.requires_base_url is False

    def test_requires_model_id(self):
        """Test provider requires model ID."""
        provider = AnthropicProviderAdapter()
        assert provider.requires_model_id is True

    def test_supports_proxy(self):
        """Test provider supports HTTP proxy."""
        provider = AnthropicProviderAdapter()
        assert provider.supports_proxy is True

    def test_default_base_url_is_none(self):
        """Test default base URL is None (Anthropic handles default)."""
        provider = AnthropicProviderAdapter()
        assert provider.default_base_url is None

    def test_has_default_model_id(self):
        """Test provider has default model ID."""
        provider = AnthropicProviderAdapter()
        assert provider.default_model_id is not None
        assert "claude" in provider.default_model_id.lower()

    def test_has_health_check_model(self):
        """Test provider specifies health check model."""
        provider = AnthropicProviderAdapter()
        assert provider.health_check_model_id is not None
        # Should use fast/cheap model for health checks
        assert "haiku" in provider.health_check_model_id.lower()

    def test_has_available_models_list(self):
        """Test provider lists available models."""
        provider = AnthropicProviderAdapter()
        assert isinstance(provider.available_models, list)
        assert len(provider.available_models) > 0
        # All should be Claude models
        assert all("claude" in model.lower() for model in provider.available_models)

    def test_has_api_key_url(self):
        """Test provider has API key acquisition URL."""
        provider = AnthropicProviderAdapter()
        assert provider.api_key_url is not None
        assert "anthropic.com" in provider.api_key_url

    def test_has_api_key_instructions(self):
        """Test provider has API key instructions."""
        provider = AnthropicProviderAdapter()
        assert isinstance(provider.api_key_instructions, list)
        assert len(provider.api_key_instructions) > 0
        # Instructions should mention signing up and creating a key
        instructions_text = " ".join(provider.api_key_instructions).lower()
        assert "sign" in instructions_text or "log in" in instructions_text
        assert "key" in instructions_text


# =============================================================================
# Test Model Creation
# =============================================================================


class TestAnthropicModelCreation:
    """Test Anthropic model instance creation."""

    @patch("osprey.models.providers.anthropic.PydanticAnthropicProvider")
    def test_create_model_basic(self, mock_provider_class):
        """Test basic model creation without HTTP client."""
        provider = AnthropicProviderAdapter()

        model = provider.create_model(
            model_id="claude-haiku-4-5-20251001",
            api_key="test-key",
            base_url=None,
            timeout=30.0,
            http_client=None,
        )

        # Should create model instance
        assert model is not None
        assert isinstance(model, AnthropicModel)
        # Provider should be called with API key
        mock_provider_class.assert_called_once()
        call_kwargs = mock_provider_class.call_args[1]
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["http_client"] is None

    @patch("osprey.models.providers.anthropic.PydanticAnthropicProvider")
    def test_create_model_with_http_client(self, mock_provider_class):
        """Test model creation with HTTP client for proxy support."""
        provider = AnthropicProviderAdapter()
        http_client = MagicMock(spec=httpx.AsyncClient)

        model = provider.create_model(
            model_id="claude-sonnet-4-5-20250929",
            api_key="test-key",
            base_url=None,
            timeout=60.0,
            http_client=http_client,
        )

        assert model is not None
        assert isinstance(model, AnthropicModel)
        # HTTP client should be passed to provider
        call_kwargs = mock_provider_class.call_args[1]
        assert call_kwargs["http_client"] == http_client

    @patch("osprey.models.providers.anthropic.PydanticAnthropicProvider")
    def test_create_model_different_model_ids(self, mock_provider_class):
        """Test model creation with various model IDs."""
        provider = AnthropicProviderAdapter()

        for model_id in ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"]:
            model = provider.create_model(
                model_id=model_id,
                api_key="test-key",
                base_url=None,
                timeout=None,
                http_client=None,
            )
            assert model is not None
            # Model should have the specified model_name
            assert model.model_name == model_id


# =============================================================================
# Test Health Checks
# =============================================================================


class TestAnthropicHealthCheck:
    """Test Anthropic provider health check functionality."""

    @patch("osprey.models.providers.anthropic.anthropic.Anthropic")
    def test_health_check_missing_api_key(self, mock_client_class):
        """Test health check fails when API key is missing."""
        provider = AnthropicProviderAdapter()

        is_healthy, message = provider.check_health(api_key=None, base_url=None)

        assert is_healthy is False
        assert "not set" in message.lower()

    @patch("osprey.models.providers.anthropic.anthropic.Anthropic")
    def test_health_check_placeholder_api_key(self, mock_client_class):
        """Test health check detects placeholder/template API keys."""
        provider = AnthropicProviderAdapter()

        # Test placeholder formats that are detected by the implementation
        placeholders = ["${ANTHROPIC_API_KEY}", "sk-ant-xxx"]

        for placeholder in placeholders:
            is_healthy, message = provider.check_health(api_key=placeholder, base_url=None)
            assert is_healthy is False
            assert "not configured" in message.lower() or "placeholder" in message.lower()

    @patch("osprey.models.providers.anthropic.anthropic.Anthropic")
    def test_health_check_success(self, mock_client_class):
        """Test successful health check."""
        provider = AnthropicProviderAdapter()

        # Mock successful API response
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello")]
        mock_client.messages.create.return_value = mock_response

        is_healthy, message = provider.check_health(
            api_key="sk-ant-valid-key", base_url=None, model_id="claude-haiku-4-5-20251001"
        )

        assert is_healthy is True
        assert "working" in message.lower() or "accessible" in message.lower()
        mock_client.messages.create.assert_called_once()

    @patch("osprey.models.providers.anthropic.anthropic.Anthropic")
    def test_health_check_authentication_error(self, mock_client_class):
        """Test health check handles authentication errors."""
        from anthropic import AuthenticationError

        provider = AnthropicProviderAdapter()

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.messages.create.side_effect = AuthenticationError(
            "Invalid API key", response=MagicMock(), body=None
        )

        is_healthy, message = provider.check_health(api_key="sk-ant-invalid-key", base_url=None)

        assert is_healthy is False
        assert "authentication" in message.lower() or "invalid" in message.lower()

    @patch("osprey.models.providers.anthropic.anthropic.Anthropic")
    def test_health_check_rate_limit(self, mock_client_class):
        """Test health check handles rate limit as healthy (key works)."""
        from anthropic import RateLimitError

        provider = AnthropicProviderAdapter()

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.messages.create.side_effect = RateLimitError(
            "Rate limited", response=MagicMock(), body=None
        )

        is_healthy, message = provider.check_health(api_key="sk-ant-key", base_url=None)

        # Rate limit means the API key is valid, just temporarily limited
        assert is_healthy is True
        assert "rate limit" in message.lower()

    @patch("osprey.models.providers.anthropic.anthropic.Anthropic")
    def test_health_check_model_not_found(self, mock_client_class):
        """Test health check handles model not found error."""
        from anthropic import NotFoundError

        provider = AnthropicProviderAdapter()

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.messages.create.side_effect = NotFoundError(
            "Model not found", response=MagicMock(), body=None
        )

        is_healthy, message = provider.check_health(
            api_key="sk-ant-key", base_url=None, model_id="nonexistent-model"
        )

        assert is_healthy is False
        assert "not found" in message.lower()

    @patch("osprey.models.providers.anthropic.anthropic.Anthropic")
    def test_health_check_timeout(self, mock_client_class):
        """Test health check handles timeout errors."""
        from anthropic import APITimeoutError

        provider = AnthropicProviderAdapter()

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.messages.create.side_effect = APITimeoutError(request=MagicMock())

        is_healthy, message = provider.check_health(api_key="sk-ant-key", base_url=None)

        assert is_healthy is False
        # Message can be "timeout" or "connection failed" depending on error handling
        assert "timeout" in message.lower() or "connection failed" in message.lower()

    @patch("osprey.models.providers.anthropic.anthropic.Anthropic")
    def test_health_check_uses_cheap_model_by_default(self, mock_client_class):
        """Test health check uses cheapest model when no model_id specified."""
        provider = AnthropicProviderAdapter()

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="OK")]
        mock_client.messages.create.return_value = mock_response

        provider.check_health(api_key="sk-ant-key", base_url=None)

        # Should use health_check_model_id (haiku)
        call_args = mock_client.messages.create.call_args[1]
        assert "haiku" in call_args["model"].lower()

    @patch("osprey.models.providers.anthropic.anthropic.Anthropic")
    def test_health_check_uses_provided_model_id(self, mock_client_class):
        """Test health check uses provided model ID if specified."""
        provider = AnthropicProviderAdapter()

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="OK")]
        mock_client.messages.create.return_value = mock_response

        provider.check_health(
            api_key="sk-ant-key", base_url=None, model_id="claude-sonnet-4-5-20250929"
        )

        call_args = mock_client.messages.create.call_args[1]
        assert call_args["model"] == "claude-sonnet-4-5-20250929"

    @patch("osprey.models.providers.anthropic.anthropic.Anthropic")
    def test_health_check_respects_timeout(self, mock_client_class):
        """Test health check passes timeout parameter."""
        provider = AnthropicProviderAdapter()

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="OK")]
        mock_client.messages.create.return_value = mock_response

        provider.check_health(api_key="sk-ant-key", base_url=None, timeout=10.0)

        call_args = mock_client.messages.create.call_args[1]
        assert call_args["timeout"] == 10.0

    @patch("osprey.models.providers.anthropic.anthropic.Anthropic")
    def test_health_check_generic_api_error(self, mock_client_class):
        """Test health check handles generic API errors."""
        from anthropic import APIConnectionError

        provider = AnthropicProviderAdapter()

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.messages.create.side_effect = APIConnectionError(request=MagicMock())

        is_healthy, message = provider.check_health(api_key="sk-ant-key", base_url=None)

        assert is_healthy is False
        assert (
            "error" in message.lower()
            or "connection" in message.lower()
            or "failed" in message.lower()
        )

    @patch("osprey.models.providers.anthropic.anthropic.Anthropic")
    def test_health_check_unexpected_error(self, mock_client_class):
        """Test health check handles unexpected exceptions."""
        provider = AnthropicProviderAdapter()

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("Unexpected error")

        is_healthy, message = provider.check_health(api_key="sk-ant-key", base_url=None)

        assert is_healthy is False
        assert "error" in message.lower() or "unexpected" in message.lower()
