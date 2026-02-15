"""Tests for Anthropic model provider adapter."""

from unittest.mock import MagicMock, patch

from osprey.models.providers.anthropic import AnthropicProviderAdapter


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


class TestAnthropicHealthCheck:
    """Test Anthropic provider health check functionality via LiteLLM."""

    def test_health_check_missing_api_key(self):
        """Test health check fails when API key is missing."""
        provider = AnthropicProviderAdapter()

        is_healthy, message = provider.check_health(api_key=None, base_url=None)

        assert is_healthy is False
        assert "not set" in message.lower()

    def test_health_check_placeholder_api_key(self):
        """Test health check detects placeholder/template API keys."""
        provider = AnthropicProviderAdapter()

        placeholders = ["${ANTHROPIC_API_KEY}", "YOUR_API_KEY_HERE"]

        for placeholder in placeholders:
            is_healthy, message = provider.check_health(api_key=placeholder, base_url=None)
            assert is_healthy is False
            assert "not configured" in message.lower() or "placeholder" in message.lower()

    @patch("osprey.models.providers.litellm_adapter.litellm.completion")
    def test_health_check_success(self, mock_completion):
        """Test successful health check via LiteLLM."""
        provider = AnthropicProviderAdapter()

        # Mock successful LiteLLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(tool_calls=None, content="Hello"))]
        mock_completion.return_value = mock_response

        is_healthy, message = provider.check_health(
            api_key="sk-ant-valid-key",
            base_url=None,
            model_id="claude-haiku-4-5-20251001",
        )

        assert is_healthy is True
        assert "accessible" in message.lower()
        mock_completion.assert_called_once()

    @patch("osprey.models.providers.litellm_adapter.litellm.completion")
    def test_health_check_authentication_error(self, mock_completion):
        """Test health check handles authentication errors."""
        import litellm

        provider = AnthropicProviderAdapter()
        mock_completion.side_effect = litellm.AuthenticationError(
            message="Invalid API key",
            llm_provider="anthropic",
            model="claude-haiku-4-5-20251001",
        )

        is_healthy, message = provider.check_health(api_key="sk-ant-invalid-key", base_url=None)

        assert is_healthy is False
        assert "authentication" in message.lower()

    @patch("osprey.models.providers.litellm_adapter.litellm.completion")
    def test_health_check_rate_limit(self, mock_completion):
        """Test health check handles rate limit as healthy (key works)."""
        import litellm

        provider = AnthropicProviderAdapter()
        mock_completion.side_effect = litellm.RateLimitError(
            message="Rate limited",
            llm_provider="anthropic",
            model="claude-haiku-4-5-20251001",
        )

        is_healthy, message = provider.check_health(api_key="sk-ant-key", base_url=None)

        # Rate limit means the API key is valid, just temporarily limited
        assert is_healthy is True
        assert "rate limit" in message.lower()


class TestAnthropicExecuteCompletion:
    """Test Anthropic provider execute_completion via LiteLLM."""

    @patch("osprey.models.providers.litellm_adapter.litellm.completion")
    def test_execute_completion_basic(self, mock_completion):
        """Test basic completion execution."""
        provider = AnthropicProviderAdapter()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(tool_calls=None, content="Hello!"))]
        mock_completion.return_value = mock_response

        result = provider.execute_completion(
            message="Say hello",
            model_id="claude-haiku-4-5-20251001",
            api_key="test-key",
            base_url=None,
        )

        assert result == "Hello!"
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == "anthropic/claude-haiku-4-5-20251001"
        assert call_kwargs["api_key"] == "test-key"

    @patch("osprey.models.providers.litellm_adapter.litellm.completion")
    def test_execute_completion_with_thinking(self, mock_completion):
        """Test completion with extended thinking enabled."""
        provider = AnthropicProviderAdapter()

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(tool_calls=None, content="Thinking response"))
        ]
        mock_completion.return_value = mock_response

        result = provider.execute_completion(
            message="Think about this",
            model_id="claude-sonnet-4",
            api_key="test-key",
            base_url=None,
            max_tokens=2000,
            enable_thinking=True,
            budget_tokens=1000,
        )

        assert result is not None
        call_kwargs = mock_completion.call_args[1]
        # Should include thinking configuration
        assert "thinking" in call_kwargs
