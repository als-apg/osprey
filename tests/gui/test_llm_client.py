"""Tests for SimpleLLMClient - singleton-free LLM routing client.

These tests verify that SimpleLLMClient works independently of the global
registry singleton, eliminating the "Registry not initialized" error.
"""

import os
from unittest.mock import Mock, patch

import pytest

from osprey.interfaces.pyqt.llm_client import SimpleLLMClient


class TestSimpleLLMClientInitialization:
    """Test SimpleLLMClient initialization and validation."""

    def test_init_with_valid_anthropic_config(self):
        """Test initialization with valid Anthropic configuration."""
        client = SimpleLLMClient(
            provider="anthropic", model_id="claude-3-sonnet-20240229", api_key="test-key"
        )

        assert client.provider == "anthropic"
        assert client.model_id == "claude-3-sonnet-20240229"
        assert client.api_key == "test-key"
        assert client.base_url is None

    def test_init_with_valid_openai_config(self):
        """Test initialization with valid OpenAI configuration."""
        client = SimpleLLMClient(
            provider="openai",
            model_id="gpt-4",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )

        assert client.provider == "openai"
        assert client.model_id == "gpt-4"
        assert client.api_key == "test-key"
        assert client.base_url == "https://api.openai.com/v1"

    def test_init_with_valid_ollama_config(self):
        """Test initialization with valid Ollama configuration."""
        client = SimpleLLMClient(
            provider="ollama", model_id="llama3.1:8b", base_url="http://localhost:11434"
        )

        assert client.provider == "ollama"
        assert client.model_id == "llama3.1:8b"
        assert client.base_url == "http://localhost:11434"

    def test_init_with_valid_argo_config(self):
        """Test initialization with valid Argo (ANL) configuration."""
        client = SimpleLLMClient(
            provider="argo",
            model_id="gpt5",
            api_key="test-key",
            base_url="https://argo-bridge.cels.anl.gov",
        )

        assert client.provider == "argo"
        assert client.model_id == "gpt5"
        assert client.api_key == "test-key"
        assert client.base_url == "https://argo-bridge.cels.anl.gov"

    def test_init_missing_provider(self):
        """Test that missing provider raises ValueError."""
        with pytest.raises(ValueError, match="Provider is required"):
            SimpleLLMClient(provider="", model_id="test-model", api_key="test-key")

    def test_init_missing_model_id(self):
        """Test that missing model_id raises ValueError."""
        with pytest.raises(ValueError, match="Model ID is required"):
            SimpleLLMClient(provider="anthropic", model_id="", api_key="test-key")

    def test_init_ollama_missing_base_url(self):
        """Test that Ollama without base_url raises ValueError."""
        with pytest.raises(ValueError, match="Base URL required for Ollama"):
            SimpleLLMClient(provider="ollama", model_id="llama3.1:8b")

    def test_init_case_insensitive_provider(self):
        """Test that provider names are case-insensitive."""
        client = SimpleLLMClient(
            provider="ANTHROPIC", model_id="claude-3-sonnet-20240229", api_key="test-key"
        )

        assert client.provider == "anthropic"


class TestSimpleLLMClientFromGUIConfig:
    """Test SimpleLLMClient.from_gui_config() factory method."""

    def test_from_gui_config_anthropic(self, tmp_path):
        """Test loading Anthropic config from gui_config.yml."""
        config_content = """
models:
  classifier:
    provider: anthropic
    model_id: claude-3-sonnet-20240229

api:
  providers:
    anthropic:
      api_key: ${ANTHROPIC_API_KEY}
"""
        config_path = tmp_path / "gui_config.yml"
        config_path.write_text(config_content)

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = SimpleLLMClient.from_gui_config(str(config_path))

        assert client.provider == "anthropic"
        assert client.model_id == "claude-3-sonnet-20240229"
        assert client.api_key == "test-key"

    def test_from_gui_config_argo(self, tmp_path):
        """Test loading Argo (ANL) config from gui_config.yml."""
        config_content = """
models:
  classifier:
    provider: argo
    model_id: gpt5

api:
  providers:
    argo:
      api_key: ${ARGO_API_KEY}
      base_url: https://argo-bridge.cels.anl.gov
"""
        config_path = tmp_path / "gui_config.yml"
        config_path.write_text(config_content)

        with patch.dict(os.environ, {"ARGO_API_KEY": "test-key"}):
            client = SimpleLLMClient.from_gui_config(str(config_path))

        assert client.provider == "argo"
        assert client.model_id == "gpt5"
        assert client.api_key == "test-key"
        assert client.base_url == "https://argo-bridge.cels.anl.gov"

    def test_from_gui_config_ollama(self, tmp_path):
        """Test loading Ollama config from gui_config.yml."""
        config_content = """
models:
  classifier:
    provider: ollama
    model_id: llama3.1:8b

api:
  providers:
    ollama:
      base_url: http://localhost:11434
"""
        config_path = tmp_path / "gui_config.yml"
        config_path.write_text(config_content)

        client = SimpleLLMClient.from_gui_config(str(config_path))

        assert client.provider == "ollama"
        assert client.model_id == "llama3.1:8b"
        assert client.base_url == "http://localhost:11434"

    def test_from_gui_config_missing_file(self):
        """Test that missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="GUI configuration file not found"):
            SimpleLLMClient.from_gui_config("/nonexistent/path/gui_config.yml")

    def test_from_gui_config_missing_classifier(self, tmp_path):
        """Test that missing classifier config raises ValueError."""
        config_content = """
models:
  other_model:
    provider: anthropic
    model_id: claude-3-sonnet-20240229
"""
        config_path = tmp_path / "gui_config.yml"
        config_path.write_text(config_content)

        with pytest.raises(ValueError, match="'classifier' model not configured"):
            SimpleLLMClient.from_gui_config(str(config_path))

    def test_from_gui_config_missing_provider_field(self, tmp_path):
        """Test that missing provider field raises ValueError."""
        config_content = """
models:
  classifier:
    model_id: claude-3-sonnet-20240229

api:
  providers:
    anthropic:
      api_key: test-key
"""
        config_path = tmp_path / "gui_config.yml"
        config_path.write_text(config_content)

        with pytest.raises(ValueError, match="missing required fields"):
            SimpleLLMClient.from_gui_config(str(config_path))

    def test_from_gui_config_missing_provider_config(self, tmp_path):
        """Test that missing provider configuration raises ValueError."""
        config_content = """
models:
  classifier:
    provider: anthropic
    model_id: claude-3-sonnet-20240229

api:
  providers:
    openai:
      api_key: test-key
"""
        config_path = tmp_path / "gui_config.yml"
        config_path.write_text(config_content)

        with pytest.raises(ValueError, match="Provider 'anthropic' not configured"):
            SimpleLLMClient.from_gui_config(str(config_path))

    def test_from_gui_config_env_var_not_set(self, tmp_path):
        """Test warning when environment variable is not set."""
        config_content = """
models:
  classifier:
    provider: anthropic
    model_id: claude-3-sonnet-20240229

api:
  providers:
    anthropic:
      api_key: ${ANTHROPIC_API_KEY}
"""
        config_path = tmp_path / "gui_config.yml"
        config_path.write_text(config_content)

        # Ensure env var is not set
        with patch.dict(os.environ, {}, clear=True):
            with patch("osprey.interfaces.pyqt.llm_client.logger") as mock_logger:
                client = SimpleLLMClient.from_gui_config(str(config_path))

                # Should log warning
                mock_logger.warning.assert_called()
                assert client.api_key is None


class TestSimpleLLMClientCalls:
    """Test SimpleLLMClient LLM API calls."""

    @patch("anthropic.Anthropic")
    def test_call_anthropic_success(self, mock_anthropic_class):
        """Test successful Anthropic API call."""
        # Setup mock
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(text="Test response")]
        mock_client.messages.create.return_value = mock_message
        mock_anthropic_class.return_value = mock_client

        # Create client and call
        client = SimpleLLMClient(
            provider="anthropic", model_id="claude-3-sonnet-20240229", api_key="test-key"
        )

        response = client.call("Test prompt", max_tokens=100, temperature=0.5)

        # Verify
        assert response == "Test response"
        mock_anthropic_class.assert_called_once_with(api_key="test-key")
        mock_client.messages.create.assert_called_once_with(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            temperature=0.5,
            messages=[{"role": "user", "content": "Test prompt"}],
        )

    @patch("openai.OpenAI")
    def test_call_openai_success(self, mock_openai_class):
        """Test successful OpenAI API call."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # Create client and call
        client = SimpleLLMClient(
            provider="openai",
            model_id="gpt-4",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )

        response = client.call("Test prompt", max_tokens=100, temperature=0.5)

        # Verify
        assert response == "Test response"
        mock_openai_class.assert_called_once_with(
            api_key="test-key", base_url="https://api.openai.com/v1"
        )
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            max_tokens=100,
            temperature=0.5,
            messages=[{"role": "user", "content": "Test prompt"}],
        )

    @patch("openai.OpenAI")
    def test_call_argo_success(self, mock_openai_class):
        """Test successful Argo (ANL) API call using OpenAI-compatible interface."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # Create client and call
        client = SimpleLLMClient(
            provider="argo",
            model_id="gpt5",
            api_key="test-key",
            base_url="https://argo-bridge.cels.anl.gov",
        )

        response = client.call("Test prompt")

        # Verify
        assert response == "Test response"
        mock_openai_class.assert_called_once_with(
            api_key="test-key", base_url="https://argo-bridge.cels.anl.gov"
        )

    @patch("ollama.Client")
    def test_call_ollama_success(self, mock_ollama_class):
        """Test successful Ollama API call."""
        # Setup mock
        mock_client = Mock()
        mock_client.chat.return_value = {"message": {"content": "Test response"}}
        mock_ollama_class.return_value = mock_client

        # Create client and call
        client = SimpleLLMClient(
            provider="ollama", model_id="llama3.1:8b", base_url="http://localhost:11434"
        )

        response = client.call("Test prompt", max_tokens=100, temperature=0.5)

        # Verify
        assert response == "Test response"
        mock_ollama_class.assert_called_once_with(host="http://localhost:11434")
        mock_client.chat.assert_called_once_with(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": "Test prompt"}],
            options={"num_predict": 100, "temperature": 0.5},
        )

    def test_call_unsupported_provider(self):
        """Test that unsupported provider raises ValueError."""
        client = SimpleLLMClient(provider="unsupported", model_id="test-model", api_key="test-key")

        # Bypass validation for this test
        client.provider = "unsupported"

        with pytest.raises(ValueError, match="Unsupported provider"):
            client.call("Test prompt")

    @patch("anthropic.Anthropic")
    def test_call_anthropic_api_error(self, mock_anthropic_class):
        """Test handling of Anthropic API errors."""
        # Setup mock to raise exception
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic_class.return_value = mock_client

        client = SimpleLLMClient(
            provider="anthropic", model_id="claude-3-sonnet-20240229", api_key="test-key"
        )

        with pytest.raises(Exception, match="API Error"):
            client.call("Test prompt")


class TestSimpleLLMClientNoSingletonDependency:
    """Test that SimpleLLMClient works without global registry singleton."""

    def test_no_registry_dependency(self):
        """Verify SimpleLLMClient works without global registry initialization.

        This is the critical test that verifies the fix for the
        "Registry not initialized" error.
        """
        # Create client WITHOUT initializing any global registry
        client = SimpleLLMClient(
            provider="anthropic", model_id="claude-3-sonnet-20240229", api_key="test-key"
        )

        # Should not raise "Registry not initialized"
        assert client.provider == "anthropic"
        assert client.model_id == "claude-3-sonnet-20240229"

    @patch("anthropic.Anthropic")
    def test_call_without_registry(self, mock_anthropic_class):
        """Verify LLM calls work without global registry.

        This test ensures routing can happen BEFORE project selection,
        which is when the global registry gets initialized.
        """
        # Setup mock
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(text="Routing response")]
        mock_client.messages.create.return_value = mock_message
        mock_anthropic_class.return_value = mock_client

        # Create client and call WITHOUT any registry
        client = SimpleLLMClient(
            provider="anthropic", model_id="claude-3-sonnet-20240229", api_key="test-key"
        )

        # Should work without registry
        response = client.call("Route this query")

        assert response == "Routing response"
        # Verify no registry was accessed
        mock_anthropic_class.assert_called_once()
