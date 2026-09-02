"""Anthropic Provider Adapter Implementation.

This provider uses LiteLLM as the backend for unified API access.
"""

from .litellm_adapter import check_litellm_health, execute_litellm_completion
from .litellm_delegating import LiteLLMDelegatingProvider

__all__ = ["AnthropicProviderAdapter", "check_litellm_health", "execute_litellm_completion"]


class AnthropicProviderAdapter(LiteLLMDelegatingProvider):
    """Anthropic AI provider implementation using LiteLLM."""

    # Metadata (single source of truth)
    name = "anthropic"
    description = "Anthropic (Claude models)"
    requires_api_key = True
    requires_base_url = False
    requires_model_id = True
    supports_proxy = True
    default_base_url = None
    default_model_id = "claude-haiku-4-5"
    health_check_model_id = "claude-haiku-4-5"
    available_models = ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"]

    # API key acquisition information
    api_key_url = "https://console.anthropic.com/"
    api_key_instructions = [
        "Sign up or log in with your account",
        "Navigate to 'API Keys' in the settings",
        "Click 'Create Key' and name your key",
        "Copy the key (shown only once!)",
    ]
    api_key_note = None

    # LiteLLM integration
    litellm_prefix = "anthropic"

    # execute_completion / check_health inherited from LiteLLMDelegatingProvider.
