"""Anthropic Provider Adapter Implementation.

This provider uses LiteLLM as the backend for unified API access.
"""

from typing import Any

from .base import BaseProvider
from .litellm_adapter import check_litellm_health, execute_litellm_completion


class AnthropicProviderAdapter(BaseProvider):
    """Anthropic AI provider implementation using LiteLLM."""

    # Metadata (single source of truth)
    name = "anthropic"
    description = "Anthropic (Claude models)"
    requires_api_key = True
    requires_base_url = False
    requires_model_id = True
    supports_proxy = True
    default_base_url = None
    default_model_id = "claude-haiku-4-5-20251001"
    health_check_model_id = "claude-haiku-4-5-20251001"
    available_models = ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"]

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

    def execute_completion(
        self,
        message: str,
        model_id: str,
        api_key: str | None,
        base_url: str | None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        thinking: dict | None = None,
        system_prompt: str | None = None,
        output_format: Any | None = None,
        **kwargs,
    ) -> str | list | Any:
        """Execute Anthropic chat completion via LiteLLM."""
        return execute_litellm_completion(
            provider=self.name,
            message=message,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            output_format=output_format,
            **kwargs,
        )

    def check_health(
        self,
        api_key: str | None,
        base_url: str | None,
        timeout: float = 5.0,
        model_id: str | None = None,
    ) -> tuple[bool, str]:
        """Check Anthropic API health via LiteLLM."""
        return check_litellm_health(
            provider=self.name,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            model_id=model_id or self.health_check_model_id,
        )
