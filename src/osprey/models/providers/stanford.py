"""Stanford AI Playground Provider Adapter Implementation.

Stanford AI Playground is an OpenAI-compatible API proxy that provides access
to multiple LLM providers (Anthropic, OpenAI, Google, DeepSeek, etc.) through
a unified endpoint at https://aiapi-prod.stanford.edu/v1.

This provider uses LiteLLM as the backend for unified API access.
"""

from typing import Any

from .base import BaseProvider
from .litellm_adapter import check_litellm_health, execute_litellm_completion


class StanfordProviderAdapter(BaseProvider):
    """Stanford AI Playground provider adapter using LiteLLM."""

    # Metadata (single source of truth)
    name = "stanford"
    description = "Stanford AI Playground (multi-provider proxy)"
    requires_api_key = True
    requires_base_url = True
    requires_model_id = True
    supports_proxy = True
    default_base_url = "https://aiapi-prod.stanford.edu/v1"
    default_model_id = "gpt-4o"
    health_check_model_id = "gpt-4.omini"  # Cheapest OpenAI model for health checks
    available_models = [
        # Anthropic Claude models
        "claude-3-7-sonnet",
        # OpenAI models
        "gpt-4o",
        "gpt-4.omini",
        "o3-mini",
        # Google models
        "gemini-2.0-flash-001",
        # DeepSeek models
        "deepseek-r1",
    ]

    # API key acquisition help
    api_key_url = "https://uit.stanford.edu/service/ai-api-gateway"
    api_key_instructions = [
        "Requires Stanford University affiliation",
        "Go to 'Get Started' -> 'Request the creation of a new API key'",
        "Log in with your Stanford credentials and complete the form",
        "Once approved, copy the API key from the notification email",
    ]
    api_key_note = "Access restricted to Stanford community"

    # LiteLLM integration - Stanford is an OpenAI-compatible proxy
    is_openai_compatible = True

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
    ) -> str | Any:
        """Execute Stanford AI chat completion via LiteLLM."""
        effective_base_url = base_url or self.default_base_url

        return execute_litellm_completion(
            provider=self.name,
            message=message,
            model_id=model_id,
            api_key=api_key,
            base_url=effective_base_url,
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
        """Check Stanford AI API health via LiteLLM."""
        effective_base_url = base_url or self.default_base_url

        return check_litellm_health(
            provider=self.name,
            api_key=api_key,
            base_url=effective_base_url,
            timeout=timeout,
            model_id=model_id or self.health_check_model_id,
        )
