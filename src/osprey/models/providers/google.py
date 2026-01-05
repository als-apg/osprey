"""Google Provider Adapter Implementation.

This provider uses LiteLLM as the backend for unified API access.
"""

from typing import Any

from .base import BaseProvider
from .litellm_adapter import check_litellm_health, execute_litellm_completion


class GoogleProviderAdapter(BaseProvider):
    """Google AI (Gemini) provider implementation using LiteLLM."""

    # Metadata (single source of truth)
    name = "google"
    description = "Google (Gemini models)"
    requires_api_key = True
    requires_base_url = False
    requires_model_id = True
    supports_proxy = True
    default_base_url = None
    default_model_id = "gemini-2.5-flash"  # Latest Flash for general use
    health_check_model_id = "gemini-2.5-flash-lite"  # Cheapest/fastest for health checks
    available_models = [
        "gemini-2.5-pro",  # Most capable Gemini 2.5 model
        "gemini-2.5-flash",  # Fast and capable, good balance
        "gemini-2.5-flash-lite",  # Fastest, most cost-effective
    ]

    # API key acquisition information
    api_key_url = "https://aistudio.google.com/app/apikey"
    api_key_instructions = [
        "Sign in with your Google account",
        "Click 'Create API key'",
        "Select a Google Cloud project or create a new one",
        "Copy the generated API key",
    ]
    api_key_note = None

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
        """Execute Google Gemini chat completion via LiteLLM."""
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
        """Check Google API health via LiteLLM."""
        return check_litellm_health(
            provider=self.name,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            model_id=model_id or self.health_check_model_id,
        )
