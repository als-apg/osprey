"""CBORG Provider Adapter Implementation.

This provider uses LiteLLM as the backend for unified API access.
CBORG is LBNL's OpenAI-compatible proxy service.
"""

from typing import Any

from .base import BaseProvider
from .litellm_adapter import check_litellm_health, execute_litellm_completion


class CBorgProviderAdapter(BaseProvider):
    """CBORG (LBNL) provider implementation using LiteLLM."""

    # Metadata (single source of truth)
    name = "cborg"
    description = "LBNL CBorg proxy (supports multiple models)"
    requires_api_key = True
    requires_base_url = True
    requires_model_id = True
    supports_proxy = True
    default_base_url = None
    default_model_id = "anthropic/claude-haiku"  # Claude Haiku via CBORG for general use
    health_check_model_id = "anthropic/claude-haiku"  # Fast and cost-effective for health checks
    available_models = [
        "anthropic/claude-sonnet",
        "anthropic/claude-haiku",
        "google/gemini-flash",
        "google/gemini-pro",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
    ]

    # API key acquisition information
    api_key_url = "https://cborg.lbl.gov"
    api_key_instructions = [
        "As a Berkeley Lab employee, go to 'API' -> 'Request API Key'",
        "Create an API key ($50/month per user allocation)",
        "Copy the key provided",
    ]
    api_key_note = "Must have affiliation with Berkeley Lab to request an API key."

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
        """Execute CBORG chat completion via LiteLLM."""
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
        """Check CBORG API health via LiteLLM."""
        return check_litellm_health(
            provider=self.name,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            model_id=model_id or self.health_check_model_id,
        )
