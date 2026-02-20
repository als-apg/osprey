"""American Science Cloud (AMSC) Provider Adapter Implementation.

This provider uses LiteLLM as the backend for unified API access.
AMSC is an OpenAI-compatible proxy service for scientific computing.
"""

from typing import Any

from .base import BaseProvider
from .litellm_adapter import check_litellm_health, execute_litellm_completion


class AMSCProviderAdapter(BaseProvider):
    """American Science Cloud (AMSC) provider implementation using LiteLLM."""

    # Metadata (single source of truth)
    name = "amsc"
    description = "American Science Cloud proxy (supports multiple models)"
    requires_api_key = True
    requires_base_url = True
    requires_model_id = True
    supports_proxy = True
    default_base_url = None
    default_model_id = "anthropic/claude-haiku"  # Claude Haiku via ASC for general use
    health_check_model_id = "anthropic/claude-haiku"  # Fast and cost-effective for health checks
    # TODO: Update available_models when the AMSC model list is confirmed
    available_models = [
        "anthropic/claude-sonnet",
        "anthropic/claude-haiku",
        "google/gemini-flash",
        "google/gemini-pro",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
    ]

    # API key acquisition information
    api_key_url = (
        "https://docs.google.com/forms/d/1xcuOTxzvwu6sEmQfNu5zxLsjaS_hMvAfr99XQzdc_nY/edit"
    )
    api_key_instructions = [
        "If you have an americansciencecloud.org Google account (workshop attendees), log in directly",
        "Otherwise, request access via GlobusAuth whitelist form with your lab ID",
        "Copy the API key provided after access is granted",
    ]
    api_key_note = (
        "Requires an americansciencecloud.org Google account or lab ID via GlobusAuth whitelist."
    )

    # LiteLLM integration - AMSC is an OpenAI-compatible proxy
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
        """Execute AMSC chat completion via LiteLLM."""
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
        """Check AMSC API health via LiteLLM."""
        return check_litellm_health(
            provider=self.name,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            model_id=model_id or self.health_check_model_id,
        )
