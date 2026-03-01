"""ALS-APG Provider Adapter Implementation.

This provider uses LiteLLM as the backend for unified API access.
ALS-APG is an OpenAI-compatible proxy service hosted on AWS for the
Advanced Light Source Accelerator Physics Group.
"""

from typing import Any

from .base import BaseProvider
from .litellm_adapter import check_litellm_health, execute_litellm_completion


class ALSAPGProviderAdapter(BaseProvider):
    """ALS Accelerator Physics Group provider implementation using LiteLLM."""

    # Metadata (single source of truth)
    name = "als-apg"
    description = "ALS Accelerator Physics Group AWS proxy (supports Anthropic models)"
    requires_api_key = True
    requires_base_url = True
    requires_model_id = True
    supports_proxy = True
    default_base_url = "https://llm.gianlucamartino.com"
    default_model_id = "claude-haiku-4-5-20251001"
    health_check_model_id = "claude-haiku-4-5-20251001"
    available_models = [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ]

    # API key acquisition information
    api_key_url = None
    api_key_instructions = [
        "Contact the ALS Accelerator Physics Group for API access.",
        "Set ALS_APG_API_KEY in your environment.",
    ]
    api_key_note = "Internal ALS-APG proxy — requires group membership."

    # LiteLLM integration - ALS-APG is an OpenAI-compatible proxy
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
        """Execute ALS-APG chat completion via LiteLLM."""
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
        """Check ALS-APG API health via LiteLLM."""
        return check_litellm_health(
            provider=self.name,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            model_id=model_id or self.health_check_model_id,
        )
