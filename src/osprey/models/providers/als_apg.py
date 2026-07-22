"""ALS-APG Provider Adapter Implementation.

This provider uses LiteLLM as the backend for unified API access.
ALS-APG is an OpenAI-compatible proxy service hosted on AWS for the
Advanced Light Source Accelerator Physics Group.
"""

from .litellm_adapter import check_litellm_health, execute_litellm_completion
from .litellm_delegating import LiteLLMDelegatingProvider

__all__ = ["ALSAPGProviderAdapter", "check_litellm_health", "execute_litellm_completion"]


class ALSAPGProviderAdapter(LiteLLMDelegatingProvider):
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
    # Note: intentionally leaves supports_native_structured_output at the None default
    # so structured-output support is auto-detected via litellm.supports_response_schema()
    # on the resolved openai/<model> id — als-apg was never in the old native-json_schema
    # whitelist, so this preserves its prior behavior.

    # execute_completion / check_health inherited from LiteLLMDelegatingProvider.
