"""OpenAI Provider Adapter Implementation.

This provider uses LiteLLM as the backend for unified API access.
"""

from .litellm_adapter import check_litellm_health, execute_litellm_completion
from .litellm_delegating import LiteLLMDelegatingProvider

__all__ = ["OpenAIProviderAdapter", "check_litellm_health", "execute_litellm_completion"]


class OpenAIProviderAdapter(LiteLLMDelegatingProvider):
    """OpenAI provider implementation using LiteLLM."""

    # Metadata (single source of truth)
    name = "openai"
    description = "OpenAI (GPT models)"
    requires_api_key = True
    requires_base_url = False
    requires_model_id = True
    supports_proxy = True
    default_base_url = "https://api.openai.com/v1"
    default_model_id = "gpt-5"  # GPT-5 for general use
    health_check_model_id = "gpt-5-nano"  # Cheapest GPT-5 model for health checks
    available_models = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]

    # API key acquisition information
    api_key_url = "https://platform.openai.com/api-keys"
    api_key_instructions = [
        "Sign up or log in to your OpenAI account",
        "Add billing information if not already set up",
        "Click '+ Create new secret key'",
        "Name your key and copy it (shown only once!)",
    ]
    api_key_note = None

    # LiteLLM integration - OpenAI models don't need a prefix in LiteLLM
    litellm_prefix = ""

    # execute_completion / check_health inherited from LiteLLMDelegatingProvider.
