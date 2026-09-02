"""Google Provider Adapter Implementation.

This provider uses LiteLLM as the backend for unified API access.
"""

from .litellm_adapter import check_litellm_health, execute_litellm_completion
from .litellm_delegating import LiteLLMDelegatingProvider

__all__ = ["GoogleProviderAdapter", "check_litellm_health", "execute_litellm_completion"]


class GoogleProviderAdapter(LiteLLMDelegatingProvider):
    """Google AI (Gemini) provider implementation using LiteLLM."""

    # Metadata (single source of truth)
    name = "google"
    description = "Google (Gemini models)"
    requires_api_key = True
    requires_base_url = False
    requires_model_id = True
    supports_proxy = True
    default_base_url = None
    default_model_id = "gemini-3.8-flash"  # Current stable Flash for general use
    health_check_model_id = "gemini-3.5-flash-lite"  # Cheapest/fastest stable, for health checks
    available_models = [
        "gemini-2.5-pro",  # Most capable stable Gemini model (no stable Pro successor yet)
        "gemini-3.8-flash",  # Current stable Flash, good balance
        "gemini-3.5-flash-lite",  # Fastest, most cost-effective stable model
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

    # LiteLLM integration - Google uses "gemini" prefix
    litellm_prefix = "gemini"

    # execute_completion / check_health inherited from LiteLLMDelegatingProvider.
