"""CBORG Provider Adapter Implementation.

This provider uses LiteLLM as the backend for unified API access.
CBORG is LBNL's OpenAI-compatible proxy service.
"""

from .litellm_adapter import check_litellm_health, execute_litellm_completion
from .litellm_delegating import LiteLLMDelegatingProvider

__all__ = ["CBorgProviderAdapter", "check_litellm_health", "execute_litellm_completion"]


class CBorgProviderAdapter(LiteLLMDelegatingProvider):
    """CBORG (LBNL) provider implementation using LiteLLM."""

    # Metadata (single source of truth)
    name = "cborg"
    description = "LBNL CBorg proxy (supports multiple models)"
    requires_api_key = True
    requires_base_url = True
    requires_model_id = True
    supports_proxy = True
    default_base_url = None
    models_probe = "bearer"
    default_model_id = "anthropic/claude-haiku"  # Claude Haiku via CBORG for general use
    health_check_model_id = "anthropic/claude-haiku"  # Fast and cost-effective for health checks

    # API key acquisition information
    api_key_url = "https://cborg.lbl.gov"
    api_key_instructions = [
        "As a Berkeley Lab employee, go to 'API' -> 'Request API Key'",
        "Create an API key ($50/month per user allocation)",
        "Copy the key provided",
    ]
    api_key_note = "Must have affiliation with Berkeley Lab to request an API key."

    # Provider facts (see BaseProvider)
    api_key_env_var = "CBORG_API_KEY"
    api_protocol = "anthropic"
    supports_interactive_login = False
    # The gateway's OpenAI route translates image parts for the models it fronts.
    supports_images = True
    supports_thinking = False
    self_hosted = False

    # LiteLLM integration - CBORG is an OpenAI-compatible proxy
    is_openai_compatible = True
    gateway = "litellm"  # a LiteLLM proxy — requests carry the acting identity
    supports_native_structured_output = True  # proxies to models with native json_schema support

    # execute_completion / check_health inherited from LiteLLMDelegatingProvider.
