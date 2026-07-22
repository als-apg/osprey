"""Stanford AI Playground Provider Adapter Implementation.

Stanford AI Playground is an OpenAI-compatible API proxy that provides access
to multiple LLM providers (Anthropic, OpenAI, Google, DeepSeek, etc.) through
a unified endpoint at https://aiapi-prod.stanford.edu/v1.

This provider uses LiteLLM as the backend for unified API access.
"""

from .litellm_adapter import check_litellm_health, execute_litellm_completion
from .litellm_delegating import LiteLLMDelegatingProvider

__all__ = ["StanfordProviderAdapter", "check_litellm_health", "execute_litellm_completion"]


class StanfordProviderAdapter(LiteLLMDelegatingProvider):
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
    health_check_model_id = "gpt-4o-mini"  # Cheapest OpenAI model for health checks
    available_models = [
        # Anthropic Claude models
        "claude-3-7-sonnet",
        # OpenAI models
        "gpt-4o",
        "gpt-4o-mini",
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
    supports_native_structured_output = True  # proxies to models with native json_schema support

    # Stanford alone resolves a missing base_url to default_base_url before
    # delegating; execute_completion / check_health are inherited from
    # LiteLLMDelegatingProvider, which applies the fallback when this flag is set.
    apply_default_base_url_fallback = True
