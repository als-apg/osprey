"""American Science Cloud Intelligent Interfaces (AMSC i2) Provider Adapter Implementation.

This provider uses LiteLLM as the backend for unified API access.
AMSC i2 is an OpenAI-compatible proxy service for scientific computing.
"""

from .litellm_adapter import check_litellm_health, execute_litellm_completion
from .litellm_delegating import LiteLLMDelegatingProvider

__all__ = ["AMSCI2ProviderAdapter", "check_litellm_health", "execute_litellm_completion"]


class AMSCI2ProviderAdapter(LiteLLMDelegatingProvider):
    """American Science Cloud Intelligent Interfaces (AMSC i2) provider implementation using LiteLLM."""

    # Metadata (single source of truth)
    name = "amsc-i2"
    description = "American Science Cloud proxy (supports multiple models)"
    requires_api_key = True
    requires_base_url = True
    requires_model_id = True
    supports_proxy = True
    default_base_url = None
    default_model_id = "claude-haiku"  # Claude Haiku via AMSC i2 for general use
    health_check_model_id = "claude-haiku"  # Fast and cost-effective for health checks
    available_models = [
        "claude-opus",
        "claude-sonnet",
        "claude-haiku",
        "gpt-oss-120b",
        "gpt-oss-20b",
    ]

    # API key acquisition information
    api_key_url = "https://api.i2-core.american-science-cloud.org/"
    api_key_instructions = [
        "If you have an americansciencecloud.org Google account, log in and go to 'API Key Manager.",
        "Otherwise, request access at https://docs.google.com/forms/d/1xcuOTxzvwu6sEmQfNu5zxLsjaS_hMvAfr99XQzdc_nY/viewform",
    ]
    api_key_note = (
        "Requires an americansciencecloud.org Google account or lab ID via GlobusAuth whitelist."
    )

    # LiteLLM integration - AMSC i2 is an OpenAI-compatible proxy
    is_openai_compatible = True
    supports_native_structured_output = True  # proxies to models with native json_schema support

    # execute_completion / check_health inherited from LiteLLMDelegatingProvider.
