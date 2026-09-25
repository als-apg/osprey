"""ALS-APG Provider Adapter Implementation.

This provider uses LiteLLM as the backend for unified API access.
ALS-APG is an OpenAI-compatible gateway that fronts Anthropic and OpenAI models,
served at ``https://llm.als.lbl.gov/v1``. That endpoint ships with the provider;
a site whose gateway is elsewhere overrides it through
``api.providers.als-apg.base_url`` or the ``ALS_APG_BASE_URL`` environment
variable.
"""

from .litellm_adapter import check_litellm_health, execute_litellm_completion
from .litellm_delegating import LiteLLMDelegatingProvider

__all__ = ["ALSAPGProviderAdapter", "check_litellm_health", "execute_litellm_completion"]


class ALSAPGProviderAdapter(LiteLLMDelegatingProvider):
    """ALS Accelerator Physics Group provider implementation using LiteLLM."""

    # Metadata (single source of truth)
    name = "als-apg"
    description = "ALS Accelerator Physics Group gateway (supports Anthropic and OpenAI models)"
    requires_api_key = True
    requires_base_url = True
    requires_model_id = True
    supports_proxy = True
    # The gateway's own endpoint, so a call that names none still reaches it.
    # ``requires_base_url`` above is what makes this value reachable at all:
    # BaseProvider.effective_base_url returns a default only for a provider
    # that requires an endpoint.
    default_base_url = "https://llm.als.lbl.gov/v1"
    # Optional override: a set ALS_APG_BASE_URL beats both config and the
    # default above, so a deployment with a baked-in URL can be pointed at
    # another gateway at runtime (accepts the URL with or without /v1).
    base_url_env_var = "ALS_APG_BASE_URL"
    default_model_id = "claude-sonnet-5"
    health_check_model_id = "claude-haiku-4-5-20251001"

    # API key acquisition information
    api_key_url = None
    api_key_instructions = [
        "Contact the ALS Accelerator Physics Group for API access.",
        "Set ALS_APG_API_KEY in your environment.",
        "Optionally set ALS_APG_BASE_URL to reach the gateway at another host.",
    ]
    api_key_note = "Internal ALS-APG proxy — requires group membership."

    # LiteLLM integration - ALS-APG is an OpenAI-compatible proxy
    is_openai_compatible = True
    # A LiteLLM proxy: requests carry the acting identity so the gateway's
    # spend logs book each call to a person, not to the deployment's key.
    gateway = "litellm"
    # Note: intentionally leaves supports_native_structured_output at the None default
    # so structured-output support is auto-detected via litellm.supports_response_schema()
    # on the resolved openai/<model> id, which is what the proxy actually serves.

    # execute_completion / check_health inherited from LiteLLMDelegatingProvider.
