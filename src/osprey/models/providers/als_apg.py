"""ALS-APG Provider Adapter Implementation.

This provider uses LiteLLM as the backend for unified API access.
ALS-APG is an OpenAI-compatible gateway that fronts Anthropic models. It has no
public endpoint: a deployment supplies its own, through
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
    description = "ALS Accelerator Physics Group gateway (supports Anthropic models)"
    requires_api_key = True
    requires_base_url = True
    requires_model_id = True
    supports_proxy = True
    # No built-in endpoint: this proxy is a deployment's own gateway, so its URL
    # is site data and there is nothing generic to fall back to. A config that
    # names none is refused by the ``requires_base_url`` gate in
    # osprey.models.completion rather than resolving to somebody else's host.
    default_base_url = None
    # Break-glass redirect: a set ALS_APG_BASE_URL beats config, so deployments
    # with a baked-in URL can be pointed at a fallback gateway at runtime
    # (accepts the URL with or without a trailing /v1). It is also the ordinary
    # way to supply the endpoint, since the shipped catalog entry reads
    # ``base_url: ${ALS_APG_BASE_URL}``.
    base_url_env_var = "ALS_APG_BASE_URL"
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
        "Set ALS_APG_BASE_URL to the gateway endpoint — there is no default.",
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
