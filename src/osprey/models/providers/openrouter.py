"""OpenRouter Provider Adapter Implementation.

This provider uses LiteLLM as the backend for unified API access.
OpenRouter is a hosted router that serves models from many vendors behind one
key, addressed as ``<vendor>/<model>`` (for example
``anthropic/claude-haiku-4.5``). Requests route through LiteLLM's native
``openrouter/`` prefix rather than its generic OpenAI-compatible path. The
public endpoint ships with the provider; a deployment that fronts it elsewhere
overrides it through ``api.providers.openrouter.base_url`` or the
``OPENROUTER_BASE_URL`` environment variable.
"""

from .litellm_adapter import check_litellm_health, execute_litellm_completion
from .litellm_delegating import LiteLLMDelegatingProvider

__all__ = ["OpenRouterProviderAdapter", "check_litellm_health", "execute_litellm_completion"]


class OpenRouterProviderAdapter(LiteLLMDelegatingProvider):
    """OpenRouter provider implementation using LiteLLM."""

    # Metadata (single source of truth)
    name = "openrouter"
    description = "OpenRouter hosted model router (supports models from many vendors)"
    requires_api_key = True
    requires_base_url = True
    requires_model_id = True
    supports_proxy = True
    # The public endpoint, so a call that names none still reaches OpenRouter.
    # ``requires_base_url`` above is what makes this value reachable at all:
    # BaseProvider.effective_base_url returns a default only for a provider
    # that requires an endpoint.
    default_base_url = "https://openrouter.ai/api/v1"
    # Optional override: a set OPENROUTER_BASE_URL beats both config and the
    # default above.
    base_url_env_var = "OPENROUTER_BASE_URL"
    default_model_id = "anthropic/claude-sonnet-4.5"
    health_check_model_id = "anthropic/claude-haiku-4.5"

    # API key acquisition information
    api_key_url = "https://openrouter.ai/settings/keys"
    api_key_instructions = [
        "Sign in at openrouter.ai and open 'Keys'",
        "Create an API key and add credit to the account",
        "Copy the key and set OPENROUTER_API_KEY",
    ]
    api_key_note = "Usage is billed per token against the OpenRouter account's credit."

    # Provider facts (see BaseProvider)
    api_key_env_var = "OPENROUTER_API_KEY"
    api_protocol = "openai"
    supports_interactive_login = False
    # Image input is a per-model capability on the router, not assumed for all.
    supports_images = False
    supports_thinking = False
    self_hosted = False

    # LiteLLM integration - routed through LiteLLM's native openrouter/ prefix
    litellm_prefix = "openrouter"
    is_openai_compatible = False
    # A hosted router, not a LiteLLM proxy: requests carry no acting identity.
    gateway = None
    # Declared explicitly rather than left to litellm's detection: the native
    # json_schema path through OpenRouter is not assumed until proven live.
    supports_native_structured_output = False

    # execute_completion / check_health inherited from LiteLLMDelegatingProvider.
