"""OpenAI Provider Adapter Implementation.

This provider uses LiteLLM as the backend for unified API access.
"""

from .litellm_adapter import check_litellm_health, execute_litellm_completion
from .litellm_delegating import LiteLLMDelegatingProvider

__all__ = ["OpenAIProviderAdapter", "check_litellm_health", "execute_litellm_completion"]

#: The chat model families that take a caller-chosen temperature. A family
#: matches its own id and every id that extends it with a hyphen (``gpt-4o``
#: matches ``gpt-4o-mini`` and ``gpt-4o-2024-08-06``).
_TEMPERATURE_FAMILIES = ("gpt-3.5", "gpt-4", "gpt-4o", "gpt-4.1", "gpt-4.5", "chatgpt-4o")


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
    default_model_id = "gpt-5.6-sol"  # Flagship for general use
    health_check_model_id = "gpt-5.6-luna"  # Cheapest listed model for health checks

    # API key acquisition information
    api_key_url = "https://platform.openai.com/api-keys"
    api_key_instructions = [
        "Sign up or log in to your OpenAI account",
        "Add billing information if not already set up",
        "Click '+ Create new secret key'",
        "Name your key and copy it (shown only once!)",
    ]
    api_key_note = None

    # Provider facts (see BaseProvider)
    api_key_env_var = "OPENAI_API_KEY"
    api_protocol = "openai"
    supports_interactive_login = False
    supports_images = True
    supports_thinking = False

    # LiteLLM integration - OpenAI models don't need a prefix in LiteLLM
    litellm_prefix = ""
    # OpenAI's API takes max_completion_tokens on every chat model and refuses
    # max_tokens on its reasoning models, whichever generation LiteLLM knows.
    max_tokens_param = "max_completion_tokens"

    @classmethod
    def accepts_temperature(cls, model_id: str) -> bool:
        """Whether a request for *model_id* carries the caller's temperature.

        OpenAI's chat families take one; its reasoning models refuse every
        temperature but their default. An id outside the chat families is
        treated as a reasoning model, so a new reasoning id is never sent one.

        Args:
            model_id: The bare OpenAI model identifier.

        Returns:
            True for an id in one of the chat families.
        """
        return any(
            model_id == family or model_id.startswith(family + "-")
            for family in _TEMPERATURE_FAMILIES
        )

    # execute_completion / check_health inherited from LiteLLMDelegatingProvider.
