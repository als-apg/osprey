"""Anthropic Provider Adapter Implementation.

This provider uses LiteLLM as the backend for unified API access.
"""

import re

from .litellm_adapter import check_litellm_health, execute_litellm_completion
from .litellm_delegating import LiteLLMDelegatingProvider

__all__ = ["AnthropicProviderAdapter", "check_litellm_health", "execute_litellm_completion"]

#: The Claude models that take a caller-chosen temperature: generation 3, and
#: generation 4 up to 4.6 (``claude-sonnet-4``, ``claude-haiku-4-5-20251001``).
#: Every later model refuses any temperature but 1.
_TAKES_TEMPERATURE = re.compile(
    r"claude-(?:3-.+|(?:opus|sonnet|haiku)-4(?:-[0-6])?(?:-\d{8}|-latest)?)"
)


class AnthropicProviderAdapter(LiteLLMDelegatingProvider):
    """Anthropic AI provider implementation using LiteLLM."""

    # Metadata (single source of truth)
    name = "anthropic"
    description = "Anthropic (Claude models)"
    requires_api_key = True
    requires_base_url = False
    supports_proxy = True
    default_base_url = None
    models_probe = "anthropic"
    models_probe_base_url = "https://api.anthropic.com"
    default_model_id = "claude-haiku-4-5"
    health_check_model_id = "claude-haiku-4-5"

    # API key acquisition information
    api_key_url = "https://console.anthropic.com/"
    api_key_instructions = [
        "Sign up or log in with your account",
        "Navigate to 'API Keys' in the settings",
        "Click 'Create Key' and name your key",
        "Copy the key (shown only once!)",
    ]
    api_key_note = None

    # Provider facts (see BaseProvider)
    api_key_env_var = "ANTHROPIC_API_KEY"
    api_protocol = "anthropic"
    # A launch with no key signs in to a Claude subscription instead.
    supports_interactive_login = True
    supports_images = True
    supports_thinking = False
    self_hosted = False

    # LiteLLM integration
    litellm_prefix = "anthropic"

    @classmethod
    def accepts_temperature(cls, model_id: str) -> bool:
        """Whether a request for *model_id* carries the caller's temperature.

        Claude models up to 4.6 take one; the later ones, Claude 5 among them,
        refuse every temperature but 1. An id outside the known generations is
        treated as a later model, so a new model is never sent one.

        Args:
            model_id: The bare Anthropic model identifier.

        Returns:
            True for a model of a generation that takes a temperature.
        """
        return _TAKES_TEMPERATURE.fullmatch(model_id) is not None

    # execute_completion / check_health inherited from LiteLLMDelegatingProvider.
