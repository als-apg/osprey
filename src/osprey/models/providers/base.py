"""Base Provider Interface for AI Model Access."""

from abc import ABC, abstractmethod
from typing import Any


class BaseProvider(ABC):
    """Abstract base class for AI model providers.

    All provider implementations must inherit from this class and implement
    the two core methods: execute_completion and check_health.

    **Metadata as Class Attributes** (SINGLE SOURCE OF TRUTH):
    Subclasses define provider metadata as class attributes. The registry
    introspects these attributes after loading the class, avoiding duplication
    between ProviderRegistration and the class itself. This follows the same
    pattern as capabilities and context classes in the framework.

    Metadata Attributes (define on subclass):
        name: Provider identifier (e.g., "anthropic", "openai")
        description: User-friendly description for display in TUI (e.g., "Anthropic (Claude models)")
        requires_api_key: Whether provider requires API key for authentication
        requires_base_url: Whether provider requires custom base URL
        requires_model_id: Whether provider requires model ID specification
        supports_proxy: Whether provider supports HTTP proxy configuration
        default_base_url: Default API endpoint URL if applicable
        default_model_id: Default model recommended for general use (used in templates)
        health_check_model_id: Cheapest/fastest model for health checks
        available_models: List of available model IDs for this provider (for TUI/selection)
        api_key_url: URL where users can obtain an API key (e.g., "https://console.anthropic.com/")
        api_key_instructions: Step-by-step instructions for obtaining an API key
        api_key_note: Additional notes or requirements (e.g., "Requires affiliation")

    This interface ensures consistent provider behavior across the framework
    while allowing provider-specific implementations.
    """

    # Metadata - subclasses MUST override these class attributes
    name: str = NotImplemented  # Provider identifier (e.g., "anthropic")
    description: str = (
        NotImplemented  # User-friendly description (e.g., "Anthropic (Claude models)")
    )
    requires_api_key: bool = NotImplemented
    requires_base_url: bool = NotImplemented
    requires_model_id: bool = NotImplemented
    supports_proxy: bool = NotImplemented
    default_base_url: str | None = None
    default_model_id: str | None = None  # Default model for templates/general use
    health_check_model_id: str | None = None  # Cheapest model for health checks
    available_models: list[str] = []  # List of available models for this provider

    # API key acquisition information (for CLI help and documentation)
    api_key_url: str | None = None  # URL where users can obtain an API key
    api_key_instructions: list[str] = []  # Step-by-step instructions for obtaining the key
    api_key_note: str | None = None  # Additional notes or requirements

    @abstractmethod
    def execute_completion(
        self,
        message: str,
        model_id: str,
        api_key: str | None,
        base_url: str | None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        thinking: dict | None = None,
        system_prompt: str | None = None,
        output_format: Any | None = None,
        **kwargs,
    ) -> str | Any:
        """Execute a direct chat completion.

        :param message: User message to send
        :param model_id: Model identifier
        :param api_key: API authentication key
        :param base_url: Custom API endpoint URL
        :param max_tokens: Maximum tokens to generate
        :param temperature: Sampling temperature
        :param thinking: Extended thinking configuration (if supported)
        :param system_prompt: System prompt (if supported)
        :param output_format: Structured output format (Pydantic model or TypedDict)
        :param kwargs: Additional provider-specific arguments
        :return: Model response text or structured output
        """
        pass

    @abstractmethod
    def check_health(
        self,
        api_key: str | None,
        base_url: str | None,
        timeout: float = 5.0,
        model_id: str | None = None,
    ) -> tuple[bool, str]:
        """Test provider connectivity and authentication.

        Makes a minimal API call to verify the API key works. For paid providers,
        uses the cheapest available model with minimal tokens (~$0.0001 per check).

        :param api_key: API authentication key
        :param base_url: Custom API endpoint URL
        :param timeout: Request timeout in seconds
        :param model_id: Optional model ID to test with (uses cheapest if not provided)
        :return: (success, message) tuple
        """
        pass
