"""AI Model Factory for Structured Generation.

Creates and configures LLM model instances for use with PydanticAI agents and structured
generation workflows. This factory handles the complexity of provider-specific initialization,
credential management, HTTP client configuration, and proxy setup across multiple AI providers.

The factory supports enterprise-grade features including connection pooling, timeout management,
and automatic HTTP proxy detection through environment variables. Each provider has specific
requirements for API keys, base URLs, and model identifiers that are validated and enforced.

.. note::
   Model instances created here are optimized for structured generation with PydanticAI.
   For direct chat completions without structured outputs, consider using
   :func:`~completion.get_chat_completion` instead.

.. seealso::
   :func:`get_model` : Main factory function for creating model instances
   :func:`~completion.get_chat_completion` : Direct chat completion interface
   :mod:`configs.config` : Provider configuration and credential management
"""

import logging
import os
from urllib.parse import urlparse

import httpx
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel

from osprey.utils.config import get_provider_config

logger = logging.getLogger(__name__)


def _validate_proxy_url(proxy_url: str) -> bool:
    """Validate HTTP proxy URL format and accessibility.

    Performs basic validation of proxy URL format to ensure it follows
    standard HTTP/HTTPS proxy URL patterns. This helps catch common
    configuration errors early and provides clear feedback.

    :param proxy_url: Proxy URL to validate
    :type proxy_url: str
    :return: True if proxy URL appears valid, False otherwise
    :rtype: bool
    """
    if not proxy_url:
        return False

    try:
        parsed = urlparse(proxy_url)
        # Check for valid scheme and netloc (host:port)
        if parsed.scheme not in ("http", "https"):
            return False
        if not parsed.netloc:
            return False
        return True
    except Exception:
        return False


def get_model(
    provider: str | None = None,
    model_config: dict | None = None,
    model_id: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    max_tokens: int = 100000,
) -> OpenAIModel | AnthropicModel | GeminiModel:
    """Create a configured LLM model instance for structured generation with PydanticAI.

    This factory function creates and configures LLM model instances optimized for
    structured generation workflows using PydanticAI agents. It handles provider-specific
    initialization, credential validation, HTTP client configuration, and proxy setup
    automatically based on environment variables and configuration files.

    The function supports flexible configuration through multiple approaches:
    - Direct parameter specification for programmatic use
    - Model configuration dictionaries from YAML files
    - Automatic credential loading from configuration system
    - Environment-based HTTP proxy detection and configuration

    Provider-specific behavior:
    - **Anthropic**: Requires API key and model ID, supports HTTP proxy
    - **Google**: Requires API key and model ID, supports HTTP proxy
    - **OpenAI**: Requires API key and model ID, supports HTTP proxy and custom base URLs
    - **Ollama**: Requires model ID and base URL, no API key needed, no proxy support
    - **CBORG**: Requires API key, model ID, and base URL, supports HTTP proxy

    :param provider: AI provider name ('anthropic', 'google', 'openai', 'ollama', 'cborg')
    :type provider: str, optional
    :param model_config: Configuration dictionary with provider, model_id, and other settings
    :type model_config: dict, optional
    :param model_id: Specific model identifier recognized by the provider
    :type model_id: str, optional
    :param api_key: API authentication key, auto-loaded from config if not provided
    :type api_key: str, optional
    :param base_url: Custom API endpoint URL, required for Ollama and CBORG
    :type base_url: str, optional
    :param timeout: Request timeout in seconds, defaults to provider configuration
    :type timeout: float, optional
    :param max_tokens: Maximum tokens for generation, defaults to 100000
    :type max_tokens: int
    :raises ValueError: If required provider, model_id, api_key, or base_url are missing
    :raises ValueError: If provider is not supported
    :return: Configured model instance ready for PydanticAI agent integration
    :rtype: Union[OpenAIModel, AnthropicModel, GeminiModel]

    .. note::
       HTTP proxy configuration is automatically detected from the HTTP_PROXY
       environment variable for supported providers. Timeout and connection
       pooling are managed through shared HTTP clients when proxies are enabled.

    .. warning::
       API keys and base URLs are validated before model creation. Ensure proper
       configuration is available through the config system or direct
       parameter specification.

    Examples:
        Basic model creation with direct parameters::

            >>> from osprey.models import get_model
            >>> model = get_model(
            ...     provider="anthropic",
            ...     model_id="claude-3-sonnet-20240229",
            ...     api_key="your-api-key"
            ... )
            >>> # Use with PydanticAI Agent
            >>> agent = Agent(model=model, output_type=YourModel)

        Using configuration dictionary from YAML::

            >>> model_config = {
            ...     "provider": "cborg",
            ...     "model_id": "anthropic/claude-sonnet",
            ...     "max_tokens": 4096,
            ...     "timeout": 30.0
            ... }
            >>> model = get_model(model_config=model_config)

        Ollama local model setup::

            >>> model = get_model(
            ...     provider="ollama",
            ...     model_id="llama3.1:8b",
            ...     base_url="http://localhost:11434"
            ... )

    .. seealso::
       :func:`~completion.get_chat_completion` : Direct chat completion without structured output
       :func:`configs.config.get_provider_config` : Provider configuration loading
       :class:`pydantic_ai.Agent` : PydanticAI agent that uses these models
       :doc:`/developer-guides/01_understanding-the-framework/02_convention-over-configuration` : Complete model setup guide
    """
    if model_config:
        provider = model_config.get("provider", provider)
        model_id = model_config.get("model_id", model_id)
        max_tokens = model_config.get("max_tokens", max_tokens)
        timeout = model_config.get("timeout", timeout)

    if not provider:
        raise ValueError("Provider must be specified either directly or via model_config")

    # Get provider from registry
    from osprey.registry import get_registry

    registry = get_registry()
    provider_class = registry.get_provider(provider)

    if not provider_class:
        raise ValueError(
            f"Unknown provider: {provider}. Use registry.list_providers() to see available providers."
        )

    # Get provider config
    provider_config = get_provider_config(provider)
    api_key = provider_config.get("api_key", api_key)
    base_url = provider_config.get("base_url", base_url)
    timeout = provider_config.get("timeout", timeout)

    # Validate requirements using provider metadata
    if provider_class.requires_api_key and not api_key:
        raise ValueError(f"API key required for {provider}")
    if provider_class.requires_base_url and not base_url:
        raise ValueError(f"Base URL required for {provider}")
    if provider_class.requires_model_id and not model_id:
        raise ValueError(f"Model ID required for {provider}")

    # Setup HTTP client (proxy + timeout)
    async_http_client: httpx.AsyncClient | None = None
    if provider_class.supports_proxy:
        proxy_url = os.getenv("HTTP_PROXY")
        if proxy_url and _validate_proxy_url(proxy_url):
            async_http_client = httpx.AsyncClient(proxy=proxy_url, timeout=timeout)
        elif timeout:
            async_http_client = httpx.AsyncClient(timeout=timeout)

    # Create model using provider
    provider_instance = provider_class()
    return provider_instance.create_model(
        model_id=model_id,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        http_client=async_http_client,
    )
