"""LangChain Model Factory for LangGraph Integration.

This module provides factory functions to create LangChain BaseChatModel instances
for use with LangGraph's create_react_agent and other LangChain-based workflows.

The factory uses osprey's configuration system (get_provider_config, get_model_config)
to create properly configured LangChain chat models that support:
- Tool calling / function calling (bind_tools)
- Structured output (with_structured_output)
- Streaming (stream, astream)
- Full LangGraph ReAct agent compatibility

Supported Providers
-------------------
Native LangChain providers (require provider-specific packages):
    - **anthropic**: ChatAnthropic (requires langchain-anthropic)
    - **openai**: ChatOpenAI (requires langchain-openai)
    - **google**: ChatGoogleGenerativeAI (requires langchain-google-genai)
    - **ollama**: ChatOllama (requires langchain-ollama)

OpenAI-compatible providers (use ChatOpenAI with custom base_url):
    - **cborg**: LBNL's CBORG proxy (https://api.cborg.lbl.gov)
    - **amsc**: American Science Cloud (https://api.i2-core.american-science-cloud.org)
    - **vllm**: Local vLLM server (OpenAI-compatible API)
    - **stanford**: Stanford AI Playground (https://aiapi-prod.stanford.edu/v1)
    - **argo**: ANL's Argo proxy (https://argo-bridge.cels.anl.gov)

Example:
    >>> from osprey.models import get_langchain_model
    >>> from langgraph.prebuilt import create_react_agent
    >>>
    >>> # Get model using osprey's config
    >>> llm = get_langchain_model(provider="anthropic", model_id="claude-sonnet-4-5-20250929")
    >>>
    >>> # Use with LangGraph ReAct agent
    >>> agent = create_react_agent(llm, tools=[...])

.. seealso::
   :func:`get_chat_completion` : Direct LiteLLM-based completion (non-LangChain)
   :mod:`osprey.utils.config` : Provider configuration management
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from osprey.utils.logger import get_logger

logger = get_logger("langchain")


# =============================================================================
# Retry Budget
# =============================================================================

# Cumulative sleep tracked per request (keyed by id(options) which is stable
# across retries within a single request() call).
_retry_cumulative: dict[int, float] = {}
_RETRY_BUDGET = 5.0


def _patch_openai_retry_budget(budget: float = _RETRY_BUDGET) -> None:
    """Cap total retry wait time per request to ``budget`` seconds.

    Two-layer approach:
    1. ``_should_retry``: skip retries where the Retry-After header alone
       exceeds the budget (e.g. CBORG 429 with Retry-After: 60 → skip immediately).
    2. ``_calculate_retry_timeout``: track cumulative sleep and cap the
       remaining wait so the total never exceeds ``budget``.

    This preserves fast recovery for transient 5xx errors (0.5-1s backoff)
    while preventing long blocking on rate limits or other time-costly retries.
    """
    try:
        from openai._base_client import AsyncAPIClient, SyncAPIClient
    except ImportError:
        return

    orig_should_retry = SyncAPIClient._should_retry
    orig_calc_timeout = SyncAPIClient._calculate_retry_timeout

    def _budgeted_should_retry(self, response):
        if not orig_should_retry(self, response):
            return False
        # If Retry-After alone exceeds budget, skip retry entirely
        retry_after = self._parse_retry_after_header(response.headers)
        if retry_after is not None and retry_after > budget:
            return False
        return True

    def _budgeted_calc_timeout(self, remaining_retries, options, response_headers=None):
        timeout = orig_calc_timeout(self, remaining_retries, options, response_headers)
        key = id(options)  # Same input_options object across retries
        cumulative = _retry_cumulative.get(key, 0.0)
        remaining = max(0, budget - cumulative)
        if remaining <= 0:
            _retry_cumulative.pop(key, None)
            return 0  # Budget exhausted
        capped = min(timeout, remaining)
        _retry_cumulative[key] = cumulative + capped
        # Prevent unbounded dict growth
        if len(_retry_cumulative) > 100:
            _retry_cumulative.clear()
        return capped

    SyncAPIClient._should_retry = _budgeted_should_retry
    AsyncAPIClient._should_retry = _budgeted_should_retry
    SyncAPIClient._calculate_retry_timeout = _budgeted_calc_timeout
    AsyncAPIClient._calculate_retry_timeout = _budgeted_calc_timeout


_patch_openai_retry_budget()


# =============================================================================
# Provider Configuration
# =============================================================================

# Native LangChain providers with their import paths
_NATIVE_PROVIDERS = {
    "anthropic": {
        "module": "langchain_anthropic",
        "class": "ChatAnthropic",
        "api_key_param": "anthropic_api_key",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "openai": {
        "module": "langchain_openai",
        "class": "ChatOpenAI",
        "api_key_param": "api_key",
        "api_key_env": "OPENAI_API_KEY",
    },
    "google": {
        "module": "langchain_google_genai",
        "class": "ChatGoogleGenerativeAI",
        "api_key_param": "google_api_key",
        "api_key_env": "GOOGLE_API_KEY",
    },
    "ollama": {
        "module": "langchain_ollama",
        "class": "ChatOllama",
        "api_key_param": None,  # Ollama doesn't need API key
        "api_key_env": None,
        "default_base_url": "http://localhost:11434",
        "base_url_env": "OLLAMA_HOST",
    },
}

# OpenAI-compatible providers (use ChatOpenAI with custom base_url)
_OPENAI_COMPATIBLE_PROVIDERS = {
    "cborg": {
        "default_base_url": "https://api.cborg.lbl.gov",
        "api_key_env": "CBORG_API_KEY",
        "description": "LBNL CBORG proxy",
    },
    "vllm": {
        "default_base_url": "http://localhost:8000/v1",
        "api_key_env": "VLLM_API_KEY",
        "api_key_default": "EMPTY",  # vLLM often doesn't require API key
        "base_url_env": "VLLM_BASE_URL",
        "description": "Local vLLM server",
    },
    "stanford": {
        "default_base_url": "https://aiapi-prod.stanford.edu/v1",
        "api_key_env": "STANFORD_API_KEY",
        "description": "Stanford AI Playground",
    },
    "argo": {
        "default_base_url": "https://argo-bridge.cels.anl.gov",
        "api_key_env": "ARGO_API_KEY",
        "api_key_from_user": True,  # Uses $USER as API key if not specified
        "base_url_env": "ARGO_BASE_URL",
        "description": "ANL Argo proxy",
    },
    "asksage": {
        "default_base_url": "https://api.civ.asksage.ai/server/v1",
        "api_key_env": "ASKSAGE_API_KEY",
        "description": "AskSage proxy",
        "model_kwargs": {
            "system_prompt": "-",
            "dataset": "none",
            "live": 0,
            "limit_references": 0,
        },
    },
    "amsc": {
        "default_base_url": "https://api.i2-core.american-science-cloud.org/v1",
        "api_key_env": "AMSC_API_KEY",
        "description": "American Science Cloud proxy",
    },
}

# Combined list of all supported providers
SUPPORTED_PROVIDERS = list(_NATIVE_PROVIDERS.keys()) + list(_OPENAI_COMPATIBLE_PROVIDERS.keys())


# =============================================================================
# Helper Functions
# =============================================================================


def _import_chat_class(module_name: str, class_name: str) -> type:
    """Dynamically import a LangChain chat class.

    :param module_name: Python module to import (e.g., 'langchain_anthropic')
    :param class_name: Class name to get from module (e.g., 'ChatAnthropic')
    :raises ImportError: If the required package is not installed
    :return: The LangChain chat model class
    """
    try:
        import importlib

        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except ImportError as e:
        install_pkg = module_name.replace("_", "-")
        raise ImportError(
            f"LangChain integration requires '{install_pkg}'. "
            f"Install it with: pip install {install_pkg}"
        ) from e


def _get_env_or_default(env_var: str | None, default: str | None = None) -> str | None:
    """Get environment variable value or return default."""
    if env_var:
        return os.environ.get(env_var, default)
    return default


# =============================================================================
# Main Factory Function
# =============================================================================


def get_langchain_model(
    provider: str | None = None,
    model_id: str | None = None,
    model_config: dict[str, Any] | None = None,
    provider_config: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a LangChain BaseChatModel instance for use with LangGraph.

    This factory function creates properly configured LangChain chat models
    that are fully compatible with LangGraph's create_react_agent and other
    LangChain-based workflows. It integrates with osprey's configuration
    system for centralized credential and model management.

    :param provider: AI provider name. Supported providers:
        - Native: 'anthropic', 'openai', 'google', 'ollama'
        - OpenAI-compatible: 'cborg', 'amsc', 'vllm', 'stanford', 'argo'
    :param model_id: Specific model identifier (e.g., 'claude-sonnet-4-5-20250929', 'gpt-4o')
    :param model_config: Optional dict with 'provider', 'model_id', 'max_tokens' keys
    :param provider_config: Optional provider config dict with 'api_key', 'base_url', etc.
        If not provided, loaded from osprey's config system.
    :param max_tokens: Maximum tokens for generation (default: 4096)
    :param temperature: Sampling temperature (default: provider-specific)
    :param kwargs: Additional keyword arguments passed to the LangChain model constructor
    :raises ValueError: If provider or model_id cannot be determined
    :raises ImportError: If the required langchain package is not installed
    :return: Configured LangChain BaseChatModel instance

    Examples:
        Basic usage with explicit provider::

            >>> llm = get_langchain_model(
            ...     provider="anthropic",
            ...     model_id="claude-sonnet-4-5-20250929"
            ... )

        Using model_config dict::

            >>> llm = get_langchain_model(model_config={
            ...     "provider": "openai",
            ...     "model_id": "gpt-4o",
            ...     "max_tokens": 8192
            ... })

        OpenAI-compatible provider (CBORG)::

            >>> llm = get_langchain_model(
            ...     provider="cborg",
            ...     model_id="anthropic/claude-sonnet"
            ... )

        Local vLLM server::

            >>> llm = get_langchain_model(
            ...     provider="vllm",
            ...     model_id="meta-llama/Llama-3-8b-instruct"
            ... )

        Use with LangGraph ReAct agent::

            >>> from langgraph.prebuilt import create_react_agent
            >>> from langchain_core.tools import tool
            >>>
            >>> @tool
            ... def search(query: str) -> str:
            ...     '''Search for information.'''
            ...     return f"Results for: {query}"
            >>>
            >>> llm = get_langchain_model(provider="anthropic", model_id="claude-sonnet-4-5-20250929")
            >>> agent = create_react_agent(llm, tools=[search])
    """
    from osprey.utils.config import get_provider_config as _get_provider_config

    # Resolve configuration from model_config dict if provided
    if model_config is not None:
        provider = model_config.get("provider", provider)
        model_id = model_config.get("model_id", model_id)
        if max_tokens is None:
            max_tokens = model_config.get("max_tokens")

    # Validate provider
    if not provider:
        raise ValueError(
            "Provider must be specified either directly or via model_config. "
            f"Supported providers: {', '.join(sorted(SUPPORTED_PROVIDERS))}"
        )

    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Provider '{provider}' not supported for LangChain models. "
            f"Supported providers: {', '.join(sorted(SUPPORTED_PROVIDERS))}"
        )

    # Load provider config from osprey's config system if not provided
    if provider_config is None:
        try:
            provider_config = _get_provider_config(provider)
        except Exception as e:
            logger.debug(f"Could not load provider config for '{provider}': {e}")
            provider_config = {}

    # Get model_id from provider config if not specified
    if not model_id:
        model_id = provider_config.get("default_model_id")
        if not model_id:
            raise ValueError(
                f"Model ID must be specified for provider '{provider}'. "
                f"Either pass model_id directly or configure default_model_id in provider config."
            )

    # Extract configuration values
    api_key = provider_config.get("api_key")
    base_url = provider_config.get("base_url")

    # Set defaults
    if max_tokens is None:
        max_tokens = 4096

    # Build model based on provider type
    if provider in _NATIVE_PROVIDERS:
        return _create_native_model(
            provider=provider,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
    else:
        return _create_openai_compatible_model(
            provider=provider,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )


def _create_native_model(
    provider: str,
    model_id: str,
    api_key: str | None,
    base_url: str | None,
    max_tokens: int,
    temperature: float | None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a native LangChain model (Anthropic, OpenAI, Google, Ollama)."""
    config = _NATIVE_PROVIDERS[provider]

    # Import the chat class
    chat_class = _import_chat_class(config["module"], config["class"])

    # Build constructor arguments
    model_kwargs: dict[str, Any] = {"model": model_id, **kwargs}

    if max_tokens is not None:
        model_kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        model_kwargs["temperature"] = temperature

    # Handle API key
    api_key_param = config.get("api_key_param")
    api_key_env = config.get("api_key_env")

    if api_key_param:  # Provider requires API key
        if api_key:
            model_kwargs[api_key_param] = api_key
        elif api_key_env and api_key_env not in os.environ:
            raise ValueError(
                f"{provider.title()} API key required. Set via provider_config, "
                f"osprey config, or {api_key_env} environment variable."
            )

    # Handle base_url for providers that support it (e.g., Ollama)
    if "default_base_url" in config:
        if base_url:
            model_kwargs["base_url"] = base_url
        else:
            base_url_env = config.get("base_url_env")
            default_url = config["default_base_url"]
            model_kwargs["base_url"] = _get_env_or_default(base_url_env, default_url)

    logger.debug(
        f"Creating LangChain model: provider={provider}, model={model_id}, "
        f"class={chat_class.__name__}"
    )

    return chat_class(**model_kwargs)


def _create_openai_compatible_model(
    provider: str,
    model_id: str,
    api_key: str | None,
    base_url: str | None,
    max_tokens: int,
    temperature: float | None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create an OpenAI-compatible model (CBORG, AMSC, vLLM, Stanford, Argo)."""
    config = _OPENAI_COMPATIBLE_PROVIDERS[provider]

    # Import ChatOpenAI
    chat_class = _import_chat_class("langchain_openai", "ChatOpenAI")

    # Build constructor arguments
    model_kwargs: dict[str, Any] = {"model": model_id, **kwargs}

    if max_tokens is not None:
        model_kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        model_kwargs["temperature"] = temperature

    # Resolve base_url
    if base_url:
        model_kwargs["base_url"] = base_url
    else:
        base_url_env = config.get("base_url_env")
        default_url = config["default_base_url"]
        model_kwargs["base_url"] = _get_env_or_default(base_url_env, default_url)

    # Resolve API key
    if api_key:
        model_kwargs["api_key"] = api_key
    else:
        api_key_env = config.get("api_key_env")
        env_key = _get_env_or_default(api_key_env)

        if env_key:
            model_kwargs["api_key"] = env_key
        elif config.get("api_key_from_user"):
            # Argo uses $USER as API key
            model_kwargs["api_key"] = os.environ.get("USER", "user")
        elif config.get("api_key_default"):
            # vLLM default
            model_kwargs["api_key"] = config["api_key_default"]
        else:
            raise ValueError(
                f"API key required for '{provider}' provider. "
                f"Set via provider_config, osprey config, or {api_key_env} environment variable."
            )

    if config.get("model_kwargs"):
        model_kwargs["model_kwargs"] = config["model_kwargs"]

    logger.debug(
        f"Creating OpenAI-compatible LangChain model: provider={provider}, "
        f"model={model_id}, base_url={model_kwargs.get('base_url')}"
    )

    return chat_class(**model_kwargs)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_langchain_model_from_name(
    model_name: str,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a LangChain model from an osprey model name.

    This is a convenience function that looks up a model by name in osprey's
    configuration and creates the corresponding LangChain model.

    :param model_name: Model name as defined in osprey's config.yml (e.g., 'default', 'fast')
    :param kwargs: Additional arguments passed to get_langchain_model
    :raises ValueError: If model_name is not found in configuration
    :return: Configured LangChain BaseChatModel instance

    Example:
        >>> # Assuming config.yml has:
        >>> # models:
        >>> #   default:
        >>> #     provider: anthropic
        >>> #     model_id: claude-sonnet-4-5-20250929
        >>> llm = get_langchain_model_from_name("default")
    """
    from osprey.utils.config import get_model_config as _get_model_config

    try:
        model_config = _get_model_config(model_name)
    except Exception as e:
        raise ValueError(
            f"Model '{model_name}' not found in configuration. "
            f"Available models are defined in config.yml under the 'models' section."
        ) from e

    return get_langchain_model(model_config=model_config, **kwargs)


def list_supported_providers() -> dict[str, str]:
    """List all supported providers with their descriptions.

    :return: Dict mapping provider name to description

    Example:
        >>> providers = list_supported_providers()
        >>> for name, desc in providers.items():
        ...     print(f"{name}: {desc}")
    """
    providers = {}

    for name in _NATIVE_PROVIDERS:
        config = _NATIVE_PROVIDERS[name]
        providers[name] = f"Native LangChain ({config['class']})"

    for name, config in _OPENAI_COMPATIBLE_PROVIDERS.items():
        providers[name] = f"OpenAI-compatible: {config['description']}"

    return providers
