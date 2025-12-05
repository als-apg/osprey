"""Direct Chat Completion Interface for Multi-Provider LLM Access.

Provides immediate access to LLM model inference across multiple providers without the
overhead of structured generation frameworks. This module handles direct chat completion
requests with support for advanced features like extended thinking, structured outputs,
and automatic TypedDict to Pydantic model conversion.

Key capabilities include:
- Direct inference access for simple use cases
- Extended thinking support for Anthropic and Google models
- Structured output generation with Pydantic models or TypedDict
- Automatic TypedDict to Pydantic conversion for seamless integration
- HTTP proxy support for enterprise environments
- Provider-specific optimization and error handling

.. note::
   This module is optimized for direct inference and simple integration scenarios.
   For complex structured generation workflows with agents, consider using
   :func:`~factory.get_model` with PydanticAI agents instead.

.. seealso::
   :func:`get_chat_completion` : Main chat completion interface
   :func:`~factory.get_model` : Model factory for structured generation
   :mod:`configs.config` : Provider configuration management
"""

import logging
import os
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field, create_model

from osprey.utils.config import get_provider_config


def _is_typed_dict(cls) -> bool:
    """Check if a class is a TypedDict by examining its attributes.

    TypedDict classes have specific attributes that distinguish them from regular
    classes and Pydantic models. This function performs a lightweight check
    without importing typing_extensions unnecessarily.

    :param cls: Class to check for TypedDict characteristics
    :type cls: type
    :return: True if the class appears to be a TypedDict, False otherwise
    :rtype: bool

    .. note::
       This check is based on the presence of __annotations__ and __total__
       attributes which are characteristic of TypedDict classes.
    """
    return hasattr(cls, "__annotations__") and hasattr(cls, "__total__")


def _convert_typed_dict_to_pydantic(typed_dict_cls) -> type[BaseModel]:
    """Convert a TypedDict class to a dynamically created Pydantic BaseModel.

    This function enables seamless integration between TypedDict-based type hints
    and Pydantic-based structured output generation. It preserves field names and
    types while adding Pydantic validation and serialization capabilities.

    The conversion process:
    1. Extracts field annotations from the TypedDict
    2. Creates Pydantic field definitions with descriptions
    3. Dynamically generates a new BaseModel class
    4. Preserves type information for validation

    :param typed_dict_cls: TypedDict class to convert to Pydantic model
    :type typed_dict_cls: type
    :raises ValueError: If the provided class is not a valid TypedDict
    :return: Dynamically created Pydantic BaseModel with equivalent structure
    :rtype: Type[BaseModel]

    .. note::
       All fields in the generated Pydantic model include basic descriptions.
       The original TypedDict class name is preserved with a "Pydantic" suffix.

    .. seealso::
       :func:`_is_typed_dict` : TypedDict detection utility
       :func:`_handle_output_conversion` : Convert results back to dict format
    """
    if not _is_typed_dict(typed_dict_cls):
        raise ValueError(f"Expected TypedDict, got {type(typed_dict_cls)}")

    # Get the annotations from the TypedDict
    annotations = getattr(typed_dict_cls, "__annotations__", {})

    # Convert to Pydantic field definitions
    field_definitions = {}
    for field_name, field_type in annotations.items():
        # Create Pydantic fields - all optional for TypedDict compatibility
        field_definitions[field_name] = (field_type, Field(description=f"Field {field_name}"))

    # Create the Pydantic model dynamically
    model_name = f"{typed_dict_cls.__name__}Pydantic"
    pydantic_model = create_model(model_name, **field_definitions)

    return pydantic_model


def _handle_output_conversion(result, is_typed_dict_output: bool):
    """Convert Pydantic model results back to dictionary format when appropriate.

    This function handles the final step of TypedDict integration by converting
    Pydantic model instances back to plain dictionaries when the original
    output_model parameter was a TypedDict. This maintains API consistency
    and expected return types for users.

    :param result: Model inference result, potentially a Pydantic model instance
    :type result: Any
    :param is_typed_dict_output: Whether original output_model was a TypedDict
    :type is_typed_dict_output: bool
    :return: Result converted to dict if needed, otherwise unchanged
    :rtype: Any

    .. note::
       Only Pydantic BaseModel instances are converted to dictionaries.
       Other result types (strings, lists) are returned unchanged.

    .. seealso::
       :func:`_convert_typed_dict_to_pydantic` : Initial TypedDict conversion
       :meth:`pydantic.BaseModel.model_dump` : Pydantic serialization method
    """
    if is_typed_dict_output and isinstance(result, BaseModel):
        return result.model_dump()
    return result


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


def get_chat_completion(
    message: str,
    max_tokens: int = 1024,
    model_config: dict | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    budget_tokens: int | None = None,
    enable_thinking: bool = False,
    output_model: type[BaseModel] | None = None,
    base_url: str | None = None,
    provider_config: dict | None = None,
    temperature: float = 0.0,
) -> str | BaseModel | list:
    """Execute direct chat completion requests across multiple AI providers.

    This function provides immediate access to LLM model inference with support for
    advanced features including extended thinking, structured outputs, and automatic
    TypedDict conversion. It handles provider-specific API differences, credential
    management, and HTTP proxy configuration transparently.

    The function supports multiple interaction patterns:
    - Simple text-to-text completion for basic use cases
    - Structured output generation with Pydantic models or TypedDict
    - Extended thinking workflows for complex reasoning tasks
    - Enterprise proxy and timeout configuration

    Provider-specific features:
    - **Anthropic**: Extended thinking with budget_tokens, content block responses
    - **Google**: Thinking configuration for enhanced reasoning
    - **OpenAI**: Structured outputs with beta chat completions API
    - **Ollama**: Local model inference with JSON schema validation
    - **CBORG**: OpenAI-compatible API with custom endpoints (LBNL-provided service)

    :param message: Input prompt or message for the LLM model
    :type message: str
    :param max_tokens: Maximum tokens to generate in the response
    :type max_tokens: int
    :param model_config: Configuration dictionary with provider and model settings
    :type model_config: dict, optional
    :param provider: AI provider name ('anthropic', 'google', 'openai', 'ollama', 'cborg')
    :type provider: str, optional
    :param model_id: Specific model identifier recognized by the provider
    :type model_id: str, optional
    :param budget_tokens: Thinking budget for Anthropic/Google extended reasoning
    :type budget_tokens: int, optional
    :param enable_thinking: Enable extended thinking capabilities where supported
    :type enable_thinking: bool
    :param output_model: Pydantic model or TypedDict for structured output validation
    :type output_model: Type[BaseModel], optional
    :param base_url: Custom API endpoint, required for Ollama and CBORG providers
    :type base_url: str, optional
    :param provider_config: Optional provider configuration dict with api_key, base_url, etc.
    :type provider_config: dict, optional
    :raises ValueError: If required provider, model_id, api_key, or base_url are missing
    :raises ValueError: If budget_tokens >= max_tokens or other invalid parameter combinations
    :raises pydantic.ValidationError: If output_model validation fails for structured outputs
    :raises anthropic.APIError: For Anthropic API-specific errors
    :raises openai.APIError: For OpenAI API-specific errors
    :raises ollama.ResponseError: For Ollama API-specific errors
    :return: Model response in format determined by provider and output_model settings
    :rtype: Union[str, BaseModel, list]

    .. note::
       Extended thinking is currently supported by Anthropic (with budget_tokens)
       and Google (with thinking_config). Other providers will log warnings if
       thinking parameters are provided.

    .. warning::
       When using structured outputs, ensure your prompt guides the model toward
       generating the expected structure. Not all models handle schema constraints
       equally well.

    Examples:
        Simple text completion::

            >>> from osprey.models import get_chat_completion
            >>> response = get_chat_completion(
            ...     message="Explain quantum computing in simple terms",
            ...     provider="anthropic",
            ...     model_id="claude-3-sonnet-20240229",
            ...     max_tokens=500
            ... )
            >>> print(response)

        Extended thinking with Anthropic::

            >>> response = get_chat_completion(
            ...     message="Solve this complex reasoning problem...",
            ...     provider="anthropic",
            ...     model_id="claude-3-sonnet-20240229",
            ...     enable_thinking=True,
            ...     budget_tokens=1000,
            ...     max_tokens=2000
            ... )
            >>> # Response includes thinking process and final answer

        Structured output with Pydantic model::

            >>> from pydantic import BaseModel
            >>> class AnalysisResult(BaseModel):
            ...     summary: str
            ...     confidence: float
            ...     recommendations: list[str]
            >>>
            >>> result = get_chat_completion(
            ...     message="Analyze this data and provide structured results",
            ...     provider="openai",
            ...     model_id="gpt-4",
            ...     output_model=AnalysisResult
            ... )
            >>> print(f"Confidence: {result.confidence}")

        Using configuration dictionary::

            >>> config = {
            ...     "provider": "ollama",
            ...     "model_id": "llama3.1:8b",
            ...     "max_tokens": 1000
            ... }
            >>> response = get_chat_completion(
            ...     message="Hello, how are you?",
            ...     model_config=config,
            ...     base_url="http://localhost:11434"
            ... )

    .. seealso::
       :func:`~factory.get_model` : Create model instances for PydanticAI agents
       :func:`configs.config.get_provider_config` : Provider configuration loading
       :class:`pydantic.BaseModel` : Base class for structured output models
       :doc:`/developer-guides/01_understanding-the-framework/02_convention-over-configuration` : Complete model configuration and usage guide
    """

    # Handle TypedDict to Pydantic conversion automatically
    original_output_model = output_model
    is_typed_dict_output = False

    if output_model is not None and _is_typed_dict(output_model):
        is_typed_dict_output = True
        output_model = _convert_typed_dict_to_pydantic(output_model)

    # Configuration setup - handle both model_config set and not set cases
    if model_config is not None:
        provider = model_config.get("provider", provider)
        model_id = model_config.get("model_id", model_id)
        max_tokens = model_config.get("max_tokens", max_tokens)
        # Get provider config after provider is determined, but prefer provided provider_config
        if provider_config is None:
            provider_config = get_provider_config(provider) if provider else {}
        base_url = provider_config.get("base_url", base_url)
        api_key = provider_config.get("api_key")
    else:
        # Set defaults when model_config is not provided
        if not provider:
            raise ValueError("Provider must be specified either directly or via model_config")
        # Use provided provider_config or get from global config
        if provider_config is None:
            provider_config = get_provider_config(provider)
        if not model_id:
            model_id = provider_config.get("default_model_id")
        if base_url is None:
            base_url = provider_config.get("base_url")
        api_key = provider_config.get("api_key")

    # Get provider from registry
    from osprey.registry import get_registry

    registry = get_registry()
    provider_class = registry.get_provider(provider)

    if not provider_class:
        raise ValueError(f"Unknown provider: {provider}")

    # Validate requirements using provider metadata
    if provider_class.requires_api_key and not api_key:
        raise ValueError(f"API key required for {provider}")
    if provider_class.requires_base_url and not base_url:
        raise ValueError(f"Base URL required for {provider}")
    if provider_class.requires_model_id and not model_id:
        raise ValueError(f"Model ID required for {provider}")

    # Set up HTTP client with proxy if needed
    http_client = None
    if provider_class.supports_proxy:
        proxy_url = os.environ.get("HTTP_PROXY")
        if proxy_url and _validate_proxy_url(proxy_url):
            http_client = httpx.Client(proxy=proxy_url)
        else:
            if proxy_url:
                logger.warning(
                    f"Invalid HTTP_PROXY URL format '{proxy_url}', ignoring proxy configuration"
                )

    # Execute completion using provider adapter
    provider_instance = provider_class()

    # Build kwargs for provider
    completion_kwargs = {
        "enable_thinking": enable_thinking,
        "budget_tokens": budget_tokens,
        "system_prompt": None,  # Not used in current implementation
        "output_format": output_model,  # Pydantic model (already converted from TypedDict if needed)
        "http_client": http_client,
        "is_typed_dict_output": is_typed_dict_output,
    }

    result = provider_instance.execute_completion(
        message=message,
        model_id=model_id,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        **completion_kwargs,
    )

    # Log API call for transparency and debugging
    from osprey.models.logging import log_api_call

    log_api_call(
        message=message,
        result=result,
        provider=provider,
        model_id=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
        budget_tokens=budget_tokens,
        output_model=output_model,
    )

    # Result is already handled by provider (TypedDict conversion if needed)
    return result
