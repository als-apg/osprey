"""Direct Chat Completion Interface for Multi-Provider LLM Access.

Provides immediate access to LLM model inference across multiple providers via LiteLLM.
This module handles direct chat completion requests with support for advanced features
like extended thinking, structured outputs, and automatic TypedDict to Pydantic
model conversion.

Key capabilities include:
- Direct inference access via LiteLLM (100+ providers)
- Extended thinking support for Anthropic and Google models
- Structured output generation with Pydantic models or TypedDict
- Automatic TypedDict to Pydantic conversion for seamless integration
- HTTP proxy support via standard environment variables

.. seealso::
   :func:`get_chat_completion` : Main chat completion interface
   :mod:`configs.config` : Provider configuration management
"""

import logging

from pydantic import BaseModel, Field, create_model

from osprey.utils.config import get_provider_config

logger = logging.getLogger(__name__)


def _is_typed_dict(cls) -> bool:
    """Check if a class is a TypedDict by examining its attributes.

    :param cls: Class to check for TypedDict characteristics
    :return: True if the class appears to be a TypedDict, False otherwise
    """
    return hasattr(cls, "__annotations__") and hasattr(cls, "__total__")


def _convert_typed_dict_to_pydantic(typed_dict_cls) -> type[BaseModel]:
    """Convert a TypedDict class to a dynamically created Pydantic BaseModel.

    :param typed_dict_cls: TypedDict class to convert to Pydantic model
    :raises ValueError: If the provided class is not a valid TypedDict
    :return: Dynamically created Pydantic BaseModel with equivalent structure
    """
    if not _is_typed_dict(typed_dict_cls):
        raise ValueError(f"Expected TypedDict, got {type(typed_dict_cls)}")

    annotations = getattr(typed_dict_cls, "__annotations__", {})

    field_definitions = {}
    for field_name, field_type in annotations.items():
        field_definitions[field_name] = (field_type, Field(description=f"Field {field_name}"))

    model_name = f"{typed_dict_cls.__name__}Pydantic"
    pydantic_model = create_model(model_name, **field_definitions)

    return pydantic_model


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
    """Execute direct chat completion requests across multiple AI providers via LiteLLM.

    This function provides immediate access to LLM model inference with support for
    advanced features including extended thinking, structured outputs, and automatic
    TypedDict conversion.

    :param message: Input prompt or message for the LLM model
    :param max_tokens: Maximum tokens to generate in the response
    :param model_config: Configuration dictionary with provider and model settings
    :param provider: AI provider name ('anthropic', 'google', 'openai', 'ollama', 'cborg', etc.)
    :param model_id: Specific model identifier recognized by the provider
    :param budget_tokens: Thinking budget for Anthropic/Google extended reasoning
    :param enable_thinking: Enable extended thinking capabilities where supported
    :param output_model: Pydantic model or TypedDict for structured output validation
    :param base_url: Custom API endpoint, required for Ollama and CBORG providers
    :param provider_config: Optional provider configuration dict with api_key, base_url, etc.
    :param temperature: Sampling temperature (0.0-2.0)
    :raises ValueError: If required provider, model_id, api_key, or base_url are missing
    :return: Model response (str, Pydantic model, or list of content blocks for thinking)

    Examples:
        Simple text completion::

            >>> from osprey.models import get_chat_completion
            >>> response = get_chat_completion(
            ...     message="Explain quantum computing",
            ...     provider="anthropic",
            ...     model_id="claude-sonnet-4",
            ... )

        Structured output::

            >>> from pydantic import BaseModel
            >>> class Result(BaseModel):
            ...     summary: str
            ...     confidence: float
            >>>
            >>> result = get_chat_completion(
            ...     message="Analyze this data",
            ...     provider="openai",
            ...     model_id="gpt-4o",
            ...     output_model=Result
            ... )
    """
    # Handle TypedDict to Pydantic conversion
    is_typed_dict_output = False
    if output_model is not None and _is_typed_dict(output_model):
        is_typed_dict_output = True
        output_model = _convert_typed_dict_to_pydantic(output_model)

    # Configuration setup
    if model_config is not None:
        provider = model_config.get("provider", provider)
        model_id = model_config.get("model_id", model_id)
        max_tokens = model_config.get("max_tokens", max_tokens)
        if provider_config is None:
            provider_config = get_provider_config(provider) if provider else {}
        base_url = provider_config.get("base_url", base_url)
        api_key = provider_config.get("api_key")
    else:
        if not provider:
            raise ValueError("Provider must be specified either directly or via model_config")
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

    # Execute completion using provider adapter (LiteLLM handles proxy via env vars)
    provider_instance = provider_class()

    completion_kwargs = {
        "enable_thinking": enable_thinking,
        "budget_tokens": budget_tokens,
        "output_format": output_model,
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

    return result
