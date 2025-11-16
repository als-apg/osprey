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
from typing import Optional, Union, Type, get_origin, get_args, get_type_hints
from typing_extensions import TypedDict
from urllib.parse import urlparse
from pydantic import BaseModel, create_model, Field
import anthropic
import openai
import ollama
import httpx
from google import genai
from google.genai import types as genai_types


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
    return hasattr(cls, '__annotations__') and hasattr(cls, '__total__')


def _convert_typed_dict_to_pydantic(typed_dict_cls) -> Type[BaseModel]:
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
    annotations = getattr(typed_dict_cls, '__annotations__', {})
    
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


def get_provider_config(provider_name: str) -> dict:
    """Get provider configuration from environment variables.
    
    Simple configuration helper that loads provider settings from environment
    variables without requiring external configuration files. This is a
    lightweight alternative to the full framework configuration system.
    
    Environment Variables:
        For each provider, the following environment variables are checked:
        - {PROVIDER}_API_KEY: API key for authentication
        - {PROVIDER}_BASE_URL: Base URL for API endpoint
        - {PROVIDER}_MODEL_ID: Default model ID
        - {PROVIDER}_MAX_TOKENS: Default max tokens (optional, defaults to 2048)
    
    Args:
        provider_name: Name of the provider ('anthropic', 'openai', 'cborg', etc.)
    
    Returns:
        Dictionary with provider configuration containing api_key, base_url,
        default_model_id, and max_tokens where available.
    
    Examples:
        >>> # For CBORG provider, set environment variables:
        >>> # CBORG_API_KEY=your_key_here
        >>> # CBORG_BASE_URL=https://api.cborg.lbl.gov
        >>> # CBORG_MODEL_ID=gpt-4
        >>> config = get_provider_config("cborg")
        >>> print(config)
        {'provider': 'cborg', 'api_key': 'your_key_here', 'base_url': '...', ...}
    
    Note:
        This is a simplified configuration approach suitable for standalone
        applications. For complex multi-application frameworks, consider using
        a full configuration management system.
    """
    provider_upper = provider_name.upper()
    
    config = {
        "provider": provider_name,
        "api_key": os.getenv(f"{provider_upper}_API_KEY"),
        "base_url": os.getenv(f"{provider_upper}_BASE_URL"),
        "default_model_id": os.getenv(f"{provider_upper}_MODEL_ID"),
        "max_tokens": int(os.getenv(f"{provider_upper}_MAX_TOKENS", "2048")),
    }
    
    # Remove None values to avoid overriding explicit parameters
    config = {k: v for k, v in config.items() if v is not None}
    
    return config


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
        if parsed.scheme not in ('http', 'https'):
            return False
        if not parsed.netloc:
            return False
        return True
    except Exception:
        return False


def _get_ollama_fallback_urls(base_url: str) -> list[str]:
    """Generate fallback URLs for Ollama based on the current base URL.
    
    This helper function generates appropriate fallback URLs to handle
    common development scenarios where the execution context (container vs local)
    doesn't match the configured Ollama URL.
    
    :param base_url: Current configured Ollama base URL
    :type base_url: str
    :return: List of fallback URLs to try in order
    :rtype: list[str]
    
    .. note::
       Fallback URLs are generated based on common patterns:
       - host.containers.internal -> localhost (container to local)
       - localhost -> host.containers.internal (local to container)
       - Generic fallbacks for other scenarios
    """
    fallback_urls = []
    
    if "host.containers.internal" in base_url:
        # Running in container but Ollama might be on localhost
        fallback_urls = [
            base_url.replace("host.containers.internal", "localhost"),
            "http://localhost:11434"
        ]
    elif "localhost" in base_url:
        # Running locally but Ollama might be in container context
        fallback_urls = [
            base_url.replace("localhost", "host.containers.internal"),
            "http://host.containers.internal:11434"
        ]
    else:
        # Generic fallbacks for other scenarios
        fallback_urls = [
            "http://localhost:11434",
            "http://host.containers.internal:11434"
        ]
    
    return fallback_urls


def get_chat_completion(
    message: str,
    max_tokens: int = 1024,
    model_config: Optional[dict] = None,
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
    budget_tokens: int | None = None,
    enable_thinking: bool = False,
    output_model: Optional[Type[BaseModel]] = None,
    base_url: Optional[str] = None, # currently used only for ollama
    provider_config: Optional[dict] = None,
) -> Union[str, BaseModel, list]:
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
        
            >>> from framework.models import get_chat_completion
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
       :func:`get_provider_config` : Provider configuration loading from environment
       :class:`pydantic.BaseModel` : Base class for structured output models
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
        # Check model_config first for base_url and api_key, then provider_config
        base_url = model_config.get("base_url") or provider_config.get("base_url", base_url)
        api_key = model_config.get("api_key") or provider_config.get("api_key")
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

    # Define provider requirements
    provider_requirements = {
        "google":    {"model_id": True, "api_key": True,  "base_url": False, "use_proxy": True},
        "anthropic": {"model_id": True, "api_key": True,  "base_url": False, "use_proxy": True},
        "openai":    {"model_id": True, "api_key": True,  "base_url": True,  "use_proxy": True},
        "ollama":    {"model_id": True, "api_key": False, "base_url": True,  "use_proxy": False},
        "cborg":     {"model_id": True, "api_key": True,  "base_url": True,  "use_proxy": True},
    }
    
    if provider not in provider_requirements:
        raise ValueError(f"Invalid provider: {provider}. Must be 'anthropic', 'cborg', 'google', 'ollama', or 'openai'.")
    
    requirements = provider_requirements[provider]
    
    # Common validation
    if requirements["model_id"] and not model_id:
        raise ValueError(f"Model ID for {provider} not provided.")
    
    if requirements["api_key"] and not api_key:
        raise ValueError(f"No API key provided for {provider}.")
    
    if requirements["base_url"] and not base_url:
        raise ValueError(f"No base URL provided for {provider}.")

    # Set up HTTP client with proxy if needed
    proxy_url = os.environ.get("HTTP_PROXY")
    should_use_proxy = False
    http_client = None
    
    if requirements["use_proxy"] and proxy_url:
        if _validate_proxy_url(proxy_url):
            should_use_proxy = True
            http_client = httpx.Client(proxy=proxy_url)
        else:
            logger.warning(f"Invalid HTTP_PROXY URL format '{proxy_url}', ignoring proxy configuration")
    
    if not should_use_proxy and requirements["use_proxy"]:
        # Only create client without proxy if no proxy was requested
        http_client = None

    # Provider-specific logic (validation already done above)
    if provider == "anthropic":
        client = anthropic.Anthropic(
            api_key=api_key,
            http_client=http_client,
        )

        request_params = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": message}],
        }

        if enable_thinking and budget_tokens is not None:
            if budget_tokens >= max_tokens:
                raise ValueError("budget_tokens must be less than max_tokens.")
            request_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget_tokens
            }
        
        message_response = client.messages.create(**request_params)
        
        if enable_thinking and "thinking" in request_params:
            return message_response.content # Returns List[ContentBlock]
        else:
            # Concatenate text from all TextBlock instances
            text_parts = [
                block.text for block in message_response.content 
                if isinstance(block, anthropic.types.TextBlock)
            ]
            return "\n".join(text_parts)
    
    # ----- GEMINI ------
    elif provider == "google":
        client = genai.Client(api_key=api_key)
        
        if not enable_thinking:
            budget_tokens = 0
            
        if budget_tokens >= max_tokens: # Assuming max_tokens is the overall limit
            raise ValueError("budget_tokens must be less than max_tokens.")

        response = client.models.generate_content(
            model=model_id,
            contents=[message], # Use the transformed messages
            config=genai_types.GenerateContentConfig(
                **({"thinking_config": genai_types.ThinkingConfig(thinking_budget=budget_tokens)}),
                max_output_tokens=max_tokens
            )
        )

        return response.text # Returns str

    # ----- OPENAI ------
    elif provider == "openai":
        if enable_thinking or budget_tokens is not None:
            logging.warning("enable_thinking and budget_tokens are not used for OpenAI provider.")

        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )

        if output_model is not None:
            # Use structured outputs with Pydantic model (recommended approach)
            response = client.beta.chat.completions.parse(
                model=model_id,
                messages=[{"role": "user", "content": message}],
                max_tokens=max_tokens,
                response_format=output_model,
            )
            if not response.choices:
                raise ValueError("OpenAI API returned empty choices list")
            result = response.choices[0].message.parsed
            return _handle_output_conversion(result, is_typed_dict_output)
        else:
            # Regular text completion
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": message}],
                max_tokens=max_tokens,
            )
            if not response.choices:
                raise ValueError("OpenAI API returned empty choices list")
            return response.choices[0].message.content

    # ----- OLLAMA ------
    elif provider == "ollama":
        if enable_thinking or budget_tokens is not None:
            # These features are not standard for Ollama's basic chat API
            # You might log a warning or simply ignore them.
            pass
        
        chat_messages = [{'role': 'user', 'content': message}]
        
        options = {}
        if max_tokens is not None: # Default is 1024
             options['num_predict'] = max_tokens
        # Other options like temperature, top_p could be added if needed

        request_args = {
            "model": model_id,
            "messages": chat_messages,
        }
        if options: # Only add options if there are any
            request_args["options"] = options

        if output_model is not None:
            # Instruct Ollama to use the Pydantic model's JSON schema for the output format.
            request_args["format"] = output_model.model_json_schema()
            # The user's prompt ('message') should ideally also guide the model
            # towards generating the desired structured output.
        
        # Ollama connection with graceful fallback for development workflows
        client = None
        used_fallback = False
        
        try:
            # First attempt: Use configured base_url
            client = ollama.Client(host=base_url)
            # Test connection with a simple health check
            client.list()  # This will fail if Ollama is not accessible
            logger.debug(f"Successfully connected to Ollama at {base_url}")
        except Exception as e:
            logger.debug(f"Failed to connect to Ollama at {base_url}: {e}")
            
            # Determine fallback URLs based on current base_url
            fallback_urls = _get_ollama_fallback_urls(base_url)
            
            # Try fallback URLs
            for fallback_url in fallback_urls:
                try:
                    logger.debug(f"Attempting fallback connection to Ollama at {fallback_url}")
                    client = ollama.Client(host=fallback_url)
                    client.list()  # Test connection
                    used_fallback = True
                    logger.warning(
                        f"⚠️  Ollama connection fallback: configured URL '{base_url}' failed, "
                        f"using fallback '{fallback_url}'. Consider updating your configuration "
                        f"for your current execution environment."
                    )
                    break
                except Exception as fallback_e:
                    logger.debug(f"Fallback attempt failed for {fallback_url}: {fallback_e}")
                    continue
            
            if client is None:
                # All connection attempts failed
                raise ValueError(
                    f"Failed to connect to Ollama at configured URL '{base_url}' "
                    f"and all fallback URLs {fallback_urls}. Please ensure Ollama is running "
                    f"and accessible, or update your configuration."
                )
        
        try:
            response = client.chat(**request_args)
        except Exception as e:
            # Provide helpful error context
            current_url = fallback_urls[0] if used_fallback else base_url
            raise ValueError(
                f"Ollama chat request failed using {current_url}. "
                f"Error: {e}. Please verify the model '{model_id}' is available."
            )
        
        # response is a dict, e.g.:
        # {'model': 'llama3.1', 'created_at': ..., 
        #  'message': {'role': 'assistant', 'content': '...'}, ...}
        ollama_content_str = response['message']['content']

        if output_model is not None:
            # Validate the JSON string from Ollama against the Pydantic model
            result = output_model.model_validate_json(ollama_content_str.strip())
            return _handle_output_conversion(result, is_typed_dict_output)
        else:
            # If no output_model was specified, return the raw string content
            return ollama_content_str

    # ----- CBORG ------
    elif provider == "cborg":
        if enable_thinking or budget_tokens is not None:
            logging.warning("enable_thinking and budget_tokens are not used for CBORG provider.")

        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )

        if output_model is not None:
            # Use structured outputs with Pydantic model (same as OpenAI implementation)
            response = client.beta.chat.completions.parse(
                model=model_id,
                messages=[{"role": "user", "content": message}],
                max_tokens=max_tokens,
                response_format=output_model,
            )
            if not response.choices:
                raise ValueError("CBORG API returned empty choices list")
            result = response.choices[0].message.parsed
            return _handle_output_conversion(result, is_typed_dict_output)
        else:
            # Regular text completion
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": message}],
                max_tokens=max_tokens,
            )
            if not response.choices:
                raise ValueError("CBORG API returned empty choices list")
            return response.choices[0].message.content 