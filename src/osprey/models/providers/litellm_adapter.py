"""LiteLLM Adapter for Unified Provider Access.

This module provides a unified interface to 100+ LLM providers through LiteLLM,
replacing Osprey's custom provider implementations with a single adapter layer.

Key features:
- Model name mapping using provider-declared attributes (litellm_prefix, is_openai_compatible)
- Structured output detection via LiteLLM's supports_response_schema()
- Extended thinking support via LiteLLM's standardized interface
- HTTP proxy configuration
- Health check utilities

Provider Integration:
    Providers declare their LiteLLM routing behavior via class attributes:
    - litellm_prefix: The LiteLLM prefix (e.g., "anthropic", "gemini")
    - is_openai_compatible: True for OpenAI-compatible endpoints (CBORG, vLLM, etc.)

    This eliminates hardcoded provider checks and allows custom providers to integrate
    without modifying this adapter.
"""

import json
import os
import warnings
from typing import TYPE_CHECKING, Any

import litellm
from pydantic import BaseModel

from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from .base import BaseProvider

# Suppress LiteLLM's verbose logging
litellm.set_verbose = False
litellm.suppress_debug_info = True

# Suppress Pydantic serialization warnings from LiteLLM
# These occur with vLLM/OpenAI-compatible providers that return fewer fields
# than LiteLLM's response models expect (harmless - response is still valid)
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
    module="pydantic.main",
)

logger = get_logger("litellm_adapter")


def get_litellm_model_name(
    provider: str,
    model_id: str,
    base_url: str | None = None,
    provider_class: "type[BaseProvider] | None" = None,
) -> str:
    """Map Osprey provider/model to LiteLLM model string.

    LiteLLM uses a prefix/model format to identify providers:
    - anthropic/claude-sonnet-4 for Anthropic
    - gemini/gemini-2.0-flash for Google
    - ollama/llama3.1:8b for Ollama
    - openai/{model} with api_base for OpenAI-compatible endpoints

    This function reads provider-declared attributes (litellm_prefix, is_openai_compatible)
    to determine the correct routing, eliminating hardcoded provider checks.

    :param provider: Osprey provider name
    :param model_id: Model identifier
    :param base_url: Custom API endpoint URL (for OpenAI-compatible providers)
    :param provider_class: Optional provider class with LiteLLM configuration attributes
    :return: LiteLLM-formatted model string
    """
    # If provider class is available, use its declared attributes
    if provider_class is not None:
        # OpenAI-compatible providers use openai/ prefix with custom api_base
        if getattr(provider_class, "is_openai_compatible", False):
            return f"openai/{model_id}"

        # Use provider-declared LiteLLM prefix
        litellm_prefix = getattr(provider_class, "litellm_prefix", None)
        if litellm_prefix is not None:
            # Empty string means no prefix (like OpenAI native)
            if litellm_prefix == "":
                return model_id
            return f"{litellm_prefix}/{model_id}"

    # Fallback for backwards compatibility when provider_class is not provided
    # This allows the adapter to work even without the provider class
    _fallback_prefixes = {
        "anthropic": "anthropic",
        "google": "gemini",
        "openai": "",  # No prefix for OpenAI
        "ollama": "ollama",
    }
    _openai_compatible = {"cborg", "stanford", "argo", "vllm", "amsc"}

    if provider in _openai_compatible:
        return f"openai/{model_id}"

    prefix = _fallback_prefixes.get(provider)
    if prefix is not None:
        if prefix == "":
            return model_id
        return f"{prefix}/{model_id}"

    # Unknown provider - use provider name as prefix (LiteLLM's default behavior)
    logger.debug(f"Provider '{provider}' using default LiteLLM routing: {provider}/{model_id}")
    return f"{provider}/{model_id}"


def execute_litellm_completion(
    provider: str,
    message: str,
    model_id: str,
    api_key: str | None,
    base_url: str | None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    **kwargs,
) -> str | BaseModel | list:
    """Execute chat completion using LiteLLM.

    This is the core completion function that handles all provider-specific
    details through LiteLLM's unified interface.

    :param provider: Osprey provider name
    :param message: User message
    :param model_id: Model identifier
    :param api_key: API key for authentication
    :param base_url: Custom API endpoint URL
    :param max_tokens: Maximum tokens to generate
    :param temperature: Sampling temperature
    :param kwargs: Additional arguments (enable_thinking, budget_tokens, output_format, etc.)
    :return: Response text, Pydantic model instance, or list of content blocks
    """
    # Pop chat_request and tools from kwargs (passed through from get_chat_completion)
    chat_request = kwargs.pop("chat_request", None)
    tools = kwargs.pop("tools", None)
    tool_choice = kwargs.pop("tool_choice", None)

    # Get LiteLLM model name
    litellm_model = get_litellm_model_name(provider, model_id, base_url)

    # Build completion kwargs
    if chat_request is not None:
        messages = chat_request.to_litellm_messages(provider=provider)
    else:
        messages = [{"role": "user", "content": message}]

    completion_kwargs: dict[str, Any] = {
        "model": litellm_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # Set API key
    if api_key:
        completion_kwargs["api_key"] = api_key

    # Set base URL for OpenAI-compatible providers
    if base_url:
        completion_kwargs["api_base"] = base_url

    # Add tools if provided
    if tools is not None:
        completion_kwargs["tools"] = tools
        completion_kwargs["tool_choice"] = tool_choice if tool_choice is not None else "auto"

    # Handle HTTP proxy from environment
    proxy_url = os.environ.get("HTTP_PROXY")
    if proxy_url:
        # LiteLLM respects standard proxy environment variables
        # No additional configuration needed
        pass

    # Handle extended thinking
    enable_thinking = kwargs.get("enable_thinking", False)
    budget_tokens = kwargs.get("budget_tokens")

    if enable_thinking and budget_tokens is not None:
        if budget_tokens >= max_tokens:
            raise ValueError("budget_tokens must be less than max_tokens")

        if provider == "anthropic":
            # Anthropic extended thinking
            completion_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
        elif provider == "google":
            # Google thinking config - handled differently
            # LiteLLM passes this through to the Google API
            completion_kwargs["thinking_config"] = {"thinking_budget": budget_tokens}

    # Allow retries for transient errors (5xx, connection) with short backoff.
    # The Retry-After cap in langchain.py prevents 60s waits on 429s.
    completion_kwargs.setdefault("num_retries", 2)

    # Handle structured output
    output_format = kwargs.get("output_format")
    is_typed_dict_output = kwargs.get("is_typed_dict_output", False)

    if output_format is not None and tools is not None:
        raise ValueError("Cannot use both 'tools' and 'output_format' simultaneously")

    if output_format is not None:
        return _handle_structured_output(
            provider=provider,
            model_id=model_id,
            litellm_model=litellm_model,
            message=message,
            completion_kwargs=completion_kwargs,
            output_format=output_format,
            is_typed_dict_output=is_typed_dict_output,
            chat_request=chat_request,
        )

    # Ollama: Use direct API to bypass LiteLLM bug #15463 with thinking models
    if provider == "ollama":
        return _execute_ollama_completion(
            model_id=model_id,
            message=message,
            base_url=completion_kwargs.get("api_base", "http://localhost:11434"),
            max_tokens=max_tokens,
        )

    # Regular text completion
    response = litellm.completion(**completion_kwargs)

    # Handle extended thinking response (returns content blocks)
    if enable_thinking and budget_tokens is not None and provider == "anthropic":
        # Return raw content blocks for thinking responses
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice.message, "content") and isinstance(choice.message.content, list):
                return choice.message.content

    # Handle tool call responses
    if hasattr(response, "choices") and response.choices:
        message = response.choices[0].message
        if hasattr(message, "tool_calls") and message.tool_calls:
            return [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

    # Extract text from response
    if hasattr(response, "choices") and response.choices:
        return response.choices[0].message.content or ""

    return ""


def _handle_structured_output(
    provider: str,
    model_id: str,
    litellm_model: str,
    message: str,
    completion_kwargs: dict[str, Any],
    output_format: type[BaseModel],
    is_typed_dict_output: bool,
    chat_request=None,
) -> BaseModel | dict:
    """Handle structured output generation.

    Uses native JSON schema support for providers that support it,
    falls back to prompt-based approach for others.

    :param provider: Provider name
    :param model_id: Model identifier
    :param litellm_model: LiteLLM-formatted model name
    :param message: Original message
    :param completion_kwargs: Base completion kwargs
    :param output_format: Pydantic model for output validation
    :param is_typed_dict_output: Whether to convert result to dict
    :param chat_request: Optional ChatCompletionRequest (preserves multi-turn messages)
    :return: Validated Pydantic model instance or dict
    """
    # Ollama: Use direct API to bypass LiteLLM bug #15463 with thinking models
    if provider == "ollama":
        base_url = completion_kwargs.get("api_base", "http://localhost:11434")
        max_tokens = completion_kwargs.get("max_tokens", 1024)
        return _execute_ollama_structured_output(
            model_id=model_id,
            message=message,
            output_format=output_format,
            base_url=base_url,
            max_tokens=max_tokens,
            is_typed_dict_output=is_typed_dict_output,
        )

    schema = output_format.model_json_schema()

    # Check if model supports native structured outputs using LiteLLM's detection
    supports_native = _supports_native_structured_output(litellm_model, provider)

    if supports_native:
        # Use LiteLLM's native structured output support
        completion_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": output_format.__name__, "schema": schema},
        }
        # Only rebuild messages when chat_request is not providing them
        if chat_request is None:
            completion_kwargs["messages"] = [{"role": "user", "content": message}]

        response = litellm.completion(**completion_kwargs)
        response_text = response.choices[0].message.content or ""
        # Clean response even for native support (some models still return Python-style booleans)
        response_text = _clean_json_response(response_text)
    else:
        # Prompt-based fallback for models without native support
        schema_instruction = (
            f"\n\nYou must respond with valid JSON that matches this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            f"Respond ONLY with the JSON object, no additional text or markdown formatting."
        )

        if chat_request is not None:
            # Append schema instruction to the last user message
            msgs = completion_kwargs["messages"]
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i]["role"] == "user":
                    content = msgs[i]["content"]
                    # Handle content that's already a list (Anthropic cache blocks)
                    if isinstance(content, list):
                        content[-1]["text"] += schema_instruction
                    else:
                        msgs[i]["content"] = content + schema_instruction
                    break
        else:
            structured_message = f"{message}{schema_instruction}"
            completion_kwargs["messages"] = [{"role": "user", "content": structured_message}]

        response = litellm.completion(**completion_kwargs)
        response_text = response.choices[0].message.content or ""

        # Clean up markdown code blocks and fix common JSON issues
        response_text = _clean_json_response(response_text)

    # Parse and validate
    try:
        result = output_format.model_validate_json(response_text)

        if is_typed_dict_output and hasattr(result, "model_dump"):
            return result.model_dump()
        return result
    except Exception as e:
        raise ValueError(
            f"Failed to parse structured output from {provider}: {e}\n"
            f"Response: {response_text[:200]}"
        ) from e


def _supports_native_structured_output(litellm_model: str, provider: str) -> bool:
    """Check if a model supports native structured outputs.

    Uses LiteLLM's built-in supports_response_schema() for detection, with
    fallback handling for OpenAI-compatible providers (CBORG, AMSC, Stanford, ARGO, vLLM)
    that support structured outputs via their proxy but aren't recognized by LiteLLM.

    :param litellm_model: LiteLLM-formatted model string (e.g., "anthropic/claude-sonnet-4")
    :param provider: Osprey provider name
    :return: True if native structured output is supported
    """
    # OpenAI-compatible providers support structured outputs via their API
    # LiteLLM can't detect this since it sees the openai/ prefix, not the actual model
    if provider in ("cborg", "stanford", "argo", "vllm", "amsc"):
        return True

    try:
        return litellm.supports_response_schema(model=litellm_model)
    except Exception:
        # If LiteLLM can't determine support, fall back to False (use prompt-based)
        return False


def _clean_json_response(text: str) -> str:
    """Clean markdown code blocks and fix common JSON issues from LLM response.

    :param text: Raw response text
    :return: Cleaned JSON string
    """
    import re

    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    # Fix Python-style booleans (True/False) to JSON-style (true/false)
    # Only replace when they appear as values (after : or ,) not inside strings
    # Use word boundaries to avoid replacing inside strings
    text = re.sub(r":\s*True\b", ": true", text)
    text = re.sub(r":\s*False\b", ": false", text)
    text = re.sub(r",\s*True\b", ", true", text)
    text = re.sub(r",\s*False\b", ", false", text)

    return text


def _execute_ollama_completion(
    model_id: str,
    message: str,
    base_url: str,
    max_tokens: int,
) -> str:
    """Direct Ollama API call for text completion.

    Bypasses LiteLLM's response handling which has a bug with thinking models
    (LiteLLM issue #15463). The Ollama API correctly returns content even when
    the model includes a 'thinking' field, but LiteLLM fails to extract it.

    :param model_id: Ollama model identifier
    :param message: User message
    :param base_url: Ollama server URL
    :param max_tokens: Maximum tokens to generate
    :return: Response text
    """
    import httpx

    url = f"{base_url.rstrip('/')}/api/chat"

    # Thinking models (like gpt-oss) need extra tokens for the thinking phase
    # Ensure minimum of 100 tokens to avoid truncation during thinking
    effective_max_tokens = max(max_tokens, 100)

    response = httpx.post(
        url,
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": message}],
            "stream": False,
            "options": {"num_predict": effective_max_tokens},
        },
        timeout=120.0,
    )
    response.raise_for_status()

    # Extract content - works correctly even with thinking field present
    data = response.json()
    return data["message"]["content"]


def _execute_ollama_structured_output(
    model_id: str,
    message: str,
    output_format: type[BaseModel],
    base_url: str,
    max_tokens: int,
    is_typed_dict_output: bool = False,
) -> BaseModel | dict:
    """Direct Ollama API call for structured output.

    Bypasses LiteLLM's response handling which has a bug with thinking models
    (LiteLLM issue #15463). The Ollama API correctly returns content even when
    the model includes a 'thinking' field, but LiteLLM fails to extract it.

    :param model_id: Ollama model identifier
    :param message: User message
    :param output_format: Pydantic model for output validation
    :param base_url: Ollama server URL
    :param max_tokens: Maximum tokens to generate
    :param is_typed_dict_output: Whether to convert result to dict
    :return: Validated Pydantic model instance or dict
    """
    import httpx

    schema = output_format.model_json_schema()

    # Build prompt with schema instruction
    structured_message = f"""{message}

You must respond with valid JSON that matches this schema:
{json.dumps(schema, indent=2)}

Respond ONLY with the JSON object, no additional text."""

    # Direct Ollama API call - bypasses LiteLLM's broken response handling
    response = httpx.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": structured_message}],
            "stream": False,
            "format": "json",
            "options": {"num_predict": max_tokens},
        },
        timeout=120.0,
    )
    response.raise_for_status()

    # Extract content - works correctly even with thinking field present
    data = response.json()
    content = data["message"]["content"]

    # Parse and validate
    try:
        result = output_format.model_validate_json(content)
        if is_typed_dict_output and hasattr(result, "model_dump"):
            return result.model_dump()
        return result
    except Exception as e:
        raise ValueError(
            f"Failed to parse structured output from Ollama: {e}\nResponse: {content[:200]}"
        ) from e


def check_litellm_health(
    provider: str,
    api_key: str | None,
    base_url: str | None,
    timeout: float = 5.0,
    model_id: str | None = None,
) -> tuple[bool, str]:
    """Check provider health using LiteLLM.

    Makes a minimal API call to verify connectivity and authentication.

    :param provider: Provider name
    :param api_key: API key
    :param base_url: Custom API endpoint
    :param timeout: Request timeout
    :param model_id: Model to test with
    :return: (success, message) tuple
    """
    if not api_key and provider not in ("ollama",):
        return False, "API key not set"

    # Check for placeholder values
    if api_key and (api_key.startswith("${") or "YOUR_API_KEY" in api_key.upper()):
        return False, "API key not configured (placeholder value detected)"

    if not model_id:
        return False, "Model ID required for health check"

    litellm_model = get_litellm_model_name(provider, model_id, base_url)

    try:
        completion_kwargs: dict[str, Any] = {
            "model": litellm_model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1,
            "timeout": timeout,
        }

        if api_key:
            completion_kwargs["api_key"] = api_key

        if base_url:
            completion_kwargs["api_base"] = base_url

        _ = litellm.completion(**completion_kwargs)
        return True, "API accessible and authenticated"

    except litellm.AuthenticationError:
        return False, "Authentication failed (invalid API key)"
    except litellm.RateLimitError:
        # Rate limited = API key works
        return True, "API key valid (rate limited, but functional)"
    except litellm.NotFoundError:
        return False, f"Model '{model_id}' not found (check model ID)"
    except litellm.BadRequestError as e:
        return False, f"Bad request: {str(e)[:50]}"
    except litellm.Timeout:
        return False, "Request timeout"
    except litellm.APIConnectionError as e:
        return False, f"Connection failed: {str(e)[:50]}"
    except litellm.APIError as e:
        return False, f"API error: {str(e)[:50]}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)[:50]}"
