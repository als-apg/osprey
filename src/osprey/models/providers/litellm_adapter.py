"""LiteLLM Adapter for Unified Provider Access.

This module provides a unified interface to 100+ LLM providers through LiteLLM,
replacing Osprey's custom provider implementations with a single adapter layer.

Key features:
- Model name mapping from Osprey format to LiteLLM format
- Extended thinking support for Anthropic and Google
- Structured output handling (native and prompt-based fallback)
- HTTP proxy configuration
- Health check utilities
"""

import json
import logging
import os
from typing import Any

import litellm
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def get_litellm_model_name(
    provider: str,
    model_id: str,
    base_url: str | None = None,
) -> str:
    """Map Osprey provider/model to LiteLLM model string.

    LiteLLM uses a prefix/model format to identify providers:
    - anthropic/claude-sonnet-4 for Anthropic
    - gemini/gemini-2.0-flash for Google
    - ollama/llama3.1:8b for Ollama
    - openai/{model} with api_base for OpenAI-compatible endpoints

    :param provider: Osprey provider name
    :param model_id: Model identifier
    :param base_url: Custom API endpoint URL (for OpenAI-compatible providers)
    :return: LiteLLM-formatted model string
    """
    # OpenAI-compatible providers (CBORG, Stanford, ARGO, vLLM)
    # These use the openai/ prefix with a custom api_base
    if provider in ("cborg", "stanford", "argo", "vllm"):
        return f"openai/{model_id}"

    # Native LiteLLM providers
    provider_prefixes = {
        "anthropic": "anthropic",
        "google": "gemini",
        "openai": "openai",
        "ollama": "ollama",
    }

    prefix = provider_prefixes.get(provider)
    if prefix:
        # OpenAI models don't need prefix in LiteLLM
        if provider == "openai":
            return model_id
        return f"{prefix}/{model_id}"

    # Fallback: return as-is
    logger.warning(f"Unknown provider '{provider}', using model_id as-is: {model_id}")
    return model_id


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
    # Get LiteLLM model name
    litellm_model = get_litellm_model_name(provider, model_id, base_url)

    # Build completion kwargs
    completion_kwargs: dict[str, Any] = {
        "model": litellm_model,
        "messages": [{"role": "user", "content": message}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # Set API key
    if api_key:
        completion_kwargs["api_key"] = api_key

    # Set base URL for OpenAI-compatible providers
    if base_url:
        completion_kwargs["api_base"] = base_url

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

    # Handle structured output
    output_format = kwargs.get("output_format")
    is_typed_dict_output = kwargs.get("is_typed_dict_output", False)

    if output_format is not None:
        return _handle_structured_output(
            provider=provider,
            model_id=model_id,
            litellm_model=litellm_model,
            message=message,
            completion_kwargs=completion_kwargs,
            output_format=output_format,
            is_typed_dict_output=is_typed_dict_output,
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

    # Check if model supports native structured outputs
    supports_native = _supports_native_structured_output(provider, model_id)

    if supports_native:
        # Use LiteLLM's native structured output support
        completion_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": output_format.__name__, "schema": schema},
        }
        completion_kwargs["messages"] = [{"role": "user", "content": message}]

        response = litellm.completion(**completion_kwargs)
        response_text = response.choices[0].message.content or ""
    else:
        # Prompt-based fallback for models without native support
        structured_message = f"""{message}

You must respond with valid JSON that matches this schema:
{json.dumps(schema, indent=2)}

Respond ONLY with the JSON object, no additional text or markdown formatting."""

        completion_kwargs["messages"] = [{"role": "user", "content": structured_message}]

        response = litellm.completion(**completion_kwargs)
        response_text = response.choices[0].message.content or ""

        # Clean up markdown code blocks
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


def _supports_native_structured_output(provider: str, model_id: str) -> bool:
    """Check if a model supports native structured outputs.

    :param provider: Provider name
    :param model_id: Model identifier
    :return: True if native structured output is supported
    """
    if provider == "anthropic":
        # Claude 4+ models support native structured outputs
        return "claude-sonnet-4" in model_id or "claude-opus-4" in model_id

    if provider == "openai":
        # GPT-4o and newer support structured outputs
        return "gpt-4o" in model_id or "gpt-4-turbo" in model_id

    # OpenAI-compatible providers (CBORG, Stanford, ARGO) support native structured outputs
    # CBORG proxies to underlying providers and supports structured output for all models
    if provider == "cborg":
        return True  # CBORG supports structured outputs via OpenAI-compatible API

    if provider in ("stanford", "argo"):
        return "gpt-4o" in model_id or "claude-sonnet-4" in model_id or "claude-opus-4" in model_id

    # vLLM supports structured outputs for most models
    if provider == "vllm":
        return True  # vLLM handles json_schema natively

    return False


def _clean_json_response(text: str) -> str:
    """Clean markdown code blocks from JSON response.

    :param text: Raw response text
    :return: Cleaned JSON string
    """
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


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
