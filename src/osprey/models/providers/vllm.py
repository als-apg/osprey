"""vLLM Provider Adapter Implementation.

This provider uses LiteLLM as the backend for unified API access.
vLLM provides high-throughput inference with an OpenAI-compatible API.

vLLM Features:
- OpenAI-compatible API (uses standard OpenAI client)
- High-throughput inference with continuous batching
- Support for reasoning models (DeepSeek R1, etc.)
- Structured outputs via json_schema
- Runs locally or on remote servers

Usage:
    Start vLLM server: vllm serve <model-name>
    Default endpoint: http://localhost:8000/v1
"""

from typing import Any

from .base import BaseProvider
from .litellm_adapter import check_litellm_health, execute_litellm_completion


class VLLMProviderAdapter(BaseProvider):
    """vLLM provider implementation using LiteLLM.

    vLLM serves models with an OpenAI-compatible API, making it easy to integrate
    with existing OpenAI-based workflows. The API key is typically not required
    (can be "EMPTY" or any placeholder value).
    """

    # Metadata (single source of truth)
    name = "vllm"
    description = "vLLM inference server (OpenAI-compatible API)"
    requires_api_key = False  # vLLM doesn't require authentication by default
    requires_base_url = True
    requires_model_id = True
    supports_proxy = True
    default_base_url = "http://localhost:8000/v1"
    default_model_id = None  # Model depends on what's served
    health_check_model_id = None  # Will query the server for available models

    # Available models depend on what's served by the vLLM instance
    # These are common model families that vLLM supports
    available_models = [
        # Example models - actual availability depends on server configuration
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "Qwen/Qwen2.5-7B-Instruct",
    ]

    # API key acquisition information
    api_key_url = "https://docs.vllm.ai/en/latest/"
    api_key_instructions = [
        "vLLM typically doesn't require an API key",
        "Start vLLM server: vllm serve <model-name>",
        "Default endpoint: http://localhost:8000/v1",
        "If authentication is configured, use the key provided by your admin",
    ]
    api_key_note = "API key optional - set VLLM_API_KEY or use 'EMPTY' as placeholder"

    # LiteLLM integration - vLLM is an OpenAI-compatible server
    is_openai_compatible = True

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
        """Execute vLLM chat completion via LiteLLM.

        vLLM uses an OpenAI-compatible API, so we route through LiteLLM's
        OpenAI adapter with a custom base_url.

        :param message: User message
        :param model_id: Model identifier (as served by vLLM)
        :param api_key: Optional API key (can be None or "EMPTY")
        :param base_url: vLLM server URL (default: http://localhost:8000/v1)
        :param max_tokens: Maximum tokens to generate
        :param temperature: Sampling temperature
        :param thinking: Optional thinking configuration for reasoning models
        :param system_prompt: Optional system prompt
        :param output_format: Optional Pydantic model for structured output
        :param kwargs: Additional arguments
        :return: Response text or structured output
        """
        # Use placeholder API key if none provided
        effective_api_key = api_key if api_key else "EMPTY"

        return execute_litellm_completion(
            provider=self.name,
            message=message,
            model_id=model_id,
            api_key=effective_api_key,
            base_url=base_url or self.default_base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            output_format=output_format,
            **kwargs,
        )

    def check_health(
        self,
        api_key: str | None,
        base_url: str | None,
        timeout: float = 5.0,
        model_id: str | None = None,
    ) -> tuple[bool, str]:
        """Check vLLM server health.

        Queries the vLLM server to verify it's running and accessible.
        If no model_id is provided, attempts to get the list of available models.

        :param api_key: Optional API key
        :param base_url: vLLM server URL
        :param timeout: Request timeout
        :param model_id: Optional model ID to test
        :return: (success, message) tuple
        """
        effective_base_url = base_url or self.default_base_url
        effective_api_key = api_key if api_key else "EMPTY"

        # First, try to get the list of models from the server
        if not model_id:
            try:
                import httpx

                models_url = f"{effective_base_url.rstrip('/v1')}/v1/models"
                response = httpx.get(models_url, timeout=timeout)
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("data", [])
                    if models:
                        model_id = models[0].get("id")
                    else:
                        return False, "vLLM server running but no models loaded"
                else:
                    return False, f"vLLM server returned {response.status_code}"
            except httpx.ConnectError:
                return False, f"Cannot connect to vLLM server at {effective_base_url}"
            except Exception as e:
                return False, f"Error querying vLLM: {str(e)[:50]}"

        if not model_id:
            return False, "No model available for health check"

        return check_litellm_health(
            provider=self.name,
            api_key=effective_api_key,
            base_url=effective_base_url,
            timeout=timeout,
            model_id=model_id,
        )
