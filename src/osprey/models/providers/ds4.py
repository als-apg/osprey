"""ds4 Provider Adapter — DwarfStar local DeepSeek V4 inference server.

DwarfStar (https://github.com/antirez/ds4) serves DeepSeek V4 Flash/Pro models
locally behind an OpenAI-compatible API. It is wire-compatible with the vLLM
adapter for plain completions and tool calls.

IMPORTANT: ds4 accepts but does NOT honor ``response_format: json_schema`` — it
returns free-form output that ignores the schema. Therefore this provider sets
``supports_native_structured_output = False`` so OSPREY uses its prompt-based JSON
fallback, which ds4 handles correctly.

Usage:
    Start ds4-server (on the inference host):
        ./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
    Default endpoint: http://127.0.0.1:8000/v1
"""

from typing import Any

from .base import BaseProvider
from .litellm_adapter import _ERR_SNIPPET, check_litellm_health, execute_litellm_completion


class DS4ProviderAdapter(BaseProvider):
    """DwarfStar (ds4) local DeepSeek V4 provider, via LiteLLM's OpenAI adapter."""

    # Metadata (single source of truth)
    name = "ds4"
    description = "DwarfStar local DeepSeek V4 server (OpenAI-compatible)"
    requires_api_key = False  # local server, no auth by default
    requires_base_url = True
    requires_model_id = True
    supports_proxy = True
    default_base_url = "http://127.0.0.1:8000/v1"
    default_model_id = "deepseek-v4-flash"
    health_check_model_id = None  # query the server for available models

    available_models = [
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    ]

    api_key_url = "https://github.com/antirez/ds4"
    api_key_instructions = [
        "ds4 is a local server and requires no API key.",
        "Start ds4-server on the inference host (default port 8000).",
        "Point base_url at the server, e.g. http://127.0.0.1:8000/v1",
        "If reaching it over SSH, forward the port: ssh -L 8000:127.0.0.1:8000 <host>",
    ]
    api_key_note = "No API key required - uses 'EMPTY' placeholder."

    # LiteLLM integration - ds4 is an OpenAI-compatible server.
    is_openai_compatible = True
    # ds4 ignores response_format json_schema -> use OSPREY's prompt-based fallback.
    supports_native_structured_output = False

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
        """Execute a ds4 chat completion via LiteLLM's OpenAI-compatible path."""
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
        """Check ds4 server health, discovering a model if none is given."""
        effective_base_url = base_url or self.default_base_url
        effective_api_key = api_key if api_key else "EMPTY"

        if not model_id:
            try:
                import httpx

                models_url = f"{effective_base_url.rstrip('/').removesuffix('/v1')}/v1/models"
                response = httpx.get(models_url, timeout=timeout)
                if response.status_code == 200:
                    models = response.json().get("data", [])
                    if models:
                        model_id = models[0].get("id")
                    else:
                        return False, "ds4 server running but no models loaded"
                else:
                    return False, f"ds4 server returned {response.status_code}"
            except httpx.ConnectError:
                return False, f"Cannot connect to ds4 server at {effective_base_url}"
            except Exception as e:
                return False, f"Error querying ds4: {str(e)[:_ERR_SNIPPET]}"

        if not model_id:
            return False, "No model available for health check"

        return check_litellm_health(
            provider=self.name,
            api_key=effective_api_key,
            base_url=effective_base_url,
            timeout=timeout,
            model_id=model_id,
        )
