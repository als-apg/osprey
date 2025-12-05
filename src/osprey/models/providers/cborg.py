"""CBORG Provider Adapter Implementation."""

import logging
from typing import Any

import httpx
import openai
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider as PydanticOpenAIProvider

from .base import BaseProvider

logger = logging.getLogger(__name__)


class CBorgProviderAdapter(BaseProvider):
    """CBORG (LBNL) provider implementation - OpenAI-compatible."""

    # Metadata (single source of truth)
    name = "cborg"
    description = "LBNL CBorg proxy (supports multiple models)"
    requires_api_key = True
    requires_base_url = True
    requires_model_id = True
    supports_proxy = True
    default_base_url = None
    default_model_id = "anthropic/claude-haiku"  # Claude Haiku via CBORG for general use
    health_check_model_id = "anthropic/claude-haiku"  # Fast and cost-effective for health checks
    available_models = [
        "anthropic/claude-sonnet",
        "anthropic/claude-haiku",
        "google/gemini-flash",
        "google/gemini-pro",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
    ]

    # API key acquisition information
    api_key_url = "https://cborg.lbl.gov"
    api_key_instructions = [
        "As a Berkeley Lab employee, go to 'API' -> 'Request API Key'",
        "Create an API key ($50/month per user allocation)",
        "Copy the key provided",
    ]
    api_key_note = "Must have affiliation with Berkeley Lab to request an API key."

    def create_model(
        self,
        model_id: str,
        api_key: str | None,
        base_url: str | None,
        timeout: float | None,
        http_client: httpx.AsyncClient | None,
    ) -> OpenAIModel:
        """Create CBORG model instance for PydanticAI."""
        if http_client:
            client_args = {"api_key": api_key, "http_client": http_client, "base_url": base_url}
            openai_client = openai.AsyncOpenAI(**client_args)
        else:
            effective_timeout = timeout if timeout is not None else 60.0
            client_args = {"api_key": api_key, "timeout": effective_timeout, "base_url": base_url}
            openai_client = openai.AsyncOpenAI(**client_args)

        model = OpenAIModel(
            model_name=model_id,
            provider=PydanticOpenAIProvider(openai_client=openai_client),
        )
        model.model_id = model_id
        return model

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
        """Execute CBORG chat completion."""
        # Check for thinking parameters (not supported by CBORG)
        enable_thinking = kwargs.get("enable_thinking", False)
        budget_tokens = kwargs.get("budget_tokens")

        if enable_thinking or budget_tokens is not None:
            logger.warning("enable_thinking and budget_tokens are not used for CBORG provider.")

        # Get http_client if provided
        http_client = kwargs.get("http_client")

        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )

        # Handle typed dict output flag
        is_typed_dict_output = kwargs.get("is_typed_dict_output", False)

        if output_format is not None:
            # Use structured outputs with Pydantic model
            response = client.beta.chat.completions.parse(
                model=model_id,
                messages=[{"role": "user", "content": message}],
                max_tokens=max_tokens,
                response_format=output_format,
            )
            if not response.choices:
                raise ValueError("CBORG API returned empty choices list")
            result = response.choices[0].message.parsed

            # Handle TypedDict conversion
            if is_typed_dict_output and hasattr(result, "model_dump"):
                return result.model_dump()
            return result
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

    def check_health(
        self,
        api_key: str | None,
        base_url: str | None,
        timeout: float = 5.0,
        model_id: str | None = None,
    ) -> tuple[bool, str]:
        """Check CBORG API health by testing /v1/models endpoint."""
        import requests

        if not api_key:
            return False, "API key not set"

        if not base_url:
            return False, "Base URL not configured"

        try:
            test_url = base_url.rstrip("/") + "/models"
            headers = {"Authorization": f"Bearer {api_key}"}

            response = requests.get(test_url, headers=headers, timeout=timeout)

            if response.status_code == 200:
                return True, "API accessible and authenticated"
            elif response.status_code == 401:
                return False, "Authentication failed (invalid API key?)"
            else:
                return False, f"API returned status {response.status_code}"

        except requests.Timeout:
            return False, "Connection timeout"
        except requests.RequestException as e:
            return False, f"Connection failed: {str(e)[:50]}"
        except Exception as e:
            return False, f"Health check failed: {str(e)[:50]}"
