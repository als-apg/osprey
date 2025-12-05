"""Ollama Provider Adapter Implementation."""

import logging
from typing import Any

import httpx
import ollama
import openai
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider as PydanticOpenAIProvider

from .base import BaseProvider

logger = logging.getLogger(__name__)


class OllamaProviderAdapter(BaseProvider):
    """Ollama local model provider implementation."""

    # Metadata (single source of truth)
    name = "ollama"
    description = "Ollama (local models)"
    requires_api_key = False
    requires_base_url = True
    requires_model_id = True
    supports_proxy = False
    default_base_url = "http://localhost:11434"
    default_model_id = "mistral:7b"  # Mistral 7B as recommended default
    health_check_model_id = "mistral:7b"  # Same for health check (local, no cost)
    available_models = ["mistral:7b", "gpt-oss:20b", "gpt-oss:120b"]

    # API key acquisition information
    api_key_url = None
    api_key_instructions = []
    api_key_note = "Ollama runs locally and does not require an API key"

    @staticmethod
    def _get_fallback_urls(base_url: str) -> list[str]:
        """Generate fallback URLs for Ollama based on the current base URL."""
        fallback_urls = []

        if "host.containers.internal" in base_url:
            # Running in container but Ollama might be on localhost
            fallback_urls = [
                base_url.replace("host.containers.internal", "localhost"),
                "http://localhost:11434",
            ]
        elif "localhost" in base_url:
            # Running locally but Ollama might be in container context
            fallback_urls = [
                base_url.replace("localhost", "host.containers.internal"),
                "http://host.containers.internal:11434",
            ]
        else:
            # Generic fallbacks for other scenarios
            fallback_urls = ["http://localhost:11434", "http://host.containers.internal:11434"]

        return fallback_urls

    @staticmethod
    def _test_connection(base_url: str) -> bool:
        """Test if Ollama is accessible at the given URL."""
        try:
            import requests

            test_url = base_url.rstrip("/") + "/v1/models"
            response = requests.get(test_url, timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def create_model(
        self,
        model_id: str,
        api_key: str | None,
        base_url: str | None,
        timeout: float | None,
        http_client: httpx.AsyncClient | None,
    ) -> OpenAIModel:
        """Create Ollama model instance with fallback support."""
        effective_base_url = base_url
        if not base_url.endswith("/v1"):
            effective_base_url = base_url.rstrip("/") + "/v1"

        # Test primary URL first
        if self._test_connection(base_url):
            logger.debug(f"Successfully connected to Ollama at {base_url}")
        else:
            logger.debug(f"Failed to connect to Ollama at {base_url}")

            # Try fallback URLs
            fallback_urls = self._get_fallback_urls(base_url)
            working_url = None

            for fallback_url in fallback_urls:
                logger.debug(f"Attempting fallback connection to Ollama at {fallback_url}")
                if self._test_connection(fallback_url):
                    working_url = fallback_url
                    logger.warning(
                        f"⚠️  Ollama connection fallback: configured URL '{base_url}' failed, "
                        f"using fallback '{fallback_url}'. Consider updating your configuration "
                        f"for your current execution environment."
                    )
                    break

            if working_url:
                effective_base_url = working_url
                if not working_url.endswith("/v1"):
                    effective_base_url = working_url.rstrip("/") + "/v1"
            else:
                # All connection attempts failed
                raise ValueError(
                    f"Failed to connect to Ollama at configured URL '{base_url}' "
                    f"and all fallback URLs {fallback_urls}. Please ensure Ollama is running "
                    f"and accessible, or update your configuration."
                )

        # Create OpenAI-compatible model
        if http_client:
            client_args = {
                "api_key": api_key or "ollama",  # Ollama doesn't need real key
                "http_client": http_client,
                "base_url": effective_base_url,
            }
            openai_client = openai.AsyncOpenAI(**client_args)
        else:
            effective_timeout = timeout if timeout is not None else 60.0
            client_args = {
                "api_key": api_key or "ollama",
                "timeout": effective_timeout,
                "base_url": effective_base_url,
            }
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
        """Execute Ollama chat completion with fallback support."""
        # Ollama connection with graceful fallback for development workflows
        client = None
        used_fallback = False

        try:
            # First attempt: Use configured base_url
            client = ollama.Client(host=base_url)
            client.list()  # Test connection
            logger.debug(f"Successfully connected to Ollama at {base_url}")
        except Exception as e:
            logger.debug(f"Failed to connect to Ollama at {base_url}: {e}")

            # Determine fallback URLs based on current base_url
            fallback_urls = self._get_fallback_urls(base_url)

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

        # Build request
        chat_messages = [{"role": "user", "content": message}]

        options = {}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        request_args = {
            "model": model_id,
            "messages": chat_messages,
        }
        if options:
            request_args["options"] = options

        if output_format is not None:
            # Instruct Ollama to use the Pydantic model's JSON schema for the output format
            request_args["format"] = output_format.model_json_schema()

        try:
            response = client.chat(**request_args)
        except Exception as e:
            current_url = fallback_urls[0] if used_fallback else base_url
            raise ValueError(
                f"Ollama chat request failed using {current_url}. "
                f"Error: {e}. Please verify the model '{model_id}' is available."
            )

        # Extract content from response
        ollama_content_str = response["message"]["content"]

        # Handle typed dict output flag
        is_typed_dict_output = kwargs.get("is_typed_dict_output", False)

        if output_format is not None:
            # Validate the JSON string from Ollama against the Pydantic model
            result = output_format.model_validate_json(ollama_content_str.strip())
            if is_typed_dict_output and hasattr(result, "model_dump"):
                return result.model_dump()
            return result
        else:
            # If no output_model was specified, return the raw string content
            return ollama_content_str

    def check_health(
        self,
        api_key: str | None,
        base_url: str | None,
        timeout: float = 5.0,
        model_id: str | None = None,
    ) -> tuple[bool, str]:
        """Check Ollama connectivity (no API key needed)."""
        if not base_url:
            return False, "Base URL not configured"

        try:
            if self._test_connection(base_url):
                return True, f"Accessible at {base_url}"
            else:
                return False, f"Not accessible at {base_url}"
        except Exception as e:
            return False, f"Connection test failed: {str(e)[:50]}"
