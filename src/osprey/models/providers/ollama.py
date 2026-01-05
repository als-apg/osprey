"""Ollama Provider Adapter Implementation.

This provider uses LiteLLM as the backend for unified API access,
while preserving Ollama-specific fallback URL logic for development workflows.
"""

import logging
from typing import Any

from .base import BaseProvider
from .litellm_adapter import execute_litellm_completion

logger = logging.getLogger(__name__)


class OllamaProviderAdapter(BaseProvider):
    """Ollama local model provider implementation using LiteLLM."""

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

    def _resolve_base_url(self, base_url: str) -> str:
        """Resolve working base URL with fallback support."""
        # Test primary URL first
        if self._test_connection(base_url):
            logger.debug(f"Successfully connected to Ollama at {base_url}")
            return base_url

        logger.debug(f"Failed to connect to Ollama at {base_url}")

        # Try fallback URLs
        fallback_urls = self._get_fallback_urls(base_url)
        for fallback_url in fallback_urls:
            logger.debug(f"Attempting fallback connection to Ollama at {fallback_url}")
            if self._test_connection(fallback_url):
                logger.warning(
                    f"Ollama connection fallback: configured URL '{base_url}' failed, "
                    f"using fallback '{fallback_url}'. Consider updating your configuration."
                )
                return fallback_url

        # All connection attempts failed
        raise ValueError(
            f"Failed to connect to Ollama at configured URL '{base_url}' "
            f"and all fallback URLs {fallback_urls}. Please ensure Ollama is running "
            f"and accessible, or update your configuration."
        )

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
        """Execute Ollama chat completion via LiteLLM with fallback support."""
        # Resolve working base URL with fallbacks
        effective_base_url = self._resolve_base_url(base_url)

        return execute_litellm_completion(
            provider=self.name,
            message=message,
            model_id=model_id,
            api_key=api_key or "ollama",  # Ollama doesn't need real key
            base_url=effective_base_url,
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
