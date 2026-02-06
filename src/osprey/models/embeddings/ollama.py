"""Ollama embedding provider implementation.

This module provides the OllamaEmbeddingProvider class for generating
embeddings using Ollama's local models.

See 01_DATA_LAYER.md Section 6.3.1 for specification.
"""

import logging
import os

from osprey.models.embeddings.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama local embedding model provider.

    Uses LiteLLM as the backend for unified API access with Ollama-specific
    fallback URL logic for development workflows.
    """

    # Metadata (single source of truth)
    name = "ollama"
    description = "Ollama (local embedding models)"
    requires_api_key = False
    requires_base_url = True
    requires_model_id = True
    supports_proxy = False
    default_base_url = "http://localhost:11434"
    default_model_id = "nomic-embed-text"
    health_check_model_id = "nomic-embed-text"
    available_models = ["nomic-embed-text", "mxbai-embed-large", "all-minilm"]

    # LiteLLM integration
    litellm_prefix = "ollama"

    @staticmethod
    def _get_fallback_urls(base_url: str) -> list[str]:
        """Generate fallback URLs for Ollama based on the current base URL.

        Supports multiple container runtimes:
        - host.docker.internal: Docker Desktop (macOS/Windows)
        - host.containers.internal: Podman
        """
        fallback_urls = []

        if "host.containers.internal" in base_url:
            # Running in Podman container but Ollama might be elsewhere
            fallback_urls = [
                base_url.replace("host.containers.internal", "host.docker.internal"),
                base_url.replace("host.containers.internal", "localhost"),
                "http://localhost:11434",
            ]
        elif "host.docker.internal" in base_url:
            # Running in Docker Desktop container but Ollama might be elsewhere
            fallback_urls = [
                base_url.replace("host.docker.internal", "host.containers.internal"),
                base_url.replace("host.docker.internal", "localhost"),
                "http://localhost:11434",
            ]
        elif "localhost" in base_url or "127.0.0.1" in base_url:
            # Running locally but Ollama might be in container context
            fallback_urls = [
                "http://host.docker.internal:11434",
                "http://host.containers.internal:11434",
            ]
        else:
            # Generic fallbacks for other scenarios
            fallback_urls = [
                "http://localhost:11434",
                "http://host.docker.internal:11434",
                "http://host.containers.internal:11434",
            ]

        return fallback_urls

    @staticmethod
    def _test_connection(base_url: str) -> bool:
        """Test if Ollama is accessible at the given URL."""
        try:
            import requests

            test_url = base_url.rstrip("/") + "/api/tags"
            response = requests.get(test_url, timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def _resolve_base_url(self, base_url: str) -> str:
        """Resolve working base URL with fallback support.

        Checks OLLAMA_HOST environment variable first as override,
        then tries the configured URL, then fallbacks.
        """
        # Check environment variable override first (set by docker-compose)
        env_url = os.environ.get("OLLAMA_HOST")
        if env_url:
            if self._test_connection(env_url):
                logger.debug(f"Successfully connected to Ollama via OLLAMA_HOST at {env_url}")
                return env_url
            logger.debug(f"OLLAMA_HOST={env_url} not accessible, trying other options")

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
        raise RuntimeError(
            f"Failed to connect to Ollama at configured URL '{base_url}' "
            f"and all fallback URLs {fallback_urls}. Please ensure Ollama is running "
            f"and accessible, or update your configuration."
        )

    def execute_embedding(
        self,
        texts: list[str],
        model_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        timeout: float = 600.0,
        **kwargs,
    ) -> list[list[float]]:
        """Generate embeddings using Ollama via LiteLLM.

        Args:
            texts: List of texts to embed
            model_id: Model identifier (e.g., "nomic-embed-text")
            api_key: Ignored for Ollama (no API key required)
            base_url: Ollama server URL (defaults to localhost:11434)
            dimensions: Optional output dimensions (not supported by all models)
            timeout: Request timeout in seconds

        Returns:
            List of embedding vectors.

        Raises:
            ValueError: Invalid input
            RuntimeError: API/network failures
        """
        if not texts:
            return []

        url = base_url or self.default_base_url
        if not url:
            raise ValueError("base_url is required for Ollama provider")

        # Resolve working URL with fallback
        resolved_url = self._resolve_base_url(url)

        try:
            import litellm

            # LiteLLM expects model format: "ollama/model_name"
            litellm_model = f"{self.litellm_prefix}/{model_id}"

            # Prepare kwargs for LiteLLM
            embed_kwargs: dict = {
                "model": litellm_model,
                "input": texts,
                "timeout": timeout,
            }

            # Add base URL for local Ollama
            if resolved_url:
                embed_kwargs["api_base"] = resolved_url

            # Add dimensions if specified and supported
            if dimensions is not None:
                embed_kwargs["dimensions"] = dimensions

            # Execute embedding request
            response = litellm.embedding(**embed_kwargs)

            # Extract embeddings from response
            embeddings = [item["embedding"] for item in response.data]

            return embeddings

        except ImportError as e:
            raise RuntimeError(
                "litellm is required for Ollama embedding support. "
                "Install with: pip install litellm"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings with Ollama: {e}") from e

    def check_health(
        self,
        api_key: str | None,
        base_url: str | None,
        model_id: str | None = None,
        timeout: float = 10.0,
    ) -> tuple[bool, str]:
        """Check if Ollama is accessible and model is available.

        Args:
            api_key: Ignored for Ollama
            base_url: Ollama server URL
            model_id: Model to verify (defaults to health_check_model_id)
            timeout: Request timeout in seconds

        Returns:
            Tuple of (healthy, message).
        """
        # Check environment variable override first
        env_url = os.environ.get("OLLAMA_HOST")
        url = env_url or base_url or self.default_base_url
        if not url:
            return (False, "No base_url configured for Ollama provider")

        # Test basic connectivity
        if not self._test_connection(url):
            # Try fallbacks
            fallback_urls = self._get_fallback_urls(url)
            for fallback_url in fallback_urls:
                if self._test_connection(fallback_url):
                    url = fallback_url
                    break
            else:
                return (False, f"Cannot connect to Ollama at {url}")

        # Check if model is available
        model = model_id or self.health_check_model_id or self.default_model_id
        if model:
            try:
                import requests

                models_url = url.rstrip("/") + "/api/tags"
                response = requests.get(models_url, timeout=timeout)
                if response.status_code == 200:
                    data = response.json()
                    available = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
                    model_base = model.split(":")[0]
                    if model_base not in available:
                        return (
                            False,
                            f"Model '{model}' not found. Available: {available}. "
                            f"Run 'ollama pull {model}' to download.",
                        )
            except Exception as e:
                return (False, f"Failed to check model availability: {e}")

        return (True, f"Ollama connected at {url}")
