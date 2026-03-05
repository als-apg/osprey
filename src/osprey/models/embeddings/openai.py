"""OpenAI embedding provider implementation.

This module provides the OpenAIEmbeddingProvider class for generating
embeddings using OpenAI's API via LiteLLM.
"""

from osprey.models.embeddings.base import BaseEmbeddingProvider
from osprey.utils.logger import get_logger

logger = get_logger("openai_embeddings")


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding model provider.

    Uses LiteLLM as the backend for unified API access to OpenAI's
    embedding models (text-embedding-3-small, text-embedding-3-large, etc.).
    """

    # Metadata (single source of truth)
    name = "openai"
    description = "OpenAI (cloud embedding models)"
    requires_api_key = True
    requires_base_url = False
    requires_model_id = True
    supports_proxy = True
    default_base_url = None
    default_model_id = "text-embedding-3-small"
    health_check_model_id = "text-embedding-3-small"
    available_models = [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ]

    # LiteLLM integration — OpenAI models use no prefix
    litellm_prefix = ""
    is_openai_compatible = True

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
        """Generate embeddings using OpenAI via LiteLLM.

        Args:
            texts: List of texts to embed
            model_id: Model identifier (e.g., "text-embedding-3-small")
            api_key: OpenAI API key
            base_url: Optional custom base URL (for Azure OpenAI, etc.)
            dimensions: Optional output dimensions (supported by v3 models)
            timeout: Request timeout in seconds

        Returns:
            List of embedding vectors.

        Raises:
            ValueError: Invalid input
            RuntimeError: API/network failures
        """
        if not texts:
            return []

        try:
            import litellm

            # OpenAI models use model_id directly (no prefix)
            embed_kwargs: dict = {
                "model": model_id,
                "input": texts,
                "timeout": timeout,
            }

            if api_key:
                embed_kwargs["api_key"] = api_key

            if base_url:
                embed_kwargs["api_base"] = base_url

            if dimensions is not None:
                embed_kwargs["dimensions"] = dimensions

            response = litellm.embedding(**embed_kwargs)

            embeddings = [item["embedding"] for item in response.data]
            return embeddings

        except ImportError as e:
            raise RuntimeError(
                "litellm is required for OpenAI embedding support. "
                "Install with: pip install litellm"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings with OpenAI: {e}") from e

    def check_health(
        self,
        api_key: str | None,
        base_url: str | None,
        model_id: str | None = None,
        timeout: float = 10.0,
    ) -> tuple[bool, str]:
        """Check if OpenAI embedding API is accessible.

        Args:
            api_key: OpenAI API key (required)
            base_url: Optional custom base URL
            model_id: Model to verify (defaults to health_check_model_id)
            timeout: Request timeout in seconds

        Returns:
            Tuple of (healthy, message).
        """
        if not api_key:
            return (False, "OpenAI API key is required")

        model = model_id or self.health_check_model_id or self.default_model_id
        if not model:
            return (False, "No model specified for health check")

        try:
            self.execute_embedding(
                texts=["health check"],
                model_id=model,
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
            )
            return (True, f"OpenAI embedding API healthy (model: {model})")
        except Exception as e:
            return (False, f"OpenAI embedding health check failed: {e}")
