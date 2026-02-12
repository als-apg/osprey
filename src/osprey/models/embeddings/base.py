"""Base embedding provider interface.

This module defines the abstract base class for ARIEL embedding providers.

See 01_DATA_LAYER.md Section 6.3.1 for specification.
"""

from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding model providers.

    Follows Osprey's BaseProvider pattern. Metadata is declared as class
    attributes (single source of truth). The registry introspects these
    attributes after loading the class.

    Metadata Attributes (define on subclass):
        name: Provider identifier (e.g., "ollama")
        description: User-friendly description
        requires_api_key: Whether provider requires API key
        requires_base_url: Whether provider requires custom base URL
        requires_model_id: Whether provider requires explicit model ID
        supports_proxy: Whether provider supports proxy configuration
        default_base_url: Default API endpoint URL if applicable
        default_model_id: Default model recommended for general use
        health_check_model_id: Model ID to use for health checks
        available_models: List of available model IDs for this provider

    LiteLLM Integration Attributes:
        litellm_prefix: LiteLLM provider prefix (e.g., "ollama")
        is_openai_compatible: True if uses OpenAI-compatible API endpoint
    """

    # === METADATA (class attributes, single source of truth) ===
    name: str
    description: str
    requires_api_key: bool
    requires_base_url: bool
    requires_model_id: bool
    supports_proxy: bool

    default_base_url: str | None = None
    default_model_id: str | None = None
    health_check_model_id: str | None = None
    available_models: list[str] = []

    # LiteLLM integration
    litellm_prefix: str | None = None
    is_openai_compatible: bool = False

    @abstractmethod
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
        """Generate embeddings for input texts.

        Args:
            texts: List of texts to embed (always list, even for single text)
            model_id: Model identifier (e.g., "nomic-embed-text")
            api_key: API key (if required by provider)
            base_url: Base URL (if required by provider)
            dimensions: Optional output dimensions (if model supports truncation)
            timeout: Request timeout in seconds (default: 600.0)

        Returns:
            List of embedding vectors, one per input text.
            Each vector is a list of floats with length = model dimension.

        Raises:
            ValueError: Missing required parameters or invalid input
            RuntimeError: API/network failures (retriable)
        """

    @abstractmethod
    def check_health(
        self,
        api_key: str | None,
        base_url: str | None,
        model_id: str | None = None,
        timeout: float = 10.0,
    ) -> tuple[bool, str]:
        """Check if provider is healthy and accessible.

        Args:
            api_key: API key (if required by provider)
            base_url: Base URL (if required by provider)
            model_id: Model to verify. If None, uses health_check_model_id.
            timeout: Request timeout in seconds (default: 10.0)

        Returns:
            Tuple of (healthy, message) where healthy is True if provider
            is accessible and model is available.
        """

    def embed_text(
        self,
        text: str,
        model_id: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ) -> list[float]:
        """Convenience method to embed a single text.

        Args:
            text: Text to embed
            model_id: Model identifier (defaults to default_model_id)
            base_url: Base URL (defaults to default_base_url)
            **kwargs: Additional arguments passed to execute_embedding

        Returns:
            Single embedding vector as list of floats.
        """
        model = model_id or self.default_model_id
        if not model:
            raise ValueError(f"model_id required for provider {self.name}")

        url = base_url or self.default_base_url

        embeddings = self.execute_embedding(
            texts=[text],
            model_id=model,
            base_url=url,
            **kwargs,
        )
        return embeddings[0]

    def embed_texts(
        self,
        texts: list[str],
        model_id: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ) -> list[list[float]]:
        """Convenience method to embed multiple texts.

        Args:
            texts: List of texts to embed
            model_id: Model identifier (defaults to default_model_id)
            base_url: Base URL (defaults to default_base_url)
            **kwargs: Additional arguments passed to execute_embedding

        Returns:
            List of embedding vectors.
        """
        model = model_id or self.default_model_id
        if not model:
            raise ValueError(f"model_id required for provider {self.name}")

        url = base_url or self.default_base_url

        return self.execute_embedding(
            texts=texts,
            model_id=model,
            base_url=url,
            **kwargs,
        )
