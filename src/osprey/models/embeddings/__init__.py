"""ARIEL embedding provider interface.

This module provides the base class for embedding providers,
concrete implementations, and a factory function for provider lookup.
"""

from osprey.models.embeddings.base import BaseEmbeddingProvider
from osprey.models.embeddings.ollama import OllamaEmbeddingProvider
from osprey.models.embeddings.openai import OpenAIEmbeddingProvider


def get_embedding_provider(name: str) -> BaseEmbeddingProvider:
    """Look up an embedding provider by name and return an instance.

    Args:
        name: Provider identifier (e.g., "ollama", "openai")

    Returns:
        Instantiated embedding provider.

    Raises:
        ValueError: If the provider name is unknown.
    """
    from osprey.models.embedding_registry import get_embedding_registry

    registry = get_embedding_registry()
    provider_cls = registry.get_provider(name)
    if provider_cls is None:
        available = registry.list_providers()
        raise ValueError(f"Unknown embedding provider: '{name}'. Available: {available}")
    return provider_cls()


__all__ = [
    "BaseEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "get_embedding_provider",
]
