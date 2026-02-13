"""ARIEL embedding provider interface.

This module provides the base class for embedding providers
and the Ollama implementation.
"""

from osprey.models.embeddings.base import BaseEmbeddingProvider
from osprey.models.embeddings.ollama import OllamaEmbeddingProvider

__all__ = [
    "BaseEmbeddingProvider",
    "OllamaEmbeddingProvider",
]
