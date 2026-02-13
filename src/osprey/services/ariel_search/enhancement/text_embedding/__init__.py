"""ARIEL text embedding enhancement module.

This module provides text embedding generation for logbook entries.
"""

from osprey.services.ariel_search.enhancement.text_embedding.embedder import (
    TextEmbeddingModule,
)
from osprey.services.ariel_search.enhancement.text_embedding.migration import (
    TextEmbeddingMigration,
)

__all__ = ["TextEmbeddingMigration", "TextEmbeddingModule"]
