"""ARIEL text embedding enhancement module.

This module provides text embedding generation for logbook entries.
"""

from osprey.services.ariel_search.enhancement.text_embedding.embedder import (
    TextEmbeddingModule,
)
from osprey.services.ariel_search.enhancement.text_embedding.hnsw_migration import (
    TextEmbeddingHnswIndexMigration,
)
from osprey.services.ariel_search.enhancement.text_embedding.migration import (
    TextEmbeddingMigration,
    create_vector_index_sql,
    legacy_vector_index_name,
    vector_index_name,
)

__all__ = [
    "TextEmbeddingHnswIndexMigration",
    "TextEmbeddingMigration",
    "TextEmbeddingModule",
    "create_vector_index_sql",
    "legacy_vector_index_name",
    "vector_index_name",
]
