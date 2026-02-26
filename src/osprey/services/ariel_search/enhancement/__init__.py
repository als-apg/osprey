"""ARIEL enhancement modules.

This module provides enhancement modules for processing logbook entries
during ingestion (text embedding, semantic processing, etc.).
"""

from osprey.services.ariel_search.enhancement.base import BaseEnhancementModule
from osprey.services.ariel_search.enhancement.factory import (
    create_enhancers_from_config,
    get_enhancer_names,
)
from osprey.services.ariel_search.enhancement.semantic_processor import (
    SemanticProcessorMigration,
)
from osprey.services.ariel_search.enhancement.text_embedding import (
    TextEmbeddingMigration,
)

__all__ = [
    "BaseEnhancementModule",
    "SemanticProcessorMigration",
    "TextEmbeddingMigration",
    "create_enhancers_from_config",
    "get_enhancer_names",
]
