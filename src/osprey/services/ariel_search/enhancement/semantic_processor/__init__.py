"""ARIEL semantic processor enhancement module.

This module provides semantic processing (keyword extraction, summarization)
for logbook entries.
"""

from osprey.services.ariel_search.enhancement.semantic_processor.migration import (
    SemanticProcessorMigration,
)
from osprey.services.ariel_search.enhancement.semantic_processor.processor import (
    SemanticProcessorModule,
    SemanticProcessorResult,
)

__all__ = [
    "SemanticProcessorMigration",
    "SemanticProcessorModule",
    "SemanticProcessorResult",
]
