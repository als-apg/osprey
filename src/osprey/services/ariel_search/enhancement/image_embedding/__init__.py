"""ARIEL image embedding enhancement module.

Embeds image attachments into the table the configured model and width name.
"""

from osprey.services.ariel_search.enhancement.image_embedding.migration import (
    ImageEmbeddingMigration,
)

__all__ = [
    "ImageEmbeddingMigration",
]
