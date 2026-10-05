"""ARIEL image caption enhancement module.

Captions each viewable picture of an entry with a vision model, in the catch-up.
"""

from osprey.services.ariel_search.enhancement.image_caption.module import (
    DEFAULT_CAPTION_PROMPT,
    DEFAULT_MAX_IMAGES_PER_ENTRY,
    DEFAULT_TIMEOUT_SECONDS,
    ImageCaptionModule,
    parse_caption_reply,
)

__all__ = [
    "DEFAULT_CAPTION_PROMPT",
    "DEFAULT_MAX_IMAGES_PER_ENTRY",
    "DEFAULT_TIMEOUT_SECONDS",
    "ImageCaptionModule",
    "parse_caption_reply",
]
