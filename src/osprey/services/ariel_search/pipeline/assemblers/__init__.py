"""Assembler implementations for ARIEL RAP pipeline.

Assemblers take retrieved items and prepare them for processing.

Available assemblers:
    - TopKAssembler: Simple top-K selection by score
    - ContextWindowAssembler: Token-aware assembly for RAG
"""

from osprey.services.ariel_search.pipeline.assemblers.context_window import (
    ContextWindowAssembler,
)
from osprey.services.ariel_search.pipeline.assemblers.topk import TopKAssembler

__all__ = [
    "ContextWindowAssembler",
    "TopKAssembler",
]
