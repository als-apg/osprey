"""Processor implementations for ARIEL RAP pipeline.

Processors transform assembled context into results.

Available processors:
    - IdentityProcessor: Pass through unchanged (keyword/semantic search)
    - SingleLLMProcessor: One LLM call (RAG answer generation)
"""

from osprey.services.ariel_search.pipeline.processors.identity import IdentityProcessor
from osprey.services.ariel_search.pipeline.processors.single_llm import (
    SingleLLMProcessor,
)

__all__ = [
    "IdentityProcessor",
    "SingleLLMProcessor",
]
