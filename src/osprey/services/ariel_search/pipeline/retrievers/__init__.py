"""Retriever implementations for ARIEL RAP pipeline.

Retrievers fetch relevant items from the database based on the query.

Available retrievers:
    - KeywordRetriever: PostgreSQL full-text search
    - SemanticRetriever: Embedding similarity search
    - HybridRetriever: Combines multiple retrievers with fusion

Fusion strategies:
    - RRFFusion: Reciprocal Rank Fusion
    - WeightedFusion: Score-weighted combination
"""

from osprey.services.ariel_search.pipeline.retrievers.hybrid import (
    HybridRetriever,
    RRFFusion,
    WeightedFusion,
)
from osprey.services.ariel_search.pipeline.retrievers.keyword import KeywordRetriever
from osprey.services.ariel_search.pipeline.retrievers.semantic import SemanticRetriever

__all__ = [
    "HybridRetriever",
    "KeywordRetriever",
    "RRFFusion",
    "SemanticRetriever",
    "WeightedFusion",
]
