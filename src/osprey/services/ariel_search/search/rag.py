"""ARIEL RAG (Retrieval-Augmented Generation) search module.

This module provides question-answering with local grounding using
retrieved logbook entries as context.

See 02_SEARCH_MODULES.md Section 5 for specification.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from osprey.services.ariel_search.prompts import RAG_PROMPT_TEMPLATE

if TYPE_CHECKING:
    from osprey.models.embeddings.base import BaseEmbeddingProvider
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.models import EnhancedLogbookEntry

logger = logging.getLogger(__name__)

# Default RAG settings
DEFAULT_MAX_ENTRIES = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.5

# Context limits per spec Section 5.5.2
MAX_CHARS_PER_ENTRY = 2000
MAX_TOTAL_CONTEXT_CHARS = 12000


def format_entry_for_context(
    entry: EnhancedLogbookEntry,
    score: float,
) -> str:
    """Format a logbook entry for inclusion in RAG context.

    Uses ENTRY #id format per spec Section 5.5.2.

    Args:
        entry: The logbook entry
        score: Similarity score

    Returns:
        Formatted string for the entry
    """
    metadata = entry.get("metadata", {})
    title = metadata.get("title", "")
    author = entry.get("author", "Unknown")
    timestamp = entry.get("timestamp")
    timestamp_str = timestamp.isoformat() if timestamp else "Unknown"

    # Use ENTRY #id | timestamp | Author: name format per spec Section 5.5.2
    header = f"ENTRY #{entry['entry_id']} | {timestamp_str} | Author: {author}"
    if title:
        header += f" | {title}"

    content = entry.get("raw_text", "")
    # Truncate long entries per spec (2000 chars/entry)
    if len(content) > MAX_CHARS_PER_ENTRY:
        content = content[:MAX_CHARS_PER_ENTRY] + "..."

    return f"{header}\n{content}\n"


async def rag_search(
    query: str,
    repository: ARIELRepository,
    config: ARIELConfig,
    embedder: BaseEmbeddingProvider,
    *,
    max_entries: int | None = None,
    similarity_threshold: float | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    **kwargs: Any,
) -> tuple[str, list[EnhancedLogbookEntry]]:
    """Execute RAG search: retrieve relevant entries and generate an answer.

    Uses semantic search to find relevant logbook entries, then uses an LLM
    to generate an answer grounded in the retrieved context.

    Args:
        query: Natural language question
        repository: ARIEL database repository
        config: ARIEL configuration
        embedder: Embedding provider (Ollama or other)
        max_entries: Maximum entries to use for context (default: 5)
        similarity_threshold: Minimum similarity for retrieval (default: 0.5)
        start_date: Filter entries after this time
        end_date: Filter entries before this time

    Returns:
        Tuple of (answer, source_entries)
        - answer: Generated answer text
        - source_entries: List of entries used as context
    """
    if not query.strip():
        return ("Please provide a question.", [])

    # Resolve settings using 3-tier resolution
    rag_config = config.search_modules.get("rag")

    if max_entries is None:
        if rag_config and rag_config.settings:
            max_entries = rag_config.settings.get(
                "max_entries_for_context",
                DEFAULT_MAX_ENTRIES,
            )
        else:
            max_entries = DEFAULT_MAX_ENTRIES

    if similarity_threshold is None:
        if rag_config and rag_config.settings:
            similarity_threshold = rag_config.settings.get(
                "similarity_threshold",
                DEFAULT_SIMILARITY_THRESHOLD,
            )
        else:
            similarity_threshold = DEFAULT_SIMILARITY_THRESHOLD

    # Get the model to use for embedding
    model_name = config.get_search_model()
    if not model_name:
        logger.warning("No search model configured for RAG")
        return ("RAG search is not properly configured.", [])

    # Get embedding provider config for base_url
    embedding_config = config.embedding
    base_url = getattr(embedding_config, "base_url", None) or embedder.default_base_url

    # Generate query embedding
    try:
        embeddings = embedder.execute_embedding(
            texts=[query],
            model_id=model_name,
            base_url=base_url,
        )
        if not embeddings or not embeddings[0]:
            logger.error("Failed to generate query embedding for RAG")
            return ("Failed to process your question.", [])

        query_embedding = embeddings[0]

    except Exception as e:
        logger.error(f"Embedding generation failed for RAG: {e}")
        return ("Failed to process your question.", [])

    # Retrieve relevant entries via semantic search
    results = await repository.semantic_search(
        query_embedding=query_embedding,
        model_name=model_name,
        max_results=max_entries,
        similarity_threshold=similarity_threshold,
        start_date=start_date,
        end_date=end_date,
    )

    if not results:
        return (
            "I don't have enough information to answer this question based on "
            "the available logbook entries.",
            [],
        )

    # Extract entries from results
    source_entries = [entry for entry, _score in results]

    # Format context from retrieved entries with total limit per spec Section 5.5.2
    context_parts = []
    total_chars = 0
    for entry, score in results:
        formatted = format_entry_for_context(entry, score)
        if total_chars + len(formatted) > MAX_TOTAL_CONTEXT_CHARS:
            # Truncate to fit within total context limit
            remaining = MAX_TOTAL_CONTEXT_CHARS - total_chars
            if remaining > 100:  # Only add if meaningful content remains
                formatted = formatted[:remaining] + "..."
                context_parts.append(formatted)
            break
        context_parts.append(formatted)
        total_chars += len(formatted)
    context = "\n---\n".join(context_parts)

    # Build the RAG prompt
    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=query,
    )

    # Generate answer using LLM
    try:
        from osprey.models.completion import get_chat_completion

        # Get RAG model config if specified
        model_config = None
        if rag_config and rag_config.model:
            model_config = {"model": rag_config.model}

        response = get_chat_completion(
            message=prompt,
            model_config=model_config,
        )

        # Handle different response types
        if isinstance(response, str):
            answer = response
        else:
            # For structured outputs or thinking blocks, convert to string
            answer = str(response)
        if not answer:
            answer = "I was unable to generate an answer."

    except ImportError:
        logger.warning("osprey.models.completion not available for RAG")
        # Fallback: return the context entries with a note
        answer = (
            "LLM not available for answer generation. "
            f"Found {len(source_entries)} relevant entries."
        )
    except Exception as e:
        logger.error(f"LLM call failed for RAG: {e}")
        answer = f"Error generating answer: {e}"

    return (answer, source_entries)
