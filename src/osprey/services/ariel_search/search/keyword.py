"""ARIEL keyword search module.

This module provides full-text search using PostgreSQL's built-in
text search capabilities with optional fuzzy matching fallback.

See 02_SEARCH_MODULES.md Section 3 for specification.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from osprey.services.ariel_search.models import SearchMode
from osprey.services.ariel_search.search.base import ParameterDescriptor, SearchToolDescriptor
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.models import EnhancedLogbookEntry

logger = get_logger("ariel")

# Allowed operators for query parsing
ALLOWED_OPERATORS = {"AND", "OR", "NOT"}
ALLOWED_FIELD_PREFIXES = {"author:", "date:"}

# Default fuzzy threshold (pg_trgm similarity)
DEFAULT_FUZZY_THRESHOLD = 0.3

# Maximum query length (GAP-C002)
MAX_QUERY_LENGTH = 1000


def _balance_quotes(query: str) -> str:
    """Balance unbalanced quotes in a query string.

    Handles queries with odd numbers of double quotes by escaping
    the unbalanced quote as literal text.

    Args:
        query: User search query that may have unbalanced quotes

    Returns:
        Query with balanced quotes
    """
    quote_count = query.count('"')
    if quote_count % 2 == 0:
        # Quotes are balanced
        return query

    # Find the position of the last unbalanced quote and escape it
    # by removing it (treating it as literal text)
    logger.warning(
        f"Unbalanced quotes detected in query, auto-balancing: {query[:50]}..."
        if len(query) > 50
        else f"Unbalanced quotes detected in query, auto-balancing: {query}"
    )

    # Strategy: Find the last quote and remove it to balance
    # This preserves the most complete quoted phrases
    last_quote_idx = query.rfind('"')
    if last_quote_idx >= 0:
        query = query[:last_quote_idx] + query[last_quote_idx + 1 :]

    return query


def parse_query(query: str) -> tuple[str, dict[str, str], list[str]]:
    """Parse user query to extract field filters, phrases, and terms.

    Args:
        query: User search query

    Returns:
        Tuple of (search_text, field_filters, phrases)
        - search_text: Query text for FTS
        - field_filters: Dict of field:value filters
        - phrases: List of quoted phrases
    """
    field_filters: dict[str, str] = {}
    phrases: list[str] = []

    # Balance unbalanced quotes before parsing (GAP-C001)
    query = _balance_quotes(query)

    # Extract quoted phrases first
    phrase_pattern = r'"([^"]+)"'
    phrase_matches = re.findall(phrase_pattern, query)
    phrases.extend(phrase_matches)

    # Remove phrases from query
    remaining = re.sub(phrase_pattern, "", query)

    # Extract field prefixes
    tokens = remaining.split()
    search_tokens: list[str] = []

    for token in tokens:
        lower_token = token.lower()

        # Check for field prefix
        if lower_token.startswith("author:"):
            field_filters["author"] = token[7:]  # After "author:"
        elif lower_token.startswith("date:"):
            field_filters["date"] = token[5:]  # After "date:"
        else:
            search_tokens.append(token)

    search_text = " ".join(search_tokens)
    return search_text, field_filters, phrases


def build_tsquery(search_text: str, phrases: list[str]) -> str:
    """Build PostgreSQL tsquery from parsed query components.

    Args:
        search_text: Main search text with operators
        phrases: List of exact phrases to match

    Returns:
        tsquery expression string
    """
    tsquery_parts: list[str] = []

    # Process main search text
    if search_text.strip():
        # Replace operators with PostgreSQL equivalents
        ts_text = search_text
        ts_text = re.sub(r"\bAND\b", "&", ts_text, flags=re.IGNORECASE)
        ts_text = re.sub(r"\bOR\b", "|", ts_text, flags=re.IGNORECASE)
        ts_text = re.sub(r"\bNOT\b", "!", ts_text, flags=re.IGNORECASE)

        # Check if text has operators
        if any(op in ts_text for op in ("&", "|", "!")):
            # Use websearch_to_tsquery for flexible parsing
            tsquery_parts.append("websearch_to_tsquery('english', %s)")
        else:
            # Use plainto_tsquery for implicit AND
            tsquery_parts.append("plainto_tsquery('english', %s)")

    # Add phrase matches
    for _phrase in phrases:
        tsquery_parts.append("phraseto_tsquery('english', %s)")

    if not tsquery_parts:
        return "plainto_tsquery('english', '')"

    # Combine with AND
    return " && ".join(tsquery_parts)


async def keyword_search(
    query: str,
    repository: ARIELRepository,
    config: ARIELConfig,
    *,
    max_results: int = 10,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    author: str | None = None,
    source_system: str | None = None,
    include_highlights: bool = True,
    fuzzy_fallback: bool = True,
    **kwargs: Any,
) -> list[tuple[EnhancedLogbookEntry, float, list[str]]]:
    """Execute keyword search against the logbook database.

    Uses PostgreSQL full-text search with optional fuzzy matching fallback.

    Args:
        query: Search query with optional operators (AND, OR, NOT)
            and field prefixes (author:, date:)
        repository: ARIEL database repository
        config: ARIEL configuration
        max_results: Maximum entries to return (default: 10)
        start_date: Filter entries after this time
        end_date: Filter entries before this time
        author: Filter by author name (ILIKE match)
        source_system: Filter by source system (exact match)
        include_highlights: Include highlighted snippets (default: True)
        fuzzy_fallback: Fall back to fuzzy search if no exact matches

    Returns:
        List of (entry, score, highlights) tuples sorted by relevance
    """
    if not query.strip():
        return []

    logger.info(
        f"keyword_search: query={query!r}, max_results={max_results}, "
        f"start_date={start_date}, end_date={end_date}"
    )

    # Truncate query if too long (GAP-C002)
    if len(query) > MAX_QUERY_LENGTH:
        original_length = len(query)
        query = query[:MAX_QUERY_LENGTH]
        logger.warning(f"Query truncated from {original_length} to {MAX_QUERY_LENGTH} characters")

    # Parse the query
    search_text, field_filters, phrases = parse_query(query)

    # Build search parameters
    params: list[Any] = []
    where_clauses: list[str] = []

    # Build tsquery
    if search_text.strip() or phrases:
        # Combine search text and phrases for tsquery
        all_search_terms = search_text
        params.append(all_search_terms)

        for phrase in phrases:
            params.append(phrase)

        # Main FTS condition
        # Note: Search is performed on raw_text which contains subject + details
        where_clauses.append(
            f"to_tsvector('english', raw_text) @@ ({build_tsquery(search_text, phrases)})"
        )

    # Add field filters
    if "author" in field_filters:
        where_clauses.append("author ILIKE %s")
        params.append(f"%{field_filters['author']}%")

    if "date" in field_filters:
        # Parse date filter (YYYY-MM or YYYY-MM-DD)
        date_val = field_filters["date"]
        if len(date_val) == 7:  # YYYY-MM
            where_clauses.append("timestamp >= %s AND timestamp < %s")
            params.append(f"{date_val}-01")
            # Next month
            year, month = int(date_val[:4]), int(date_val[5:7])
            if month == 12:
                next_month = f"{year + 1}-01-01"
            else:
                next_month = f"{year}-{month + 1:02d}-01"
            params.append(next_month)
        else:  # YYYY-MM-DD
            where_clauses.append("DATE(timestamp) = %s")
            params.append(date_val)

    # Add time range filters
    if start_date:
        where_clauses.append("timestamp >= %s")
        params.append(start_date)
    if end_date:
        where_clauses.append("timestamp <= %s")
        params.append(end_date)

    # Add metadata filters (from web UI filter panel)
    if author and "author" not in field_filters:
        where_clauses.append("author ILIKE %s")
        params.append(f"%{author}%")
    if source_system:
        where_clauses.append("source_system = %s")
        params.append(source_system)

    # Execute search via repository
    results = await repository.keyword_search(
        where_clauses=where_clauses,
        params=params,
        search_text=search_text or " ".join(phrases),
        max_results=max_results,
        include_highlights=include_highlights,
    )

    # If no results and fuzzy fallback enabled, try fuzzy search
    if not results and fuzzy_fallback and (search_text.strip() or phrases):
        results = await repository.fuzzy_search(
            search_text=search_text or " ".join(phrases),
            threshold=DEFAULT_FUZZY_THRESHOLD,
            max_results=max_results,
            start_date=start_date,
            end_date=end_date,
        )

    logger.info(f"keyword_search: returning {len(results)} results")
    return results


# === Tool descriptor for agent auto-discovery ===


class KeywordSearchInput(BaseModel):
    """Input schema for keyword search tool."""

    query: str = Field(
        description="Search terms. Supports phrases in quotes, AND/OR/NOT operators."
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum results to return",
    )
    start_date: datetime | None = Field(
        default=None,
        description="Filter entries created after this time (inclusive)",
    )
    end_date: datetime | None = Field(
        default=None,
        description="Filter entries created before this time (inclusive)",
    )


def format_keyword_result(
    entry: EnhancedLogbookEntry,
    score: float,
    highlights: list[str],
) -> dict[str, Any]:
    """Format a keyword search result for agent consumption.

    Args:
        entry: EnhancedLogbookEntry
        score: Relevance score
        highlights: Highlighted snippets

    Returns:
        Formatted dict for agent
    """
    timestamp = entry.get("timestamp")
    return {
        "entry_id": entry.get("entry_id"),
        "timestamp": timestamp.isoformat() if timestamp is not None else None,
        "author": entry.get("author"),
        "text": entry.get("raw_text", "")[:500],  # Truncate for agent
        "title": entry.get("metadata", {}).get("title"),
        "score": score,
        "highlights": highlights,
    }


def get_parameter_descriptors() -> list[ParameterDescriptor]:
    """Return tunable parameter descriptors for the capabilities API."""
    return [
        ParameterDescriptor(
            name="include_highlights",
            label="Include Highlights",
            description="Include highlighted snippets in search results",
            param_type="bool",
            default=True,
            section="Options",
        ),
        ParameterDescriptor(
            name="fuzzy_fallback",
            label="Fuzzy Fallback",
            description="Fall back to fuzzy matching when no exact matches are found",
            param_type="bool",
            default=True,
            section="Options",
        ),
    ]


def get_tool_descriptor() -> SearchToolDescriptor:
    """Return the descriptor for auto-discovery by the agent executor."""
    return SearchToolDescriptor(
        name="keyword_search",
        description=(
            "Fast text-based lookup using full-text search. "
            "Use for specific terms, equipment names, PV names, or phrases. "
            "Supports quoted phrases and AND/OR/NOT operators."
        ),
        search_mode=SearchMode.KEYWORD,
        args_schema=KeywordSearchInput,
        execute=keyword_search,
        format_result=format_keyword_result,
    )
