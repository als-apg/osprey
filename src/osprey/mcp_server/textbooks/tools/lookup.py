"""MCP tool: textbook_lookup — primary routing tool for concept/term/equation queries."""

import json
import logging

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.textbooks.indexer import search_indexes
from osprey.mcp_server.textbooks.server import mcp

logger = logging.getLogger("osprey.mcp_server.textbooks.tools.lookup")


@mcp.tool()
async def textbook_lookup(
    query: str,
    book: str | None = None,
    max_results: int = 10,
) -> str:
    """Look up a concept, term, or equation in textbook indexes.

    Searches across concept maps, alphabetical indexes, section headings, and
    equation tags simultaneously.  Returns ranked matches with locations for
    use with textbook_read_section.

    Args:
        query: Concept, term, or equation tag to look up
            (e.g. "Courant-Snyder invariant", "betatron tune", "4.5").
        book: Restrict search to a specific book (partial name match).
            Omit to search all books.
        max_results: Maximum number of matches to return (default 10).

    Returns:
        JSON with ranked matches including type, term, book, chapter,
        section, and line number.
    """
    if not query or not query.strip():
        return json.dumps(
            make_error("validation_error", "Empty query.", ["Provide a concept or term to look up."])
        )

    matches = search_indexes(query.strip(), book_name=book, max_results=max_results)

    if not matches:
        return json.dumps({
            "query": query,
            "matches": [],
            "suggestion": (
                "No matches found in indexes. Try textbook_search() for full-text "
                "search, or use broader/alternative terminology."
            ),
        })

    # Remove internal score from output
    for m in matches:
        m.pop("score", None)

    return json.dumps({"query": query, "matches": matches})
