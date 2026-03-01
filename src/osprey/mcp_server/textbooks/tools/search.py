"""MCP tool: textbook_search — full-text grep fallback."""

import json
import logging
import re

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.textbooks.indexer import get_book, get_books
from osprey.mcp_server.textbooks.server import mcp

logger = logging.getLogger("osprey.mcp_server.textbooks.tools.search")


@mcp.tool()
async def textbook_search(
    query: str,
    book: str | None = None,
    chapter: str | None = None,
    context_lines: int = 3,
    max_results: int = 20,
) -> str:
    """Full-text search across textbook chapter files.

    Use this as a fallback when textbook_lookup returns no results — it
    searches the raw text of chapter files using regex matching.

    Args:
        query: Search pattern (supports regex).
        book: Restrict to a specific book (partial name match).
        chapter: Restrict to a specific chapter file stem.
        context_lines: Number of context lines before/after each match (default 3).
        max_results: Maximum matches to return (default 20).

    Returns:
        JSON with matching lines, context, file paths, and line numbers.
    """
    if not query or not query.strip():
        return json.dumps(
            make_error("validation_error", "Empty query.", ["Provide a search pattern."])
        )

    context_lines = max(0, min(10, context_lines))
    max_results = max(1, min(50, max_results))

    try:
        pattern = re.compile(re.escape(query.strip()), re.IGNORECASE)
    except re.error:
        # If the query itself is a valid regex, try that
        try:
            pattern = re.compile(query.strip(), re.IGNORECASE)
        except re.error as exc:
            return json.dumps(make_error("validation_error", f"Invalid regex: {exc}"))

    books_to_search: list = []
    if book:
        b = get_book(book)
        if b:
            books_to_search = [b]
    else:
        books_to_search = list(get_books().values())

    if not books_to_search:
        return json.dumps(
            make_error("not_found", f"Book not found: {book}", ["Use textbook_overview() to list books."])
        )

    results: list[dict] = []

    for b in books_to_search:
        chapters_to_search = b.chapters + (["back_matter"] if "back_matter" in b.sections else [])
        if chapter:
            chapters_to_search = [c for c in chapters_to_search if chapter.lower() in c.lower()]

        for ch in chapters_to_search:
            ch_file = b.path / f"{ch}.md"
            if not ch_file.exists():
                continue

            file_lines = ch_file.read_text(encoding="utf-8").splitlines()

            for i, line in enumerate(file_lines):
                if pattern.search(line):
                    start = max(0, i - context_lines)
                    end = min(len(file_lines), i + context_lines + 1)
                    context = file_lines[start:end]

                    results.append({
                        "book": b.name,
                        "chapter": ch,
                        "line": i + 1,
                        "match": line.strip(),
                        "context": "\n".join(context),
                    })

                    if len(results) >= max_results:
                        break
            if len(results) >= max_results:
                break
        if len(results) >= max_results:
            break

    return json.dumps({
        "query": query,
        "results_found": len(results),
        "matches": results,
    })
