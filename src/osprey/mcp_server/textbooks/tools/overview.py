"""MCP tool: textbook_overview — orientation tool for available books."""

import json
import logging

from osprey.mcp_server.textbooks.indexer import get_book, get_books
from osprey.mcp_server.textbooks.server import mcp

logger = logging.getLogger("osprey.mcp_server.textbooks.tools.overview")


@mcp.tool()
async def textbook_overview(book: str | None = None) -> str:
    """Get an overview of available textbooks or a specific book's contents.

    Two modes:
    - No args: list all available books with chapter counts and topic summaries.
    - With book: return chapter listing with summaries for that book.

    Args:
        book: Book name to get details for (partial match).
            Omit to list all available books.

    Returns:
        Markdown-formatted overview of available content.
    """
    books = get_books()

    if not books:
        return "No textbooks loaded. Check TEXTBOOKS_ROOT environment variable."

    if book:
        b = get_book(book)
        if b is None:
            return json.dumps({
                "error": True,
                "message": f"Book not found: {book}",
                "available_books": list(books.keys()),
            })

        lines = [f"# {b.name}\n"]
        lines.append(f"- **Chapters**: {len(b.chapters)}")
        lines.append(f"- **Indexed concepts**: {len(b.concepts)}")
        lines.append(f"- **Index terms**: {len(b.terms)}")
        lines.append(f"- **Equations**: {len(b.equations)}")
        lines.append("")
        lines.append("## Chapters\n")

        for ch in b.chapters:
            summary = b.chapter_summaries.get(ch, "")
            section_count = len(b.sections.get(ch, []))
            lines.append(f"### {ch}")
            if summary:
                lines.append(summary)
            lines.append(f"*{section_count} sections*\n")

        return "\n".join(lines)

    # List all books
    lines = ["# Available Textbooks\n"]
    for name, b in sorted(books.items()):
        lines.append(f"## {name}")
        lines.append(f"- Chapters: {len(b.chapters)}")
        lines.append(f"- Concepts: {len(b.concepts)}")
        lines.append(f"- Terms: {len(b.terms)}")
        lines.append(f"- Equations: {len(b.equations)}")

        # List chapter names
        if b.chapters:
            ch_names = ", ".join(ch.replace("_", " ").replace("ch", "Ch") for ch in b.chapters[:5])
            if len(b.chapters) > 5:
                ch_names += f", ... (+{len(b.chapters) - 5} more)"
            lines.append(f"- Topics: {ch_names}")
        lines.append("")

    return "\n".join(lines)
