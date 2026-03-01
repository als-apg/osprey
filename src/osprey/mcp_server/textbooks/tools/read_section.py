"""MCP tool: textbook_read_section — surgical section reader."""

import json
import logging

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.textbooks.indexer import find_section_bounds, get_book
from osprey.mcp_server.textbooks.server import mcp

logger = logging.getLogger("osprey.mcp_server.textbooks.tools.read_section")


@mcp.tool()
async def textbook_read_section(
    book: str,
    chapter: str,
    section: str | None = None,
    start_line: int | None = None,
    num_lines: int = 80,
) -> str:
    """Read a section from a textbook chapter.

    Two modes:
    - By section name: reads from the heading to the next heading of equal
      or higher level (automatically calculates boundaries).
    - By line number: reads num_lines starting at start_line.

    Args:
        book: Book name (partial match, e.g. "Fundamentals").
        chapter: Chapter file stem (e.g. "ch04_high_energy_accelerators").
        section: Section heading to read (e.g. "Courant-Snyder Invariant").
            If provided, start_line/num_lines are ignored.
        start_line: Line number to start reading from (1-indexed).
        num_lines: Number of lines to read (default 80, max 300).

    Returns:
        The section content as markdown text with LaTeX equations preserved.
    """
    book_index = get_book(book)
    if book_index is None:
        return json.dumps(
            make_error(
                "not_found",
                f"Book not found: {book}",
                ["Use textbook_overview() to see available books."],
            )
        )

    # Validate chapter exists
    chapter_file = book_index.path / f"{chapter}.md"
    if not chapter_file.exists():
        available = [c for c in book_index.chapters if chapter.lower() in c.lower()]
        return json.dumps(
            make_error(
                "not_found",
                f"Chapter not found: {chapter}",
                [f"Did you mean: {a}?" for a in available[:3]]
                or [f"Available chapters: {', '.join(book_index.chapters[:5])}..."],
            )
        )

    num_lines = max(1, min(300, num_lines))

    if section:
        # Mode 1: Read by section name
        bounds = find_section_bounds(book_index, chapter, section)
        if bounds is None:
            return json.dumps(
                make_error(
                    "not_found",
                    f"Section not found: {section} in {chapter}",
                    [
                        "Use textbook_lookup() to find the correct section name.",
                        f"Chapter has {len(book_index.sections.get(chapter, []))} sections.",
                    ],
                )
            )
        start_line, end_line = bounds
        num_lines = end_line - start_line + 1
    elif start_line is None:
        start_line = 1

    # Read the specified lines
    content_lines = []
    with open(chapter_file, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i < start_line:
                continue
            if i > start_line + num_lines - 1:
                break
            content_lines.append(line.rstrip())

    if not content_lines:
        return json.dumps(
            make_error(
                "not_found",
                f"No content at line {start_line} in {chapter}",
                ["Check the line number — it may be beyond the end of the chapter."],
            )
        )

    header = f"# {book_index.name} / {chapter}"
    if section:
        header += f" / {section}"
    header += f"\n# Lines {start_line}–{start_line + len(content_lines) - 1}"

    return f"{header}\n\n" + "\n".join(content_lines)
