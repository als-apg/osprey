"""MCP tool: papers_get — retrieve full paper details by texkey."""

import json
import logging
from pathlib import Path

from osprey.mcp_server.accelpapers.db import get_connection
from osprey.mcp_server.accelpapers.server import mcp
from osprey.mcp_server.common import make_error

logger = logging.getLogger("osprey.mcp_server.accelpapers.tools.get_paper")


@mcp.tool()
async def papers_get(
    texkey: str,
    include_full_text: bool = True,
    include_content: bool = False,
) -> str:
    """Get full details for a specific paper by its texkey identifier.

    Returns metadata from the database. Optionally includes the full text
    and/or the raw JSON content (figures, tables, sections) from the source file.

    Args:
        texkey: The paper's texkey identifier (e.g. "Abe:2004wp").
        include_full_text: Include extracted full text (default True).
        include_content: Include raw JSON content sections from source file (default False).

    Returns:
        JSON with complete paper metadata and optional content.
    """
    if not texkey or not texkey.strip():
        return json.dumps(
            make_error("validation_error", "Empty texkey.", ["Provide a paper texkey."])
        )

    try:
        conn = get_connection()
        row = conn.execute("SELECT * FROM papers WHERE texkey = ?", [texkey.strip()]).fetchone()

        if row is None:
            conn.close()
            return json.dumps(
                make_error(
                    "not_found",
                    f"Paper not found: {texkey}",
                    ["Check the texkey spelling.", "Use papers_search to find papers."],
                )
            )

        paper = {
            "texkey": row["texkey"],
            "title": row["title"],
            "abstract": row["abstract"],
            "year": row["year"],
            "earliest_date": row["earliest_date"],
            "conference": row["conference"],
            "first_author": row["first_author"],
            "all_authors": row["all_authors"],
            "affiliations": row["affiliations"],
            "keywords": row["keywords"],
            "doi": row["doi"],
            "citation_count": row["citation_count"],
            "paper_id": row["paper_id"],
            "document_type": row["document_type"],
            "inspire_url": row["inspire_url"],
            "pdf_url": row["pdf_url"],
            "arxiv_id": row["arxiv_id"],
            "journal_title": row["journal_title"],
            "publisher": row["publisher"],
            "num_pages": row["num_pages"],
        }

        if include_full_text:
            paper["full_text"] = row["full_text"]

        # Optionally load raw content from JSON file
        if include_content:
            json_path = row["json_path"]
            if json_path and Path(json_path).exists():
                try:
                    with open(json_path) as f:
                        source_data = json.load(f)
                    paper["content"] = source_data.get("content", {})
                except (json.JSONDecodeError, OSError) as exc:
                    paper["content_error"] = f"Could not load source file: {exc}"
            else:
                paper["content_error"] = "Source JSON file not found on disk"

        conn.close()

        return json.dumps(paper, default=str)

    except Exception as exc:
        logger.exception("papers_get failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to retrieve paper: {exc}",
                ["Check that the AccelPapers database has been indexed."],
            )
        )
