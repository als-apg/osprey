"""MCP tool: papers_get — retrieve full paper details by texkey."""

import json
import logging
from pathlib import Path

from osprey.mcp_server.accelpapers import db
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

    Returns metadata from Typesense. Optionally includes the full text
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
        client = db.get_client()
        collection = db.get_collection_name()

        try:
            doc = client.collections[collection].documents[texkey.strip()].retrieve()
        except Exception:
            return json.dumps(
                make_error(
                    "not_found",
                    f"Paper not found: {texkey}",
                    ["Check the texkey spelling.", "Use papers_search to find papers."],
                )
            )

        paper = {
            "texkey": doc.get("id", ""),
            "title": doc.get("title", ""),
            "abstract": doc.get("abstract", ""),
            "year": doc.get("year"),
            "earliest_date": doc.get("earliest_date", ""),
            "conference": doc.get("conference", ""),
            "first_author": doc.get("first_author", ""),
            "all_authors": doc.get("all_authors", ""),
            "affiliations": doc.get("affiliations", ""),
            "keywords": doc.get("keywords", ""),
            "doi": doc.get("doi", ""),
            "citation_count": doc.get("citation_count", 0),
            "paper_id": doc.get("paper_id", ""),
            "document_type": doc.get("document_type", ""),
            "inspire_url": doc.get("inspire_url", ""),
            "pdf_url": doc.get("pdf_url", ""),
            "arxiv_id": doc.get("arxiv_id", ""),
            "journal_title": doc.get("journal_title", ""),
            "publisher": doc.get("publisher", ""),
            "num_pages": doc.get("num_pages"),
        }

        if include_full_text:
            paper["full_text"] = doc.get("full_text", "")

        # Optionally load raw content from JSON file
        if include_content:
            json_path = doc.get("json_path", "")
            if json_path and Path(json_path).exists():
                try:
                    with open(json_path) as f:
                        source_data = json.load(f)
                    paper["content"] = source_data.get("content", {})
                except (json.JSONDecodeError, OSError) as exc:
                    paper["content_error"] = f"Could not load source file: {exc}"
            else:
                paper["content_error"] = "Source JSON file not found on disk"

        return json.dumps(paper, default=str)

    except Exception as exc:
        logger.exception("papers_get failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to retrieve paper: {exc}",
                ["Check that the Typesense server is running and the collection is indexed."],
            )
        )
