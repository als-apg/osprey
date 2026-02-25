"""MCP tool: papers_search_author — author-focused search via Typesense."""

import json
import logging

from osprey.mcp_server.accelpapers import db
from osprey.mcp_server.accelpapers.server import mcp
from osprey.mcp_server.accelpapers.tools._filters import build_filter_string
from osprey.mcp_server.common import make_error

logger = logging.getLogger("osprey.mcp_server.accelpapers.tools.search_author")


@mcp.tool()
async def papers_search_author(
    author: str,
    max_results: int = 20,
    year_min: int | None = None,
    year_max: int | None = None,
) -> str:
    """Search for papers by a specific author.

    Searches the author names field for matching authors.
    Results are ranked by citation count.

    Args:
        author: Author name to search for (e.g. "Wiedemann", "Chao, Alexander").
        max_results: Maximum results to return (1-100, default 20).
        year_min: Filter papers from this year onward.
        year_max: Filter papers up to this year.

    Returns:
        JSON with matching papers sorted by citation count.
    """
    if not author or not author.strip():
        return json.dumps(
            make_error("validation_error", "Empty author name.", ["Provide an author name."])
        )

    max_results = max(1, min(100, max_results))

    try:
        client = db.get_client()
        collection = db.get_collection_name()

        search_params: dict = {
            "q": author.strip(),
            "query_by": "all_authors",
            "sort_by": "citation_count:desc",
            "per_page": max_results,
            "exclude_fields": "embedding,full_text",
        }

        filter_str = build_filter_string(year_min=year_min, year_max=year_max)
        if filter_str:
            search_params["filter_by"] = filter_str

        result = client.collections[collection].documents.search(search_params)

        results = []
        for hit in result.get("hits", []):
            doc = hit["document"]
            results.append(
                {
                    "texkey": doc.get("id", ""),
                    "title": doc.get("title", ""),
                    "first_author": doc.get("first_author", ""),
                    "all_authors": doc.get("all_authors", ""),
                    "year": doc.get("year"),
                    "conference": doc.get("conference", ""),
                    "citation_count": doc.get("citation_count", 0),
                    "document_type": doc.get("document_type", ""),
                    "doi": doc.get("doi", ""),
                    "inspire_url": doc.get("inspire_url", ""),
                    "journal_title": doc.get("journal_title", ""),
                    "arxiv_id": doc.get("arxiv_id", ""),
                }
            )

        return json.dumps(
            {
                "author_query": author,
                "filters": {
                    k: v
                    for k, v in {
                        "year_min": year_min,
                        "year_max": year_max,
                    }.items()
                    if v is not None
                },
                "results_found": len(results),
                "papers": results,
            }
        )

    except Exception as exc:
        logger.exception("papers_search_author failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Author search failed: {exc}",
                ["Check that the Typesense server is running and the collection is indexed."],
            )
        )
