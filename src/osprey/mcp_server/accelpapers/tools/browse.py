"""MCP tool: papers_browse — browse papers by conference/year with sorting."""

import json
import logging

from osprey.mcp_server.accelpapers import db
from osprey.mcp_server.accelpapers.server import mcp
from osprey.mcp_server.accelpapers.tools._filters import build_filter_string
from osprey.mcp_server.common import make_error

logger = logging.getLogger("osprey.mcp_server.accelpapers.tools.browse")

VALID_SORT_FIELDS = {"year", "citation_count", "first_author", "title"}
VALID_SORT_ORDERS = {"asc", "desc"}


@mcp.tool()
async def papers_browse(
    conference: str | None = None,
    year: int | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    document_type: str | None = None,
    sort_by: str = "citation_count",
    sort_order: str = "desc",
    max_results: int = 20,
    offset: int = 0,
) -> str:
    """Browse accelerator physics papers by conference, year, or document type.

    Unlike papers_search, this does NOT use full-text search — it filters on
    faceted fields for fast, structured browsing. At least one filter is required.

    Args:
        conference: Filter by conference name (e.g. "IPAC2023").
        year: Filter by exact year.
        year_min: Filter papers from this year onward.
        year_max: Filter papers up to this year.
        document_type: Filter by type (e.g. "article", "conference paper").
        sort_by: Sort field: "citation_count", "year", "first_author", "title" (default: citation_count).
        sort_order: Sort direction: "asc" or "desc" (default: desc).
        max_results: Maximum results (1-100, default 20).
        offset: Skip first N results for pagination (default 0).

    Returns:
        JSON with matching papers and total count.
    """
    # Require at least one filter
    if not any([conference, year, year_min, year_max, document_type]):
        return json.dumps(
            make_error(
                "validation_error",
                "At least one filter is required for browsing.",
                [
                    "Specify conference, year, year_min/year_max, or document_type.",
                    "Use papers_search for full-text search instead.",
                ],
            )
        )

    if sort_by not in VALID_SORT_FIELDS:
        sort_by = "citation_count"
    if sort_order.lower() not in VALID_SORT_ORDERS:
        sort_order = "desc"
    max_results = max(1, min(100, max_results))
    offset = max(0, offset)

    try:
        client = db.get_client()
        collection = db.get_collection_name()

        search_params: dict = {
            "q": "*",
            "sort_by": f"{sort_by}:{sort_order}",
            "per_page": max_results,
            "page": (offset // max_results) + 1,
            "exclude_fields": "embedding,full_text",
        }

        filter_str = build_filter_string(
            conference=conference,
            year=year,
            year_min=year_min,
            year_max=year_max,
            document_type=document_type,
        )
        if filter_str:
            search_params["filter_by"] = filter_str

        result = client.collections[collection].documents.search(search_params)
        total = result.get("found", 0)

        results = []
        for hit in result.get("hits", []):
            doc = hit["document"]
            abstract = doc.get("abstract", "") or ""
            if len(abstract) > 300:
                abstract = abstract[:300] + "..."
            results.append({
                "texkey": doc.get("id", ""),
                "title": doc.get("title", ""),
                "first_author": doc.get("first_author", ""),
                "year": doc.get("year"),
                "conference": doc.get("conference", ""),
                "citation_count": doc.get("citation_count", 0),
                "document_type": doc.get("document_type", ""),
                "doi": doc.get("doi", ""),
                "inspire_url": doc.get("inspire_url", ""),
                "journal_title": doc.get("journal_title", ""),
                "arxiv_id": doc.get("arxiv_id", ""),
                "abstract": abstract,
            })

        return json.dumps({
            "filters": {
                k: v for k, v in {
                    "conference": conference,
                    "year": year,
                    "year_min": year_min,
                    "year_max": year_max,
                    "document_type": document_type,
                }.items() if v is not None
            },
            "sort_by": sort_by,
            "sort_order": sort_order,
            "total_matching": total,
            "offset": offset,
            "results_returned": len(results),
            "papers": results,
        })

    except Exception as exc:
        logger.exception("papers_browse failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Browse failed: {exc}",
                ["Check that the Typesense server is running and the collection is indexed."],
            )
        )
