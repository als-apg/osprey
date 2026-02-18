"""MCP tool: papers_browse — browse papers by conference/year with sorting."""

import json
import logging

from osprey.mcp_server.accelpapers.db import get_connection
from osprey.mcp_server.accelpapers.server import mcp
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
    indexed columns for fast, structured browsing. At least one filter is required.

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
        conn = get_connection()

        where_parts = []
        params: list = []

        if conference:
            where_parts.append("conference = ?")
            params.append(conference)
        if year is not None:
            where_parts.append("year = ?")
            params.append(year)
        if year_min is not None:
            where_parts.append("year >= ?")
            params.append(year_min)
        if year_max is not None:
            where_parts.append("year <= ?")
            params.append(year_max)
        if document_type:
            where_parts.append("document_type = ?")
            params.append(document_type)

        where_clause = "WHERE " + " AND ".join(where_parts)

        # Get total count
        count_row = conn.execute(
            f"SELECT COUNT(*) as c FROM papers {where_clause}", params
        ).fetchone()
        total = count_row["c"]

        # Get results
        sql = f"""
            SELECT texkey, title, first_author, year, conference,
                   citation_count, document_type, doi, inspire_url,
                   journal_title, arxiv_id, abstract
            FROM papers
            {where_clause}
            ORDER BY {sort_by} {sort_order}
            LIMIT ? OFFSET ?
        """

        rows = conn.execute(sql, params + [max_results, offset]).fetchall()

        results = []
        for row in rows:
            r = {
                "texkey": row["texkey"],
                "title": row["title"],
                "first_author": row["first_author"],
                "year": row["year"],
                "conference": row["conference"],
                "citation_count": row["citation_count"],
                "document_type": row["document_type"],
                "doi": row["doi"],
                "inspire_url": row["inspire_url"],
                "journal_title": row["journal_title"],
                "arxiv_id": row["arxiv_id"],
            }
            # Include truncated abstract for browse
            abstract = row["abstract"] or ""
            if len(abstract) > 300:
                abstract = abstract[:300] + "..."
            r["abstract"] = abstract
            results.append(r)

        conn.close()

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
                ["Check that the AccelPapers database has been indexed."],
            )
        )
