"""MCP tool: papers_search_author — author-focused FTS5 search."""

import json
import logging

from osprey.mcp_server.accelpapers.db import get_connection
from osprey.mcp_server.accelpapers.server import mcp
from osprey.mcp_server.common import make_error

logger = logging.getLogger("osprey.mcp_server.accelpapers.tools.search_author")


@mcp.tool()
async def papers_search_author(
    author: str,
    max_results: int = 20,
    year_min: int | None = None,
    year_max: int | None = None,
) -> str:
    """Search for papers by a specific author using full-text search.

    Searches the author names field using FTS5 column filtering for precise
    author matching. Results are ranked by citation count.

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
        conn = get_connection()

        # FTS5 column filter: search only the all_authors column
        safe_author = author.strip()
        fts_query = f"all_authors:{safe_author}"

        where_parts = []
        params: list = []

        if year_min is not None:
            where_parts.append("p.year >= ?")
            params.append(year_min)
        if year_max is not None:
            where_parts.append("p.year <= ?")
            params.append(year_max)

        where_clause = ""
        if where_parts:
            where_clause = "AND " + " AND ".join(where_parts)

        sql = f"""
            SELECT p.texkey, p.title, p.first_author, p.all_authors, p.year,
                   p.conference, p.citation_count, p.document_type, p.doi,
                   p.inspire_url, p.journal_title, p.arxiv_id
            FROM papers_fts fts
            JOIN papers p ON p.rowid = fts.rowid
            WHERE papers_fts MATCH ?
            {where_clause}
            ORDER BY p.citation_count DESC
            LIMIT ?
        """

        rows = conn.execute(sql, [fts_query] + params + [max_results]).fetchall()

        results = []
        for row in rows:
            results.append({
                "texkey": row["texkey"],
                "title": row["title"],
                "first_author": row["first_author"],
                "all_authors": row["all_authors"],
                "year": row["year"],
                "conference": row["conference"],
                "citation_count": row["citation_count"],
                "document_type": row["document_type"],
                "doi": row["doi"],
                "inspire_url": row["inspire_url"],
                "journal_title": row["journal_title"],
                "arxiv_id": row["arxiv_id"],
            })

        conn.close()

        return json.dumps({
            "author_query": author,
            "filters": {
                k: v for k, v in {
                    "year_min": year_min,
                    "year_max": year_max,
                }.items() if v is not None
            },
            "results_found": len(results),
            "papers": results,
        })

    except Exception as exc:
        logger.exception("papers_search_author failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Author search failed: {exc}",
                ["Check that the AccelPapers database has been indexed."],
            )
        )
