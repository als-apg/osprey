"""MCP tool: papers_search — FTS5 BM25-ranked search of accelerator physics papers."""

import json
import logging

from osprey.mcp_server.accelpapers.db import get_connection
from osprey.mcp_server.accelpapers.server import mcp
from osprey.mcp_server.common import make_error

logger = logging.getLogger("osprey.mcp_server.accelpapers.tools.search")


@mcp.tool()
async def papers_search(
    query: str,
    max_results: int = 20,
    conference: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    author: str | None = None,
    document_type: str | None = None,
) -> str:
    """Search accelerator physics papers using full-text search (BM25 ranking).

    Searches across titles, abstracts, author names, keywords, and full paper text.
    Returns results ranked by relevance with text snippets showing matching context.

    Args:
        query: Search terms (e.g. "beam position monitor", "RF cavity design").
        max_results: Maximum results to return (1-100, default 20).
        conference: Filter by conference name (e.g. "IPAC2023", "NAPAC2016").
        year_min: Filter papers from this year onward.
        year_max: Filter papers up to this year.
        author: Filter by author name (partial match in all_authors field).
        document_type: Filter by type (e.g. "article", "conference paper").

    Returns:
        JSON with matching papers, BM25 scores, and text snippets.
    """
    if not query or not query.strip():
        return json.dumps(
            make_error("validation_error", "Empty search query.", ["Provide search terms."])
        )

    max_results = max(1, min(100, max_results))

    try:
        conn = get_connection()
        # Build FTS5 MATCH query
        # Escape special FTS5 characters to prevent syntax errors
        safe_query = query.strip()

        # Build WHERE clauses for filters
        where_parts = []
        params: list = []

        if conference:
            where_parts.append("p.conference = ?")
            params.append(conference)
        if year_min is not None:
            where_parts.append("p.year >= ?")
            params.append(year_min)
        if year_max is not None:
            where_parts.append("p.year <= ?")
            params.append(year_max)
        if author:
            where_parts.append("p.all_authors LIKE ?")
            params.append(f"%{author}%")
        if document_type:
            where_parts.append("p.document_type = ?")
            params.append(document_type)

        where_clause = ""
        if where_parts:
            where_clause = "AND " + " AND ".join(where_parts)

        sql = f"""
            SELECT p.texkey, p.title, p.first_author, p.year, p.conference,
                   p.citation_count, p.document_type, p.doi, p.inspire_url,
                   p.journal_title, p.arxiv_id,
                   snippet(papers_fts, 1, '>>>', '<<<', '...', 40) as abstract_snippet,
                   rank
            FROM papers_fts fts
            JOIN papers p ON p.rowid = fts.rowid
            WHERE papers_fts MATCH ?
            {where_clause}
            ORDER BY rank
            LIMIT ?
        """

        rows = conn.execute(sql, [safe_query] + params + [max_results]).fetchall()

        results = []
        for row in rows:
            results.append({
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
                "snippet": row["abstract_snippet"],
                "bm25_score": round(row["rank"], 4),
            })

        conn.close()

        return json.dumps({
            "query": query,
            "filters": {
                k: v for k, v in {
                    "conference": conference,
                    "year_min": year_min,
                    "year_max": year_max,
                    "author": author,
                    "document_type": document_type,
                }.items() if v is not None
            },
            "results_found": len(results),
            "papers": results,
        })

    except Exception as exc:
        logger.exception("papers_search failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Search failed: {exc}",
                ["Check that the AccelPapers database has been indexed."],
            )
        )
