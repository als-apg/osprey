"""MCP tool: papers_search — hybrid BM25 + vector search of accelerator physics papers."""

import json
import logging

from osprey.mcp_server.accelpapers import db
from osprey.mcp_server.accelpapers.server import mcp
from osprey.mcp_server.accelpapers.tools._filters import build_filter_string
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
    """Search accelerator physics papers using hybrid BM25 + vector search.

    Searches across titles, abstracts, author names, keywords, and full paper text.
    Results are ranked by a blend of keyword relevance (BM25) and semantic similarity.

    Args:
        query: Search terms (e.g. "beam position monitor", "RF cavity design").
        max_results: Maximum results to return (1-100, default 20).
        conference: Filter by conference name (e.g. "IPAC2023", "NAPAC2016").
        year_min: Filter papers from this year onward.
        year_max: Filter papers up to this year.
        author: Filter by author name (partial match in all_authors field).
        document_type: Filter by type (e.g. "article", "conference paper").

    Returns:
        JSON with matching papers, relevance scores, and text snippets.
    """
    if not query or not query.strip():
        return json.dumps(
            make_error("validation_error", "Empty search query.", ["Provide search terms."])
        )

    max_results = max(1, min(100, max_results))

    try:
        client = db.get_client()
        collection = db.get_collection_name()

        search_params: dict = {
            "q": query.strip(),
            "query_by": "title,abstract,all_authors,keywords,full_text,embedding",
            "prefix": "true,true,true,true,true,false",
            "highlight_full_fields": "abstract",
            "highlight_start_tag": ">>>",
            "highlight_end_tag": "<<<",
            "per_page": max_results,
            "exclude_fields": "embedding,full_text",
        }

        filter_str = build_filter_string(
            conference=conference,
            year_min=year_min,
            year_max=year_max,
            author=author,
            document_type=document_type,
        )
        if filter_str:
            search_params["filter_by"] = filter_str

        result = client.collections[collection].documents.search(search_params)

        results = []
        for hit in result.get("hits", []):
            doc = hit["document"]
            # Extract highlighted abstract snippet if available
            highlight = hit.get("highlight", {})
            snippet = ""
            if "abstract" in highlight:
                abs_highlight = highlight["abstract"]
                if isinstance(abs_highlight, dict):
                    snippet = abs_highlight.get("snippet", "")
                elif isinstance(abs_highlight, list) and abs_highlight:
                    snippet = abs_highlight[0].get("snippet", "")

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
                "snippet": snippet,
                "text_match_score": hit.get("text_match_info", {}).get("score", 0),
            })

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
                ["Check that the Typesense server is running and the collection is indexed."],
            )
        )
