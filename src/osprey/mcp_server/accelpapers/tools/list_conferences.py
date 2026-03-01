"""MCP tool: papers_list_conferences — list conferences with paper counts."""

import json
import logging

from osprey.mcp_server.accelpapers import db
from osprey.mcp_server.accelpapers.server import mcp
from osprey.mcp_server.errors import make_error

logger = logging.getLogger("osprey.mcp_server.accelpapers.tools.list_conferences")


@mcp.tool()
async def papers_list_conferences(
    pattern: str | None = None,
    min_papers: int = 1,
    max_results: int = 50,
) -> str:
    """List conferences in the accelerator physics paper database.

    Returns conferences with paper counts, sorted by count descending.
    Use pattern to filter by conference name (e.g. "IPAC" to find all IPAC conferences).

    Args:
        pattern: Optional filter pattern for conference names (case-insensitive substring match).
        min_papers: Minimum number of papers to include a conference (default 1).
        max_results: Maximum conferences to return (1-500, default 50).

    Returns:
        JSON with conference names and paper counts.
    """
    max_results = max(1, min(500, max_results))
    min_papers = max(1, min_papers)

    try:
        client = db.get_client()
        collection = db.get_collection_name()

        search_params: dict = {
            "q": "*",
            "facet_by": "conference",
            "max_facet_values": 500,
            "per_page": 0,
        }

        result = client.collections[collection].documents.search(search_params)

        facet_counts = result.get("facet_counts", [])
        facets: list[dict] = []
        if facet_counts:
            facets = facet_counts[0].get("counts", [])

        # Post-filter: pattern (substring match) and min_papers (count threshold)
        if pattern:
            pattern_upper = pattern.upper()
            facets = [f for f in facets if pattern_upper in f["value"].upper()]

        facets = [f for f in facets if f["count"] >= min_papers]
        facets = facets[:max_results]

        conferences = [{"conference": f["value"], "paper_count": f["count"]} for f in facets]

        return json.dumps(
            {
                "pattern": pattern,
                "min_papers": min_papers,
                "conferences_found": len(conferences),
                "conferences": conferences,
            }
        )

    except Exception as exc:
        logger.exception("papers_list_conferences failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"List conferences failed: {exc}",
                ["Check that the Typesense server is running and the collection is indexed."],
            )
        )
