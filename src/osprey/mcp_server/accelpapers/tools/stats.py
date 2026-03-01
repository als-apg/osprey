"""MCP tool: papers_stats — collection statistics from Typesense."""

import json
import logging

from osprey.mcp_server.accelpapers import db
from osprey.mcp_server.accelpapers.server import mcp
from osprey.mcp_server.errors import make_error

logger = logging.getLogger("osprey.mcp_server.accelpapers.tools.stats")


@mcp.tool()
async def papers_stats() -> str:
    """Get summary statistics for the accelerator physics paper database.

    Returns statistics including total papers, year range, conference count,
    author count, and document type breakdown computed from Typesense facets.

    Returns:
        JSON with database statistics.
    """
    try:
        client = db.get_client()
        collection = db.get_collection_name()

        # Get collection info for total document count
        collection_info = client.collections[collection].retrieve()
        total_papers = collection_info.get("num_documents", 0)

        if total_papers == 0:
            return json.dumps(
                make_error(
                    "not_found",
                    "No papers in collection — database may not be indexed.",
                    ["Run the indexer first: python -m osprey.mcp_server.accelpapers index"],
                )
            )

        # Facet query for year, document_type, conference, first_author
        search_params: dict = {
            "q": "*",
            "facet_by": "year,document_type,conference,first_author",
            "max_facet_values": 500,
            "per_page": 0,
        }
        result = client.collections[collection].documents.search(search_params)

        facet_data: dict[str, list[dict]] = {}
        for facet in result.get("facet_counts", []):
            facet_data[facet["field_name"]] = facet.get("counts", [])

        # Year range
        year_values = [int(f["value"]) for f in facet_data.get("year", []) if f["value"]]
        year_min = min(year_values) if year_values else None
        year_max = max(year_values) if year_values else None

        # Conference count
        num_conferences = len(facet_data.get("conference", []))

        # Author count
        num_authors = len(facet_data.get("first_author", []))

        # Document type breakdown
        document_types = {f["value"]: f["count"] for f in facet_data.get("document_type", [])}

        # Top 10 conferences by paper count
        conference_facets = facet_data.get("conference", [])
        top_conferences = {f["value"]: f["count"] for f in conference_facets[:10]}

        stats = {
            "total_papers": total_papers,
            "year_min": year_min,
            "year_max": year_max,
            "num_conferences": num_conferences,
            "num_authors": num_authors,
            "document_types": document_types,
            "top_conferences": top_conferences,
        }

        return json.dumps({"statistics": stats})

    except Exception as exc:
        logger.exception("papers_stats failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Stats retrieval failed: {exc}",
                ["Check that the Typesense server is running and the collection is indexed."],
            )
        )
