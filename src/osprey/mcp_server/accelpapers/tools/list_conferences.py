"""MCP tool: papers_list_conferences — list conferences with paper counts."""

import json
import logging

from osprey.mcp_server.accelpapers.db import get_connection
from osprey.mcp_server.accelpapers.server import mcp
from osprey.mcp_server.common import make_error

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
        pattern: Optional filter pattern for conference names (case-insensitive LIKE match).
        min_papers: Minimum number of papers to include a conference (default 1).
        max_results: Maximum conferences to return (1-500, default 50).

    Returns:
        JSON with conference names and paper counts.
    """
    max_results = max(1, min(500, max_results))
    min_papers = max(1, min_papers)

    try:
        conn = get_connection()

        if pattern:
            sql = """
                SELECT conference, COUNT(*) as paper_count
                FROM papers
                WHERE conference != '' AND conference LIKE ?
                GROUP BY conference
                HAVING paper_count >= ?
                ORDER BY paper_count DESC
                LIMIT ?
            """
            rows = conn.execute(sql, [f"%{pattern}%", min_papers, max_results]).fetchall()
        else:
            sql = """
                SELECT conference, COUNT(*) as paper_count
                FROM papers
                WHERE conference != ''
                GROUP BY conference
                HAVING paper_count >= ?
                ORDER BY paper_count DESC
                LIMIT ?
            """
            rows = conn.execute(sql, [min_papers, max_results]).fetchall()

        conferences = [
            {"conference": row["conference"], "paper_count": row["paper_count"]}
            for row in rows
        ]

        conn.close()

        return json.dumps({
            "pattern": pattern,
            "min_papers": min_papers,
            "conferences_found": len(conferences),
            "conferences": conferences,
        })

    except Exception as exc:
        logger.exception("papers_list_conferences failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"List conferences failed: {exc}",
                ["Check that the AccelPapers database has been indexed."],
            )
        )
