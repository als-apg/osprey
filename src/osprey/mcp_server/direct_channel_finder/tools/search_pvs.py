"""MCP tool: search_pvs — search for PVs matching a glob pattern with filters."""

import asyncio
import json
import logging
from dataclasses import asdict

from osprey.mcp_server.direct_channel_finder.server import make_error, mcp

logger = logging.getLogger("osprey.mcp_server.direct_channel_finder.tools.search_pvs")


@mcp.tool()
def search_pvs(
    pattern: str,
    record_type: str | None = None,
    ioc: str | None = None,
    description_contains: str | None = None,
    page: int = 1,
    page_size: int = 100,
) -> str:
    """Search for PVs matching a glob pattern with optional filters.

    Args:
        pattern: Glob pattern for PV names (e.g., "SR:*:BPM:*", "SR:01:*").
            Use ``*`` to match any sequence of characters within a segment.
        record_type: Filter by EPICS record type (e.g., "ai", "ao").
        ioc: Filter by IOC name.
        description_contains: Filter by substring in PV description.
        page: Page number (1-indexed, default 1).
        page_size: Results per page (1-200, default 100).

    Returns:
        JSON with matching PVs, pagination info, and total count.
    """
    try:
        from osprey.mcp_server.direct_channel_finder.server_context import get_dcf_context

        registry = get_dcf_context()
        backend = registry.backend

        page_size = max(1, min(page_size, 200))
        page = max(1, page)

        result = asyncio.run(
            backend.search(
                pattern,
                record_type=record_type,
                ioc=ioc,
                description_contains=description_contains,
                page=page,
                page_size=page_size,
            )
        )

        return json.dumps(
            {
                "records": [asdict(r) for r in result.records],
                "total_count": result.total_count,
                "has_more": result.has_more,
                "page": result.page,
                "page_size": result.page_size,
                "pattern": pattern,
                "filters": {
                    k: v
                    for k, v in {
                        "record_type": record_type,
                        "ioc": ioc,
                        "description_contains": description_contains,
                    }.items()
                    if v is not None
                },
            }
        )

    except RuntimeError as exc:
        return json.dumps(
            make_error(
                "backend_not_available",
                str(exc),
                suggestions=[
                    "Check that config.yml has channel_finder.direct.backend configured",
                    "Supported backends: mock",
                ],
            )
        )
    except Exception as exc:
        logger.exception("search_pvs failed")
        return json.dumps(
            make_error(
                "search_error",
                f"Search failed: {exc}",
                suggestions=["Check the pattern syntax (glob patterns, e.g., 'SR:*:BPM:*')"],
            )
        )
