"""MCP tool: get_pv_metadata — retrieve detailed metadata for specific PVs."""

import asyncio
import json
import logging
from dataclasses import asdict

from osprey.mcp_server.direct_channel_finder.server import make_error, mcp

logger = logging.getLogger("osprey.mcp_server.direct_channel_finder.tools.get_pv_metadata")


@mcp.tool()
def get_pv_metadata(pv_names: list[str]) -> str:
    """Get detailed metadata for specific PV names.

    Args:
        pv_names: List of exact PV names to look up (max 100).

    Returns:
        JSON with metadata records for each found PV.
        PVs not found in the backend are listed separately.
    """
    try:
        if len(pv_names) > 100:
            return json.dumps(
                make_error(
                    "too_many_pvs",
                    f"Requested {len(pv_names)} PVs, maximum is 100.",
                    suggestions=["Split into multiple calls of 100 PVs or fewer"],
                )
            )

        from osprey.mcp_server.direct_channel_finder.registry import get_dcf_registry

        registry = get_dcf_registry()
        backend = registry.backend

        records = asyncio.run(backend.get_metadata(pv_names))
        found_names = {r.name for r in records}
        not_found = [name for name in pv_names if name not in found_names]

        return json.dumps(
            {
                "records": [asdict(r) for r in records],
                "found_count": len(records),
                "not_found": not_found,
                "not_found_count": len(not_found),
            }
        )

    except RuntimeError as exc:
        return json.dumps(
            make_error(
                "backend_not_available",
                str(exc),
                suggestions=[
                    "Check that config.yml has channel_finder.direct.backend configured",
                ],
            )
        )
    except Exception as exc:
        logger.exception("get_pv_metadata failed")
        return json.dumps(
            make_error(
                "metadata_error",
                f"Metadata lookup failed: {exc}",
                suggestions=["Verify the PV names are correct"],
            )
        )
