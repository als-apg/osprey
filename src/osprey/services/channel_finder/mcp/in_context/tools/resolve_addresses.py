"""MCP tool: resolve_addresses -- resolve channel names to PV addresses.

Resolves descriptive channel names into actual EPICS PV addresses by calling
database.get_channel(name)["address"].
"""

import json
import logging

from osprey.services.channel_finder.mcp.in_context.registry import get_cf_ic_registry
from osprey.services.channel_finder.mcp.in_context.server import make_error, mcp

logger = logging.getLogger("osprey.services.channel_finder.mcp.in_context.tools.resolve_addresses")


@mcp.tool()
def resolve_addresses(channels: list[str]) -> str:
    """Resolve channel names to their control system PV addresses.

    Channel names (descriptive PascalCase identifiers) differ from PV addresses
    (the actual EPICS process variable names used by the control system).
    Call this AFTER matching channels to get the real addresses for submit_response.

    Args:
        channels: List of channel names from the database.

    Returns:
        JSON with resolved addresses and any unresolved names.
    """
    try:
        registry = get_cf_ic_registry()
        db = registry.database

        resolved = []
        unresolved = []

        for name in channels:
            ch = db.get_channel(name)
            if ch is not None and "address" in ch:
                resolved.append({"channel": name, "address": ch["address"]})
            else:
                unresolved.append(name)

        return json.dumps(
            {
                "resolved": resolved,
                "addresses": [r["address"] for r in resolved],
                "unresolved": unresolved,
                "total": len(channels),
                "valid_count": len(resolved),
                "invalid_count": len(unresolved),
            }
        )

    except ValueError as exc:
        return json.dumps(make_error("validation_error", str(exc)))
    except Exception as exc:
        logger.exception("resolve_addresses failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to resolve addresses: {exc}",
                [
                    "Check that the channel database is loaded.",
                    "Verify channel names are exact matches from get_channels.",
                ],
            )
        )
