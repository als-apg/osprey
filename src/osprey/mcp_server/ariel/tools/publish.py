"""MCP tool: entry_publish — publish an existing ARIEL entry to the facility logbook."""

import json
import logging

from osprey.mcp_server.ariel.server import make_error, mcp
from osprey.mcp_server.ariel.server_context import get_ariel_context

logger = logging.getLogger("osprey.mcp_server.ariel.tools.publish")


@mcp.tool()
async def entry_publish(
    entry_id: str,
    logbook: str | None = None,
) -> str:
    """Publish an existing ARIEL entry to the configured facility logbook.

    Writes through to the upstream source (e.g., facility logbook, JSON file)
    and ingests back with the facility-assigned ID. The ARIEL database entry
    will have the canonical ID from the source of truth.

    Args:
        entry_id: The ID of the existing ARIEL entry to publish.
        logbook: Target logbook name (required by some facility APIs).

    Returns:
        JSON with the facility-assigned entry_id, source_system, sync_status, and message.
    """
    if not entry_id or not entry_id.strip():
        return make_error(
                "validation_error",
                "entry_id is required.",
                ["Provide a valid entry ID."],
            )

    try:
        registry = get_ariel_context()
        service = await registry.service()

        result = await service.publish_entry(entry_id, logbook=logbook)

        return json.dumps(
            {
                "entry_id": result.entry_id,
                "source_system": result.source_system,
                "sync_status": result.sync_status.value,
                "message": result.message,
            },
            default=str,
        )

    except KeyError:
        return make_error(
                "not_found",
                f"Entry {entry_id} not found.",
                ["Check the entry_id is correct."],
            )
    except NotImplementedError as exc:
        return make_error(
                "not_supported",
                str(exc),
                ["The configured adapter does not support writing entries."],
            )
    except Exception as exc:
        logger.exception("entry_publish failed for %s", entry_id)
        return make_error(
                "internal_error",
                f"Failed to publish entry: {exc}",
                ["Check the ARIEL service logs for details."],
            )
