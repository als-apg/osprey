"""MCP tool: status — report ARIEL service health and statistics."""

import json
import logging

from osprey.interfaces.ariel.mcp.registry import get_ariel_registry
from osprey.interfaces.ariel.mcp.server import make_error, mcp

logger = logging.getLogger("osprey.interfaces.ariel.mcp.tools.status")


@mcp.tool()
async def status() -> str:
    """Get ARIEL service status including health, database connectivity,
    entry counts, embedding tables, and enabled modules.

    Returns:
        JSON with comprehensive service status.
    """
    try:
        registry = get_ariel_registry()
        service = await registry.service()

        status = await service.get_status()

        # Serialize ARIELStatusResult (dataclass — use attribute access)
        return json.dumps(
            {
                "healthy": status.healthy,
                "database_connected": status.database_connected,
                "database_uri": status.database_uri,
                "entry_count": status.entry_count,
                "embedding_tables": [
                    {
                        "table_name": t.table_name,
                        "entry_count": t.entry_count,
                        "dimension": t.dimension,
                        "is_active": t.is_active,
                    }
                    for t in status.embedding_tables
                ],
                "active_embedding_model": status.active_embedding_model,
                "enabled_search_modules": status.enabled_search_modules,
                "enabled_pipelines": status.enabled_pipelines,
                "enabled_enhancement_modules": status.enabled_enhancement_modules,
                "last_ingestion": status.last_ingestion,
                "errors": status.errors,
            },
            default=str,
        )

    except Exception as exc:
        logger.exception("status failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to get status: {exc}",
                [
                    "Check ARIEL database connectivity.",
                    "Verify config.yml has correct ariel.database settings.",
                ],
            )
        )
