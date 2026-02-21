"""MCP tool: inspect_fields — inspect field structure of a device family.

PROMPT-PROVIDER: This tool's docstring is a static prompt visible to Claude Code.
  Future: source from FrameworkPromptProvider.get_middle_layer_prompt_builder()
  Facility-customizable: field type descriptions (Monitor, Setpoint, etc.),
  system/family name examples
"""

import json
import logging

from osprey.services.channel_finder.mcp.middle_layer.registry import get_cf_ml_registry
from osprey.services.channel_finder.mcp.middle_layer.server import make_error, mcp

logger = logging.getLogger("osprey.services.channel_finder.mcp.middle_layer.tools.inspect_fields")


@mcp.tool()
def inspect_fields(
    system: str,
    family: str,
    field: str | None = None,
) -> str:
    """Inspect the field structure of a device family.

    Shows what fields are available (Monitor, Setpoint, etc.) and their types.
    Optionally drill into a specific field to see its subfields.

    Args:
        system: System name (e.g., "SR").
        family: Family name (e.g., "BPM").
        field: Optional specific field to inspect deeper.

    Returns:
        JSON with field names mapped to type and description info.
    """
    try:
        registry = get_cf_ml_registry()
        fields = registry.database.inspect_fields(system, family, field)

        return json.dumps({"fields": fields})

    except ValueError as exc:
        return json.dumps(
            make_error(
                "validation_error",
                str(exc),
                [
                    "Use list_systems to see available systems.",
                    "Use list_families to see families in a system.",
                ],
            )
        )
    except Exception as exc:
        logger.exception("inspect_fields failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to inspect fields: {exc}",
                ["Check that the channel finder database is configured."],
            )
        )
