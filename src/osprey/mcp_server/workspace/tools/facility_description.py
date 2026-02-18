"""MCP tool: facility_description.

Reads the facility description file (.claude/rules/facility.md) and returns
its content so sub-agents can understand facility-specific context, terminology,
and operational details.
"""

import json
import logging

from osprey.mcp_server.common import make_error, resolve_config_path
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.facility_description")


@mcp.tool()
async def facility_description() -> str:
    """Get facility description and context.

    Reads the facility description from .claude/rules/facility.md in the
    project root. This file contains facility identity, systems, terminology,
    and operational context that helps agents understand the environment
    they are operating in.

    Returns:
        JSON with facility description content, or an error envelope if the
        file is not found.
    """
    try:
        # Derive project root from config.yml location
        project_root = resolve_config_path().parent
        facility_file = project_root / ".claude" / "rules" / "facility.md"

        if not facility_file.exists():
            return json.dumps(
                make_error(
                    "not_found",
                    "Facility description file not found at .claude/rules/facility.md",
                    [
                        "Run `osprey claude regen` to create the file.",
                        "Or create .claude/rules/facility.md manually.",
                        "The file should describe your facility's identity, "
                        "systems, terminology, and operational context.",
                    ],
                )
            )

        content = facility_file.read_text(encoding="utf-8")

        return json.dumps(
            {
                "facility_description": content,
                "source": str(facility_file),
            }
        )

    except Exception as exc:
        logger.exception("facility_description failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to read facility description: {exc}",
                ["Check that .claude/rules/facility.md is readable."],
            )
        )
