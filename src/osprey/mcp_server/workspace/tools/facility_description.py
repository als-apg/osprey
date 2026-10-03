"""MCP tool: facility_description.

Returns the hand-written facility description (.claude/rules/facility.md) and
the build's generated facts page (data/facility_facts.md) so sub-agents can
understand facility-specific context, terminology, and operational details.
"""

import json
import logging
from pathlib import Path

from fastmcp.exceptions import ToolError

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.workspace.server import mcp
from osprey.utils.workspace import resolve_config_path

logger = logging.getLogger("osprey.mcp_server.tools.facility_description")


def _read_page(path: Path) -> str | None:
    """Read one page of the project.

    Args:
        path: The page's file.

    Returns:
        The page's text, or ``None`` when the project holds no such file.
    """
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


@mcp.tool()
async def facility_description() -> str:
    """Get facility description and context.

    Reads two pages from the project root: the hand-written description in
    .claude/rules/facility.md (facility identity, systems, terminology, and
    operational context) and the facts page the build generates in
    data/facility_facts.md (identity, place levels, device classes, models and
    channel count).

    Returns:
        JSON with ``facility_description`` (the hand-written text, or null),
        ``generated`` (the facts page's text, or null) and ``source`` (the path
        of each), or an error envelope if neither file is found.
    """
    try:
        from osprey.facility.views.facts import FACTS_PAGE

        project_root = resolve_config_path().parent
        facility_file = project_root / ".claude" / "rules" / "facility.md"
        generated_source = f"data/{FACTS_PAGE}"

        content = _read_page(facility_file)
        generated = _read_page(project_root / "data" / FACTS_PAGE)

        if content is None and generated is None:
            return make_error(
                "not_found",
                f"No facility description found at .claude/rules/facility.md or {generated_source}",
                [
                    f"Run `osprey build` to write {generated_source}.",
                    "Or create .claude/rules/facility.md manually.",
                    "The file should describe your facility's identity, "
                    "systems, terminology, and operational context.",
                ],
            )

        return json.dumps(
            {
                "facility_description": content,
                "generated": generated,
                "source": {
                    "facility_description": None if content is None else str(facility_file),
                    "generated": generated_source,
                },
            }
        )

    except ToolError:
        raise
    except Exception as exc:
        logger.exception("facility_description failed")
        return make_error(
            "internal_error",
            f"Failed to read facility description: {exc}",
            ["Check that .claude/rules/facility.md and data/facility_facts.md are readable."],
        )
