"""MCP tools: list_panels, switch_panel.

list_panels returns the panels available in the Web Terminal.
switch_panel directs the Web Terminal to activate a specific panel tab.
"""

import json
import logging

from osprey.mcp_server.http import notify_panel_focus, web_terminal_url
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.panel")


@mcp.tool()
async def list_panels() -> str:
    """List the panels available in the Web Terminal.

    Returns the enabled built-in panels (e.g. artifacts, ariel, tuning,
    channel-finder, lattice) and any custom panels defined in config.yml.

    Returns:
        JSON with ``panels`` list, each entry having ``id`` and ``label``.
    """
    import urllib.request

    base = web_terminal_url()
    try:
        with urllib.request.urlopen(f"{base}/api/panels", timeout=3) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        logger.warning("list_panels: web terminal unreachable: %s", exc)
        return json.dumps(
            {
                "status": "error",
                "message": "Web Terminal is not running — panel list unavailable.",
            }
        )

    # Built-in panel labels keyed by id
    labels = {
        "artifacts": "WORKSPACE",
        "ariel": "ARIEL",
        "tuning": "TUNING",
        "channel-finder": "CHANNELS",
        "lattice": "LATTICE",
    }

    panels = [
        {"id": pid, "label": labels.get(pid, pid.upper())}
        for pid in data.get("enabled", [])
    ]
    for cp in data.get("custom", []):
        panels.append({"id": cp["id"], "label": cp.get("label", cp["id"].upper())})

    return json.dumps({"status": "success", "panels": panels})


@mcp.tool()
async def switch_panel(panel_id: str, url: str | None = None) -> str:
    """Switch the Web Terminal to show a specific panel tab.

    Use this when the user asks to open, show, or switch to a panel.
    Also useful after producing content relevant to a particular panel.

    IMPORTANT: Always call list_panels first to discover the actual panel
    IDs available in this deployment.  Do NOT guess panel IDs — they
    vary between deployments and include custom panels.

    Args:
        panel_id: Panel identifier returned by list_panels (e.g.
            'artifacts', 'events', 'ariel', 'tuning', etc.).
        url: Optional URL to navigate the panel iframe to.

    Returns:
        JSON with status confirmation.
    """
    notify_panel_focus(panel_id, url=url)
    return json.dumps({"status": "success", "panel": panel_id})
