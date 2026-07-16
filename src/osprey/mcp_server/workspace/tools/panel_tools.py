"""MCP tools: list_panels, show_panel, hide_panel, register_panel, switch_panel.

list_panels returns the panels available in the Web Terminal together with
their visibility and active state.
show_panel / hide_panel toggle a panel's visibility flag.
register_panel dynamically registers a new panel (requires deployment opt-in).
switch_panel directs the Web Terminal to activate a specific panel tab.
"""

import json
import logging

from osprey.mcp_server.http import (
    notify_panel_focus,
    notify_panel_register,
    notify_panel_visibility,
    web_terminal_url,
)
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.panel")


def _fetch_panels(base: str) -> dict | None:
    """Fetch and parse ``GET /api/panels`` from the Web Terminal.

    Single source of truth for the panel-inventory fetch shared by
    :func:`list_panels` and :func:`_fetch_known_panel_ids`.

    Returns:
        The parsed JSON payload, or ``None`` when the web terminal is
        unreachable (the caller decides how to surface that).
    """
    import urllib.request

    try:
        with urllib.request.urlopen(f"{base}/api/panels", timeout=3) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        logger.warning("panel fetch: web terminal unreachable: %s", exc)
        return None


@mcp.tool()
async def list_panels() -> str:
    """List the panels available in the Web Terminal.

    Returns the enabled built-in panels (e.g. artifacts, ariel,
    channel-finder, lattice, okf, events) and any custom panels defined in
    config.yml, together with their current visibility state and the
    currently active panel.

    The current panel inventory (ids, labels, visibility, active tab) is
    provided in your context at session start.  Call this tool to refresh
    the live state (e.g. after visibility changes) or when the inventory
    is not present.  Panel IDs vary between deployments and include custom
    panels defined in config.yml.

    Returns:
        JSON with ``status``, ``active`` (id of the currently visible panel,
        or null), ``panels`` list, and ``presets`` (config-defined layouts —
        each ``{name, panels: [id, ...]}``; empty unless the deployment defines
        any).  Each panel entry has ``id``, ``label``, and ``visible`` (bool —
        whether the tab is shown in the tab bar).  To honor a request like "set
        up for machine setup," look up the named preset and compose the result
        with ``show_panel``/``hide_panel`` — the same primitive a human's click
        resolves to.
    """
    base = web_terminal_url()
    data = _fetch_panels(base)
    if data is None:
        return json.dumps(
            {
                "status": "error",
                "message": "Web Terminal is not running — panel list unavailable.",
            }
        )

    # Built-in labels come from the server response (SSOT: osprey.profiles.web_panels).
    server_labels: dict[str, str] = data.get("labels", {})
    visible_ids: set[str] = set(data.get("visible", []))

    panels = [
        {
            "id": pid,
            "label": server_labels.get(pid, pid.upper()),
            "visible": pid in visible_ids,
        }
        for pid in data.get("enabled", [])
    ]
    for cp in data.get("custom", []):
        cid = cp["id"]
        panels.append(
            {
                "id": cid,
                "label": cp.get("label", cid.upper()),
                "visible": cid in visible_ids,
            }
        )

    return json.dumps(
        {
            "status": "success",
            "active": data.get("active"),
            "panels": panels,
            "presets": data.get("presets", []),
        }
    )


def _fetch_known_panel_ids(base: str) -> set[str] | None:
    """Fetch /api/panels and return the set of known panel ids.

    Returns None when the web terminal is unreachable.
    """
    data = _fetch_panels(base)
    if data is None:
        return None

    known: set[str] = set(data.get("enabled", []))
    for cp in data.get("custom", []):
        known.add(cp["id"])
    return known


def _set_panel_visibility(panel_id: str, visible: bool) -> str:
    """Validate ``panel_id`` against the live inventory, then toggle its tab.

    Shared by :func:`show_panel` and :func:`hide_panel`, which differ only in the
    ``visible`` flag.  Returns the JSON string those tools hand back to the agent:
    a structured error when the web terminal is unreachable or the id is unknown,
    otherwise a success payload echoing the new ``visible`` state.
    """
    base = web_terminal_url()
    known = _fetch_known_panel_ids(base)
    if known is None:
        return json.dumps(
            {
                "status": "error",
                "message": "Web Terminal is not running — panel list unavailable.",
            }
        )
    if panel_id not in known:
        return json.dumps(
            {
                "status": "error",
                "message": (
                    f"Unknown panel '{panel_id}'. Call list_panels to see available panels."
                ),
            }
        )
    notify_panel_visibility(panel_id, visible)
    return json.dumps({"status": "success", "panel": panel_id, "visible": visible})


@mcp.tool()
async def show_panel(panel_id: str) -> str:
    """Make a panel visible in the Web Terminal tab bar.

    Use this when the user asks to show, reveal, or enable a panel tab.

    Panel IDs are provided in your context at session start; call
    ``list_panels`` to refresh the live state if needed.  Do NOT guess
    panel IDs — they vary between deployments.

    Args:
        panel_id: Panel identifier (e.g. 'artifacts', 'ariel', 'lattice').

    Returns:
        JSON with status confirmation and the updated ``visible`` state.
    """
    return _set_panel_visibility(panel_id, True)


@mcp.tool()
async def hide_panel(panel_id: str) -> str:
    """Hide a panel from the Web Terminal tab bar.

    Use this when the user asks to hide, remove, or collapse a panel tab.

    Panel IDs are provided in your context at session start; call
    ``list_panels`` to refresh the live state if needed.  Do NOT guess
    panel IDs — they vary between deployments.

    Args:
        panel_id: Panel identifier (e.g. 'artifacts', 'ariel', 'lattice').

    Returns:
        JSON with status confirmation and the updated ``visible`` state.
    """
    return _set_panel_visibility(panel_id, False)


@mcp.tool()
async def register_panel(
    panel_id: str,
    label: str,
    url: str,
    path: str = "/",
    health_endpoint: str | None = None,
) -> str:
    """Register a new panel with the Web Terminal dynamically.

    Use this to add a custom panel (e.g. a Grafana dashboard or local web
    service) at runtime without restarting OSPREY.

    The current panel inventory is available in your context at session start;
    call ``list_panels`` to refresh if needed before registering a new panel.

    NOTE: Registration may be disabled by deployment policy.  The deployment
    must set ``web.allow_runtime_panels: true`` in config.yml for this tool
    to succeed.  A clear error is returned when registration is disabled so
    the operator can update the configuration.

    Args:
        panel_id: Unique panel identifier (used as the tab key).
        label: Human-readable panel name shown in the UI tab bar.
        url: Upstream URL the Web Terminal will proxy or embed.
        path: Sub-path to open inside *url* on first load (default ``"/"``).
        health_endpoint: Optional URL polled to determine panel health.
            Omit to skip health checking.

    Returns:
        JSON with status and the registered panel URL on success, or a
        structured error describing why registration was rejected.
    """
    result = notify_panel_register(panel_id, label, url, path, health_endpoint)
    if result["ok"]:
        return json.dumps(
            {
                "status": "success",
                "panel": panel_id,
                "url": result["data"].get("url"),
            }
        )
    status = result.get("status")
    if status is None:
        return json.dumps({"status": "error", "message": "Web Terminal is not running."})
    if status == 403:
        return json.dumps(
            {
                "status": "error",
                "message": (
                    "Runtime panel registration is disabled (set web.allow_runtime_panels: true)."
                ),
            }
        )
    if status == 422:
        return json.dumps({"status": "error", "message": result.get("detail", "URL rejected")})
    # Catch-all for unexpected server errors
    return json.dumps(
        {"status": "error", "message": result.get("detail", f"Server error ({status}).")}
    )


@mcp.tool()
async def switch_panel(panel_id: str, url: str | None = None) -> str:
    """Switch the Web Terminal to show a specific panel tab.

    Use this when the user asks to open, show, or switch to a panel.
    Also useful after producing content relevant to a particular panel.

    Panel IDs are provided in your context at session start; call
    ``list_panels`` to refresh the live state if needed.  Do NOT guess
    panel IDs — they vary between deployments.

    Args:
        panel_id: Panel identifier returned by list_panels (e.g.
            'artifacts', 'events', 'ariel', 'lattice', etc.).
        url: Optional URL to navigate the panel iframe to.

    Returns:
        JSON with status confirmation.
    """
    notify_panel_focus(panel_id, url=url)
    return json.dumps({"status": "success", "panel": panel_id})
