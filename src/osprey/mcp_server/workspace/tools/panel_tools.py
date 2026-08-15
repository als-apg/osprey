"""MCP tools for the Web Terminal's panels.

A panel sits on two independent axes, and each verb here names the one it moves:

- **rail membership** — can the operator launch this panel in one click?
  ``add_panel_to_rail`` / ``remove_panel_from_rail``.
- **on screen** — is the panel's tile actually in front of the operator?
  ``open_panel`` / ``close_panel``.

Keeping the axes apart is the whole point of the names. The pairs used to be
``show_panel``/``hide_panel``/``switch_panel``, where ``show_panel`` moved rail
membership despite its name, ``hide_panel`` moved both axes at once, and the
only verb that put anything on screen was the one named for switching. Nothing
could close a tile without also making the panel unlaunchable, and the
docstrings spent more lines redirecting to each other than describing what they
did.

``list_panels`` reports both axes. ``register_panel`` adds a panel the
deployment did not ship (requires opt-in). ``arrange_workspace`` sets a whole
tile layout in one declarative call.
"""

import functools
import json

import anyio

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import (
    fetch_panels,
    notify_panel_arrange,
    notify_panel_close,
    notify_panel_focus,
    notify_panel_register,
    notify_panel_visibility,
)
from osprey.mcp_server.workspace.server import mcp


@mcp.tool()
async def list_panels() -> str:
    """List the panels available in the Web Terminal, and what is on screen.

    Two different things, easy to confuse:

    - ``visible`` — the panel has an entry in the **launcher rail**, so the
      operator can open it in one click.  It says nothing about whether the
      panel is on screen right now.
    - ``open_tiles`` — the service tiles **actually on screen**, in
      left-to-right reading order.  This is what the operator is looking at.

    Covers the enabled built-in panels (e.g. artifacts, ariel, channel-finder,
    lattice, okf, events) and any custom panels defined in config.yml.  The
    inventory is provided in your context at session start; call this tool to
    refresh the live state, or when you need to know what is on screen before
    rearranging it.  Panel IDs vary between deployments — do not guess them.

    How far to trust ``open_tiles``: it is whatever a browser last reported,
    and three cases read differently.

    - A list with ``open_tiles_dock: true`` is a real report — including an
      empty list, which means the operator closed every service tile.
    - ``open_tiles: null`` with ``open_tiles_dock: false`` means someone is
      watching but their client has no dock shell and cannot report tile
      order: occupancy is unknown, not empty.
    - ``open_tiles_dock: null`` (with a null age) means no client has ever
      reported — nobody is watching.

    ``open_tiles_age_s`` counts seconds since the arrangement last *changed*,
    so a large age means either that the operator has not moved a tile in a
    while or that their browser is gone; the two are indistinguishable from
    here.  Treat an old report as a hint, not as fact.

    To honor a request like "set up for machine setup", look up the named
    layout in ``presets`` and apply it with ``arrange_workspace(preset=...)``
    — the same operation a human's "Layouts" click performs.

    Returns:
        JSON with ``status``, ``active`` (the last focus the server recorded —
        an operator's tab click, an ``open_panel``, or the focus an arrange
        resolved to; it is an intent, so a client that could not honor it may
        be showing something else), ``panels``, ``open_tiles``,
        ``open_tiles_age_s``,
        ``open_tiles_dock``, and ``presets`` (config-defined layouts — each
        ``{name, panels: [id, ...]}``; empty unless the deployment defines
        any).  Each panel entry has ``id``, ``label`` and ``visible``.
    """
    data = await anyio.to_thread.run_sync(fetch_panels)
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
            "open_tiles": data.get("open_tiles"),
            "open_tiles_age_s": data.get("open_tiles_age_s"),
            "open_tiles_dock": data.get("open_tiles_dock"),
            "presets": data.get("presets", []),
        }
    )


def _fetch_known_panel_ids() -> set[str] | None:
    """Fetch /api/panels and return the set of known panel ids.

    Returns None when the web terminal is unreachable.
    """
    data = fetch_panels()
    if data is None:
        return None

    known: set[str] = set(data.get("enabled", []))
    for cp in data.get("custom", []):
        known.add(cp["id"])
    return known


def _set_rail_membership(panel_id: str, on_rail: bool) -> str:
    """Validate ``panel_id`` against the live inventory, then toggle its rail entry.

    Shared by :func:`add_panel_to_rail` and :func:`remove_panel_from_rail`, which
    differ only in the flag.  Returns the JSON string those tools hand back to
    the agent: a structured error when the web terminal is unreachable or the id
    is unknown, otherwise a success payload echoing the new ``on_rail`` state.

    Blocking: it makes two HTTP calls, so async callers run it in a worker
    thread rather than inline.
    """
    known = _fetch_known_panel_ids()
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
    notify_panel_visibility(panel_id, on_rail)
    return json.dumps({"status": "success", "panel": panel_id, "on_rail": on_rail})


@mcp.tool()
async def add_panel_to_rail(panel_id: str) -> str:
    """Make a panel launchable from the Web Terminal's rail, in one click.

    Rail membership only — this does NOT put the panel on screen.  Use
    ``open_panel`` for that.

    Use when the user asks to make a panel available, or to bring back one they
    took off the rail earlier.

    In the Simple web UI, which has the same rail but one locked workspace slot
    instead of a dock, adding a panel while the page is chat-only also brings up
    the workspace column — on that panel, when the slot is still empty.

    Panel IDs are provided in your context at session start; call
    ``list_panels`` to refresh the live state if needed.  Do NOT guess
    panel IDs — they vary between deployments.

    Args:
        panel_id: Panel identifier (e.g. 'artifacts', 'ariel', 'lattice').

    Returns:
        JSON with status confirmation and the panel's new ``on_rail`` state.
    """
    return await anyio.to_thread.run_sync(_set_rail_membership, panel_id, True)


@mcp.tool()
async def remove_panel_from_rail(panel_id: str) -> str:
    """Take a panel off the Web Terminal's rail, so it can no longer be launched.

    Any tile of it also closes on every connected client — a panel the operator
    cannot launch must not be left stranded on screen.  To clear the screen
    while keeping the panel one click away, use ``close_panel`` instead.

    Use when the user asks to remove a panel or get rid of it for good.

    Panel IDs are provided in your context at session start; call
    ``list_panels`` to refresh the live state if needed.  Do NOT guess
    panel IDs — they vary between deployments.

    Args:
        panel_id: Panel identifier (e.g. 'artifacts', 'ariel', 'lattice').

    Returns:
        JSON with status confirmation and the panel's new ``on_rail`` state.
    """
    return await anyio.to_thread.run_sync(_set_rail_membership, panel_id, False)


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
    service) at runtime without restarting OSPREY.  The new panel joins the
    launcher rail; it is not opened on screen — follow with ``open_panel``
    once its upstream is ready.

    The current panel inventory is available in your context at session start;
    call ``list_panels`` to refresh if needed before registering a new panel.

    NOTE: Registration may be disabled by deployment policy.  The deployment
    must set ``web.allow_runtime_panels: true`` in config.yml for this tool
    to succeed.  A clear error is returned when registration is disabled so
    the operator can update the configuration.

    Args:
        panel_id: Unique panel identifier (used as the rail-entry key).
        label: Human-readable panel name shown on the launcher rail.
        url: Upstream URL the Web Terminal will proxy or embed.
        path: Sub-path to open inside *url* on first load (default ``"/"``).
        health_endpoint: Optional URL polled to determine panel health.
            Omit to skip health checking.

    Returns:
        JSON with status and the registered panel URL on success, or a
        structured error describing why registration was rejected.
    """
    result = await anyio.to_thread.run_sync(
        notify_panel_register, panel_id, label, url, path, health_endpoint
    )
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
async def open_panel(panel_id: str, url: str | None = None) -> str:
    """Put a panel on screen in front of the operator.

    The "look at this" verb: use it when the user asks to open or go to a panel,
    and after producing something they should see (a new artifact, a drafted
    plan, a lattice view).

    Additive in the expert (dock) UX — an open tile is never evicted:

    - the panel's tile is already on screen → it is focused;
    - the panel is on the rail but has no tile → a tile opens *beside* the one
      the operator is currently on;
    - the panel is not on the rail → it is added to the rail, then opened beside.

    In the Simple web UI there is a single workspace slot, so the panel takes it
    over (and the workspace column appears if the page was chat-only).

    A panel whose backend is unhealthy does not open: nothing appears and focus
    is unchanged.

    One panel.  For an explicit multi-tile end state ("lattice next to
    artifacts", "set up for injection") use ``arrange_workspace`` instead.

    Panel IDs are provided in your context at session start; call
    ``list_panels`` to refresh the live state if needed.  Do NOT guess
    panel IDs — they vary between deployments.

    Args:
        panel_id: Panel identifier returned by list_panels (e.g.
            'artifacts', 'events', 'ariel', 'lattice', etc.).
        url: Optional URL to navigate the panel iframe to before opening it.

    Returns:
        JSON with status confirmation.
    """
    await anyio.to_thread.run_sync(functools.partial(notify_panel_focus, panel_id, url=url))
    return json.dumps({"status": "success", "panel": panel_id})


@mcp.tool()
async def close_panel(panel_id: str) -> str:
    """Take a panel's tile off the screen, leaving it one click away on the rail.

    The operator can reopen it from the rail with its state intact.  Use when
    the user asks to close, dismiss, or clear a panel but keep it available.  To
    take it off the rail as well use ``remove_panel_from_rail``; to state a whole
    layout at once use ``arrange_workspace``.

    Closing a panel with no tile open is a no-op, not an error.

    Panel IDs are provided in your context at session start; call
    ``list_panels`` to refresh the live state if needed.  Do NOT guess
    panel IDs — they vary between deployments.

    Args:
        panel_id: Panel identifier (e.g. 'artifacts', 'ariel', 'lattice').

    Returns:
        JSON with status confirmation.
    """
    await anyio.to_thread.run_sync(notify_panel_close, panel_id)
    return json.dumps({"status": "success", "panel": panel_id})


@mcp.tool()
async def arrange_workspace(
    tiles: list[str] | None = None,
    focus: str | None = None,
    preset: str | None = None,
) -> str:
    """Lay out the whole workspace: exactly these tiles on screen, in this order.

    Use this when the user describes an arrangement rather than a single panel
    — "put lattice next to artifacts", "just the orbit tools", "set up for
    injection".  For putting one panel on screen, use ``open_panel`` instead.

    This is a declarative end state, not a set of steps.  After it is applied:

    - exactly the listed panels have a tile, left to right in the order given;
    - panels not listed have their tiles closed; with ``tiles`` they **keep**
      their launcher-rail entry so the operator can reopen them, while a
      ``preset`` also drops non-members from the rail;
    - listed panels that were not in the rail are added to it;
    - the terminal tile is never moved, closed, or reordered.

    Pass either ``tiles`` or ``preset``, never both.  A ``preset`` names a
    layout from the deployment's config (``list_panels`` reports the available
    ones); pruning the rail to its members is exactly what an operator gets
    from clicking that layout in "Layouts".

    Focus is applied with a health fallback in the browser: the requested panel
    if it is healthy, otherwise the first healthy tile in the list, otherwise
    focus is left where it is — unless the arrangement closed the focused tile,
    in which case nothing is focused.  Health is only observable client-side,
    so the response echoes the focus that was *requested*, which may not be the
    one the operator ends up on.

    Args:
        tiles: Panel ids to leave open, in left-to-right order.  Must name at
            least one panel — an arrangement never empties the workspace.
        focus: Optional panel to focus; must be one of the resulting tiles.
        preset: Name of a configured layout to apply instead of ``tiles``.

    Returns:
        JSON with ``status``, the applied ``tiles`` and requested ``focus``,
        ``preset`` when one was applied, and ``open_tiles_age_s`` /
        ``open_tiles_dock``.  Those last two describe the *last layout report*,
        not the arrangement just requested — usually the state from before this
        call, since the client reports back on its own schedule.  A null
        ``open_tiles_dock`` means no client has ever reported, so there may be
        nobody there to apply this.  The age counts from the last time the
        arrangement *changed*, so a large value does not by itself mean the
        browser is gone.  Read the result back with ``list_panels`` if you need
        to confirm what landed.
    """
    result = await anyio.to_thread.run_sync(
        functools.partial(notify_panel_arrange, tiles=tiles, preset=preset, focus=focus)
    )

    if not result["ok"]:
        if result.get("status") is None:
            return make_error(
                "web_terminal_unreachable",
                "Web Terminal is not running — the workspace cannot be arranged.",
                ["Continue without the workspace, or ask the operator to start it."],
            )
        return make_error(
            "arrange_rejected",
            f"The arrangement was rejected: {result.get('detail', '')}",
            [
                "Call list_panels for the valid panel ids and the configured presets.",
                "Pass exactly one of 'tiles' or 'preset'.",
                "A focus target must be one of the arranged tiles.",
            ],
        )

    data = result.get("data", {})
    panels = await anyio.to_thread.run_sync(fetch_panels) or {}
    response: dict = {
        "status": "success",
        "tiles": data.get("tiles", tiles),
        "focus": data.get("focus", focus),
        "open_tiles_age_s": panels.get("open_tiles_age_s"),
        "open_tiles_dock": panels.get("open_tiles_dock"),
    }
    if data.get("preset") is not None:
        response["preset"] = data["preset"]
    return json.dumps(response)
