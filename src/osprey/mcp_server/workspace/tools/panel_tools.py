"""MCP tools for the Web Terminal's panels.

A panel sits on two independent axes, and each verb here names the one it moves:

- **rail membership** — can the operator launch this panel in one click?
  ``add_panel_to_rail`` / ``remove_panel_from_rail``.
- **on screen** — is a tile in front of the operator showing the panel? A tile
  shows one panel. ``open_panel``, ``open_panel_set``, ``arrange_workspace``,
  ``drop_panel`` and ``close_panel``.

The on-screen verbs are the dock ops in OSPREY's naming style (``open``,
``open_set``, ``open_arrangement``, ``drop``, ``close``). Every one goes through
:func:`osprey.mcp_server.http.panel_op`, the one agent dock-write path: the web
terminal relays the op to the browser that holds the dock, and the tool answers
with the resulting layout and its ``rev`` — or with a named refusal, in which
case nothing changed. The one exception is ``no_dock_answer``: the browser did
not answer in time and may still apply the op, so its outcome is unknown until
``list_panels`` is read.

``list_panels`` reports both axes, the full layout and the ``rev`` that
``drop_panel``, the one verb that names tile ids, takes as ``base_rev``.
``register_panel`` adds a panel the deployment did not ship (requires opt-in).
"""

import functools
import json
from typing import Any, Literal
from urllib.parse import urlsplit

import anyio

from osprey.interfaces.panel_urls import (
    Origin,
    agent_hop_panel_ids,
    agent_hop_url_refusal,
    is_loopback_host,
    navigation_url_refusal,
    normalise_navigation_url,
)
from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import (
    NO_WEB_TERMINAL,
    fetch_panels,
    notify_panel_register,
    notify_panel_visibility,
    panel_op,
)
from osprey.mcp_server.workspace.server import mcp
from osprey.profiles.sidecar_panels import SIDECAR_PANELS


@mcp.tool()
async def list_panels() -> str:
    """List the panels available in the Web Terminal, and the dock's layout.

    Two different things, easy to confuse:

    - ``visible`` — the panel has an entry in the **launcher rail**, so the
      operator can open it in one click.  It says nothing about whether the
      panel is on screen right now.
    - ``layout`` — what is **actually on screen**: the dock's tiles, each
      showing one panel (the terminal's tile included).

    ``layout`` is a tree.  A tile is ``{tile, tabs, showing, size, from?}``:
    ``tile`` is its id, ``tabs`` lists the one panel it shows and ``showing``
    names that panel again, e.g. ``{"tile": "t2", "tabs": ["ariel"],
    "showing": "ariel"}``.  A split is ``{split: "row"|"column", items,
    size?}``; ``size`` is a fraction of the parent.  ``popouts`` lists the
    panels the operator moved into windows of their own: they are on screen
    but in no tile, and no write tool moves or closes them (``popped_out``) —
    an ``open_panel`` brings the window forward.  ``layout`` is null when no
    browser with the dock has reported.  ``rev`` is the layout's revision:
    pass it as ``base_rev`` to ``drop_panel``, which names tile ids — a write
    against an older ``rev`` is refused ``stale_base``.  Every write tool
    returns the new layout and ``rev``, so two steps in a row need no second
    read.  A write the browser did not answer in time fails with
    ``no_dock_answer``: the browser may still apply it, so read the panels
    again before you retry it.

    How far to trust ``layout``:

    - ``layout_dock: true`` — a real report.
    - ``layout: null`` with ``layout_dock: false`` — someone is watching
      without a dock shell: occupancy is unknown, not empty.
    - ``layout_dock: null`` (with a null age) — no client ever reported.

    ``layout_age_s`` counts seconds since the arrangement last *changed*.

    Covers the enabled built-in panels (e.g. artifacts, ariel, channel-finder,
    okf, events) and any custom panels defined in config.yml.  Panel
    IDs vary between deployments — do not guess them.

    To honor a request like "set up for machine setup", look up the named
    layout in ``presets`` and apply it with ``arrange_workspace(preset=...)``
    — the same operation a human's "Layouts" click performs.

    Returns:
        JSON with ``status``, ``active`` (the last focus the server recorded),
        ``panels`` (each ``id``, ``label``, ``visible``), ``layout``,
        ``popouts``, ``rev``, ``layout_age_s``, ``layout_dock`` and ``presets``
        (config-defined layouts, each with a ``name`` and its ``panels``; empty
        unless the deployment defines any).
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
            "layout": data.get("layout"),
            "popouts": data.get("popouts", []),
            "rev": data.get("rev"),
            "layout_age_s": data.get("layout_age_s"),
            "layout_dock": data.get("layout_dock"),
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

    Panel IDs are provided in your context at session start; call
    ``list_panels`` to refresh the live state if needed.  Do NOT guess
    panel IDs — they vary between deployments.

    Args:
        panel_id: Panel identifier (e.g. 'artifacts', 'ariel', 'okf').

    Returns:
        JSON with status confirmation and the panel's new ``on_rail`` state.
    """
    return await anyio.to_thread.run_sync(_set_rail_membership, panel_id, True)


@mcp.tool()
async def remove_panel_from_rail(panel_id: str) -> str:
    """Take a panel off the Web Terminal's rail, so it can no longer be launched.

    Any tile showing it also closes on every connected client — a panel the
    operator cannot launch must not be left stranded on screen.  To clear the
    screen while keeping the panel one click away, use ``close_panel`` instead.

    Use when the user asks to remove a panel or get rid of it for good.

    Panel IDs are provided in your context at session start; call
    ``list_panels`` to refresh the live state if needed.  Do NOT guess
    panel IDs — they vary between deployments.

    Args:
        panel_id: Panel identifier (e.g. 'artifacts', 'ariel', 'okf').

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


#: The one reason after which the op may still be applied: the browser was sent
#: it and did not answer in time, so its outcome is unknown, not refused.
_NO_DOCK_ANSWER = "no_dock_answer"

#: Next steps per named refusal; a reason not listed gets :data:`_DEFAULT_HINT`.
_REFUSAL_HINTS: dict[str, str] = {
    "stale_base": (
        "The layout changed since your read: call list_panels for the current layout "
        "and rev, then retry with that rev as base_rev."
    ),
    "no_dock_client": (
        "No browser with the dock is answering: ask the operator to open the Web "
        "Terminal, or continue without the workspace."
    ),
    _NO_DOCK_ANSWER: (
        "The browser may still apply the op: call list_panels, and retry only if "
        "its layout does not show the change."
    ),
    NO_WEB_TERMINAL: "Continue without the workspace, or ask the operator to start it.",
    "terminal_not_agent_movable": (
        "The Agent pane is always its own tile: move or close service panels "
        "instead, and put them beside its tile."
    ),
    "terminal_in_layout": (
        "Leave the terminal out of the layout: arrangements place service panels "
        "and leave the terminal where it is."
    ),
    "too_many_for_side": (
        "The panels do not fit side by side: give an exact layout with fewer tiles, "
        "or open them one at a time."
    ),
    "one_panel_per_tile": (
        "A tile holds one panel: put panels side by side (open_panel_set, or a layout "
        "with one panel per tile), or use open_panel to replace what a tile shows."
    ),
}
_DEFAULT_HINT = "Call list_panels for the current layout, rev, panel ids and presets."

#: Which tool reports a refusal of each op.
_OP_TOOLS = {
    "drop": "drop_panel",
    "close": "close_panel",
    "open": "open_panel",
    "open_set": "open_panel_set",
    "open_arrangement": "arrange_workspace",
}

#: How ``arrange_workspace`` opens a preset's panels: side by side, or by the
#: preset's own layout.
OpenAs = Literal["side", "layout"]


def _refuse(op: str, reason: str, detail: str, rev: int | None = None) -> None:
    """Raise the structured error for an op refused for a named ``reason``.

    The ``error_type`` is the reason itself (``stale_base``,
    ``terminal_not_agent_movable``, ...), so an agent can branch on it. A
    refusal changed nothing; ``no_dock_answer`` is not a refusal but an
    unknown outcome, and its message says so.
    """
    details: dict[str, Any] = {"op": op, "reason": reason, "detail": detail}
    if rev is not None:
        details["rev"] = rev
    if reason == _NO_DOCK_ANSWER:
        message = f"{_OP_TOOLS[op]} got no answer ({reason}): {detail} The outcome is unknown."
    else:
        message = f"{_OP_TOOLS[op]} refused ({reason}): {detail} Nothing changed."
    make_error(reason, message, [_REFUSAL_HINTS.get(reason, _DEFAULT_HINT)], details)


async def _run_panel_op(
    op: str, args: dict, *, base_rev: int | None = None, echo: dict | None = None
) -> str:
    """Run one dock op through :func:`panel_op`; the shared body of the write tools.

    Returns:
        JSON ``{"status": "success", **echo, "layout", "popouts", "rev"}`` once
        the dock applied the op.

    Raises:
        ToolError: The op was refused, the web terminal is unreachable, or the
            browser did not answer in time (``no_dock_answer``); the envelope's
            ``error_type`` is the named reason.
    """
    result = await anyio.to_thread.run_sync(
        functools.partial(panel_op, op, args, base_rev=base_rev)
    )
    if not result.get("ok"):
        _refuse(
            op,
            str(result.get("reason") or "invalid_request"),
            str(result.get("detail") or ""),
            result.get("rev"),
        )
    return json.dumps(
        {
            "status": "success",
            **(echo or {}),
            "layout": result.get("layout"),
            "popouts": result.get("popouts", []),
            "rev": result.get("rev"),
        }
    )


def _layout_members(layout: object) -> list[str]:
    """The panel ids a panel-id layout names, in reading order, each once."""
    found: list[str] = []

    def walk(node: object) -> None:
        if not isinstance(node, dict):
            return
        tabs = node.get("tabs")
        if isinstance(tabs, list):
            found.extend(tab for tab in tabs if isinstance(tab, str) and tab not in found)
        items = node.get("items")
        if isinstance(items, list):
            for item in items:
                walk(item)

    walk(layout)
    return found


def _indexed_layout(layout: object, members: list[str]) -> object:
    """*layout* with every panel id in its ``tabs`` (and ``showing``) replaced by its member index.

    The op routes take a collection-form layout whose keys are member indexes;
    agents write panel ids. Anything that names no member is left as it is, so
    the route refuses it under its own named reason (``terminal_in_layout``,
    ``bad_key``, ...).
    """
    index = {member: position for position, member in enumerate(members)}

    def key(value: object) -> object:
        return index.get(value, value) if isinstance(value, str) else value

    if not isinstance(layout, dict):
        return layout
    out = dict(layout)
    if isinstance(out.get("tabs"), list):
        out["tabs"] = [key(tab) for tab in out["tabs"]]
    if "showing" in out:
        out["showing"] = key(out["showing"])
    if isinstance(out.get("items"), list):
        out["items"] = [_indexed_layout(item, members) for item in out["items"]]
    return out


def _agent_hop_panel_ids() -> frozenset[str]:
    """The sidecar panels whose own MCP transport the agent drives directly.

    :func:`~osprey.interfaces.panel_urls.agent_hop_panel_ids` of the registry
    the gate and the panel proxy read. Such a panel has a tool of its own for
    what is on its screen, so a url pushed into its iframe from here would
    bypass that tool's input checks and attribution.
    """
    return agent_hop_panel_ids(SIDECAR_PANELS)


def _deployment_ports() -> tuple[range, int | None]:
    """The deployment's port block and the web terminal's own port.

    Every port OSPREY publishes lies in the block (``osprey.port_layout``); the
    terminal's port may be moved out of it by ``OSPREY_WEB_PORT``. A config that
    cannot be read yields the default block and no terminal port.
    """
    from osprey.port_layout import BLOCK_SIZE, resolve_port_base

    terminal_port: int | None = None
    try:
        from osprey.mcp_server.http import web_terminal_url
        from osprey_connectors.workspace import load_osprey_config

        base = resolve_port_base(load_osprey_config())
        terminal_port = urlsplit(web_terminal_url()).port
    except Exception:  # an unreadable config falls back to the default block
        base = resolve_port_base(None)
    return range(base, base + BLOCK_SIZE), terminal_port


def _is_sidecar_origin(origin: Origin) -> bool:
    """Whether *origin* can be a panel sidecar's own address.

    A sidecar listens on an ephemeral loopback port the terminal never
    publishes: the browser only ever reaches it through ``/panel/<id>``. So a
    loopback address (:func:`~osprey.interfaces.panel_urls.is_loopback_host`,
    in any spelling) outside the deployment's port block, and not the
    terminal's own port, is taken as a sidecar's. This guess only words the
    agent's error; the terminal judges the url again against the addresses it
    is configured with, and refuses it there whatever this says.
    """
    _scheme, host, port = origin
    if not is_loopback_host(host):
        return False
    block, terminal_port = _deployment_ports()
    return not (port in block or port == terminal_port)


def _agent_hop_url_refusal(panel_id: str, url: str, hop_ids: frozenset[str]) -> str | None:
    """Why *url* may not be pushed into a panel by ``open_panel``, or ``None``.

    The terminal's own rule (:func:`~osprey.interfaces.panel_urls.agent_hop_url_refusal`),
    run here first so the refusal can name the panel's own tool.

    Args:
        panel_id: The panel the agent asked to open.
        url: The url, already passed through ``normalise_navigation_url``.
        hop_ids: The agent-hop panel ids (:func:`_agent_hop_panel_ids`).
    """
    return agent_hop_url_refusal(panel_id, url, hop_ids=hop_ids, is_hop_origin=_is_sidecar_origin)


@mcp.tool()
async def open_panel(panel_id: str, url: str | None = None) -> str:
    """Put a panel on screen in front of the operator.

    The "look at this" verb: use it when the user asks to open or go to a panel,
    and after producing something they should see (a new artifact, a drafted
    plan, a channel search).

    A tile shows one panel, so an open replaces what a tile shows:

    - the panel is already on screen → it is focused where it is;
    - otherwise it takes the service tile the operator used last, replacing
      the panel that tile showed; the replaced panel stays on the rail, one
      click away, with its state intact;
    - with no service tile open it opens as the first service tile beside the
      Agent pane, which is never the target;
    - a panel not on the rail is added to it once it is on screen.

    A panel whose backend is unhealthy does not open, and a panel living in its
    own pop-out window is brought to the front there, with the layout unchanged.

    One panel.  For several at once, side by side, use ``open_panel_set``; for
    an exact arrangement or a configured preset use ``arrange_workspace``; to
    put a panel in a tile of its own beside a given tile, replacing nothing,
    use ``drop_panel``.

    Panel IDs are provided in your context at session start; call
    ``list_panels`` to refresh the live state if needed.  Do NOT guess
    panel IDs — they vary between deployments.

    Args:
        panel_id: Panel identifier returned by list_panels (e.g.
            'artifacts', 'events', 'ariel', 'okf', etc.).
        url: Optional URL to navigate the panel iframe to as it opens: a
            path under the panel's own mount (``/panel/ariel/#/draft/7``) or
            an ``http(s)://`` URL on the panel's own server, which opens
            through the mount. Anything else (another path on this origin, a
            foreign host, a ``javascript:``, ``data:`` or scheme-relative
            ``//host`` URL) is refused with an ``invalid_url`` error and
            nothing is opened.
            A panel that has its own workspace tool (CANOPY:
            ``set_workspace``) takes no url: a url for it, a url under
            ``/panel/<that panel>/`` for any panel, or its sidecar's own address
            raises an ``agent_hop_panel_url`` error and opens nothing. Open such
            a panel without a url and change its screen with its tool.

    Returns:
        JSON with ``status``, ``panel``, the resulting ``layout`` and its
        ``rev``; or a structured error whose ``error_type`` names the refusal
        (``unknown_member``, ``no_dock_client``, ...).
    """
    args: dict[str, Any] = {"member": panel_id}
    if url is not None:
        # Judged here as well as by the terminal's route, so a refused url is
        # an error the agent sees rather than a success nothing applied, and the
        # agent-hop refusal can name the panel's own tool.
        normalised = normalise_navigation_url(url)
        hop_ids = _agent_hop_panel_ids()
        hop_refusal = _agent_hop_url_refusal(panel_id, normalised, hop_ids)
        if hop_refusal is not None:
            make_error(
                "agent_hop_panel_url",
                f"URL refused: {hop_refusal}. Nothing was opened.",
                [
                    "Change what the panel shows with its set_workspace tool "
                    "(read it first with get_workspace).",
                    f"Call open_panel('{panel_id}') without a url to bring the panel on screen."
                    if panel_id in hop_ids
                    else "Open a panel with its own id and no url to bring it on screen.",
                ],
                {"panel": panel_id, "agent_hop_panels": sorted(hop_ids)},
            )
        refusal = navigation_url_refusal(normalised)
        if refusal is not None:
            make_error(
                "invalid_url",
                f"URL refused: {refusal}. Nothing was opened.",
                [
                    "Pass a path under the panel's mount (/panel/<id>/...) or an http(s) URL on its server."
                ],
                {"panel": panel_id},
            )
        args["url"] = url
    return await _run_panel_op("open", args, echo={"panel": panel_id})


@mcp.tool()
async def drop_panel(source: dict[str, Any], target: dict[str, Any], base_rev: int) -> str:
    """Put a panel, or a whole tile, at a specific place in the dock.

    The agent's version of a person's drag, with the same result: every drop a
    person can make, you can make, plus the ``border`` target, which a drag
    has no pointer zone for.  A tile shows one panel, so every drop makes a
    tile of its own: nothing is dropped into a tile.  Read the layout first
    (``list_panels``) and pass its ``rev`` as ``base_rev``.

    ``source`` — what moves:

    - ``{"new": {"panel": "<panel id>"}}`` — a panel that is not open yet;
    - ``{"tab": "<panel id>"}`` — a panel on screen, by its id (its tile goes
      with it, and the space goes to one neighbour); a panel in a pop-out
      window is in no tile and is refused ``popped_out``;
    - ``{"tile": "<tile id>"}`` — a whole tile, keeping its id.

    ``target`` — where it lands (tile ids come from ``layout``):

    - ``{"tile": id, "edge": "left"|"right"|"top"|"bottom"}`` — split that
      tile into two equal halves, the panel on that side;
    - ``{"border": {"tile": id, "side": "left"|"right"|"top"|"bottom"}}`` — a
      new tile inserted between that tile and its neighbour on that side,
      taking an equal share of the row or column;
    - ``{"outer": "left"|"right"|"top"|"bottom"}`` — a tile spanning the full
      height or width of the dock, taking one third of it.

    A target inside a tile (``{"tile": id, "zone": ...}``) is refused
    ``one_panel_per_tile``: to show a panel in a tile in place of another, use
    ``open_panel``.  The terminal, and the tile holding it, is never a source
    (refused ``terminal_not_agent_movable``); dropping services beside the
    terminal's tile is allowed.  A drop onto itself changes nothing.

    Args:
        source: What moves, as above.
        target: Where it lands, as above.
        base_rev: The ``rev`` of the layout the tile ids were read from.

    Returns:
        JSON with ``status``, the resulting ``layout`` and its ``rev``; or a
        structured error whose ``error_type`` names the refusal
        (``stale_base``, ``unknown_tile``, ``invalid_source``,
        ``invalid_target``, ``one_panel_per_tile``, ...).
    """
    return await _run_panel_op("drop", {"source": source, "target": target}, base_rev=base_rev)


@mcp.tool()
async def close_panel(panel_id: str) -> str:
    """Close the tile showing a panel, leaving the panel one click away on the rail.

    Use when the user asks to close, dismiss or clear a panel but keep it
    available.  The tile's space goes to one neighbour.  To take the panel off
    the rail as well use ``remove_panel_from_rail``.  The terminal is never
    closed by an agent (``terminal_not_agent_movable``), and a panel in a
    pop-out window is refused ``popped_out``: the operator closes that window.

    Args:
        panel_id: The panel's id (e.g. 'artifacts', 'ariel', 'okf').

    Returns:
        JSON with ``status``, ``panel``, the resulting ``layout`` and its
        ``rev``; or a structured error whose ``error_type`` names the refusal
        (``unknown_key`` for a panel no tile shows, ...).
    """
    return await _run_panel_op("close", {"key": panel_id}, echo={"panel": panel_id})


@mcp.tool()
async def open_panel_set(panels: list[str]) -> str:
    """Open several panels side by side in one call.

    One tile each, in a row after the focused tile, sharing its space equally
    (a near-square grid when a row is too narrow; refused ``too_many_for_side``
    when even that does not fit).  For an exact arrangement use
    ``arrange_workspace(layout=...)``.  The terminal is never part of a set and
    is left where it is.

    Args:
        panels: The panel ids, each once, never ``terminal``.

    Returns:
        JSON with ``status``, ``panels``, the resulting ``layout`` and its
        ``rev``; or a structured error whose ``error_type`` names the refusal
        (``unknown_member``, ``duplicate_member``, ``terminal_in_layout``,
        ``too_many_for_side``, ...).
    """
    args: dict[str, Any] = {"members": list(panels), "as": "side"}
    return await _run_panel_op("open_set", args, echo={"panels": list(panels)})


@mcp.tool()
async def arrange_workspace(
    preset: str | None = None,
    layout: dict[str, Any] | None = None,
    open_as: OpenAs | None = None,
) -> str:
    """Lay out the service panels: a configured preset, or an exact layout.

    Use this when the user describes an arrangement rather than a single panel
    — "put okf next to artifacts", "ariel above the logbook", "set up for
    injection".  For one panel use ``open_panel``; for a small move (one panel
    beside a tile) use ``drop_panel``.

    Pass exactly one of:

    - ``preset`` — the name of a configured layout (``list_panels`` reports
      them in ``presets``).  It opens as configured; ``open_as="side"`` opens
      its panels side by side instead.  Like the operator's "Layouts" click,
      the rail then holds exactly the preset's panels.
    - ``layout`` — an exact arrangement: a tree whose tiles each list one
      panel id, e.g. ``{"split": "row", "items": [{"tabs": ["ariel"]},
      {"tabs": ["okf"]}, {"tabs": ["artifacts"], "size": 2}]}`` (no tile
      ids; ``size`` is a relative weight).  A tile listing several panels is
      refused ``one_panel_per_tile``.  Listed panels that are not on the rail
      join it.

    The result replaces the current service tiles.  The terminal is never part
    of a layout (refused ``terminal_in_layout``) and keeps its tile, place and
    size.

    Args:
        preset: Name of a configured layout.
        layout: An exact arrangement, as above.
        open_as: With ``preset`` only: ``"side"`` opens its panels side by
            side, ``"layout"`` by its own layout (the configured way).

    Returns:
        JSON with ``status``, ``preset`` or ``panels``, the resulting
        ``layout`` and its ``rev``; or a structured error whose ``error_type``
        names the refusal (``unknown_arrangement``, ``unknown_member``,
        ``terminal_in_layout``, ``one_panel_per_tile``, ``member_missing``,
        ``no_dock_client``, ...).
    """
    if (preset is None) == (layout is None):
        _refuse(
            "open_arrangement",
            "invalid_request",
            "pass exactly one of 'preset' or 'layout'.",
        )
    if preset is not None:
        args: dict[str, Any] = {"name": preset}
        if open_as is not None:
            args["as"] = open_as
        return await _run_panel_op("open_arrangement", args, echo={"preset": preset})
    if open_as not in (None, "layout"):
        _refuse(
            "open_arrangement",
            "invalid_request",
            "open_as applies to a preset; a layout opens as that layout.",
        )
    members = _layout_members(layout)
    if not members:
        _refuse(
            "open_arrangement",
            "invalid_layout",
            "the layout names no panel; a tile lists its panel id in 'tabs'.",
        )
    return await _run_panel_op(
        "open_set",
        {"members": members, "as": "layout", "layout": _indexed_layout(layout, members)},
        echo={"panels": members},
    )
