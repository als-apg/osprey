"""Panel configuration, server URL, health, and MCP introspection routes."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import logging
import os
import socket
import time
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from osprey.interfaces.common_middleware import apply_url_prefix, compute_url_prefix
from osprey.interfaces.web_terminal.routes.agent_activity import record_activity
from osprey.profiles.web_panels import BUILTIN_PANEL_LABELS, BUILTIN_PANELS

logger = logging.getLogger(__name__)

router = APIRouter()


def _prefix_path(path: str) -> str:
    """Prepend the per-container URL prefix to a root-absolute path.

    Thin wrapper over :func:`apply_url_prefix` (the shared prefix contract):
    an already-absolute URL passes through unchanged so an external panel URL
    is never corrupted, and an empty prefix is a no-op.
    """
    return apply_url_prefix(compute_url_prefix(), path)


@router.get("/health")
async def health(request: Request):
    """Health check endpoint."""
    from osprey import __version__

    session_id = getattr(request.app.state, "server_session_id", None)
    return {
        "status": "healthy",
        "service": "web_terminal",
        "session_id": session_id,
        "version": __version__,
    }


@router.get("/api/session")
async def session_probe():
    """Authenticated liveness probe for the operator's session.

    A tiny credentialed endpoint the frontend polls after a channel drops, so it
    can tell an expired session apart from a flaky network. It is deliberately
    **not** in the middleware exempt set: the auth gate runs ahead of this
    handler, so a request carrying a live operator credential (the operator
    secret header or a valid session cookie) reaches here and gets a 200, while
    an absent or expired credential is refused with 401 by the gate itself and
    never reaches this body.

    That app-level 401 is the whole point. Single-user deployments have no nginx
    perimeter in front of the app, so ``/health`` — which is exempt so the
    container healthcheck can curl it credential-less — always answers 200 and
    can never signal an expired session. This route can, because it sits behind
    the same auth gate every other API route does.

    Returns:
        ``{"status": "ok"}`` with a 200. The body carries no session detail; the
        status code (200 here, or 401 from the gate) is the entire signal.
    """
    return {"status": "ok"}


@router.get("/api/artifact-server")
async def artifact_server_config(request: Request):
    """Return the artifact gallery server URL for iframe embedding."""
    url = getattr(request.app.state, "artifact_server_url", None)
    proxy_url = f"{compute_url_prefix()}/panel/artifacts" if url else None
    return {"url": proxy_url, "available": proxy_url is not None}


@router.get("/api/type-registry")
async def get_type_registry():
    """Return the OSPREY type registry for tool/category color mapping."""
    from osprey.stores.type_registry import registry_to_api_dict

    return registry_to_api_dict()


@router.get("/api/ariel-server")
async def ariel_server_config(request: Request):
    """Return the ARIEL logbook server URL for iframe embedding."""
    url = getattr(request.app.state, "ariel_server_url", None)
    proxy_url = f"{compute_url_prefix()}/panel/ariel" if url else None
    return {"url": proxy_url, "available": proxy_url is not None}


@router.get("/api/channel-finder-server")
async def channel_finder_server_config(request: Request):
    """Return the Channel Finder server URL for iframe embedding."""
    url = getattr(request.app.state, "channel_finder_server_url", None)
    proxy_url = f"{compute_url_prefix()}/panel/channel-finder" if url else None
    return {"url": proxy_url, "available": proxy_url is not None}


@router.get("/api/lattice-server")
async def lattice_server_config(request: Request):
    """Return the lattice dashboard server URL for iframe embedding."""
    url = getattr(request.app.state, "lattice_dashboard_server_url", None)
    proxy_url = f"{compute_url_prefix()}/panel/lattice" if url else None
    return {"url": proxy_url, "available": proxy_url is not None}


@router.get("/api/okf-server")
async def okf_server_config(request: Request):
    """Return the OKF knowledge panel server URL for iframe embedding."""
    url = getattr(request.app.state, "okf_server_url", None)
    proxy_url = f"{compute_url_prefix()}/panel/okf" if url else None
    return {"url": proxy_url, "available": proxy_url is not None}


@router.get("/api/system-health-server")
async def system_health_server_config(request: Request):
    """Return the System Health dashboard server URL for iframe embedding."""
    url = getattr(request.app.state, "system_health_server_url", None)
    proxy_url = f"{compute_url_prefix()}/panel/system-health" if url else None
    return {"url": proxy_url, "available": proxy_url is not None}


def _browser_panel_url(cp: dict) -> str:
    """The browser-facing URL for a custom panel, prefixed with ``compute_url_prefix()``.

    Keyed off the explicit ``discovered`` marker, not URL shape: a discovered
    static bundle is served same-origin from disk at its own ``/panel-static/{id}/``
    URL, so that URL is used verbatim (beyond the prefix).  Every other custom
    panel is URL-backed and routed through the reverse proxy at ``/panel/{id}``
    so its raw upstream origin never reaches the browser (this preserves the
    pre-discovery behavior exactly, and avoids treating a protocol-relative
    ``//host`` URL as same-origin).
    """
    prefix = compute_url_prefix()
    if cp.get("discovered"):
        url = cp.get("url")
        if isinstance(url, str) and url:
            return f"{prefix}{url}"
        return f"{prefix}/panel-static/{cp['id']}/"
    return f"{prefix}/panel/{cp['id']}"


def _project_key(project_cwd: str | None) -> str:
    """Return a stable, opaque per-project key for client-side layout persistence.

    The key is the first 16 hex chars of the sha256 digest of the *resolved*
    project directory path. Resolving first means equivalent paths (symlinks,
    trailing slashes, ``.`` segments) collapse to one key, so the same project
    yields the same key across server restarts, while distinct projects differ.

    Used by the client as the ``osprey-dock-layout-<project_key>`` localStorage
    suffix, so one browser origin can persist a separate dock layout per project.
    """
    resolved = str(Path(project_cwd).resolve()) if project_cwd else ""
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]


def _workspace_has_artifacts(base: Path | None) -> bool:
    """True when the agent workspace holds at least one regular file.

    Dot-entries (``.DS_Store``, ``.gitkeep``, hidden dirs) don't count — a
    workspace seeded only with housekeeping files must still read empty, so
    the simple UX's chat-only first boot isn't defeated by scaffolding. The
    walk stops at the first hit; unreadable directories are skipped.
    """
    if base is None:
        return False
    stack = [os.fspath(base)]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.name.startswith("."):
                        continue
                    if entry.is_file(follow_symlinks=False):
                        return True
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
        except OSError:
            continue
    return False


@router.get("/api/panels")
async def get_panels(request: Request):
    """Return the full panel state in one payload.

    All custom panel URLs are rewritten to ``<prefix>/panel/{id}`` so the
    browser routes through the reverse proxy (``<prefix>`` is the empty string
    outside multi-user deployments).  The originals in ``app.state`` are left
    untouched — the proxy reads those for forwarding.

    Response shape::

        {
            "enabled":  [...],          # list of enabled builtin panel id strings
            "custom":   [...],          # custom panel dicts (url rewritten to /panel/<id>)
            "default":  str|None,       # profile-pinned cold-load focus target
            "visible":  [...],          # enabled + custom ids minus hidden: true panels
            "active":   str|None,       # currently focused panel id
            "labels":   {id: label},    # display labels for enabled built-in panels
            "allow_runtime_panels": bool,  # whether the human "+" may add a URL panel
            "presets":  [...],          # config-defined layouts: [{"name", "panels": [id,...]}]
            "ui_mode":  str,            # resolved web.ui_mode ("expert" | "simple")
            "project_key": str,         # stable 16-hex per-project key (layout persistence)
            "workspace_has_artifacts": bool,  # any non-hidden file under the agent workspace
            "open_tiles": [...]|None,   # last-reported service tiles, reading order
            "open_tiles_age_s": float|None,   # seconds since that report (None = never)
            "open_tiles_dock": bool|None,     # reporting client had a dock shell (None = never)
            "docs_url": str,            # target of the rail's Documentation control
            "feedback_github_repo": str,   # "owner/repo" for the prefilled new-issue URL
            "feedback_email": str,      # recipient of the prefilled mailto: draft
        }

    ``project_key`` is an opaque, stable per-project identifier (16 hex chars,
    a truncated sha256 of the resolved project directory). The client uses it as
    the ``osprey-dock-layout-<project_key>`` localStorage suffix so dock layouts
    persist independently per project on a shared browser origin.

    ``ui_mode`` mirrors the server-rendered ``<html data-ui-mode>`` attribute so
    the client can read the resolved mode after boot. First paint must never
    depend on this field — the SSR attribute is the authoritative first-paint
    rung; this is the API-side echo for later client mode resolution.

    ``rail_position`` mirrors the server-rendered ``<html data-rail-position>``
    the same way, and travels with two companions: ``family_rail_defaults``
    (``app.FAMILY_RAIL_DEFAULTS``, the theme-family -> rail-position coupling
    that gives the retro family a top tab strip) and
    ``rail_position_configured`` (whether ``web.rail_position`` was set
    explicitly, which outranks that coupling). ``rail-position.js`` reads both
    so a live theme-family switch can move the rail without the browser
    carrying its own copy of the coupling.

    ``presets`` is the config-defined "Layouts" list (``web.presets``), resolved
    at startup against the live panel set and carried in config order. It is
    empty unless a deployment opts in, so the "+" popover renders unchanged by
    default. Each entry is ``{"name": <label>, "panels": [<member id>, ...]}``.

    ``allow_runtime_panels`` mirrors the config gate the ``POST /api/panels/register``
    route enforces, so the frontend can show or hide the "new panel from URL" input
    without first attempting a registration that would 403.

    ``default`` is not validated here — the frontend falls back to
    ``DEFAULT_PANEL_FALLBACK`` when it is unknown so a typo doesn't leave the
    user staring at a blank tabset.

    ``active`` mirrors ``GET /api/panel-focus`` so the agent can read back the
    full panel state in a single round-trip.

    ``labels`` covers only the enabled built-in panels; custom panels carry
    their own ``label`` field in the ``custom`` list.

    ``workspace_has_artifacts`` reports whether the agent workspace already
    holds any (non-hidden) file. The simple UX uses it to decide whether the
    first paint is chat-only (empty workspace) or includes the WORKSPACE
    panel; recomputed per request so a reload after the first artifact lands
    sees ``true``.

    ``open_tiles`` is what a browser last reported through
    ``POST /api/panel-layout``: the service tiles actually on screen, in spatial
    reading order. It is distinct from ``visible`` — that is launcher-rail
    membership, this is occupancy. ``open_tiles_age_s`` is the seconds elapsed
    since the occupancy last *changed* (a deduped repeat report does not reset
    it) and ``open_tiles_dock`` is whether the reporting client had a dock
    shell.

    Read together, the three fields carry three distinct states, and a
    consumer must not collapse them:

    - **Never reported** — all three ``null``. No client has ever checked in;
      nothing is known about the screen. Not the same as an empty workspace.
    - **Unknown occupancy** — ``open_tiles: null`` with a numeric
      ``open_tiles_age_s`` and ``open_tiles_dock: false``. A client is watching
      but runs without the dock shell, so it cannot report tile order.
    - **Known occupancy** — ``open_tiles`` is a list (possibly ``[]``, meaning
      the operator genuinely closed every service tile) with a numeric age and
      ``open_tiles_dock: true``.

    So ``[]`` always means known-empty and ``null`` always means unknown; the
    age tells the two flavours of unknown apart.

    ``docs_url``, ``feedback_github_repo`` and ``feedback_email`` are the
    resolved ``web.docs_url`` / ``web.feedback.*`` values that address the two
    utility controls at the far end of the rail: the Documentation link target,
    and the two outbound channels the feedback dialog offers (a prefilled
    GitHub issue and a prefilled ``mailto:`` draft). The store ceiling
    (``web.feedback.max_store_bytes``) is deliberately not echoed — it governs
    server-side pruning and nothing in the browser acts on it.
    """
    enabled = list(getattr(request.app.state, "enabled_panels", set()))
    custom_raw = getattr(request.app.state, "custom_panels", [])
    custom = [{**cp, "url": _browser_panel_url(cp)} for cp in custom_raw]
    default = getattr(request.app.state, "default_panel", None)
    visible = getattr(request.app.state, "visible_panels", enabled)
    active = getattr(request.app.state, "active_panel", None)
    labels = {pid: BUILTIN_PANEL_LABELS[pid] for pid in enabled if pid in BUILTIN_PANEL_LABELS}
    allow_runtime = bool(getattr(request.app.state, "allow_runtime_panels", False))
    presets = list(getattr(request.app.state, "panel_presets", []))
    # Echo the resolved UI mode (server-rendered onto <html data-ui-mode>).
    # "expert" default mirrors app.DEFAULT_UI_MODE — kept as a literal here to
    # avoid a routes->app import cycle.
    ui_mode = getattr(request.app.state, "web_ui_mode", "expert")
    # Echo the resolved rail position (server-rendered onto
    # <html data-rail-position>). "left" default mirrors
    # app.DEFAULT_RAIL_POSITION — a literal for the same import-cycle reason.
    rail_position = getattr(request.app.state, "web_rail_position", "left")
    # The theme-family -> rail-position coupling, plus whether config pinned a
    # position of its own. The client needs both to decide whether a live theme
    # switch may move the rail; app.FAMILY_RAIL_DEFAULTS stays the only
    # definition of the coupling (imported lazily for the same cycle reason).
    from osprey.interfaces.web_terminal.app import FAMILY_RAIL_DEFAULTS

    rail_position_configured = getattr(request.app.state, "web_rail_position_configured", False)
    project_key = _project_key(getattr(request.app.state, "project_cwd", None))
    has_artifacts = _workspace_has_artifacts(getattr(request.app.state, "workspace_dir", None))
    # Tile occupancy as last reported by a browser, with its freshness. The
    # timestamp is never exposed raw: an age is what a consumer can reason
    # about without knowing the server's clock. ``None`` survives untouched —
    # it is the "unknown occupancy" state, distinct from a known-empty list.
    open_tiles = getattr(request.app.state, "open_tiles", None)
    open_tiles_ts = getattr(request.app.state, "open_tiles_ts", None)
    open_tiles_age = None if open_tiles_ts is None else time.time() - open_tiles_ts
    open_tiles_dock = getattr(request.app.state, "open_tiles_dock", None)
    # Utility-cluster targets. The defaults mirror app.DEFAULT_DOCS_URL /
    # DEFAULT_FEEDBACK_GITHUB_REPO / DEFAULT_FEEDBACK_EMAIL, spelled as
    # literals here for the same routes->app import-cycle reason as above.
    docs_url = getattr(request.app.state, "docs_url", "https://als-apg.github.io/osprey")
    feedback_github_repo = getattr(request.app.state, "feedback_github_repo", "als-apg/osprey")
    feedback_email = getattr(request.app.state, "feedback_email", "thellert@lbl.gov")
    return {
        "enabled": enabled,
        "custom": custom,
        "default": default,
        "visible": visible,
        "active": active,
        "labels": labels,
        "allow_runtime_panels": allow_runtime,
        "presets": presets,
        "ui_mode": ui_mode,
        "rail_position": rail_position,
        "rail_position_configured": rail_position_configured,
        "family_rail_defaults": dict(FAMILY_RAIL_DEFAULTS),
        "project_key": project_key,
        "workspace_has_artifacts": has_artifacts,
        "open_tiles": open_tiles,
        "open_tiles_age_s": open_tiles_age,
        "open_tiles_dock": open_tiles_dock,
        "docs_url": docs_url,
        "feedback_github_repo": feedback_github_repo,
        "feedback_email": feedback_email,
    }


def _known_panel_ids(request: Request) -> set[str]:
    """Return the set of valid panel ids: enabled built-ins plus custom panels.

    Single source of truth for the membership check shared by the focus and
    visibility endpoints — both reject panel ids that are not in this set.
    """
    known: set[str] = set(getattr(request.app.state, "enabled_panels", set()))
    known |= {p["id"] for p in getattr(request.app.state, "custom_panels", [])}
    return known


#: Dock id of the native terminal/chat tile (``PANEL_TERMINAL`` in
#: ``dock-workspace.js``). It carries no server-side panel state and is never a
#: service tile, so every layout verb refuses it explicitly rather than relying
#: on it being absent from :func:`_known_panel_ids` — a custom panel squatting
#: the id must not become a way to move or close the operator's terminal.
_TERMINAL_PANEL_ID = "terminal"


def _mirror_agent_panel_activity(request: Request, tool: str, panel: str) -> None:
    """Record an agent-origin panel command in the agent-activity history ring.

    Panel commands reach the browser as their own SSE frames (``panel_focus``,
    ``panel_visibility``, ...), never through ``POST /api/agent-activity``, so
    without this they are invisible to a client that reads
    ``GET /api/agent-activity/recent`` after connecting late.  The row goes in
    through that route's own ``record_activity``, so a consumer feeds it
    through the handler it already uses for the SSE ``agent_activity`` stream.

    ``tool`` is synthetic: the panel routes carry no tool name of their own, so
    the caller supplies the MCP verb the action corresponds to (``open_panel``,
    ``close_panel``, ``add_panel_to_rail``, ``remove_panel_from_rail``,
    ``arrange_workspace``, ``register_panel``) and the frontend words the entry
    from it.

    Nothing is broadcast — this is history only.  Callers must invoke it for
    agent-origin requests exactly once per action, and never for human ones: a
    human's own gestures are not the agent's activity.

    Args:
        request: Incoming FastAPI request carrying ``app.state``.
        tool: Synthetic tool name naming the action.
        panel: The panel id the action targeted.
    """
    record_activity(request, tool, {"kind": "panel", "panel": panel})


class PanelFocusRequest(BaseModel):
    panel: str
    url: str | None = None
    source: Literal["agent"] | None = None


@router.get("/api/panel-focus")
async def get_panel_focus(request: Request):
    """Return the currently active panel."""
    active = getattr(request.app.state, "active_panel", None)
    return {"active_panel": active}


@router.post("/api/panel-focus")
async def set_panel_focus(body: PanelFocusRequest, request: Request):
    """Set the active panel; broadcast a focus event only for agent switches.

    Attribution decides the frame's fate. An ``source: "agent"`` switch is a
    command every client must apply, so it broadcasts. A source-less POST is a
    human gesture REPORT (panel-commands.js's ``setPanelFocus``): the server
    mirrors ``active_panel`` for the agent's gaze and broadcasts nothing —
    one operator's tab switches never move another client's workspace, and
    the gesturing client applies its own focus locally rather than riding an
    echo.

    ``body.url`` (e.g. from an agent-invoked ``open_panel`` MCP call) is
    run through ``_prefix_path()`` before broadcast so a root-absolute path
    lands inside the user's own mount; an already-absolute URL is left
    untouched.

    An **agent** focus on a panel that is **not** in the launcher rail also
    adds it there, emitting a ``panel_visibility`` frame *before* the focus
    frame. An agent's ``open_panel`` may name a panel the operator took off
    the rail; without the membership update the rail entry would exist only on
    the client that happened to apply the focus, so ``list_panels`` would keep
    reporting the panel invisible, a reload would drop both the entry and the
    agent-opened tile, and a late-connecting client would never see it at all.
    Ordering is deliberate: clients add the rail entry, then apply focus to it.

    A panel already in the rail — which is the only kind a human can click —
    changes nothing and emits no visibility frame. A **source-less** focus on
    a non-member is dropped whole: a human can only gesture at a panel already
    on the rail, so such a report is always a fire-and-forget straggler that a
    concurrent arrange overtook, and applying it would resurrect the panel the
    arrange just pruned.

    An agent switch is also mirrored into the activity history ring as one
    ``open_panel`` row. When the open additionally adds rail membership,
    only the focus is mirrored: the pair of frames is one agent action, and
    history counts actions, not frames.

    Args:
        body: ``panel`` (panel id), optional ``url`` to load, and optional
            ``source`` attribution.
        request: Incoming FastAPI request carrying ``app.state``.

    Returns:
        ``{"status": "ok", "active_panel": <id>}``

    Raises:
        HTTPException: 422 when ``panel`` is not a known enabled or custom id.
    """
    if body.panel not in _known_panel_ids(request):
        raise HTTPException(status_code=422, detail=f"Unknown panel: {body.panel}")

    # Membership, resolved the same way ``get_panels`` resolves ``visible`` —
    # an unset list means "the enabled built-ins are the rail", so a panel the
    # read route calls visible is never treated here as a non-member.
    stored_visible = getattr(request.app.state, "visible_panels", None)
    if stored_visible is None:
        visible_panels = list(getattr(request.app.state, "enabled_panels", set()))
    else:
        visible_panels = list(stored_visible)
    adds_membership = body.panel not in visible_panels
    if adds_membership and body.source != "agent":
        # A source-less focus is a human gesture report, and a human can only
        # gesture at a panel already on the rail — new membership arrives via
        # ``/api/panels/register`` or an agent ``open_panel``, never a human
        # focus. A source-less report naming a non-member is therefore a
        # fire-and-forget straggler that a concurrent arrange overtook;
        # applying it would resurrect the pruned panel on every client's rail
        # and steal the active slot, so the whole write is dropped.
        active = getattr(request.app.state, "active_panel", None)
        return {"status": "ok", "active_panel": active}
    if adds_membership:
        visible_panels.append(body.panel)
        request.app.state.visible_panels = visible_panels

    request.app.state.active_panel = body.panel
    if adds_membership:
        visibility_event: dict = {
            "type": "panel_visibility",
            "panel": body.panel,
            "visible": True,
        }
        if body.source:
            visibility_event["source"] = body.source
        request.app.state.broadcaster.broadcast(visibility_event)

    if body.source == "agent":
        event: dict = {"type": "panel_focus", "panel": body.panel, "source": body.source}
        if body.url:
            event["url"] = _prefix_path(body.url)
        _mirror_agent_panel_activity(request, "open_panel", body.panel)
        request.app.state.broadcaster.broadcast(event)
    return {"status": "ok", "active_panel": body.panel}


class PanelVisibilityRequest(BaseModel):
    panel: str
    visible: bool
    source: Literal["agent"] | None = None


@router.post("/api/panel-visibility")
async def set_panel_visibility(body: PanelVisibilityRequest, request: Request):
    """Show or hide a panel and broadcast the change via SSE.

    An agent-origin change is also mirrored into the activity history ring, as
    an ``add_panel_to_rail`` or ``remove_panel_from_rail`` row depending on the
    flag, so a client reading the history can word it the way it words the live
    frame.

    Args:
        body: ``panel`` (panel id) and ``visible`` (desired visibility).
        request: Incoming FastAPI request carrying ``app.state``.

    Returns:
        ``{"status": "ok", "panel": <id>, "visible": <bool>}``

    Raises:
        HTTPException: 422 when ``panel`` is not a known enabled or custom id.
    """
    if body.panel not in _known_panel_ids(request):
        raise HTTPException(status_code=422, detail=f"Unknown panel: {body.panel}")
    visible_panels: list[str] = list(getattr(request.app.state, "visible_panels", []))
    if body.visible:
        if body.panel not in visible_panels:
            visible_panels.append(body.panel)
    else:
        visible_panels = [p for p in visible_panels if p != body.panel]
    request.app.state.visible_panels = visible_panels
    event: dict = {"type": "panel_visibility", "panel": body.panel, "visible": body.visible}
    if body.source:
        event["source"] = body.source
    if body.source == "agent":
        _mirror_agent_panel_activity(
            request,
            "add_panel_to_rail" if body.visible else "remove_panel_from_rail",
            body.panel,
        )
    request.app.state.broadcaster.broadcast(event)
    return {"status": "ok", "panel": body.panel, "visible": body.visible}


class PanelCloseRequest(BaseModel):
    panel: str
    source: Literal["agent"] | None = None


@router.post("/api/panel-close")
async def close_panel(body: PanelCloseRequest, request: Request):
    """Close a panel's tile on every connected client, leaving the rail alone.

    The on-screen counterpart to ``/api/panel-focus``, and deliberately not a
    flag on ``/api/panel-visibility``: rail membership is server-owned state
    that this route must not touch, while which tiles are open is per-client
    layout. A panel with no tile open closes to a no-op in the browser rather
    than an error here — the server does not track per-client tile occupancy
    closely enough to tell the two apart, and refusing would make the verb
    depend on a stale report.

    Args:
        body: ``panel`` (panel id) and the optional ``source`` marker.
        request: Incoming FastAPI request carrying ``app.state``.

    Returns:
        ``{"status": "ok", "panel": <id>}``

    Raises:
        HTTPException: 422 when ``panel`` is not a known enabled or custom id.
    """
    if body.panel not in _known_panel_ids(request):
        raise HTTPException(status_code=422, detail=f"Unknown panel: {body.panel}")
    event: dict = {"type": "panel_close", "panel": body.panel}
    if body.source:
        event["source"] = body.source
    if body.source == "agent":
        _mirror_agent_panel_activity(request, "close_panel", body.panel)
    request.app.state.broadcaster.broadcast(event)
    return {"status": "ok", "panel": body.panel}


class PanelArrangeRequest(BaseModel):
    tiles: list[str] | None = None
    preset: str | None = None
    focus: str | None = None
    source: Literal["agent"] | None = None


def _resolve_preset_tiles(request: Request, name: str, known: set[str]) -> list[str]:
    """Resolve a preset name to its member panel ids, fail-safe filtered.

    Mirrors ``computePresetDiff`` in ``panel-presets.js``: members are filtered
    to the known ids (and the terminal id is dropped) so a typo'd or disabled
    member in config is skipped rather than breaking the whole layout. Config
    order is preserved — it is the left-to-right tile order clients apply.

    Args:
        request: Incoming request carrying ``app.state.panel_presets``.
        name: The requested preset name.
        known: Valid panel ids from :func:`_known_panel_ids`.

    Returns:
        The preset's surviving member ids, in config order.

    Raises:
        HTTPException: 422 when no preset carries ``name``, or when none of its
            members survives filtering (applying it would strand an empty
            workspace).
    """
    presets: list[dict] = list(getattr(request.app.state, "panel_presets", []))
    match = next((p for p in presets if p.get("name") == name), None)
    if match is None:
        available = [p.get("name") for p in presets]
        raise HTTPException(
            status_code=422,
            detail=f"Unknown preset: {name!r}. Available presets: {available}",
        )
    members = [pid for pid in match.get("panels", []) if pid in known and pid != _TERMINAL_PANEL_ID]
    if not members:
        raise HTTPException(
            status_code=422,
            detail=(f"Preset {name!r} has no known members. Valid panel ids: {sorted(known)}"),
        )
    return members


def _resolve_requested_tiles(tiles: list[str], known: set[str]) -> list[str]:
    """Validate an explicitly requested tile list, preserving its order.

    Explicit tiles are validated strictly — unlike preset members, which are
    config-authored and filtered fail-safe. An agent that names a panel that
    does not exist gets told so rather than silently receiving a different
    workspace than it asked for, matching the focus and visibility routes.

    Args:
        tiles: Requested panel ids, in the left-to-right order to apply.
        known: Valid panel ids from :func:`_known_panel_ids`.

    Returns:
        ``tiles`` with duplicates collapsed (first occurrence wins).

    Raises:
        HTTPException: 422 when ``tiles`` is empty, names the terminal tile, or
            names any id outside ``known``.
    """
    if not tiles:
        raise HTTPException(
            status_code=422,
            detail="tiles must not be empty — an arrangement must leave at least one tile open",
        )
    if _TERMINAL_PANEL_ID in tiles:
        raise HTTPException(
            status_code=422,
            detail=(
                f"The terminal tile ({_TERMINAL_PANEL_ID!r}) is not a service panel "
                "and cannot be arranged"
            ),
        )
    unknown = [pid for pid in tiles if pid not in known]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown panel ids: {unknown}. Valid panel ids: {sorted(known)}",
        )
    deduped: list[str] = []
    for pid in tiles:
        if pid not in deduped:
            deduped.append(pid)
    return deduped


@router.post("/api/panel-arrange")
async def arrange_panels(body: PanelArrangeRequest, request: Request):
    """Request a whole-workspace tile arrangement and broadcast it via SSE.

    Declarative end state: exactly the resolved tiles are open, left to right,
    in the order given. Clients apply it as a deterministic rebuild of the
    service-tile region; the terminal tile is never touched.

    Exactly one of ``tiles`` and ``preset`` must be supplied. ``preset`` names
    an entry of ``web.presets`` (``app.state.panel_presets``); its members are
    resolved here so the human "Layouts" click and an agent preset call are one
    server operation. A preset additionally sets ``prune_rail`` on the
    broadcast, giving presets membership-exclusive semantics
    (non-members leave the launcher rail); a ``tiles`` request never removes
    rail membership, it only adds any listed non-member.

    Focus caveat: the server records the *requested* focus in
    ``app.state.active_panel`` and passes it through the broadcast unchanged.
    Panel health is only observable in the browser, so clients apply the
    healthy-fallback rule (requested panel if healthy, else the first healthy
    listed tile, else no focus change). A read-back of ``active`` therefore
    reports what was asked for, which may differ from what a client could
    actually focus.

    With no ``focus`` given — every human "Layouts" click, and any agent
    arrangement that does not name one — the first resolved tile is recorded
    instead. An arrangement always lands focus somewhere, and ``tiles[0]`` is
    what the client's healthy-fallback rule picks in the common case. Leaving
    ``active_panel`` untouched would strand it on a panel the arrangement just
    closed, so a preset click could leave ``active`` naming a panel that the
    very same ``GET /api/panels`` response omits from ``visible``. The
    broadcast still carries ``focus`` only when one was requested, leaving the
    client's fallback rule in charge of what is actually focused on screen.

    An agent arrangement is mirrored into the activity history ring as a single
    ``arrange_workspace`` row targeting the recorded focus panel.

    Args:
        body: ``tiles`` (explicit ids, left-to-right) **or** ``preset`` (a
            configured layout name), an optional ``focus`` target that must be
            one of the resolved tiles, and an optional ``source`` attribution.
        request: Incoming FastAPI request carrying ``app.state``.

    Returns:
        ``{"status": "ok", "tiles": [...], "focus": <id|None>,
        "preset": <name|None>, "prune_rail": <bool>}``

    Raises:
        HTTPException: 422 when neither or both of ``tiles``/``preset`` are
            given, when ``tiles`` is empty, when any requested id is unknown or
            is the terminal tile, when the preset name is unknown or resolves
            to no known member, or when ``focus`` is not one of the resolved
            tiles.
    """
    if (body.tiles is None) == (body.preset is None):
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one of 'tiles' or 'preset'",
        )

    known = _known_panel_ids(request)
    if body.preset is not None:
        tiles = _resolve_preset_tiles(request, body.preset, known)
        prune_rail = True
    else:
        tiles = _resolve_requested_tiles(body.tiles or [], known)
        prune_rail = False

    if body.focus is not None and body.focus not in tiles:
        raise HTTPException(
            status_code=422,
            detail=f"Focus target {body.focus!r} is not among the arranged tiles: {tiles}",
        )

    # Rail membership. A tiles request is additive (a panel the operator can
    # still reach from the rail keeps its entry); a preset prunes to its
    # members, keeping the existing rail order for the ones that survive.
    visible_panels: list[str] = list(getattr(request.app.state, "visible_panels", []))
    if prune_rail:
        visible_panels = [pid for pid in visible_panels if pid in tiles]
    visible_panels += [pid for pid in tiles if pid not in visible_panels]
    request.app.state.visible_panels = visible_panels

    # An arrangement always lands focus somewhere, so the recorded active panel
    # must move with it. Leaving it alone would let ``active`` name a panel the
    # same response no longer lists as visible — the state consumers read.
    request.app.state.active_panel = body.focus or tiles[0]

    event: dict = {"type": "panel_arrange", "tiles": tiles}
    if body.focus is not None:
        event["focus"] = body.focus
    if prune_rail:
        event["prune_rail"] = True
    if body.source:
        event["source"] = body.source
    if body.source == "agent":
        # One row for the whole arrangement, targeting the panel focus lands on
        # — the same id recorded as ``active_panel`` above, so the history entry
        # names the tile the operator's eye is sent to.
        _mirror_agent_panel_activity(request, "arrange_workspace", body.focus or tiles[0])
    request.app.state.broadcaster.broadcast(event)
    return {
        "status": "ok",
        "tiles": tiles,
        "focus": body.focus,
        "preset": body.preset,
        "prune_rail": prune_rail,
    }


class PanelLayoutRequest(BaseModel):
    tiles: list[str]
    dock: bool


@router.post("/api/panel-layout")
async def report_panel_layout(body: PanelLayoutRequest, request: Request):
    """Record which service tiles a client currently has on screen.

    This is the reporter's endpoint, not a command: it is how the browser tells
    the server what the operator's workspace actually looks like, so the agent
    stops guessing tile occupancy from rail membership. Nothing is broadcast —
    a human's tile gestures are never pushed to other clients (report-only,
    last-writer-wins), and reports are agent-facing state only.

    ``tiles`` is the service-tile list in spatial reading order (rows
    top-then-left); the terminal tile is never part of it. An empty list is a
    valid report from a dock client — it means the operator closed every
    service tile.

    ``dock`` is the reporting client's capability flag, and it decides how the
    report is *recorded*. A client running without the dock shell cannot see
    tile order, so the ``{"tiles": [], "dock": false}`` it sends is a presence
    signal, not an observation of an empty workspace. Recording that ``[]``
    verbatim would let a fallback client clobber a dock client's live list with
    a confident "nothing is open". So ``dock: false`` stores occupancy as
    **unknown** (``open_tiles = None``) while still stamping the timestamp and
    the flag — ``GET /api/panels`` then reports ``open_tiles: null`` with a
    numeric age, meaning "a client is watching but cannot report tile order".
    The mapping happens here, not on the wire. ``dock: true`` records the list
    verbatim.

    Content dedupe: a report whose *recorded* occupancy and ``dock`` flag both
    equal the stored state is a no-op — the stored timestamp is deliberately
    *not* bumped, so ``open_tiles_age_s`` keeps measuring when the layout last
    **changed**. This is the server half of the convergence contract: the
    client also skips posting a report equal to its last acknowledged one, so
    an applied arrangement settles after at most one extra round-trip instead
    of looping. Two successive dock-less reports therefore dedupe against each
    other, since both record the same unknown occupancy.

    Args:
        body: ``tiles`` (service panel ids in reading order) and ``dock``
            (whether the reporting client has a dock shell).
        request: Incoming FastAPI request carrying ``app.state``.

    Returns:
        ``{"status": "ok", "tiles": [...]|None, "dock": <bool>,
        "updated": <bool>}`` — ``tiles`` echoes what was *recorded*, so it is
        ``None`` for a dock-less report, and ``updated`` is ``False`` when the
        report was deduped.

    Raises:
        HTTPException: 422 when a reported id is unknown or is the terminal
            tile.
    """
    known = _known_panel_ids(request)
    if _TERMINAL_PANEL_ID in body.tiles:
        raise HTTPException(
            status_code=422,
            detail=(
                f"The terminal tile ({_TERMINAL_PANEL_ID!r}) is not a service panel "
                "and must not be reported"
            ),
        )
    unknown = [pid for pid in body.tiles if pid not in known]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown panel ids: {unknown}. Valid panel ids: {sorted(known)}",
        )

    # A dock-less client cannot see tile order, so its report says only "someone
    # is watching" — the occupancy it carries is recorded as unknown (``None``)
    # rather than as a confident empty list that consumers would read as
    # "the operator closed everything".
    occupancy: list[str] | None = list(body.tiles) if body.dock else None

    stored_tiles = getattr(request.app.state, "open_tiles", None)
    stored_dock = getattr(request.app.state, "open_tiles_dock", None)
    if stored_tiles == occupancy and stored_dock == body.dock:
        return {"status": "ok", "tiles": stored_tiles, "dock": stored_dock, "updated": False}

    request.app.state.open_tiles = occupancy
    request.app.state.open_tiles_dock = body.dock
    request.app.state.open_tiles_ts = time.time()
    return {"status": "ok", "tiles": occupancy, "dock": body.dock, "updated": True}


# ---- Runtime panel registration ---- #

_SCHEME_DEFAULT_PORT: dict[str, int] = {"http": 80, "https": 443}


def _allowlist_matches(host: str, port: int | None, scheme: str, allowlist: list[str]) -> bool:
    """Return True if ``host:port`` matches an entry in ``allowlist``.

    Allowlist entries are pre-lowercased ``host[:port]`` strings.  Port
    matching semantics:

    - Entry **without** a port: matches ``host`` at **any** URL port.
    - Entry **with** a port: matches only when the URL's *effective* port
      equals the entry port.  Implicit default ports (``http`` → 80,
      ``https`` → 443) are normalized before comparison, so an allowlist
      entry ``"grafana.lan:80"`` matches ``http://grafana.lan/``.

    Entry parsing uses ``urlparse`` internally so that bracketed IPv6
    literals (e.g. ``"[2001:db8::1]:9090"``) are handled correctly.  The
    allowlist host must match the ``parsed.hostname`` form (no brackets).

    Args:
        host: Lowercased hostname extracted from the candidate URL (no brackets
            for IPv6 literals).
        port: Explicit URL port, or ``None`` if omitted.
        scheme: URL scheme (``"http"`` or ``"https"``) for default-port lookup.
        allowlist: Pre-lowercased ``host[:port]`` strings from config.

    Returns:
        True when at least one allowlist entry matches.
    """
    effective_port = port if port is not None else _SCHEME_DEFAULT_PORT.get(scheme)
    for entry in allowlist:
        try:
            # Prepend a dummy scheme so urlparse recognises bare "host" and "[ipv6]:port".
            _ep = urlparse(f"http://{entry}")
            entry_host = (_ep.hostname or "").lower()
            entry_port = _ep.port  # None when omitted
        except Exception:
            continue
        if entry_host != host:
            continue
        if entry_port is None:
            # No port in entry — match any port.
            return True
        if entry_port == effective_port:
            return True
    return False


async def _validate_panel_url(raw_url: str, allowlist: list[str] | None) -> str | None:
    """Validate a panel URL for SSRF-relevant categories.

    Classification is based on **resolved** addresses, not the input string.
    This defeats numeric-encoding bypasses such as ``http://2130706433/``
    (decimal loopback) and DNS names that resolve to blocked ranges.

    Rejects:
    - Non-http/https schemes.
    - Hosts that cannot be resolved (DNS failure → 422).
    - Any resolved address that is loopback (127.0.0.0/8, ::1),
      link-local / cloud-metadata (169.254.0.0/16 incl. 169.254.169.254,
      fe80::/10), or unspecified (0.0.0.0/::).
    - IPv4-mapped IPv6 addresses (e.g. ``::ffff:169.254.169.254``) are
      unwrapped and then classified as their IPv4 equivalents.

    Permits: ordinary private LAN ranges (10/8, 172.16/12, 192.168/16) —
    real Grafana dashboards live there.

    Out of scope: DNS rebinding *after* registration (host is validated at
    registration time only; rebinding is a network-layer concern).

    ``getaddrinfo`` is executed in a thread pool so the async event loop is
    not blocked.  Tests can mock ``socket.getaddrinfo`` directly.

    Args:
        raw_url: The user-supplied URL to inspect.
        allowlist: Optional pre-lowercased ``host[:port]`` strings; ``None``
            means no allowlist is configured (allow all non-blocked hosts).

    Returns:
        A human-readable error string when the URL is rejected, or ``None``
        when it passes all checks.
    """
    try:
        parsed = urlparse(raw_url)
    except Exception:
        return "Invalid URL"

    if parsed.scheme not in ("http", "https"):
        return f"URL scheme must be http or https, got: {parsed.scheme!r}"

    host = (parsed.hostname or "").lower()
    if not host:
        return "URL must include a host"

    # Resolve and classify all returned addresses.
    try:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None, socket.getaddrinfo, host, parsed.port or 0, 0, socket.SOCK_STREAM
        )
    except OSError:
        return f"Could not resolve host: {host!r}"

    if not results:
        return f"Could not resolve host: {host!r}"

    for _family, _type, _proto, _canonname, sockaddr in results:
        raw_addr = sockaddr[0]
        ip = ipaddress.ip_address(raw_addr)
        # Unwrap IPv4-mapped IPv6 (e.g. ::ffff:169.254.169.254).
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        if ip.is_loopback or ip.is_link_local or ip.is_unspecified:
            return (
                f"Resolved address {raw_addr!r} for host {host!r} is not permitted "
                "(loopback, link-local, cloud-metadata, or unspecified)"
            )

    # Allowlist enforcement (after address validation so blocked IPs are always caught).
    if allowlist is not None and not _allowlist_matches(
        host, parsed.port, parsed.scheme, allowlist
    ):
        netloc_key = host + (f":{parsed.port}" if parsed.port else "")
        return f"Host {netloc_key!r} is not in the configured runtime_panel_allowlist"

    return None


class PanelRegisterRequest(BaseModel):
    id: str
    label: str
    url: str
    path: str = "/"
    health_endpoint: str | None = None
    source: Literal["agent"] | None = None


@router.post("/api/panels/register")
async def register_panel(body: PanelRegisterRequest, request: Request):
    """Dynamically register a custom panel at runtime.

    Requires ``web.allow_runtime_panels: true`` in config.  The supplied URL
    is validated to reject loopback, link-local, and cloud-metadata targets,
    and is stored raw server-side so the proxy can forward to it.  The URL
    exposed to the browser (in broadcasts and ``GET /api/panels``) is rewritten
    to ``<prefix>/panel/{id}`` (``<prefix>`` is the empty string outside
    multi-user deployments).

    If a custom panel with the same ``id`` already exists it is replaced
    atomically (remove-then-append) so the proxy always returns the first match.

    An agent-origin registration is mirrored into the activity history ring as
    a ``register_panel`` row.

    Args:
        body: Panel registration fields: ``id``, ``label``, ``url`` (raw),
            ``path`` (default ``"/"``), ``health_endpoint`` (optional).
        request: Incoming FastAPI request carrying ``app.state``.

    Returns:
        ``{"status": "ok", "id": <id>, "label": <label>, "url": "<prefix>/panel/<id>"}``

    Raises:
        HTTPException: 403 when runtime panel registration is disabled.
        HTTPException: 422 when the URL fails security validation or the host
            is not in the configured allowlist.
    """
    if not getattr(request.app.state, "allow_runtime_panels", False):
        raise HTTPException(
            status_code=403,
            detail="Runtime panel registration is disabled. Set web.allow_runtime_panels: true to enable.",
        )

    # Reserve both built-in ids and config-defined panel ids. Config-defined ids
    # are derived from the live custom-panel state (never a second stored field
    # that could drift) via the ``configDefined`` marker the config loader stamps.
    # Without this, a runtime registration could squat a config panel's id and the
    # remove-then-append below would silently repoint it — e.g. redirecting the
    # EVENTS panel (which the proxy credentials server-side) at an attacker URL.
    config_panel_ids = {
        cp["id"]
        for cp in getattr(request.app.state, "custom_panels", [])
        if cp.get("configDefined")
    }
    if body.id in BUILTIN_PANELS or body.id in config_panel_ids:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Panel id {body.id!r} collides with a built-in or config-defined "
                "panel; choose a different id."
            ),
        )

    allowlist: list[str] | None = getattr(request.app.state, "runtime_panel_allowlist", None)
    url_error = await _validate_panel_url(body.url, allowlist)
    if url_error:
        raise HTTPException(status_code=422, detail=url_error)

    # Replace-by-id: remove any existing entry before appending so the proxy
    # (which returns the first match) always sees the freshest registration.
    custom_panels: list[dict] = list(getattr(request.app.state, "custom_panels", []))
    custom_panels = [cp for cp in custom_panels if cp.get("id") != body.id]
    custom_panels.append(
        {
            "id": body.id,
            "label": body.label,
            "url": body.url,  # stored RAW; proxy reads this for forwarding
            "healthEndpoint": body.health_endpoint,
            "path": body.path,
        }
    )
    request.app.state.custom_panels = custom_panels

    # Ensure the new panel is visible
    visible_panels: list[str] = list(getattr(request.app.state, "visible_panels", []))
    if body.id not in visible_panels:
        visible_panels.append(body.id)
    request.app.state.visible_panels = visible_panels

    browser_url = f"{compute_url_prefix()}/panel/{body.id}"
    event: dict = {
        "type": "panel_register",
        "id": body.id,
        "label": body.label,
        "url": browser_url,  # rewritten for the browser
        "healthEndpoint": body.health_endpoint,
        "path": body.path,
    }
    if body.source:
        event["source"] = body.source
    if body.source == "agent":
        _mirror_agent_panel_activity(request, "register_panel", body.id)
    request.app.state.broadcaster.broadcast(event)
    return {"status": "ok", "id": body.id, "label": body.label, "url": browser_url}


@router.get("/panel-static/{panel_id}/{path:path}")
async def serve_discovered_panel(panel_id: str, path: str, request: Request):
    """Serve a discovered local panel bundle's files, same-origin.

    Backs the ``/panel-static/{id}/`` URL that :mod:`panel_discovery` assigns to
    discovered static panels.  Distinct from the reverse-proxy path
    (``/panel/{id}``, for URL-backed panels): this reads files straight from the
    panel bundle directory recorded in ``app.state.discovered_panel_dirs``.

    Fail-closed: an unknown id, a path that escapes the bundle directory
    (``..`` traversal), or a missing file all return 404 — never a file outside
    the bundle.  The bare ``/panel-static/{id}/`` request (empty ``path``) serves
    the manifest's ``entry`` file.

    Args:
        panel_id: The discovered panel id.
        path: Path within the bundle; empty serves the manifest entry.
        request: Incoming request carrying ``app.state.discovered_panel_dirs``.

    Returns:
        A :class:`FileResponse` for the requested asset, or a 404 ``Response``.
    """
    panels = getattr(request.app.state, "discovered_panel_dirs", {})
    panel = panels.get(panel_id)
    if panel is None:
        return Response(content=f"Panel '{panel_id}' is not available", status_code=404)

    base = Path(panel.directory).resolve()
    target = (base / (path or panel.entry)).resolve()

    # Fail-closed traversal guard: the resolved target must stay inside the
    # bundle and must be a regular file.
    if not target.is_relative_to(base) or not target.is_file():
        return Response(content="Not found", status_code=404)

    return FileResponse(target)


@router.post("/api/terminal/restart")
async def restart_terminal(request: Request):
    """Terminate the current PTY session (and operator session if active).

    The existing WebSocket reconnection logic automatically respawns
    a fresh PTY with the updated config.
    """
    pty_registry = request.app.state.pty_registry
    operator_registry = request.app.state.operator_registry

    # Terminate all PTY sessions (single-user model)
    pty_registry.cleanup_all()
    logger.info("All PTY sessions terminated for restart")

    # Terminate all operator sessions if active
    try:
        await operator_registry.cleanup_all()
    except Exception:
        pass  # May not have active operator sessions

    return {"status": "ok", "message": "Terminal session terminated — reconnecting"}


# ---- MCP Server Introspection ---- #


@router.get("/api/mcp-servers")
async def get_mcp_servers(request: Request):
    """Return enriched MCP server metadata with tool lists."""
    from osprey.mcp_server.introspect import get_mcp_servers_cached

    project_dir = request.app.state.project_cwd
    mcp_json_path = __import__("pathlib").Path(project_dir) / ".mcp.json"

    if not mcp_json_path.exists():
        return []

    try:
        servers = await get_mcp_servers_cached(mcp_json_path, project_dir, timeout=10.0)
        return servers
    except Exception:
        logger.warning("mcp-servers: introspection failed", exc_info=True)
        return []
