"""Panel configuration, server URL, health, and MCP introspection routes."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import ipaddress
import logging
import re
import socket
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from osprey.interfaces.artifacts.inline_assets import self_contained
from osprey.interfaces.common_middleware import (
    ASSET_CORS_HEADERS,
    MOUNT_SEGMENT_RE,
    PANEL_CAP_SEGMENT,
    apply_url_prefix,
    compute_url_prefix,
)
from osprey.interfaces.opaque_origin import document_headers
from osprey.interfaces.panel_urls import (
    Origin,
    agent_hop_panel_ids,
    agent_hop_url_refusal,
    canonical_origin,
    navigation_url_refusal,
    normalise_navigation_url,
    same_place,
)
from osprey.interfaces.web_terminal import panel_cap
from osprey.interfaces.web_terminal.dock_layout import (
    REASON_TEXT,
    layout_keys,
    layout_refusal,
    normalize_layout,
    same_layout,
)
from osprey.interfaces.web_terminal.feedback_destination import resolve_feedback_destination
from osprey.interfaces.web_terminal.panel_sandbox import apply_panel_sandbox
from osprey.interfaces.web_terminal.routes.agent_activity import record_activity
from osprey.interfaces.web_terminal.routes.proxy import _backend_is_loopback, _resolve_panel_url
from osprey.profiles.web_panels import (
    BUILTIN_PANEL_LABELS,
    BUILTIN_PANELS,
    JUPYTER_PANEL_ID,
    SIDECAR_PANELS,
    panel_id_refusal,
)
from osprey.registry.web import panel_url_state_attr

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
    """Health check endpoint, with the card's control-identity fields.

    See :func:`osprey.deployment.control_identity.health_fields`.
    """
    from osprey import __version__
    from osprey.deployment.control_identity import health_fields

    session_id = getattr(request.app.state, "server_session_id", None)
    return {
        "status": "healthy",
        "service": "web_terminal",
        "session_id": session_id,
        "version": __version__,
        **health_fields(),
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


def _sidecar_panel_config(request: Request, panel_id: str) -> dict:
    """The config body of a sidecar panel: its proxy URL and its start outcome.

    ``url``/``available`` follow the published backend URL, exactly as for a
    companion panel. ``state`` is what the terminal recorded for the sidecar
    (``None`` when it was never launched), and ``message`` the operator-facing
    sentence for it, spelled once in :mod:`~osprey.interfaces.web_terminal.sidecar_status`.
    """
    from osprey.interfaces.web_terminal.sidecar_status import status_message

    state = request.app.state
    url = getattr(state, panel_url_state_attr(panel_id), None)
    proxy_url = f"{compute_url_prefix()}/panel/{panel_id}" if url else None
    status = getattr(state, "sidecar_status", {}).get(panel_id)
    return {
        "url": proxy_url,
        "available": proxy_url is not None,
        "state": status.state if status is not None else None,
        "message": status_message(panel_id, status),
    }


@router.get("/api/jupyter-server")
async def jupyter_server_config(request: Request):
    """Return the notebook sidecar URL for iframe embedding, and its start outcome."""
    return _sidecar_panel_config(request, JUPYTER_PANEL_ID)


@router.get("/api/panels/{panel_id}/server")
async def sidecar_server_config(panel_id: str, request: Request):
    """Return a sidecar panel's URL for iframe embedding, and its start outcome.

    One route for every panel in ``SIDECAR_PANELS``, so a sidecar added to the
    registry needs no config route of its own. A sidecar that is switched off
    answers unavailable, as a companion panel's config route does; an id that
    names no sidecar is a 404.
    """
    if panel_id not in SIDECAR_PANELS:
        raise HTTPException(
            status_code=404, detail=f"{panel_id} is not a panel this terminal starts"
        )
    return _sidecar_panel_config(request, panel_id)


@router.post("/api/panels/{panel_id}/start")
async def start_sidecar_panel(panel_id: str, request: Request):
    """Start a panel sidecar again, one attempt at a time.

    The operator's retry for a sidecar that failed to start or stopped later.
    A request while an attempt is in flight joins it; a request for a running
    sidecar changes nothing. Answers the sidecar's config body: 200 once it
    runs, 202 while it is starting or after it failed.
    """
    from osprey.interfaces.web_terminal.app import request_sidecar_start

    enabled: set[str] = getattr(request.app.state, "enabled_panels", set())
    if panel_id not in SIDECAR_PANELS or panel_id not in enabled:
        raise HTTPException(
            status_code=404, detail=f"{panel_id} is not a panel this terminal starts"
        )
    status = await request_sidecar_start(request.app, panel_id)
    return JSONResponse(
        _sidecar_panel_config(request, panel_id),
        status_code=200 if status.state == "running" else 202,
    )


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
        return f"{prefix}{cp['url']}"
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


def _loopback_backend_ports(request: Request, panel_ids: Iterable[str]) -> dict[str, int]:
    """The port of each panel whose backend is this host's loopback, served at its root.

    The agent's tools name a backend by its loopback address (the artifact
    gallery's ``gallery_url``), and an agent answer may repeat it. That address
    is dead in a browser on another machine, so the chat maps a loopback
    link on one of these ports to the panel instead. Only the port leaves the
    server: a backend that is not a loopback literal, or that serves under a
    path or query (so a link path would not map 1:1 onto ``/panel/<id>/...``),
    is left out, and its links keep opening in a new tab.
    """
    ports: dict[str, int] = {}
    for panel_id in panel_ids:
        url = _resolve_panel_url(request, panel_id)
        if not url or not _backend_is_loopback(url):
            continue
        parsed = urlparse(url)
        if parsed.path not in ("", "/") or parsed.query:
            continue
        try:
            port = parsed.port
        except ValueError:
            continue
        ports[panel_id] = port or (443 if parsed.scheme == "https" else 80)
    return ports


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
            "presets":  [...],          # config-defined layouts: [{"name", "panels", "layout"}]
            "project_key": str,         # stable 16-hex per-project key (layout persistence)
            "layout":   {...}|None,     # the current dock client's dock layout (None = no dock reports)
            "popouts":  [...],          # panels in their own windows, per that client
            "rev":      int,            # its revision, minted per changed report (0 = never)
            "layout_age_s": float|None,  # seconds since that report (None = never)
            "layout_dock": bool|None,    # reporting client had a dock shell (None = never)
            "docs_url": str,            # target of the sidebar's Documentation control
            "feedback_trackers": [...],  # [{"kind", "label", "repo"|"url"}], render order
            "feedback_email": str,      # recipient of the prefilled mailto: draft
            "feedback_deployment": {...},  # identity lines for a report's metadata
            "feedback_escalation_url": str,  # prefilled upstream issue; "" upstream
            "loopback_ports": {id: int},  # backend port of each loopback-hosted panel
        }

    ``project_key`` is an opaque, stable per-project identifier (16 hex chars,
    a truncated sha256 of the resolved project directory). The client uses it as
    the ``osprey-dock-layout-<project_key>`` localStorage suffix so dock layouts
    persist independently per project on a shared browser origin.

    ``presets`` is the config-defined "Layouts" list (``web.presets``), resolved
    at startup against the live panel set and carried in config order. It is
    empty unless a deployment opts in, so the "+" popover renders unchanged by
    default. Each entry is ``{"name": <label>, "panels": [<member id>, ...],
    "layout": <dock layout>}``: ``layout`` is the collection form with panel-id
    tabs, pruned to the panels known now and normalised, and ``panels`` its tabs
    in reading order.

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

    ``layout`` is what a browser last reported through
    ``POST /api/panel-layout``: the tiles actually on screen. It is distinct
    from ``visible`` — that is launcher-rail membership, this is occupancy.
    ``popouts`` names the panels that client shows in windows of their own;
    they are not in ``layout``. ``layout_age_s`` is the seconds elapsed since
    the layout or the pop-outs last *changed* (a deduped repeat report does not
    reset it) and ``layout_dock`` is whether the reporting client had a dock
    shell.

    Read together, the three fields carry three distinct states, and a
    consumer must not collapse them:

    - **Never reported** — all three ``null``. No client has ever checked in;
      nothing is known about the screen. Not the same as an empty workspace.
    - **Unknown occupancy** — ``layout: null`` with a numeric ``layout_age_s``
      and ``layout_dock: false``. A client is watching but runs without the
      dock shell, so it cannot report tiles.
    - **Known occupancy** — ``layout`` is a tree (possibly holding only the
      terminal) with a numeric age and ``layout_dock: true``.

    So the age tells the two flavours of a null ``layout`` apart.

    ``docs_url``, ``feedback_trackers`` and ``feedback_email`` are the resolved
    ``web.docs_url`` / ``web.feedback.*`` values that address the two utility
    controls in the sidebar footer: the Documentation link target, and
    the outbound channels the feedback dialog offers (one prefilled new-issue
    link per configured tracker — ``web.feedback.trackers`` plus the
    ``web.feedback.github_repo`` sugar, already merged server-side — and a
    prefilled ``mailto:`` draft).

    ``loopback_ports`` maps a panel id to its backend's port, for each panel
    whose backend is a loopback literal served at its root (see
    :func:`_loopback_backend_ports`). Nothing else about a backend is exposed.
    """
    enabled = list(getattr(request.app.state, "enabled_panels", set()))
    custom_raw = getattr(request.app.state, "custom_panels", [])
    custom = [{**cp, "url": _browser_panel_url(cp)} for cp in custom_raw]
    default = getattr(request.app.state, "default_panel", None)
    visible = list(getattr(request.app.state, "visible_panels", enabled))
    active = getattr(request.app.state, "active_panel", None)
    labels = {pid: BUILTIN_PANEL_LABELS[pid] for pid in enabled if pid in BUILTIN_PANEL_LABELS}
    # Whether the Config panel's SERVER surface is live (web.config_panel.enabled).
    # `false` means /api/config and /api/claude-setup refuse every verb with 403,
    # so this payload must not advertise the panel in any form: the client reads
    # the flag to decide whether to render the Config tab at all, and a tab
    # rendered against a refusing surface is a dead control, not a gated one.
    # An app whose state carries no flag has refused the surface, as the
    # routes behind it do, so absence reads as False.
    config_panel_enabled = bool(getattr(request.app.state, "config_panel_enabled", False))
    # Whether the scaffold gallery's write surface is live
    # (web.scaffold_gallery.write_enabled). `false` means every write/delete
    # verb under /api/scaffold answers 403, so the browser must stop painting
    # the create/claim/save/delete/register controls that reach for them; the
    # gallery reads this flag to do that (static/js/scaffold/write-gate.js).
    # Absent reads as False, as the routes' own gate does.
    scaffold_write_enabled = bool(getattr(request.app.state, "scaffold_write_enabled", False))
    if not config_panel_enabled:
        # Belt and braces for the id itself. ``config`` is not a built-in panel
        # (it is a drawer tab, not a dock tile), so nothing puts it in these
        # lists today — but a config-defined custom panel may claim any id, and
        # a disabled deployment whose payload still named ``config`` would hand
        # the client back exactly the entry this key exists to withhold.
        enabled = [pid for pid in enabled if pid != _CONFIG_PANEL_ID]
        visible = [pid for pid in visible if pid != _CONFIG_PANEL_ID]
        custom = [cp for cp in custom if cp.get("id") != _CONFIG_PANEL_ID]
        labels.pop(_CONFIG_PANEL_ID, None)
        default = None if default == _CONFIG_PANEL_ID else default
        active = None if active == _CONFIG_PANEL_ID else active
    allow_runtime = bool(getattr(request.app.state, "allow_runtime_panels", False))
    presets = list(getattr(request.app.state, "panel_presets", []))
    project_key = _project_key(getattr(request.app.state, "project_cwd", None))
    # The dock layout as last reported by a browser (``layout``/``rev`` below)
    # and its freshness. The timestamp is never exposed raw: an age is what a
    # consumer can reason about without knowing the server's clock.
    layout_ts = getattr(request.app.state, "layout_ts", None)
    layout_age = None if layout_ts is None else time.time() - layout_ts
    layout_dock = getattr(request.app.state, "layout_dock", None)
    # Utility-cluster targets. The fallbacks come from the same resolver the
    # lifespan runs, called with nothing — which is the deployment that
    # configured nothing, i.e. the one the OSPREY project owns. No default is
    # re-typed here and no derivation is repeated, so the two paths cannot
    # drift. The leaf is importable from a route because it imports neither
    # the app nor any route; the cycle that once forced literals here
    # (app -> routes -> panels) does not arise.
    unconfigured = resolve_feedback_destination()
    docs_url = getattr(request.app.state, "docs_url", unconfigured.docs_url)
    feedback_trackers = getattr(request.app.state, "feedback_trackers", unconfigured.trackers)
    feedback_email = getattr(request.app.state, "feedback_email", unconfigured.email)
    # What the browser stamps into a prefilled draft's metadata block, and the
    # link a maintainer forwards a framework bug by. Both are resolved once at
    # startup; the browser renders them and invents nothing. Absent state (or a
    # deployment the OSPREY project already owns) leaves them empty, and an
    # empty metadata value is dropped by the body composer.
    identity = getattr(request.app.state, "deployment_identity", None)
    feedback_deployment = identity.build_lines() if identity is not None else {}
    feedback_escalation_url = getattr(request.app.state, "feedback_escalation_url", "")
    # Onboarding tour: the resolved invite policy, the derived capability
    # list for the "Ask in plain language" card, and whether the logbook
    # (ARIEL panel) is available. All resolved at startup (web.tour /
    # OSPREY_WEB_TOUR; config + enabled panels) — the browser renders these
    # facts and never invents its own. "once" default mirrors
    # app.DEFAULT_TOUR_POLICY — a literal for the routes->app import-cycle
    # reason above.
    tour = {
        "policy": getattr(request.app.state, "web_tour_policy", "once"),
        "capabilities": list(getattr(request.app.state, "tour_capabilities", [])),
        "logbook": bool(getattr(request.app.state, "tour_logbook", False)),
    }
    return {
        "enabled": enabled,
        "custom": custom,
        "default": default,
        "visible": visible,
        "active": active,
        "labels": labels,
        "allow_runtime_panels": allow_runtime,
        "presets": presets,
        "project_key": project_key,
        "layout": getattr(request.app.state, "dock_layout", None),
        "popouts": list(getattr(request.app.state, "dock_popouts", [])),
        "rev": getattr(request.app.state, "dock_layout_rev", 0),
        "layout_age_s": layout_age,
        "layout_dock": layout_dock,
        "docs_url": docs_url,
        "feedback_trackers": [dict(tracker) for tracker in feedback_trackers],
        "feedback_email": feedback_email,
        "feedback_deployment": feedback_deployment,
        "feedback_escalation_url": feedback_escalation_url,
        "config_panel_enabled": config_panel_enabled,
        "scaffold_write_enabled": scaffold_write_enabled,
        # The config file that exists but could not be read, which closed both
        # gates above; the browser names the file from this key and invents nothing.
        "config_unreadable_path": (
            str(p) if (p := getattr(request.app.state, "config_unreadable_path", None)) else None
        ),
        "tour": tour,
        # Whether this deployment lets a user run on a personal Claude login,
        # and whether this terminal's user has one: the browser's nudge
        # ("/login here and your usage bills to you") shows on allowed-and-absent
        # and never on a guess. Both decided at startup in app.py.
        "personal_login": {
            "allowed": bool(getattr(request.app.state, "personal_login_allowed", False)),
            "present": bool(getattr(request.app.state, "personal_login_present", False)),
        },
        "loopback_ports": _loopback_backend_ports(
            request, [*enabled, *(cp["id"] for cp in custom if cp.get("id"))]
        ),
    }


def _known_panel_ids(request: Request) -> set[str]:
    """Return the set of valid panel ids: enabled built-ins plus custom panels.

    Single source of truth for the membership check shared by the focus and
    visibility endpoints — both reject panel ids that are not in this set.
    """
    known: set[str] = set(getattr(request.app.state, "enabled_panels", set()))
    known |= {p["id"] for p in getattr(request.app.state, "custom_panels", [])}
    return known


def agent_hop_refusal(request: Request, member: str, url: str) -> str | None:
    """Why *url* may not reach panel *member*, judged against this terminal's panels; or ``None``.

    :func:`~osprey.interfaces.panel_urls.agent_hop_url_refusal`, where an
    agent-hop panel's process is any listener its configured backend names, in
    any spelling the browser reads as that listener (``127.1``,
    ``0x7f.0.0.1``, ``[::ffff:127.0.0.1]``, ``localhost.``, ...:
    :func:`~osprey.interfaces.panel_urls.same_place`). That much names the
    refusal. What fails closed is the check after it: an op admits an http(s)
    URL only on the member's own configured backend
    (:func:`~osprey.interfaces.panel_urls.member_navigation_url`, exact origin),
    and a registration admits no loopback URL at all, so an address this rule
    does not recognise is refused there, never admitted by guessing.

    Args:
        request: The request; its app's state holds the panels' backends.
        member: The panel the url is for (the id being registered, for a
            registration).
        url: The url, already passed through ``normalise_navigation_url``.
    """
    hop_ids = agent_hop_panel_ids(SIDECAR_PANELS)
    hop_origins = [
        origin
        for panel_id in hop_ids
        if (backend := _resolve_panel_url(request, panel_id))
        and (origin := canonical_origin(backend)) is not None
    ]

    def is_hop_origin(origin: Origin) -> bool:
        return any(same_place(origin, hop) for hop in hop_origins)

    return agent_hop_url_refusal(member, url, hop_ids=hop_ids, is_hop_origin=is_hop_origin)


#: Panel id of the Config surface. It is not a built-in dock panel — the Config
#: tab is static drawer markup served by ``index.html`` — so this is the id the
#: payload must never carry while ``web.config_panel.enabled`` is false, whether
#: it arrives as a built-in, a config-defined custom panel, or a focus target.
_CONFIG_PANEL_ID = "config"

#: Dock id of the native terminal/chat tile (``PANEL_TERMINAL`` in
#: ``dock-workspace.js``). It carries no server-side panel state and is never a
#: service tile, so every layout verb refuses it explicitly rather than relying
#: on it being absent from :func:`_known_panel_ids` — a custom panel squatting
#: the id must not become a way to move or close the operator's terminal.
_TERMINAL_PANEL_ID = "terminal"


def _mirror_agent_panel_activity(request: Request, tool: str, panel: str) -> None:
    """Record an agent-origin panel command in the agent-activity history buffer.

    Panel commands reach the browser as their own SSE frames (``panel_op``,
    ``panel_visibility``, ...), never through ``POST /api/agent-activity``, so
    without this they are invisible to a client that reads
    ``GET /api/agent-activity/recent`` after connecting late.  The row goes in
    through that route's own ``record_activity``, so a consumer feeds it
    through the handler it already uses for the SSE ``agent_activity`` stream.

    ``tool`` is synthetic: the panel routes carry no tool name of their own, so
    the caller supplies the MCP verb the action corresponds to (one of
    :data:`~osprey.interfaces.web_terminal.routes.panel_ops.OP_ACTIVITY_TOOLS`,
    ``add_panel_to_rail``, ``remove_panel_from_rail``
    or ``register_panel``) and the frontend words the entry from it.

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
    """A person's tab switch, reported by ``panel-commands.js``.

    Extra fields are refused: the agent's ``source``/``url`` form of this body is
    retired (an agent opens a panel through ``POST /api/panel-op/open``), and an
    old caller must hear so rather than be read as a person's report.
    """

    model_config = ConfigDict(extra="forbid")

    panel: str


@router.get("/api/panel-focus")
async def get_panel_focus(request: Request):
    """Return the currently active panel."""
    active = getattr(request.app.state, "active_panel", None)
    return {"active_panel": active}


@router.post("/api/panel-focus")
async def set_panel_focus(body: PanelFocusRequest, request: Request):
    """Mirror a person's tab switch as ``active_panel``; broadcast nothing.

    This is a gesture REPORT, not a command: the server records the active
    panel for the agent's gaze, and one operator's tab switches never move
    another client's workspace. The gesturing client applies its own focus.

    A report naming a panel that is not on the launcher rail is dropped whole:
    a person can only gesture at a panel already on the rail, so such a report
    is a fire-and-forget straggler that a concurrent arrangement overtook, and
    applying it would put the panel the arrangement just pruned back in the
    active slot. Membership is resolved the way ``get_panels`` resolves
    ``visible`` (an unset list means the enabled built-ins are the rail).

    Args:
        body: ``panel`` (panel id).
        request: Incoming FastAPI request carrying ``app.state``.

    Returns:
        ``{"status": "ok", "active_panel": <id|None>}``

    Raises:
        HTTPException: 422 when ``panel`` is not a known enabled or custom id,
            or the body carries any other field.
    """
    if body.panel not in _known_panel_ids(request):
        raise HTTPException(status_code=422, detail=f"Unknown panel: {body.panel}")
    stored_visible = getattr(request.app.state, "visible_panels", None)
    if stored_visible is None:
        visible_panels = list(getattr(request.app.state, "enabled_panels", set()))
    else:
        visible_panels = list(stored_visible)
    if body.panel not in visible_panels:
        active = getattr(request.app.state, "active_panel", None)
        return {"status": "ok", "active_panel": active}
    request.app.state.active_panel = body.panel
    return {"status": "ok", "active_panel": body.panel}


class PanelVisibilityRequest(BaseModel):
    """A rail-membership change.

    ``index`` places a panel being ADDED at that position of the rail's panel
    entries (the Agent entry is not counted), clamped to the list. It is how an
    operator's Undo puts a removed entry back where it was; without it an added
    panel goes to the end. It is ignored when hiding and when the panel is
    already a member, which keeps its place.
    """

    panel: str
    visible: bool
    source: Literal["agent"] | None = None
    index: int | None = Field(default=None, ge=0)


@router.post("/api/panel-visibility")
async def set_panel_visibility(body: PanelVisibilityRequest, request: Request):
    """Show or hide a panel and broadcast the change via SSE.

    An agent-origin change is also mirrored into the activity history buffer, as
    an ``add_panel_to_rail`` or ``remove_panel_from_rail`` row depending on the
    flag, so a client reading the history can word it the way it words the live
    frame.

    A positioned addition carries its clamped ``index`` on the frame, so every
    client places the entry at the same position of its own rail rather than
    only the client that asked; the frame of an addition that changed nothing
    carries none, so no client moves an entry that was already there.

    Args:
        body: ``panel`` (panel id), ``visible`` (desired visibility) and the
            optional ``index`` an addition is placed at.
        request: Incoming FastAPI request carrying ``app.state``.

    Returns:
        ``{"status": "ok", "panel": <id>, "visible": <bool>}``

    Raises:
        HTTPException: 422 when ``panel`` is not a known enabled or custom id.
    """
    if body.panel not in _known_panel_ids(request):
        raise HTTPException(status_code=422, detail=f"Unknown panel: {body.panel}")
    visible_panels: list[str] = list(getattr(request.app.state, "visible_panels", []))
    index: int | None = None
    if body.visible:
        if body.panel not in visible_panels and body.index is not None:
            index = min(body.index, len(visible_panels))
            visible_panels.insert(index, body.panel)
        elif body.panel not in visible_panels:
            visible_panels.append(body.panel)
    else:
        visible_panels = [p for p in visible_panels if p != body.panel]
    request.app.state.visible_panels = visible_panels
    event: dict = {"type": "panel_visibility", "panel": body.panel, "visible": body.visible}
    if index is not None:
        event["index"] = index
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


class RailOrderRequest(BaseModel):
    """Move a rail member so it sits directly above ``before``.

    ``before`` names another member; ``None`` moves the panel to the foot of
    the rail. The Agent entry is not a member, so it is never a target and
    always stays at the head.
    """

    panel: str
    before: str | None = None


@router.post("/api/rail-order")
async def set_rail_order(body: RailOrderRequest, request: Request):
    """Reorder the rail and broadcast the whole resulting order via SSE.

    The rail's order is server-owned like its membership, so every client and
    every reload shows the same rail. The frame carries the full order rather
    than the one move, which makes applying it idempotent and lets a client
    that missed an earlier frame converge from this one.

    Args:
        body: ``panel`` (the member to move) and ``before`` (the member it
            should sit above, or ``None`` for the foot of the rail).
        request: Incoming FastAPI request carrying ``app.state``.

    Returns:
        ``{"status": "ok", "order": [<id>, ...]}``

    Raises:
        HTTPException: 422 when ``panel`` or ``before`` is not on the rail, or
            when ``before`` names the panel itself.
    """
    # Membership, resolved the same way ``get_panels`` resolves ``visible``.
    stored_visible = getattr(request.app.state, "visible_panels", None)
    if stored_visible is None:
        visible_panels = sorted(getattr(request.app.state, "enabled_panels", set()))
    else:
        visible_panels = list(stored_visible)
    if body.panel not in visible_panels:
        raise HTTPException(status_code=422, detail=f"Not on the rail: {body.panel}")
    if body.before is not None and (body.before == body.panel or body.before not in visible_panels):
        raise HTTPException(status_code=422, detail=f"Cannot place before: {body.before}")
    order = [p for p in visible_panels if p != body.panel]
    order.insert(order.index(body.before) if body.before else len(order), body.panel)
    request.app.state.visible_panels = order
    request.app.state.broadcaster.broadcast({"type": "rail_order", "order": order})
    return {"status": "ok", "order": order}


class PanelLayoutRequest(BaseModel):
    layout: dict | None
    dock: bool
    client_id: str = Field(min_length=1, max_length=128)
    popouts: list[str] = Field(default_factory=list)


@router.post("/api/panel-layout")
async def report_panel_layout(body: PanelLayoutRequest, request: Request):
    """Record the dock layout a client currently shows, and who that client is.

    This is the reporter's endpoint, not a command: it is how the browser's dock
    tells the server what the operator's workspace actually looks like. The body
    is the holder's dock layout tree (tiles, their tabs, each tile's showing tab and
    the terminal tile, keyed ``"terminal"``), the panels that client shows in
    pop-out windows of their own (``popouts``, which the tree never names), the
    page's ``client_id`` and the client's ``dock`` capability flag. Nothing is
    broadcast — a human's tile gestures are never pushed to other clients
    (report-only, last-writer-wins).

    A ``dock: true`` report must carry a layout that passes
    :func:`~osprey.interfaces.web_terminal.dock_layout.layout_refusal` and whose
    every key other than the terminal is a known panel id; a pop-out is a known
    panel id that is not in the tree. The layout is stored normalised, its
    ``client_id`` becomes the current dock client, and a changed layout or
    pop-out set mints the next ``rev``. A client running without the dock
    shell cannot see the layout, so ``dock: false`` records the layout as
    **unknown** (``None``) with no current client while still stamping the
    timestamp — "someone is watching but cannot report tiles".

    Content dedupe: a report whose recorded layout, pop-outs and ``dock`` flag
    equal the stored state is a no-op — neither ``rev`` nor the timestamp moves (a
    different ``client_id`` still becomes current). This is the server half of
    the convergence contract: the client also skips posting a layout equal to
    its last acknowledged one, so an applied arrangement settles instead of
    looping.

    Returns:
        ``{"status": "ok", "layout": {...}|None, "popouts": [...], "dock": <bool>,
        "rev": <int>, "updated": <bool>}`` — ``layout`` and ``popouts`` echo what
        was *recorded* and ``rev`` is the stored revision, which the client
        keeps as its ``ackedRev``.

    Raises:
        HTTPException: 422 when the layout breaks a dock layout rule, a key or
            a pop-out is unknown, a pop-out is also in the tree, or a dock
            report carries no layout.
    """
    layout = _checked_dock_layout(request, body.layout) if body.dock else None
    popouts = _checked_popouts(request, body.popouts, layout) if body.dock else []
    rev, updated = _record_dock_report(
        request.app.state, body.client_id, layout, body.dock, popouts
    )
    return {
        "status": "ok",
        "layout": layout,
        "popouts": popouts,
        "dock": body.dock,
        "rev": rev,
        "updated": updated,
    }


def _checked_dock_layout(request: Request, layout: dict | None) -> dict:
    """``layout`` normalised, once it passes the dock layout rules and names only known panels.

    Raises:
        HTTPException: 422 when the layout is missing, breaks a dock layout rule or
            names a key that is neither the terminal nor a known panel id.
    """
    if layout is None:
        raise HTTPException(status_code=422, detail="A dock report must carry its layout")
    refusal = layout_refusal(layout)
    if refusal is not None:
        reason, where = refusal
        raise HTTPException(status_code=422, detail=f"{where}: {reason}: {REASON_TEXT[reason]}")
    known = _known_panel_ids(request)
    unknown = [key for key in layout_keys(layout) if key != _TERMINAL_PANEL_ID and key not in known]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown panel ids: {unknown}. Valid panel ids: {sorted(known)}",
        )
    return normalize_layout(layout)


def _checked_popouts(request: Request, popouts: list[str], layout: dict | None) -> list[str]:
    """``popouts`` sorted and deduplicated, once each names a known panel outside ``layout``.

    Raises:
        HTTPException: 422 when a pop-out is the terminal or an unknown id, or is
            also a tab of the tree (a panel is in a window or in a tile, never both).
    """
    known = _known_panel_ids(request)
    bad = [key for key in popouts if key == _TERMINAL_PANEL_ID or key not in known]
    if bad:
        raise HTTPException(status_code=422, detail=f"Unknown pop-out panel ids: {bad}")
    docked = set(layout_keys(layout))
    both = sorted(set(popouts) & docked)
    if both:
        raise HTTPException(
            status_code=422, detail=f"A pop-out cannot also be a tab of the layout: {both}"
        )
    return sorted(set(popouts))


def _record_dock_report(
    state, client_id: str, layout: dict | None, dock: bool, popouts: list[str]
) -> tuple[int, bool]:
    """Store a client's report as the current one; return ``(rev, updated)``.

    The reporting client becomes the current dock client (none for a
    ``dock: false`` report). A report equal to the stored layout, pop-outs and
    ``dock`` flag changes neither ``rev`` nor the timestamp; any other report
    mints the next ``rev`` — a panel leaving for its own window, or coming
    back, is a change the agent reads. Layouts are compared the way the browser
    compares its own reports (:func:`~osprey.interfaces.web_terminal.dock_layout.same_layout`,
    sizes up to the engine's tolerance), so the read-back of a tree the dock
    just applied is the same report and never a revision of its own. Shared by
    the report route and the op-result route, so an agent op's result and a
    person's report move the same revision.
    """
    state.dock_client_id = client_id if dock else None
    rev = getattr(state, "dock_layout_rev", 0)
    if (
        same_layout(getattr(state, "dock_layout", None), layout)
        and list(getattr(state, "dock_popouts", [])) == popouts
        and getattr(state, "layout_dock", None) == dock
    ):
        return rev, False
    rev += 1
    state.dock_layout = layout
    state.dock_popouts = list(popouts)
    state.dock_layout_rev = rev
    state.layout_dock = dock
    state.layout_ts = time.time()
    return rev, True


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
    literals (e.g. ``"[2001:db8::1]:8443"``) are handled correctly.  The
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


def _normalize_ip(raw_addr: str | int) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse one resolved address into a comparable :mod:`ipaddress` object.

    A scope id (``fe80::1%en0``, as ``getsockname`` reports it on a link-local
    interface) is dropped and an IPv4-mapped IPv6 address (``::ffff:10.0.0.5``)
    is unwrapped, so the two spellings of one address compare equal wherever
    they meet.

    Args:
        raw_addr: The address from a ``sockaddr``, typed ``str | int`` as
            ``getaddrinfo`` reports it.

    Returns:
        The parsed address, or ``None`` when the value is not a string or the
        string is not an IP literal.
    """
    if not isinstance(raw_addr, str):
        return None
    try:
        ip = ipaddress.ip_address(raw_addr.split("%", 1)[0])
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


#: How long a probe result stands before the interfaces are re-read. A host's
#: own addresses change on the timescale of a DHCP lease or a VPN coming up,
#: not of a panel registration, so a minute is short enough to follow a real
#: change and long enough that a burst of registrations pays for one probe.
_HOST_ADDRS_TTL_SECONDS = 60.0

#: ``(monotonic timestamp, addresses)`` of the last probe, or ``None`` before
#: the first one. Monotonic because this is an age, not a wall-clock date: a
#: clock step must not make a fresh entry look expired or an old one fresh.
_host_addrs_cache: tuple[float, frozenset[ipaddress.IPv4Address | ipaddress.IPv6Address]] | None = (
    None
)


def reset_host_addrs_cache() -> None:
    """Forget the last probe, so the next validation probes the interfaces again.

    The memo is process-wide: without a reset, one probe's answer stands for
    every panel validation in the process until the TTL runs out.
    """
    global _host_addrs_cache
    _host_addrs_cache = None


def _host_interface_addresses() -> frozenset[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Best-effort, dependency-free discovery of this host's own addresses.

    Two independent, zero-traffic probes (the idiom the loopback-chokepoint
    e2e test uses, brought into production for panel validation):

    1. ``getaddrinfo(gethostname(), None)`` — every address the host's own
       name resolves to, which catches interfaces with no default route.
    2. The UDP-connect trick — ``connect()`` on a ``SOCK_DGRAM`` socket only
       makes the kernel pick a route and sends no packet, and
       ``getsockname()`` then reports the local address that route would use.
       Run once per family so a v6-only interface is seen too.

    Either probe failing (offline host, no default route, unresolvable
    hostname) is expected rather than exceptional, so both are wrapped: the
    function fails OPEN with an empty set, and the loopback / link-local /
    unspecified checks in :func:`_validate_panel_url` still apply.

    The result is memoized for :data:`_HOST_ADDRS_TTL_SECONDS`. Probe 1 is a
    name resolution with no timeout of its own — on a host whose resolver is
    slow or unreachable it blocks for however long the system resolver takes —
    and running that unconditionally on every registration would put an
    unbounded stall in the request path. :func:`reset_host_addrs_cache` forgets
    the memo.

    Blocking (it resolves and opens sockets), so callers on the event loop run
    it in a thread pool.

    Returns:
        The discovered addresses, normalized by :func:`_normalize_ip`.
    """
    global _host_addrs_cache

    cached = _host_addrs_cache
    if cached is not None and (time.monotonic() - cached[0]) < _HOST_ADDRS_TTL_SECONDS:
        return cached[1]

    addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()

    try:
        for *_prefix, sockaddr in socket.getaddrinfo(socket.gethostname(), None):
            ip = _normalize_ip(sockaddr[0])
            if ip is not None:
                addresses.add(ip)
    except OSError:
        pass

    # RFC 5737 / RFC 3849 documentation addresses. Route selection works the
    # same for them as for any off-link destination and no packet is sent, so
    # nothing here depends on the destination existing — a documentation
    # address just makes that explicit, where a real public resolver's address
    # reads like traffic to a third party that never happens.
    for family, destination in (
        (socket.AF_INET, ("192.0.2.1", 80)),
        (socket.AF_INET6, ("2001:db8::1", 80)),
    ):
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as probe:
                probe.connect(destination)
                ip = _normalize_ip(probe.getsockname()[0])
            if ip is not None:
                addresses.add(ip)
        except OSError:
            pass

    resolved = frozenset(addresses)
    _host_addrs_cache = (time.monotonic(), resolved)
    return resolved


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
    - Any resolved address that is one of the deploy host's OWN interface
      addresses (:func:`_host_interface_addresses`). The panel proxy fetches
      server-side from inside the deployment, so a panel pointed at this
      host's LAN address is a route back into the deployment's own web
      terminals — nginx injects a per-user terminal secret on every request
      under the ``open`` auth posture, which would make that a route into a
      neighbour's terminal. The refusal is deliberately host-wide rather than
      port-scoped: it costs an operator the ability to register an unrelated
      dashboard that happens to be co-hosted with the deployment (the error
      names the workarounds), and buys not having to trust a port number to
      stay pointed at something other than the deployment's own front door.
    - IPv4-mapped IPv6 addresses (e.g. ``::ffff:169.254.169.254``) are
      unwrapped and then classified as their IPv4 equivalents.

    Permits: ordinary private LAN ranges (10/8, 172.16/12, 192.168/16) on
    hosts other than this one — real Grafana dashboards live there.

    Out of scope: DNS rebinding *after* registration (host is validated at
    registration time only; rebinding is a network-layer concern); other
    hosts inside the same deployment fleet, which are not distinguishable
    from a genuine dashboard host from here.

    ``getaddrinfo`` and the interface-address probes are executed in a thread
    pool so the async event loop is not blocked.  Tests can mock
    ``socket.getaddrinfo`` and ``_host_interface_addresses`` directly.

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

    # Probed lazily, and at most once per call: an address rejected by a
    # categorical check below never reaches the own-address comparison, so the
    # common refusals (loopback, metadata) cost no interface probe at all.
    own_addresses: frozenset[ipaddress.IPv4Address | ipaddress.IPv6Address] | None = None

    for _family, _type, _proto, _canonname, sockaddr in results:
        raw_addr = sockaddr[0]
        ip = _normalize_ip(raw_addr)
        if ip is None:
            return f"Resolved address {raw_addr!r} for host {host!r} is not an IP address"
        if ip.is_loopback or ip.is_link_local or ip.is_unspecified:
            return (
                f"Resolved address {raw_addr!r} for host {host!r} is not permitted "
                "(loopback, link-local, cloud-metadata, or unspecified)"
            )
        if own_addresses is None:
            # Fails open (empty set) when the host cannot be probed; the
            # categorical checks above stand on their own.
            own_addresses = await loop.run_in_executor(None, _host_interface_addresses)
        if ip in own_addresses:
            return (
                f"Resolved address {raw_addr!r} for host {host!r} is the deployment "
                "host itself; a panel proxied back into this deployment is not "
                "permitted. Host the dashboard on another machine, or use the "
                "deployment's landing page entries instead."
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
    is normalised as the browser's URL parser reads it
    (:func:`~osprey.interfaces.panel_urls.normalise_navigation_url`, the
    navigation guard), validated to reject non-http(s) schemes and loopback,
    link-local, and cloud-metadata targets, and is stored in that normalised
    form server-side so the proxy can forward to it.  The URL
    exposed to the browser (in broadcasts and ``GET /api/panels``) is rewritten
    to ``<prefix>/panel/{id}`` (``<prefix>`` is the empty string outside
    multi-user deployments).

    If a custom panel with the same ``id`` already exists it is replaced
    atomically (remove-then-append) so the proxy always returns the first match.

    An agent-origin registration is mirrored into the activity history buffer as
    a ``register_panel`` row.

    Args:
        body: Panel registration fields: ``id``, ``label``, ``url`` (raw),
            ``path`` (default ``"/"``), ``health_endpoint`` (optional).
        request: Incoming FastAPI request carrying ``app.state``.

    Returns:
        ``{"status": "ok", "id": <id>, "label": <label>, "url": "<prefix>/panel/<id>"}``

    Raises:
        HTTPException: 403 when runtime panel registration is disabled.
        HTTPException: 422 when the id is one no request for the panel could be
            routed with, when the id collides with a built-in or config-defined
            panel, when the URL fails security validation, or when the host is
            not in the configured allowlist.
    """
    if not getattr(request.app.state, "allow_runtime_panels", False):
        raise HTTPException(
            status_code=403,
            detail="Runtime panel registration is disabled. Set web.allow_runtime_panels: true to enable.",
        )

    # The class a panel id is spellable in, the same one a config-declared id
    # is held to (:func:`~osprey.profiles.web_panels.panel_id_refusal`). A
    # registration arrives after that gate and reaches the same places: the
    # proxy's own ``/panel/<id>`` paths and the forwarded-prefix header of every
    # hop it makes for this panel, escaped for neither. An id outside the class
    # would register cleanly and then fail every request the panel serves. The
    # wording is this route's own, because a registration has no config block to
    # name back to the caller.
    if panel_id_refusal(body.id) is not None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Panel id {body.id!r} must match {MOUNT_SEGMENT_RE.pattern!r}: the id is a "
                "URL path segment and a header value on every proxied request for the panel."
            ),
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

    # Judged and stored in the form a browser (and the proxy) will read it: a
    # raw string the checks below would misread (a tab inside the scheme, a
    # ``\`` for a ``/``) is normalised first, so what is validated is what is
    # used (the navigation guard).
    panel_url = normalise_navigation_url(body.url)
    url_error = navigation_url_refusal(panel_url)
    if url_error is None:
        hop_refusal = agent_hop_refusal(request, body.id, panel_url)
        if hop_refusal is not None:
            raise HTTPException(status_code=422, detail=f"agent_hop_panel_url: {hop_refusal}")
        allowlist: list[str] | None = getattr(request.app.state, "runtime_panel_allowlist", None)
        url_error = await _validate_panel_url(panel_url, allowlist)
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
            "url": panel_url,  # stored normalised; proxy reads this for forwarding
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


@router.get("/api/panels/{panel_id}/cap")
async def mint_panel_cap(panel_id: str, request: Request):
    """Mint the C-CAP frame URL of one opaque panel for the calling login.

    The hub calls this whenever it (re)builds an opaque panel's iframe — on
    load, after a reload, or for a panel a broadcast announced — since a cap is
    bound to the login that asked for it and is never shared between browsers
    (so it is never part of ``GET /api/panels`` or a broadcast frame).

    Returns:
        ``{"id": <id>, "cap_url": "<prefix>/panel-static|panel/<id>/~c/<cap>/"}``,
        or ``404`` for a panel that is not opaque or a caller naming no login.
    """
    url = panel_cap.cap_url(request, panel_id)
    if url is None:
        raise HTTPException(status_code=404, detail=f"No capability path for panel {panel_id!r}")
    return {"id": panel_id, "cap_url": url}


def _resolve_bundle_file(request: Request, panel_id: str, path: str) -> Path | None:
    """The bundle file ``path`` names in discovered panel ``panel_id``, or ``None``.

    Fail-closed: an unknown id, a path that escapes the bundle directory
    (``..`` traversal) or a missing file all answer ``None`` — never a file
    outside that one bundle. An empty ``path`` names the manifest's ``entry``.
    """
    panels = getattr(request.app.state, "discovered_panel_dirs", {})
    panel = panels.get(panel_id)
    if panel is None:
        return None
    base = Path(panel.directory).resolve()
    target = (base / (path or panel.entry)).resolve()
    if not target.is_relative_to(base) or not target.is_file():
        return None
    return target


@router.get(f"/panel-static/{{panel_id}}/{PANEL_CAP_SEGMENT}/{{cap}}/{{path:path}}")
async def serve_discovered_panel_cap(panel_id: str, cap: str, path: str, request: Request):
    """Serve a discovered panel's bundle under its C-CAP capability path.

    Declared above :func:`serve_discovered_panel`, which would otherwise read
    ``~c/<cap>/…`` as a file path. The auth gate exempts this path
    (:func:`~osprey.interfaces.common_middleware.is_panel_cap_path`); the cap is
    validated here instead of the cookie, and a forged, foreign-login, expired
    or other-panel cap answers ``403``. Only this one panel's bundle is read.

    Every response carries ``Access-Control-Allow-Origin: *`` and no
    credentials — the bundle's own module scripts and fetches arrive from its
    opaque origin — and every non-inert type carries ``opaque_headers()``, so a
    document opened here directly still runs at an opaque origin.
    """
    if not panel_cap.verify_request_cap(request, panel_id, cap, "static"):
        return Response(content="Forbidden", status_code=403)
    target = _resolve_bundle_file(request, panel_id, path)
    if target is None:
        return Response(content="Not found", status_code=404, headers=dict(ASSET_CORS_HEADERS))
    response = FileResponse(target)
    response.headers.update(ASSET_CORS_HEADERS)
    response.headers.update(document_headers(response.media_type))
    return response


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

    A bundle under ``<project>/panels`` is agent-writable whatever its manifest
    says, so every non-inert type carries ``opaque_headers()`` here too, not only
    under its cap path: opened at this cookie-gated URL (a new tab, a stale
    link) the document still runs at an opaque origin. Inert types get
    ``nosniff``.

    Args:
        panel_id: The discovered panel id.
        path: Path within the bundle; empty serves the manifest entry.
        request: Incoming request carrying ``app.state.discovered_panel_dirs``.

    Every document this route answers — the entry page included, on every
    status — goes out in an opaque sandbox origin (``panel_sandbox``): a
    discovered bundle is files on disk that an agent may have written, so it
    declares no entry page and never runs with the terminal's own origin. This
    route is never allowlisted.

    A sandboxed page carries no cookie, so on a gated deployment any
    subresource it asks the terminal for answers 401. The panel contract
    (``design_system.panels.validator``) has every page reference the shared
    theme by the same canonical tags, so an HTML page is served with exactly
    those made self-contained (:func:`_self_contained_page`); everything else a
    panel needs must be inline in the page itself, as in the design-system
    reference panel.

    Returns:
        A :class:`FileResponse` for the requested asset, or a 404 ``Response``.
    """
    response = _discovered_panel_file(panel_id, path, request)
    apply_panel_sandbox(response, path, ())
    return response


def _discovered_panel_file(panel_id: str, path: str, request: Request) -> Response:
    """The bundle file :func:`serve_discovered_panel` answers with, or its 404."""
    panels = getattr(request.app.state, "discovered_panel_dirs", {})
    if panels.get(panel_id) is None:
        return Response(content=f"Panel '{panel_id}' is not available", status_code=404)

    target = _resolve_bundle_file(request, panel_id, path)
    if target is None:
        return Response(content="Not found", status_code=404)

    page = _self_contained_page(target)
    if page is not None:
        return HTMLResponse(page, headers=document_headers("text/html"))
    response = FileResponse(target)
    response.headers.update(document_headers(response.media_type))
    return response


#: The shared web-font stylesheet link the panel contract requires. Its fonts
#: are ~1.5 MB of TrueType, too heavy to inline into every page, so a sandboxed
#: page drops the link and renders in the tokens' system-font fallbacks.
_SHARED_FONTS_LINK_RE = re.compile(
    r"""<link\b[^>]*\bhref\s*=\s*["']/static/fonts/fonts\.css["'][^>]*>""", re.IGNORECASE
)


def _self_contained_page(target: Path) -> str | None:
    """A discovered panel's HTML page with its canonical shared-theme references inlined.

    The design-system scripts and stylesheets become inline blocks, first-party
    module imports become ``data:`` modules (``artifacts.inline_assets``, the
    artifact pages' inliner), and the shared-font link is dropped. ``None`` for
    a file that is not HTML, and for a page that names a design-system asset
    that does not exist — that page is served as written.
    """
    if target.suffix.lower() not in {".html", ".htm"}:
        return None
    try:
        page = target.read_text(encoding="utf-8")
        return self_contained(_SHARED_FONTS_LINK_RE.sub("", page))
    except (OSError, UnicodeDecodeError, ValueError):
        logger.warning("discovered panel page %s served as written", target, exc_info=True)
        return None


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
    # May not have active operator sessions
    with contextlib.suppress(Exception):
        await operator_registry.cleanup_all()

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
        return await get_mcp_servers_cached(mcp_json_path, project_dir, timeout=10.0)
    except Exception:
        logger.warning("mcp-servers: introspection failed", exc_info=True)
        return []
