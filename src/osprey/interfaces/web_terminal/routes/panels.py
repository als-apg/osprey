"""Panel configuration, server URL, health, and MCP introspection routes."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from osprey.profiles.web_panels import BUILTIN_PANEL_LABELS, BUILTIN_PANELS

logger = logging.getLogger(__name__)

router = APIRouter()


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


@router.get("/api/artifact-server")
async def artifact_server_config(request: Request):
    """Return the artifact gallery server URL for iframe embedding."""
    url = getattr(request.app.state, "artifact_server_url", None)
    proxy_url = "/panel/artifacts" if url else None
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
    proxy_url = "/panel/ariel" if url else None
    return {"url": proxy_url, "available": proxy_url is not None}


@router.get("/api/tuning-server")
async def tuning_server_config(request: Request):
    """Return the tuning panel server URL for iframe embedding."""
    url = getattr(request.app.state, "tuning_server_url", None)
    proxy_url = "/panel/tuning" if url else None
    return {"url": proxy_url, "available": proxy_url is not None}


@router.get("/api/channel-finder-server")
async def channel_finder_server_config(request: Request):
    """Return the Channel Finder server URL for iframe embedding."""
    url = getattr(request.app.state, "channel_finder_server_url", None)
    proxy_url = "/panel/channel-finder" if url else None
    return {"url": proxy_url, "available": proxy_url is not None}


@router.get("/api/lattice-server")
async def lattice_server_config(request: Request):
    """Return the lattice dashboard server URL for iframe embedding."""
    url = getattr(request.app.state, "lattice_dashboard_server_url", None)
    proxy_url = "/panel/lattice" if url else None
    return {"url": proxy_url, "available": proxy_url is not None}


@router.get("/api/panels")
async def get_panels(request: Request):
    """Return the full panel state in one payload.

    All custom panel URLs are rewritten to ``/panel/{id}`` so the browser
    routes through the reverse proxy.  The originals in ``app.state`` are
    left untouched — the proxy reads those for forwarding.

    Response shape::

        {
            "enabled":  [...],          # list of enabled builtin panel id strings
            "custom":   [...],          # custom panel dicts (url rewritten to /panel/<id>)
            "default":  str|None,       # profile-pinned cold-load focus target
            "visible":  [...],          # enabled + custom ids minus hidden: true panels
            "active":   str|None,       # currently focused panel id
            "labels":   {id: label},    # display labels for enabled built-in panels
        }

    ``default`` is not validated here — the frontend falls back to
    ``DEFAULT_PANEL_FALLBACK`` when it is unknown so a typo doesn't leave the
    user staring at a blank tabset.

    ``active`` mirrors ``GET /api/panel-focus`` so the agent can read back the
    full panel state in a single round-trip.

    ``labels`` covers only the enabled built-in panels; custom panels carry
    their own ``label`` field in the ``custom`` list.
    """
    enabled = list(getattr(request.app.state, "enabled_panels", set()))
    custom_raw = getattr(request.app.state, "custom_panels", [])
    custom = [{**cp, "url": f"/panel/{cp['id']}"} for cp in custom_raw]
    default = getattr(request.app.state, "default_panel", None)
    visible = getattr(request.app.state, "visible_panels", enabled)
    active = getattr(request.app.state, "active_panel", None)
    labels = {pid: BUILTIN_PANEL_LABELS[pid] for pid in enabled if pid in BUILTIN_PANEL_LABELS}
    return {
        "enabled": enabled,
        "custom": custom,
        "default": default,
        "visible": visible,
        "active": active,
        "labels": labels,
    }


def _known_panel_ids(request: Request) -> set[str]:
    """Return the set of valid panel ids: enabled built-ins plus custom panels.

    Single source of truth for the membership check shared by the focus and
    visibility endpoints — both reject panel ids that are not in this set.
    """
    known: set[str] = set(getattr(request.app.state, "enabled_panels", set()))
    known |= {p["id"] for p in getattr(request.app.state, "custom_panels", [])}
    return known


class PanelFocusRequest(BaseModel):
    panel: str
    url: str | None = None


@router.get("/api/panel-focus")
async def get_panel_focus(request: Request):
    """Return the currently active panel."""
    active = getattr(request.app.state, "active_panel", None)
    return {"active_panel": active}


@router.post("/api/panel-focus")
async def set_panel_focus(body: PanelFocusRequest, request: Request):
    """Set the active panel and broadcast a focus event via SSE."""
    if body.panel not in _known_panel_ids(request):
        raise HTTPException(status_code=422, detail=f"Unknown panel: {body.panel}")
    request.app.state.active_panel = body.panel
    event: dict = {"type": "panel_focus", "panel": body.panel}
    if body.url:
        event["url"] = body.url
    request.app.state.broadcaster.broadcast(event)
    return {"status": "ok", "active_panel": body.panel}


class PanelVisibilityRequest(BaseModel):
    panel: str
    visible: bool


@router.post("/api/panel-visibility")
async def set_panel_visibility(body: PanelVisibilityRequest, request: Request):
    """Show or hide a panel and broadcast the change via SSE.

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
    request.app.state.broadcaster.broadcast(
        {"type": "panel_visibility", "panel": body.panel, "visible": body.visible}
    )
    return {"status": "ok", "panel": body.panel, "visible": body.visible}


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


@router.post("/api/panels/register")
async def register_panel(body: PanelRegisterRequest, request: Request):
    """Dynamically register a custom panel at runtime.

    Requires ``web.allow_runtime_panels: true`` in config.  The supplied URL
    is validated to reject loopback, link-local, and cloud-metadata targets,
    and is stored raw server-side so the proxy can forward to it.  The URL
    exposed to the browser (in broadcasts and ``GET /api/panels``) is rewritten
    to ``/panel/{id}``.

    If a custom panel with the same ``id`` already exists it is replaced
    atomically (remove-then-append) so the proxy always returns the first match.

    Args:
        body: Panel registration fields: ``id``, ``label``, ``url`` (raw),
            ``path`` (default ``"/"``), ``health_endpoint`` (optional).
        request: Incoming FastAPI request carrying ``app.state``.

    Returns:
        ``{"status": "ok", "id": <id>, "label": <label>, "url": "/panel/<id>"}``

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

    if body.id in BUILTIN_PANELS:
        raise HTTPException(
            status_code=422,
            detail=f"Panel id {body.id!r} collides with a built-in panel; choose a different id.",
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

    request.app.state.broadcaster.broadcast(
        {
            "type": "panel_register",
            "id": body.id,
            "label": body.label,
            "url": f"/panel/{body.id}",  # rewritten for the browser
            "healthEndpoint": body.health_endpoint,
            "path": body.path,
        }
    )
    return {"status": "ok", "id": body.id, "label": body.label, "url": f"/panel/{body.id}"}


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
