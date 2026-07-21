"""Reverse proxy for companion panel servers running inside the container.

Companion servers (artifact gallery, ARIEL, channel-finder, lattice, OKF)
bind to 127.0.0.1 inside the Docker container.  The browser cannot reach them
directly, so this proxy forwards ``/panel/{panel_id}/{path}`` to the internal
server and rewrites root-absolute paths in HTML/JS/CSS responses.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re

import httpx
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse

from osprey.interfaces.web_terminal.url_prefix import compute_url_prefix
from osprey.utils.http_proxy import HOP_BY_HOP

logger = logging.getLogger(__name__)

router = APIRouter()

# Paths that companion servers commonly use as root-absolute references.
# Only these prefixes are rewritten inside string delimiters to avoid
# false positives on arbitrary ``/`` characters.
_REWRITE_PREFIXES = (
    "/static/",
    "/design-system/",
    "/api/",
    "/files/",
    "/health",
    "/checks",
    "/ws/",
    "/assets/",
    "/dashboard/",
    # Tiled's built-in web UI (served under /ui/) references root-absolute
    # /ui/assets/... bundle paths.
    "/ui/",
    # Event dispatcher endpoints. Without these, the dashboard served through
    # /panel/events/* would call `/webhook/...` at the browser origin (the web
    # terminal) instead of going through this proxy — the dispatcher never
    # sees the request and the user gets a 404.
    "/webhook/",
    "/retry/",
    "/trigger/",
)

# Content types eligible for path rewriting.
_REWRITABLE_TYPES = {
    "text/html",
    "text/javascript",
    "application/javascript",
    "text/css",
}

# The proxy-wide caching default, applied to any proxied response whose
# upstream set no Cache-Control of its own. Panels ship unversioned asset
# filenames (panel.js, not panel-<hash>.js), and a header-less response lets
# the browser cache heuristically — so a panel redeploy silently doesn't
# reach an operator's already-open browser. Filled in with setdefault, never
# overridden: an upstream that made a deliberate caching decision (e.g. the
# artifact gallery's immutable versioned vendor bundles) keeps it.
_DEFAULT_NO_CACHE = "no-cache, no-store, must-revalidate"

# Panel ID → app.state attribute name
_PANEL_STATE_MAP = {
    "artifacts": "artifact_server_url",
    "ariel": "ariel_server_url",
    "channel-finder": "channel_finder_server_url",
    "lattice": "lattice_dashboard_server_url",
    "okf": "okf_server_url",
    "system-health": "system_health_server_url",
}


def _resolve_panel_url(request: Request, panel_id: str) -> str | None:
    """Map a panel ID to its internal server URL, or ``None`` if unavailable."""
    attr = _PANEL_STATE_MAP.get(panel_id)
    if attr:
        return getattr(request.app.state, attr, None)

    # Custom panels: look up by ID in the custom panels list.
    for cp in getattr(request.app.state, "custom_panels", []):
        if cp.get("id") == panel_id:
            url = cp.get("url", "")
            return url if url else None

    return None


def _panel_is_config_defined(request: Request, panel_id: str) -> bool:
    """True only if ``panel_id`` resolves to a config-declared custom panel.

    Runtime registrations (POST /api/panels/register) never carry the
    ``configDefined`` marker, so a registration that squats a config panel's id
    cannot inherit that panel's server-side credential injection below.
    """
    for cp in getattr(request.app.state, "custom_panels", []):
        if cp.get("id") == panel_id:
            return bool(cp.get("configDefined"))
    return False


def _panel_json_rewrite_paths(request: Request, panel_id: str) -> tuple[str, ...]:
    """The panel's configured ``rewrite_json_paths`` suffixes (usually empty)."""
    for cp in getattr(request.app.state, "custom_panels", []):
        if cp.get("id") == panel_id:
            paths = cp.get("rewriteJsonPaths") or ()
            if isinstance(paths, (list, tuple)):
                return tuple(str(p).rstrip("/") for p in paths if p)
            return (str(paths).rstrip("/"),)
    return ()


def _rewrite_content(body: str, panel_id: str, outer_prefix: str = "") -> str:
    """Rewrite root-absolute paths inside string delimiters for proxied content.

    Only touches known prefixes inside ``"``, ``'``, or backtick delimiters.
    CDN URLs (``https://…``), protocol-relative (``//…``), and data URIs
    are unaffected because the pattern requires a delimiter immediately
    before the ``/``.

    Args:
        body: The response body to rewrite.
        panel_id: The panel ID this content was proxied from.
        outer_prefix: The per-user mount prefix (``/u/<user>`` or ``""``, see
            ``compute_url_prefix()``). Prepended so a panel's internal
            assets/APIs resolve under the outer prefix too, not just
            ``/panel/<id>``. Empty prefix ⇒ unchanged (pre-refactor) output.
    """
    prefix = f"{outer_prefix}/panel/{panel_id}"

    for path in _REWRITE_PREFIXES:
        # Match path inside string delimiters: "/static/..." → "/panel/id/static/..."
        # The lookbehind ensures we only match after a quote character.
        body = re.sub(
            r"""(?<=["'`])""" + re.escape(path),
            prefix + path,
            body,
        )

    # Rewrite bare '/api' (without trailing slash) inside delimiters.
    body = re.sub(r"""(?<=["'`])/api(?=["'`])""", f"{prefix}/api", body)

    # Rewrite href="/" to href="/panel/{id}/"
    body = body.replace('href="/"', f'href="{prefix}/"')
    body = body.replace("href='/'", f"href='{prefix}/'")

    return body


@router.api_route(
    "/panel/{panel_id}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
)
async def proxy_panel(panel_id: str, path: str, request: Request):
    """Forward a request to the companion panel server."""
    backend_url = _resolve_panel_url(request, panel_id)
    if not backend_url:
        return Response(
            content=f"Panel '{panel_id}' is not available",
            status_code=404,
        )

    outer_prefix = compute_url_prefix()

    # Build the target URL.
    target = f"{backend_url.rstrip('/')}/{path}"
    if request.url.query:
        target = f"{target}?{request.url.query}"

    client: httpx.AsyncClient = request.app.state.proxy_client

    # Forward headers, dropping host (httpx sets it), hop-by-hop, and
    # accept-encoding. httpx must negotiate content-codings it can itself
    # decode (it transparently decompresses gzip/br); forwarding the
    # browser's Accept-Encoding list (which includes zstd) lets upstreams
    # pick zstd, which httpx cannot decode — it passes the compressed bytes
    # through unchanged while the proxy strips the content-encoding header
    # below, so the browser renders raw compressed garbage.
    fwd_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP and k.lower() not in ("host", "accept-encoding")
    }
    fwd_headers["x-forwarded-prefix"] = f"{outer_prefix}/panel/{panel_id}"

    # The event-dispatcher dashboard endpoints are bearer-gated. Inject the
    # dispatcher token server-side for the EVENTS panel only, so the browser
    # never holds it (and other panels are unaffected). The web-terminal process
    # picks up EVENT_DISPATCHER_TOKEN from the project .env via load_dotenv.
    # Gated on config origin, not the id string: the token must follow the
    # config-defined EVENTS panel, never a runtime-registered squat of the id.
    if panel_id == "events" and _panel_is_config_defined(request, "events"):
        token = os.environ.get("EVENT_DISPATCHER_TOKEN", "")
        if token:
            fwd_headers["authorization"] = f"Bearer {token}"

    try:
        body = await request.body()

        # SSE: stream the response.
        if "text/event-stream" in request.headers.get("accept", ""):
            upstream = await client.send(
                client.build_request(
                    method=request.method,
                    url=target,
                    headers=fwd_headers,
                    content=body if body else None,
                    # SSE streams idle between events — disable the read
                    # timeout so quiet periods don't kill the connection.
                    timeout=httpx.Timeout(None, connect=5.0),
                ),
                stream=True,
            )

            async def _stream():
                try:
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
                finally:
                    await upstream.aclose()

            resp_headers = {
                k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP
            }
            resp_headers.setdefault("cache-control", _DEFAULT_NO_CACHE)
            return StreamingResponse(
                _stream(),
                status_code=upstream.status_code,
                headers=resp_headers,
                media_type="text/event-stream",
            )

        # Standard request.
        resp = await client.request(
            method=request.method,
            url=target,
            headers=fwd_headers,
            content=body if body else None,
        )
    except httpx.RequestError as exc:
        logger.warning("Proxy error for panel %s: %s", panel_id, exc)
        return Response(
            content=f"Upstream connection failed: {exc}",
            status_code=502,
        )

    # Filter response headers; fill in the no-cache default when the upstream
    # made no caching decision of its own (see _DEFAULT_NO_CACHE).
    resp_headers = {k: v for k, v in resp.headers.items() if k.lower() not in HOP_BY_HOP}
    resp_headers.setdefault("cache-control", _DEFAULT_NO_CACHE)

    content_type = resp.headers.get("content-type", "")
    base_type = content_type.split(";")[0].strip().lower()

    # Rewrite root-absolute paths in HTML/JS/CSS — skip vendor assets
    # (large, immutable, no OSPREY paths to rewrite).
    if base_type in _REWRITABLE_TYPES and "/vendor/" not in path:
        text = resp.text
        text = _rewrite_content(text, panel_id, outer_prefix)
        return Response(
            content=text,
            status_code=resp.status_code,
            headers=resp_headers,
            media_type=content_type,
        )

    # JSON is normally passed through untouched (rewriting arbitrary API
    # payloads would corrupt data). But some backends bootstrap their web UI
    # from a JSON config endpoint that carries root-absolute paths (e.g. an
    # ``api_url`` the SPA uses as its API base) — served through this proxy,
    # the UI would then call the *browser* origin and silently break. Panels
    # opt specific endpoints in via ``web.panels.<id>.rewrite_json_paths``
    # (a list of path suffixes); only those responses get the same
    # known-prefix literal rewrite HTML/JS/CSS receive.
    if base_type == "application/json":
        stripped = path.rstrip("/")
        if any(stripped.endswith(sfx) for sfx in _panel_json_rewrite_paths(request, panel_id)):
            return Response(
                content=_rewrite_content(resp.text, panel_id, outer_prefix),
                status_code=resp.status_code,
                headers=resp_headers,
                media_type=content_type,
            )

    # JSON, images, binary — pass through unchanged.
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=resp_headers,
        media_type=content_type if content_type else None,
    )


@router.api_route(
    "/panel/{panel_id}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
)
async def proxy_panel_root(panel_id: str, request: Request):
    """Proxy the root path of a companion panel (no trailing path segment)."""
    return await proxy_panel(panel_id, "", request)


@router.websocket("/panel/{panel_id}/{path:path}")
async def proxy_panel_ws(panel_id: str, path: str, websocket: WebSocket):
    """Forward a WebSocket connection to the companion panel server."""
    backend_url = _resolve_panel_url(websocket, panel_id)
    if not backend_url:
        await websocket.close(code=4004, reason=f"Panel '{panel_id}' not available")
        return

    # Convert http(s) backend URL to ws(s)
    ws_url = backend_url.rstrip("/").replace("http://", "ws://").replace("https://", "wss://")
    target = f"{ws_url}/{path}"

    try:
        import websockets
    except ImportError:
        logger.error("websockets package not installed — cannot proxy WebSocket")
        await websocket.close(code=4500, reason="WebSocket proxy unavailable")
        return

    await websocket.accept()

    try:
        async with websockets.connect(target) as upstream:

            async def client_to_upstream():
                # receive() (not receive_text()) — binary-protocol panels
                # (e.g. the noVNC/RFB phoebus stream) send bytes frames,
                # which receive_text() rejects, killing the relay.
                try:
                    while True:
                        msg = await websocket.receive()
                        if msg.get("type") == "websocket.disconnect":
                            break
                        data = msg.get("bytes")
                        if data is None:
                            data = msg.get("text")
                        if data is not None:
                            await upstream.send(data)
                except WebSocketDisconnect:
                    pass

            async def upstream_to_client():
                try:
                    async for msg in upstream:
                        if isinstance(msg, str):
                            await websocket.send_text(msg)
                        else:
                            await websocket.send_bytes(msg)
                except websockets.ConnectionClosed:
                    pass

            # Run both directions concurrently; when either side closes,
            # cancel the other.
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(client_to_upstream()),
                    asyncio.create_task(upstream_to_client()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()

    except Exception as exc:
        logger.warning("WebSocket proxy error for panel %s: %s", panel_id, exc)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
