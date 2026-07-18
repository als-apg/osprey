"""Sidecar passthrough relay for the Bluesky bridge's plan-draft endpoints.

Task 3.1 (sidecar-draft-relay) of the agent-plan-draft plan. This module only
defines ``router``; ``osprey.services.bluesky_panels.app`` mounts it alongside
``read_proxy``/``launch``/``health``, so every route here reads the shared
``httpx.AsyncClient`` and resolved bridge base URL from ``request.app.state``
at request time (set in the app's lifespan) -- exactly the ``read_proxy``
pattern.

Every ``/draft`` route (GET/PATCH/DELETE) is a thin, verbatim passthrough of
the bridge's JSON body and HTTP status code -- including 4xx bodies such as a
409 ``no_draft``/``expected_plan_name`` mismatch or a 422 per-field
validation error -- mirroring ``read_proxy``'s passthrough convention
(nothing here recomputes or reinterprets a bridge-owned field). A PATCH
request's JSON body is forwarded to the bridge unchanged.

``GET /draft/events`` relays the bridge's Server-Sent-Events stream to the
browser -- the browser never talks to the bridge directly, only this sidecar
does. This mirrors the SSE hop in
``osprey.interfaces.web_terminal.routes.proxy.proxy_panel``:
``client.build_request(...)`` + ``client.send(req, stream=True)`` with a
per-request ``httpx.Timeout(None, connect=5.0)`` override, since the shared
client's 15s default read timeout would kill the connection during the
bridge's ~15s heartbeat-interval idle periods.

A connection-level failure to reach the bridge (refused connection, DNS
failure, timeout, ...) is translated into a 502 with the same fixed detail
body ``read_proxy`` uses for its own bridge-unreachable case -- it never
surfaces here as an uncaught 500. HTTP-level error responses from the bridge
itself (404, 409, 422, ...) are not exceptions to httpx at all; they flow
straight through and are relayed as-is.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

router = APIRouter()

_UNREACHABLE_BODY = {"detail": "bluesky bridge unreachable"}

# Hop-by-hop headers that must not be relayed onto the browser-facing SSE
# response (matches osprey.interfaces.web_terminal.routes.proxy._HOP_BY_HOP).
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-encoding",
        "content-length",
    }
)


async def _relay(request: Request, method: str) -> JSONResponse:
    """Relay ``method /draft`` onto the bridge and return its body/status verbatim.

    For ``PATCH``, the incoming request's raw JSON body is forwarded to the
    bridge unchanged (byte-for-byte, not re-serialized). Query params (none
    in the current bridge contract) are forwarded verbatim for the same
    reason ``read_proxy._forward_get`` does: forward-compatibility without
    special-casing.
    """
    client: httpx.AsyncClient = request.app.state.client
    bridge_url: str = request.app.state.bridge_url

    content: bytes | None = None
    headers: dict[str, str] | None = None
    if method == "PATCH":
        content = await request.body()
        headers = {"content-type": request.headers.get("content-type", "application/json")}

    try:
        response = await client.request(
            method,
            f"{bridge_url}/draft",
            params=request.query_params,
            content=content,
            headers=headers,
        )
    except httpx.RequestError:
        return JSONResponse(content=_UNREACHABLE_BODY, status_code=502)

    try:
        body = response.json()
    except ValueError:
        body = None

    return JSONResponse(content=body, status_code=response.status_code)


@router.get("/draft")
async def get_draft(request: Request) -> JSONResponse:
    """Relay the current plan draft (or ``null``) and its revision, verbatim."""
    return await _relay(request, "GET")


@router.patch("/draft")
async def patch_draft(request: Request) -> JSONResponse:
    """Relay a draft patch (merge/remove/plan-name change) to the bridge.

    The request body is forwarded to the bridge unchanged; the bridge's
    response body (``{revision, changed[], plan_name}`` on success, or a
    409/422 error body) and status code are relayed back verbatim.
    """
    return await _relay(request, "PATCH")


@router.delete("/draft")
async def delete_draft(request: Request) -> JSONResponse:
    """Relay a draft clear request to the bridge; idempotent, per the bridge contract."""
    return await _relay(request, "DELETE")


@router.get("/draft/events")
async def draft_events(request: Request) -> Response:
    """Relay the bridge's ``/draft/events`` SSE stream to the browser.

    The browser never talks to the bridge directly -- this sidecar is the
    sole hop. Uses a per-request ``httpx.Timeout(None, connect=5.0)`` to
    disable the shared client's 15s default read timeout, since the stream
    idles between the bridge's ~15s heartbeat comment frames.
    """
    client: httpx.AsyncClient = request.app.state.client
    bridge_url: str = request.app.state.bridge_url

    try:
        upstream = await client.send(
            client.build_request(
                "GET",
                f"{bridge_url}/draft/events",
                timeout=httpx.Timeout(None, connect=5.0),
            ),
            stream=True,
        )
    except httpx.RequestError:
        return JSONResponse(content=_UNREACHABLE_BODY, status_code=502)

    async def _stream() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()

    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_BY_HOP}

    return StreamingResponse(
        _stream(),
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type="text/event-stream",
    )
