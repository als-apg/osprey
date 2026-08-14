"""Read-only GET proxy onto the Bluesky bridge for the bluesky panels sidecar.

This module defines ``router`` only; ``osprey.interfaces.bluesky_panels.app``
mounts it and publishes the shared ``httpx.AsyncClient`` and resolved bridge
base URL onto ``app.state.client`` / ``app.state.bridge_url``, which every
route here reads at request time.

Every route here is a thin, verbatim passthrough: the bridge's JSON body and
HTTP status code (including 404/409 error shapes) are relayed unchanged --
nothing here recomputes ``row_count``/``truncated``/``partial`` or any other
bridge-owned field. This mirrors the read side of the bridge contract at
``osprey.services.bluesky_bridge.app``:

- ``GET /bridge/health`` -> the bridge's ``GET /health``
- ``GET /plans``
- ``GET /plans/{name}/source``
- ``GET /runs`` (``limit`` query param)
- ``GET /runs/{run_id}``
- ``GET /runs/{run_id}/data`` (``max_rows``/``offset``/``tail`` query params)

Every path mirrors the bridge's own except the first: the sidecar serves its
OWN ``GET /health`` (its container healthcheck, in
``osprey.interfaces.bluesky_panels.app``), so the bridge's health document --
which carries the ``capability`` record panels read to decide whether to offer
execution at all -- is relayed one level down at ``/bridge/health`` rather than
shadowing it. The body is still passed through untouched.

No write verbs are exposed here (no ``POST /runs``, no ``/launch``, no
``/stop``) -- this router is GET-only by construction.

A connection-level failure to reach the bridge (refused connection, DNS
failure, timeout, ...) is translated into a 502 with a fixed detail body,
mirroring the error-translation spirit of
``osprey.mcp_server.bluesky.server_context._http_get_json``'s
``bluesky_bridge_unreachable`` handling -- it never surfaces here as an
uncaught 500. HTTP-level error responses from the bridge itself (404, 409,
...) are not exceptions to httpx at all; they flow straight through and are
relayed as-is.

Responses only ever carry the bridge's own body content and prefix-relative
paths (path params/query params passed through verbatim) -- no absolute URLs
are ever emitted, since panels consume these prefix-relative.
"""

from __future__ import annotations

from urllib.parse import quote

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from osprey.interfaces.bluesky_panels._shared import UNREACHABLE_BODY, safe_json

router = APIRouter()


async def _forward_get(request: Request, path: str) -> JSONResponse:
    """GET ``path`` on the Bluesky bridge and relay its JSON body/status verbatim.

    ``path`` must already be a properly-escaped bridge-relative path (path
    segments taken from the incoming request are quoted by the caller before
    being interpolated in). The incoming request's query params are forwarded
    unchanged.
    """
    client: httpx.AsyncClient = request.app.state.client
    bridge_url: str = request.app.state.bridge_url

    try:
        response = await client.get(f"{bridge_url}{path}", params=request.query_params)
    except httpx.RequestError:
        return JSONResponse(content=UNREACHABLE_BODY, status_code=502)

    body = safe_json(response)

    return JSONResponse(content=body, status_code=response.status_code)


@router.get("/bridge/health")
async def bridge_health(request: Request) -> JSONResponse:
    """Relay the bridge's health document, ``capability`` record and all.

    Named ``/bridge/health`` because the sidecar's own ``/health`` is its
    container healthcheck and answers for this process, not the bridge's. A
    502 here (bridge unreachable) is itself the honest answer to "can plans
    execute" -- the caller cannot reach the bridge, so it must not offer
    execution.
    """
    return await _forward_get(request, "/health")


@router.get("/plans")
async def list_plans(request: Request) -> JSONResponse:
    return await _forward_get(request, "/plans")


@router.get("/plans/{name}/source")
async def get_plan_source(request: Request, name: str) -> JSONResponse:
    return await _forward_get(request, f"/plans/{quote(name, safe='')}/source")


@router.get("/devices")
async def list_devices(request: Request) -> JSONResponse:
    return await _forward_get(request, "/devices")


@router.get("/runs")
async def list_runs(request: Request) -> JSONResponse:
    return await _forward_get(request, "/runs")


@router.get("/runs/{run_id}")
async def get_run(request: Request, run_id: str) -> JSONResponse:
    return await _forward_get(request, f"/runs/{quote(run_id, safe='')}")


@router.get("/runs/{run_id}/data")
async def get_run_data(request: Request, run_id: str) -> JSONResponse:
    return await _forward_get(request, f"/runs/{quote(run_id, safe='')}/data")
