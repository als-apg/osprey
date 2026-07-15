"""Read-only GET proxy onto the Bluesky bridge for the scan panels sidecar.

Task 1.2 (sidecar-read-proxy) of the Phase-6 "Operator Interfaces" plan. This
module only defines ``router``; a separate integration task wires it onto the
app built in ``osprey.services.scan_panels.app`` (task 1.1), which already
publishes the shared ``httpx.AsyncClient`` and resolved bridge base URL onto
``app.state.client`` / ``app.state.bridge_url``.

Every route here is a thin, verbatim passthrough: the bridge's JSON body and
HTTP status code (including 404/409 error shapes) are relayed unchanged --
nothing here recomputes ``row_count``/``truncated``/``partial`` or any other
bridge-owned field. This mirrors the read side of the bridge contract at
``osprey.services.bluesky_bridge.app``:

- ``GET /plans``
- ``GET /plans/{name}/source``
- ``GET /runs`` (``limit`` query param)
- ``GET /runs/{run_id}``
- ``GET /runs/{run_id}/data`` (``max_rows``/``offset``/``tail`` query params)

No write verbs are exposed here (no ``POST /runs``, no ``/promote``, no
``/stop``) -- this router is GET-only by construction.

A connection-level failure to reach the bridge (refused connection, DNS
failure, timeout, ...) is translated into a 502 with a fixed detail body,
mirroring the error-translation spirit of
``osprey.mcp_server.scan.server_context._http_get_json``'s
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

router = APIRouter()

_UNREACHABLE_BODY = {"detail": "bluesky bridge unreachable"}


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
        return JSONResponse(content=_UNREACHABLE_BODY, status_code=502)

    try:
        body = response.json()
    except ValueError:
        body = None

    return JSONResponse(content=body, status_code=response.status_code)


@router.get("/plans")
async def list_plans(request: Request) -> JSONResponse:
    return await _forward_get(request, "/plans")


@router.get("/plans/{name}/source")
async def get_plan_source(request: Request, name: str) -> JSONResponse:
    return await _forward_get(request, f"/plans/{quote(name, safe='')}/source")


@router.get("/runs")
async def list_runs(request: Request) -> JSONResponse:
    return await _forward_get(request, "/runs")


@router.get("/runs/{run_id}")
async def get_run_status(request: Request, run_id: str) -> JSONResponse:
    return await _forward_get(request, f"/runs/{quote(run_id, safe='')}")


@router.get("/runs/{run_id}/data")
async def read_run_data(request: Request, run_id: str) -> JSONResponse:
    return await _forward_get(request, f"/runs/{quote(run_id, safe='')}/data")
