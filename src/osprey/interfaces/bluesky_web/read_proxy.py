"""Read-only GET proxy onto the Bluesky bridge for the bluesky-web sidecar.

This module defines ``router`` only; ``osprey.interfaces.bluesky_web.app``
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
- ``GET /runs/{run_id}/figure`` (no query params)

Every path mirrors the bridge's own except the first: the sidecar serves its
OWN ``GET /health`` (its container healthcheck, in
``osprey.interfaces.bluesky_web.app``), so the bridge's health document --
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

Every route takes an optional ``lane`` query parameter naming which PLAN LANE
the read is addressed to -- a two-lane deployment runs two bridges answering
these same paths about two different machines. It is consumed here and never
forwarded, defaults to lane 1 (the only lane a single-lane deployment has, so
nothing about those requests changes), and a lane this deployment does not
render is answered 404 rather than relayed from the other lane.

Responses only ever carry the bridge's own body content and prefix-relative
paths (path params/query params passed through verbatim) -- no absolute URLs
are ever emitted, since panels consume these prefix-relative.
"""

from __future__ import annotations

from urllib.parse import quote

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from osprey.bluesky_bridge_connection import LANE_ONE
from osprey.interfaces.bluesky_web._shared import UNREACHABLE_BODY, safe_json

router = APIRouter()

#: Query parameter naming which PLAN LANE a read is addressed to. A lane is a
#: whole bridge stack bound at render time to one control-system target, and a
#: two-lane deployment has two of them answering the same paths about two
#: different machines. Absent means lane 1 -- which is every request a
#: single-lane deployment can produce, so nothing about those changes.
#:
#: Consumed here and stripped: it addresses the sidecar, not the bridge, and
#: the bridge has no parameter by that name.
LANE_QUERY_PARAM = "lane"

#: The 404 body for a lane this deployment does not render. A distinct answer
#: from the 502 above on purpose: "I cannot reach that bridge" and "there is no
#: such bridge here" are different claims, and the one thing neither may
#: degrade into is relaying the read from the OTHER lane -- a run listing
#: labelled with the wrong machine is worse than no listing.
UNKNOWN_LANE_BODY: dict[str, str] = {"detail": "unknown bluesky lane"}


def resolve_lane_bridge_url(request: Request, lane: str | None) -> str | None:
    """The bridge base URL serving ``lane``, or ``None`` if there is no such lane.

    Reads ``app.state.bridge_urls`` -- the per-lane mapping the sidecar's app
    publishes on a two-lane deployment -- and falls back to the single
    ``app.state.bridge_url`` for lane 1, which is the only URL a single-lane
    deployment publishes and the only lane it can be asked about.

    Deliberately returns ``None`` rather than lane 1's URL for an unrecognized
    lane: relaying one lane's reads under another lane's name is the confusion
    the lane axis exists to remove. An EMPTY lane is not unrecognized, though —
    it names no lane at all, and is treated as lane 1 exactly like ``None``.

    :param request: The incoming request, for its ``app.state``
    :param lane: A lane's ``services.<lane>`` key, or ``None``/``""`` for lane 1
    :return: The base URL with any trailing slash stripped, or ``None``
    """
    state = request.app.state
    if not lane or lane == LANE_ONE:
        return str(state.bridge_url).rstrip("/")
    urls = getattr(state, "bridge_urls", None)
    url = urls.get(lane) if isinstance(urls, dict) else None
    return str(url).rstrip("/") if url else None


async def _forward_get(request: Request, path: str) -> JSONResponse:
    """GET ``path`` on the Bluesky bridge and relay its JSON body/status verbatim.

    ``path`` must already be a properly-escaped bridge-relative path (path
    segments taken from the incoming request are quoted by the caller before
    being interpolated in). The incoming request's query params are forwarded
    unchanged, except :data:`LANE_QUERY_PARAM`, which selects WHICH lane's
    bridge is asked and is consumed here.
    """
    client: httpx.AsyncClient = request.app.state.client

    # `or None`: an empty `?lane=` is a caller that named no lane, not a caller
    # that named a lane called "". Answering that 404 would refuse a request
    # whose bare form (`?lane` omitted entirely) is served — a difference the
    # caller cannot see and nothing here means to draw.
    lane = request.query_params.get(LANE_QUERY_PARAM) or None
    bridge_url = resolve_lane_bridge_url(request, lane)
    if bridge_url is None:
        return JSONResponse(content=UNKNOWN_LANE_BODY, status_code=404)

    # multi_items(), not a dict: repeated query keys are preserved exactly as
    # they arrived, which is what "forwarded unchanged" has always meant here.
    params = [(k, v) for k, v in request.query_params.multi_items() if k != LANE_QUERY_PARAM]

    try:
        response = await client.get(f"{bridge_url}{path}", params=params)
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


@router.get("/runs/{run_id}/figure")
async def get_run_figure(request: Request, run_id: str) -> JSONResponse:
    """Relay the bridge's rendered figure for a run, body untouched.

    The bridge answers 200 for any run it knows about -- a figure it could not
    draw from the plan's own ``render`` still comes back as a figure carrying a
    ``reason``, so a ``reason`` is a default view here, not an error. Only a run
    neither the live buffer nor Tiled knows about 404s, with the same body
    ``/data`` uses for an unknown run.
    """
    return await _forward_get(request, f"/runs/{quote(run_id, safe='')}/figure")
