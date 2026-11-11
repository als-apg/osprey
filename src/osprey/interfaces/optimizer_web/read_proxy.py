"""Read-only GET proxy onto the optimizer service for the optimizer-web sidecar.

Every route is a verbatim passthrough of an explicit allowlist: the service's
JSON body and status (refusals included) and the incoming query string. The
one response header relayed is ``X-Config-Revision``: a start must quote back
the revision the panel was shown, so it has to reach the browser rather than
be refetched. The service's health sits at ``/service/health`` because the
sidecar's own ``/health`` is its container healthcheck. A connection failure
answers 502, a timeout 504 (the service may still be working); neither
surfaces as a 500. Writes live in ``osprey.interfaces.optimizer_web.relay``.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

#: The 502 body every route returns when the service cannot be reached at all.
UNREACHABLE_BODY: dict[str, str] = {"detail": "optimizer service unreachable"}

#: The 504 body for a request the service took but did not answer in time.
TIMED_OUT_BODY: dict[str, str] = {
    "detail": (
        "the optimizer service did not answer in time; it may still be working "
        "— check the job before retrying"
    )
}

#: The service's response header naming a draft's revision (the service's
#: ``routers.jobs.CONFIG_REVISION_HEADER``, restated: the framework never
#: imports the optimizer package).
CONFIG_REVISION_HEADER = "X-Config-Revision"

#: Upstream response headers relayed to the browser. Everything else (framing,
#: server, date) is the sidecar's own to emit.
_RELAYED_RESPONSE_HEADERS: tuple[str, ...] = (CONFIG_REVISION_HEADER,)


def service_url(request: Request) -> str:
    """The optimizer base URL the app's lifespan published, trailing slash stripped."""
    return str(request.app.state.optimizer_url).rstrip("/")


def safe_json(response: httpx.Response) -> Any:
    """``response`` parsed as JSON, or ``None`` for a body that is not JSON."""
    try:
        return response.json()
    except ValueError:
        return None


def relay_response(response: httpx.Response) -> JSONResponse:
    """The service's answer as the browser receives it: body, status, revision."""
    headers = {
        name: response.headers[name]
        for name in _RELAYED_RESPONSE_HEADERS
        if name in response.headers
    }
    return JSONResponse(
        content=safe_json(response), status_code=response.status_code, headers=headers
    )


def failure_response(exc: httpx.RequestError) -> JSONResponse:
    """The answer for a request that got no response: 504 on a timeout, else 502.

    A connect timeout never reached the service, so it reads as unreachable.
    """
    if isinstance(exc, httpx.TimeoutException) and not isinstance(exc, httpx.ConnectTimeout):
        return JSONResponse(content=TIMED_OUT_BODY, status_code=504)
    return JSONResponse(content=UNREACHABLE_BODY, status_code=502)


def forwarded_params(request: Request) -> httpx.QueryParams:
    """The request's query string as the service receives it, repeats and order kept."""
    return httpx.QueryParams(request.url.query)


def segment(value: str) -> str:
    """One path segment taken from the request, escaped for the upstream path."""
    return quote(value, safe="")


async def _forward_get(request: Request, path: str) -> JSONResponse:
    """GET ``path`` on the service and relay its answer verbatim.

    ``path`` must already be escaped (path params go through :func:`segment`).
    No request header is forwarded: a read carries no credential.
    """
    client: httpx.AsyncClient = request.app.state.client
    try:
        response = await client.get(
            f"{service_url(request)}{path}", params=forwarded_params(request)
        )
    except httpx.RequestError as exc:
        return failure_response(exc)
    return relay_response(response)


@router.get("/service/health")
async def service_health(request: Request) -> JSONResponse:
    """Relay the service's own ``GET /health``; a 502 means it is not reachable."""
    return await _forward_get(request, "/health")


@router.get("/environments")
async def list_environments(request: Request) -> JSONResponse:
    return await _forward_get(request, "/environments")


@router.get("/environments/{name}")
async def get_environment(request: Request, name: str) -> JSONResponse:
    return await _forward_get(request, f"/environments/{segment(name)}")


@router.get("/environments/{name}/status")
async def get_environment_status(request: Request, name: str) -> JSONResponse:
    return await _forward_get(request, f"/environments/{segment(name)}/status")


@router.get("/environments/{name}/variables")
async def get_environment_variables(request: Request, name: str) -> JSONResponse:
    return await _forward_get(request, f"/environments/{segment(name)}/variables")


@router.get("/environments/{name}/presets")
async def get_environment_presets(request: Request, name: str) -> JSONResponse:
    return await _forward_get(request, f"/environments/{segment(name)}/presets")


@router.get("/environments/{name}/initial-bounds")
async def get_environment_initial_bounds(request: Request, name: str) -> JSONResponse:
    return await _forward_get(request, f"/environments/{segment(name)}/initial-bounds")


# Declared before /jobs/{job_id}, as on the service, so "current" is not read
# as a job id.
@router.get("/jobs/current")
async def get_current_job(request: Request) -> JSONResponse:
    return await _forward_get(request, "/jobs/current")


@router.get("/jobs/{job_id}")
async def get_job(request: Request, job_id: str) -> JSONResponse:
    return await _forward_get(request, f"/jobs/{segment(job_id)}")


@router.get("/jobs/{job_id}/config")
async def get_job_config(request: Request, job_id: str) -> JSONResponse:
    """Relay a draft's config with its ``X-Config-Revision`` header.

    The panel quotes that header back on Start, so the run binds to the
    revision the operator was shown.
    """
    return await _forward_get(request, f"/jobs/{segment(job_id)}/config")


@router.get("/models")
async def list_models(request: Request) -> JSONResponse:
    return await _forward_get(request, "/models")
