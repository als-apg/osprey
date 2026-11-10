"""Write relay onto the optimizer service for the optimizer-web sidecar.

Bodies and outcomes pass through verbatim: the service owns every rule about
what a write may do, and its refusals reach the panel untouched. The forwarded
header set is built from scratch, so nothing credential-shaped is taken from
the browser. Two headers are added: ``X-Osprey-Owner`` (the account the auth
gate matched, through the service's own allowlist) on every write, and
``X-Launch-Token`` (this container's ``OPTIMIZER_LAUNCH_TOKEN``) on start and
resume only. Start forwards nothing but the ``config_revision`` the panel
rendered the draft under, refused 400 here without one; the sidecar never
fetches a revision itself, so Start approves what the operator read.
"""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from osprey.interfaces.common_middleware import OWNER_STATE_KEY
from osprey.interfaces.optimizer_web.read_proxy import (
    failure_response,
    forwarded_params,
    relay_response,
    segment,
    service_url,
)
from osprey.optimizer_connection import OPTIMIZER_LAUNCH_TOKEN_ENV_VAR
from osprey.utils.owner_header import OWNER_HEADER, owner_from_header

router = APIRouter(prefix="/relay")

LAUNCH_TOKEN_HEADER = "X-Launch-Token"  # noqa: S105 -- names a header, holds no secret

#: The query parameter a start names the displayed draft revision by.
CONFIG_REVISION_PARAM = "config_revision"

#: The 400 body for a start that names no revision.
MISSING_REVISION_BODY: dict[str, str] = {
    "detail": (
        "config_revision is required: Start must name the draft revision the panel "
        "displayed (the X-Config-Revision of GET /jobs/{job_id}/config)"
    )
}


async def _forward_write(
    request: Request,
    method: str,
    path: str,
    *,
    arm: bool = False,
    params: httpx.QueryParams | None = None,
) -> JSONResponse:
    """Relay ``method path`` to the service and return its answer verbatim.

    The raw body is forwarded byte for byte, with its content type only when
    there is a body. ``params`` replaces the incoming query string when given.
    ``arm`` adds the launch token; the owner rides every call.
    """
    client: httpx.AsyncClient = request.app.state.client

    content = await request.body()
    headers: dict[str, str] = {}
    if content:
        headers["content-type"] = request.headers.get("content-type", "application/json")
    owner = owner_from_header(getattr(request.state, OWNER_STATE_KEY, None))
    if owner:
        headers[OWNER_HEADER] = owner
    if arm:
        token = os.environ.get(OPTIMIZER_LAUNCH_TOKEN_ENV_VAR)
        if token:
            headers[LAUNCH_TOKEN_HEADER] = token

    try:
        response = await client.request(
            method,
            f"{service_url(request)}{path}",
            params=forwarded_params(request) if params is None else params,
            content=content or None,
            headers=headers,
        )
    except httpx.RequestError as exc:
        return failure_response(exc)
    return relay_response(response)


@router.post("/optimizations")
async def create_draft(request: Request) -> JSONResponse:
    """Relay a new draft. Ungated on the service: a draft writes no setpoint."""
    return await _forward_write(request, "POST", "/optimizations")


@router.put("/jobs/{job_id}/config")
async def put_job_config(request: Request, job_id: str) -> JSONResponse:
    """Relay a draft edit; the service answers 409 once the job has left CREATED."""
    return await _forward_write(request, "PUT", f"/jobs/{segment(job_id)}/config")


@router.post("/jobs/{job_id}/start")
async def start_job(request: Request, job_id: str) -> JSONResponse:
    """Relay a start, armed, at the revision the browser names.

    400 without calling the service when no ``config_revision`` was sent; the
    service answers 409 when the draft changed since that revision. No other
    query parameter is forwarded, so a browser cannot add a flag (such as
    ``override_unhealthy``) the panel never showed.
    """
    revision = request.query_params.get(CONFIG_REVISION_PARAM)
    if not revision:
        return JSONResponse(content=MISSING_REVISION_BODY, status_code=400)
    return await _forward_write(
        request,
        "POST",
        f"/jobs/{segment(job_id)}/start",
        arm=True,
        params=httpx.QueryParams({CONFIG_REVISION_PARAM: revision}),
    )


@router.post("/jobs/{job_id}/resume")
async def resume_job(request: Request, job_id: str) -> JSONResponse:
    """Relay a resume, armed: resuming moves the machine again."""
    return await _forward_write(request, "POST", f"/jobs/{segment(job_id)}/resume", arm=True)


@router.post("/jobs/{job_id}/pause")
async def pause_job(request: Request, job_id: str) -> JSONResponse:
    """Relay a pause. Never armed: stopping motion must not wait on a credential."""
    return await _forward_write(request, "POST", f"/jobs/{segment(job_id)}/pause")


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(request: Request, job_id: str) -> JSONResponse:
    """Relay a cancel. Never armed, for the same reason as pause."""
    return await _forward_write(request, "POST", f"/jobs/{segment(job_id)}/cancel")
