"""The scan panels sidecar's sole write route: ``POST /runs/execute``.

This is task 1.3 (sidecar-execute-route) of the Phase-6 "Operator Interfaces"
plan. It is the ONLY route in the sidecar that can start a real scan, and it
composes two Bluesky bridge calls server-side so the browser never sees (and
the sidecar never accepts) a promote token:

1. ``POST {bridge}/runs`` — record a launch *intent*, returning a ``Run``
   dict that carries the new ``id``.
2. ``POST {bridge}/runs/{run_id}/promote`` with header ``X-Promote-Token`` —
   promote that intent into a running scan.

The promote token is resolved entirely in-process (mirroring
``osprey.mcp_server.scan.server_context.ScanContext._resolve_promote_token``):
``BLUESKY_PROMOTE_TOKEN`` env var, then ``scan.promote_token`` in
``config.yml``, then ``None``. No request field ever supplies it, and it is
never echoed back in a response.

An unarmed deployment (no token resolved locally, or the bridge itself
rejects the token with 503/403) is treated as an INERT, renderable state —
``{"status": "writes_not_armed", ...}`` with HTTP 200 — not an error: the
panel shows a banner, and an agent probing this route gets a clean signal
rather than a 500 to work around. When no token is resolved locally, the
bridge is never contacted at all, so an unarmed deployment never creates an
orphan intent.

Deliberately exposes exactly one route. No ``/stop`` route (that would let
the browser halt a scan without going through the agent's own tooling) and
no bare, unbounded ``POST /runs`` passthrough (create-intent is always
paired with promote for a named plan here, never exposed on its own).
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

router = APIRouter()

_WRITES_NOT_ARMED_BODY: dict[str, str] = {
    "status": "writes_not_armed",
    "detail": "writes are not armed on this deployment",
}


class ExecuteRequest(BaseModel):
    """Request body for ``POST /runs/execute``. No token field — see module docstring."""

    plan_name: str
    plan_args: dict[str, Any] = Field(default_factory=dict)


def _resolve_promote_token() -> str | None:
    """Resolve the Bluesky bridge promote token, entirely in-process.

    Mirrors ``osprey.mcp_server.scan.server_context.ScanContext._resolve_promote_token``
    and ``osprey.services.scan_panels.app._resolve_bridge_url``'s lazy-import
    style so this module and the scan MCP agree on which token arms which
    bridge instance. Resolution order:

    1. ``BLUESKY_PROMOTE_TOKEN`` env var — wins outright.
    2. ``scan.promote_token`` in config.yml (local/dev convenience only).
    3. ``None`` — this route refuses (inertly) before contacting the bridge.
    """
    token = os.environ.get("BLUESKY_PROMOTE_TOKEN")
    if token:
        return token

    from osprey.utils.workspace import load_osprey_config

    config = load_osprey_config()
    token = config.get("scan", {}).get("promote_token")
    return str(token) if token else None


def _bridge_error_message(body: object, status_code: int) -> str:
    """Extract the bridge's FastAPI ``detail`` message, falling back to the status."""
    if isinstance(body, dict) and body.get("detail"):
        return str(body["detail"])
    return f"Bluesky bridge returned HTTP {status_code}."


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {}


@router.post("/runs/execute")
async def execute_run(payload: ExecuteRequest, request: Request) -> JSONResponse:
    """Compose intent-create + promote into the sidecar's one write route.

    Reads the shared ``httpx.AsyncClient`` and bridge base URL off
    ``request.app.state`` (set by the app's lifespan, task 1.1) rather than
    opening a new connection per request.
    """
    token = _resolve_promote_token()
    if not token:
        # No token resolved locally: refuse before ever contacting the
        # bridge, so an unarmed deployment never creates an orphan intent.
        return JSONResponse(_WRITES_NOT_ARMED_BODY, status_code=200)

    client: httpx.AsyncClient = request.app.state.client
    bridge_url: str = request.app.state.bridge_url

    try:
        create_response = await client.post(
            f"{bridge_url}/runs",
            json={"plan_name": payload.plan_name, "plan_args": payload.plan_args},
        )
    except httpx.HTTPError:
        return JSONResponse({"detail": "bluesky bridge unreachable"}, status_code=502)

    if create_response.status_code >= 400:
        body = _safe_json(create_response)
        return JSONResponse(
            {"detail": _bridge_error_message(body, create_response.status_code)},
            status_code=create_response.status_code,
        )

    run_body = _safe_json(create_response)
    run_id = run_body.get("id") if isinstance(run_body, dict) else None
    if not run_id:
        return JSONResponse(
            {"detail": "bluesky bridge returned no run id"}, status_code=502
        )

    try:
        promote_response = await client.post(
            f"{bridge_url}/runs/{run_id}/promote",
            headers={"X-Promote-Token": token},
        )
    except httpx.HTTPError:
        return JSONResponse({"detail": "bluesky bridge unreachable"}, status_code=502)

    promote_status = promote_response.status_code
    promote_body = _safe_json(promote_response)

    if promote_status in (503, 403):
        # Bridge unarmed, or our resolved token doesn't match the bridge's —
        # for the human and the agent alike this is the same inert state as
        # "no token resolved locally", not a 500.
        return JSONResponse(_WRITES_NOT_ARMED_BODY, status_code=200)
    if promote_status == 409:
        return JSONResponse(
            {"detail": _bridge_error_message(promote_body, promote_status)},
            status_code=409,
        )
    if promote_status != 200:
        return JSONResponse(
            {"detail": _bridge_error_message(promote_body, promote_status)},
            status_code=promote_status,
        )

    result_status = (
        promote_body.get("status", "running") if isinstance(promote_body, dict) else "running"
    )
    return JSONResponse({"run_id": run_id, "status": result_status}, status_code=200)
