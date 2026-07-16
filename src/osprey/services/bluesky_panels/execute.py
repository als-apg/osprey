"""The bluesky panels sidecar's sole write route: ``POST /runs/execute``.

This is task 1.3 (sidecar-execute-route) of the Phase-6 "Operator Interfaces"
plan, extended by task 3.2 (execute-revision-gate) of the agent-plan-draft
plan. It is the ONLY route in the sidecar that can start a real scan, and it
composes two Bluesky bridge calls server-side so the browser never sees (and
the sidecar never accepts) a promote token:

1. ``POST {bridge}/runs`` — record a launch *intent*, returning a ``Run``
   dict that carries the new ``id``.
2. ``POST {bridge}/runs/{run_id}/promote`` with header ``X-Promote-Token`` —
   promote that intent into a running plan.

The promote token is resolved entirely in-process (mirroring
``osprey.mcp_server.bluesky.server_context.BridgeContext._resolve_promote_token``):
``BLUESKY_PROMOTE_TOKEN`` env var, then ``bluesky.promote_token`` in
``config.yml``, then ``None``. No request field ever supplies it, and it is
never echoed back in a response.

An unarmed deployment (no token resolved locally, or the bridge itself
rejects the token with 503/403) is treated as an INERT, renderable state —
``{"status": "writes_not_armed", ...}`` with HTTP 200 — not an error: the
panel shows a banner, and an agent probing this route gets a clean signal
rather than a 500 to work around. When no token is resolved locally, the
bridge is never contacted at all — not even for a draft-mode snapshot read —
so an unarmed deployment never creates an orphan intent.

Deliberately exposes exactly one route. No ``/stop`` route (that would let
the browser halt a scan without going through the agent's own tooling) and
no bare, unbounded ``POST /runs`` passthrough (create-intent is always
paired with promote for a named plan here, never exposed on its own).

``ExecuteRequest`` supports two mutually exclusive launch modes (see its own
docstring): **draft mode** (``draft_revision``) pins a single ``GET
{bridge}/draft`` snapshot server-side and launches exactly that snapshot's
``plan_name``/``plan_args`` — never anything from the request body — with a
409 on a stale/mismatched/cleared revision; **manual mode**
(``plan_name``/``plan_args``) behaves exactly as it did before this module
gained draft support. Everything downstream of mode resolution (create
intent, promote, the unarmed/inert mapping, 502/409 conventions) is
unchanged by which mode supplied the launched args.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

router = APIRouter()

_WRITES_NOT_ARMED_BODY: dict[str, str] = {
    "status": "writes_not_armed",
    "detail": "writes are not armed on this deployment",
}

_UNREACHABLE_BODY: dict[str, str] = {"detail": "bluesky bridge unreachable"}

# Machine-readable discriminator for a stale/mismatched draft-revision pin.
# The panel branches on this exact string; the bridge's own relayed promote
# 409 (see below) carries no ``code`` at all, so the two are distinguishable
# without the panel having to parse prose.
_STALE_DRAFT_REVISION_CODE = "stale_draft_revision"


class ExecuteRequest(BaseModel):
    """Request body for ``POST /runs/execute``. No token field — see module docstring.

    Exactly one of two mutually exclusive launch modes must be supplied:

    - **Draft mode** — ``draft_revision`` pins a specific bridge plan-draft
      revision. The launched ``plan_name``/``plan_args`` come from a single
      server-side ``GET {bridge}/draft`` snapshot, never from this request
      body; a snapshot revision that doesn't match ``draft_revision``
      (including a since-cleared/null draft) is a 409, not a silent
      fallback to whatever the body happened to carry.
    - **Manual mode** — ``plan_name`` (+ optional ``plan_args``) supplies the
      plan to launch directly, exactly as before draft mode existed.

    Supplying both a ``draft_revision`` and any manual-mode field, or
    supplying neither, is a 422: the launched args must have exactly one
    source, never an "ignored but present" body field that could act as a
    second, silently-dropped source.
    """

    draft_revision: int | None = None
    plan_name: str | None = None
    plan_args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_exactly_one_launch_mode(self) -> ExecuteRequest:
        draft_mode = self.draft_revision is not None
        # An explicitly-supplied ``plan_args`` counts as a manual-mode field
        # even without ``plan_name`` -- draft mode must reject a stray body
        # arg rather than silently ignore it (module docstring: "never an
        # ignored but present body field"). ``model_fields_set`` (not the
        # value itself) is what distinguishes "the caller sent plan_args" from
        # "plan_args defaulted to {}", since both look like ``{}`` by value.
        # An explicitly-supplied ``plan_name: null`` alongside ``draft_revision``
        # must also count as a manual-mode field: without checking
        # ``model_fields_set`` for ``plan_name`` too, ``{"draft_revision": 5,
        # "plan_name": null}`` looked identical (by value) to plain draft mode
        # and slipped through as valid, breaking XOR symmetry with the
        # ``plan_args`` check just above. Omitting ``plan_name`` entirely
        # alongside ``draft_revision`` must still validate -- only an
        # explicitly-present ``plan_name`` (null or not) counts here.
        manual_field_present = (
            self.plan_name is not None
            or "plan_args" in self.model_fields_set
            or "plan_name" in self.model_fields_set
        )
        if draft_mode and manual_field_present:
            raise ValueError(
                "supply exactly one of draft_revision or plan_name/plan_args, not both -- "
                "draft_revision launches a pinned bridge snapshot and plan_name/plan_args "
                "launch directly; they cannot be combined"
            )
        if not draft_mode and self.plan_name is None:
            raise ValueError(
                "supply exactly one of draft_revision or plan_name (plan_args optional) -- "
                "neither was given"
            )
        return self


def _resolve_promote_token() -> str | None:
    """Resolve the Bluesky bridge promote token, entirely in-process.

    Mirrors ``osprey.mcp_server.bluesky.server_context.BridgeContext._resolve_promote_token``
    and ``osprey.services.bluesky_panels.app._resolve_bridge_url``'s lazy-import
    style so this module and the Bluesky MCP agree on which token arms which
    bridge instance. Resolution order:

    1. ``BLUESKY_PROMOTE_TOKEN`` env var — wins outright.
    2. ``bluesky.promote_token`` in config.yml (local/dev convenience only).
    3. ``None`` — this route refuses (inertly) before contacting the bridge.
    """
    token = os.environ.get("BLUESKY_PROMOTE_TOKEN")
    if token:
        return token

    from osprey.utils.workspace import load_osprey_config

    config = load_osprey_config()
    token = config.get("bluesky", {}).get("promote_token")
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


async def _fetch_draft_snapshot(
    client: httpx.AsyncClient, bridge_url: str, draft_revision: int
) -> tuple[str, dict[str, Any]] | JSONResponse:
    """Take one ``GET {bridge}/draft`` snapshot and pin it against *draft_revision*.

    Returns the snapshot's ``(plan_name, plan_args)`` on a revision match.
    Returns a 409 ``JSONResponse`` with ``code: "stale_draft_revision"`` on a
    mismatch -- this includes a since-cleared draft (the bridge's revision
    counter bumps on ``DELETE`` and never resets, so a pinned revision from
    before the clear can never match again) and, defensively, a null draft
    even if its revision happened to coincide with *draft_revision* (there is
    no ``plan_name``/``plan_args`` to launch either way). Returns a 502
    ``JSONResponse`` if the bridge itself is unreachable, matching this
    route's existing bridge-unreachable convention.

    The response status is checked *before* the body is interpreted: a
    non-200 ``GET /draft`` (e.g. a bridge 500) is relayed as-is -- mirroring
    ``execute_run``'s own create-intent convention just below (``POST
    /runs``'s ``status_code >= 400`` check) -- rather than falling through to
    ``_safe_json``, which would parse a 5xx error body to a dict lacking
    ``"draft"`` and get misreported as a 409 ``stale_draft_revision``.
    """
    try:
        response = await client.get(f"{bridge_url}/draft")
    except httpx.HTTPError:
        return JSONResponse(_UNREACHABLE_BODY, status_code=502)

    if response.status_code >= 400:
        body = _safe_json(response)
        return JSONResponse(
            {"detail": _bridge_error_message(body, response.status_code)},
            status_code=response.status_code,
        )

    body = _safe_json(response)
    snapshot_draft = body.get("draft") if isinstance(body, dict) else None
    snapshot_revision = body.get("revision") if isinstance(body, dict) else None

    if snapshot_draft is None or snapshot_revision != draft_revision:
        return JSONResponse(
            {
                "code": _STALE_DRAFT_REVISION_CODE,
                "detail": (
                    f"draft revision {draft_revision!r} is stale "
                    f"(current bridge draft revision is {snapshot_revision!r})"
                ),
            },
            status_code=409,
        )

    plan_name = snapshot_draft.get("plan_name")
    if not plan_name:
        # A non-null draft lacking a plan_name is a malformed bridge
        # response -- same treatment as any other bridge data the route
        # can't act on (cf. execute_run's "bluesky bridge returned no run
        # id" 502 guard): report it as a bridge error rather than forwarding
        # a knowingly-unlaunchable {"plan_name": null, ...} to create-intent.
        return JSONResponse(
            {"detail": "bluesky bridge draft snapshot has no plan_name"},
            status_code=502,
        )

    return plan_name, snapshot_draft.get("plan_args") or {}


@router.post("/runs/execute")
async def execute_run(payload: ExecuteRequest, request: Request) -> JSONResponse:
    """Compose intent-create + promote into the sidecar's one write route.

    Reads the shared ``httpx.AsyncClient`` and bridge base URL off
    ``request.app.state`` (set by the app's lifespan, task 1.1) rather than
    opening a new connection per request.

    In draft mode (``payload.draft_revision`` set), the launched
    ``plan_name``/``plan_args`` come from a single bridge draft snapshot
    (:func:`_fetch_draft_snapshot`) taken here, after the arming check --
    never from ``payload`` itself. In manual mode they come straight from
    ``payload.plan_name``/``payload.plan_args``, exactly as before draft mode
    existed.
    """
    token = _resolve_promote_token()
    if not token:
        # No token resolved locally: refuse before ever contacting the
        # bridge (not even for a draft-mode snapshot read), so an unarmed
        # deployment never creates an orphan intent.
        return JSONResponse(_WRITES_NOT_ARMED_BODY, status_code=200)

    client: httpx.AsyncClient = request.app.state.client
    bridge_url: str = request.app.state.bridge_url

    if payload.draft_revision is not None:
        snapshot = await _fetch_draft_snapshot(client, bridge_url, payload.draft_revision)
        if isinstance(snapshot, JSONResponse):
            return snapshot
        plan_name, plan_args = snapshot
    else:
        plan_name, plan_args = payload.plan_name, payload.plan_args

    try:
        create_response = await client.post(
            f"{bridge_url}/runs",
            json={"plan_name": plan_name, "plan_args": plan_args},
        )
    except httpx.HTTPError:
        return JSONResponse(_UNREACHABLE_BODY, status_code=502)

    if create_response.status_code >= 400:
        body = _safe_json(create_response)
        return JSONResponse(
            {"detail": _bridge_error_message(body, create_response.status_code)},
            status_code=create_response.status_code,
        )

    run_body = _safe_json(create_response)
    run_id = run_body.get("id") if isinstance(run_body, dict) else None
    if not run_id:
        return JSONResponse({"detail": "bluesky bridge returned no run id"}, status_code=502)

    try:
        promote_response = await client.post(
            f"{bridge_url}/runs/{run_id}/promote",
            headers={"X-Promote-Token": token},
        )
    except httpx.HTTPError:
        return JSONResponse(_UNREACHABLE_BODY, status_code=502)

    promote_status = promote_response.status_code
    promote_body = _safe_json(promote_response)

    if promote_status in (503, 403):
        # Bridge unarmed, or our resolved token doesn't match the bridge's —
        # for the human and the agent alike this is the same inert state as
        # "no token resolved locally", not a 500.
        return JSONResponse(_WRITES_NOT_ARMED_BODY, status_code=200)
    if promote_status != 200:
        # Any other promote failure (including the bridge's own 409, which —
        # unlike the sidecar-minted stale_draft_revision 409 — carries no
        # ``code``) is relayed body-and-status verbatim.
        return JSONResponse(
            {"detail": _bridge_error_message(promote_body, promote_status)},
            status_code=promote_status,
        )

    result_status = (
        promote_body.get("status", "running") if isinstance(promote_body, dict) else "running"
    )
    return JSONResponse({"run_id": run_id, "status": result_status}, status_code=200)
