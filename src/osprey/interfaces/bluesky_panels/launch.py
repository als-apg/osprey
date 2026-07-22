"""The bluesky panels sidecar's sole write route: ``POST /runs/launch``.

This is the sidecar's single run-launch route of the "Operator Interfaces"
work, extended by the draft-launch work. It is the ONLY route in the sidecar
that can start a real scan. It has two mutually exclusive launch modes (see
``LaunchRequest``), and each rides its own Bluesky bridge primitive
server-side so the browser never sees (and the sidecar never accepts) a
launch token:

- **Manual mode** composes two bridge calls:

  1. ``POST {bridge}/runs`` — record a ``pending`` run, returning a ``Run``
     dict that carries the new ``id``.
  2. ``POST {bridge}/runs/{run_id}/launch`` with header ``X-Launch-Token`` —
     launch that pending run into a running plan.

- **Draft mode** makes ONE bridge call: ``POST {bridge}/draft/run`` with
  header ``X-Launch-Token`` and body ``{"draft_revision": <int>}``. The
  bridge launches its own held draft snapshot for that pinned revision under
  a single server-side lock — closing the GET-then-POST race the old
  snapshot-read → create pending run → launch composition had. The launched
  ``plan_name``/``plan_args`` come from the bridge's held draft, never from
  this request body.

The launch token is resolved entirely in-process via the shared
``osprey.bluesky_bridge_connection.resolve_launch_token`` (the same resolver the
Bluesky MCP server uses, so the two never drift on which token arms which
bridge): ``BLUESKY_LAUNCH_TOKEN`` env var, then ``bluesky.launch_token`` in
``config.yml``, then ``None``. No request field ever supplies it, and it is
never echoed back in a response.

An unarmed deployment (no token resolved locally, or the bridge itself
rejects the token with 503/403) is treated as an INERT, renderable state —
``{"status": "writes_not_armed", ...}`` with HTTP 200 — not an error: the
panel shows a banner, and an agent probing this route gets a clean signal
rather than a 500 to work around. When no token is resolved locally, the
bridge is never contacted at all — in either mode — so an unarmed deployment
never creates an orphan pending run or launch.

Deliberately exposes exactly one route. No ``/stop`` route (that would let
the browser halt a scan without going through the agent's own tooling) and
no bare, unbounded ``POST /runs`` passthrough (the pending-run create is
always paired with a launch for a named plan here, never exposed on its own).

``LaunchRequest`` supports the two launch modes above (see its own
docstring): **draft mode** (``draft_revision``) launches the bridge's pinned
draft snapshot for that revision — never anything from the request body —
with the bridge minting a 409 on a stale/mismatched/cleared revision
(``stale_draft_revision``) or a revision already launched
(``draft_revision_already_launched``); both are relayed to the panel at 409,
unwrapped from the bridge's nested ``detail`` to the panel client's top-level
``{"code", "detail", "revision"}`` shape. **Manual mode**
(``plan_name``/``plan_args``) behaves exactly as it did before this module
gained draft support.
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from osprey.bluesky_bridge_connection import (
    bridge_error_message,
    resolve_launch_token,
    unwrap_bridge_conflict_detail,
)
from osprey.interfaces.bluesky_panels._shared import UNREACHABLE_BODY, safe_json

router = APIRouter()

_WRITES_NOT_ARMED_BODY: dict[str, str] = {
    "status": "writes_not_armed",
    "detail": "writes are not armed on this deployment",
}


class LaunchRequest(BaseModel):
    """Request body for ``POST /runs/launch``. No token field — see module docstring.

    Exactly one of two mutually exclusive launch modes must be supplied:

    - **Draft mode** — ``draft_revision`` pins a specific bridge plan-draft
      revision. The launched ``plan_name``/``plan_args`` come from the
      bridge's own held draft snapshot for that revision (via ``POST
      {bridge}/draft/run``), never from this request body; a revision that
      doesn't match the bridge's current draft (including a
      since-cleared/null draft), or one already launched, is a 409, not a
      silent fallback to whatever the body happened to carry.
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
    def _check_exactly_one_launch_mode(self) -> LaunchRequest:
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


async def _launch_draft(
    client: httpx.AsyncClient, bridge_url: str, draft_revision: int, token: str
) -> JSONResponse:
    """Launch the bridge's held draft snapshot for *draft_revision* in one call.

    Delegates to ``POST {bridge}/draft/run`` (header ``X-Launch-Token``),
    which pins, mints, launches, and records the run under a single
    server-side lock -- so the sidecar never reads the draft and then races a
    separate create + launch against it. The launched ``plan_name``/``plan_args``
    are the bridge's, never this request's.

    Response mapping mirrors manual mode's launch handling:

    - **503 / 403** -> the same inert ``writes_not_armed`` 200 as "no token
      resolved locally": the bridge is unarmed, or the resolved token doesn't
      match the bridge's.
    - **409** -> relayed at 409, but UNWRAPPED. The bridge nests its
      discriminator under FastAPI's ``detail`` (``{"detail": {"code": ...,
      "detail": ..., "revision": <int>}}``); the panel client classifies on a
      TOP-LEVEL ``code`` (``panels/plan/draft-client.js``), so this projects
      the nested detail up to the panel's ``{"code", "detail", "revision"}``
      shape for BOTH ``stale_draft_revision`` and
      ``draft_revision_already_launched``.
    - **any other non-200** (e.g. a 500 post-mint launch failure, a 404) ->
      relayed with the bridge's status and ``detail`` message.
    - **200** -> the minted run record, projected to this route's
      ``{"run_id", "status"}`` success shape.

    A connection-level failure is this route's existing 502
    ``bluesky bridge unreachable``.
    """
    try:
        response = await client.post(
            f"{bridge_url}/draft/run",
            json={"draft_revision": draft_revision},
            headers={"X-Launch-Token": token},
        )
    except httpx.HTTPError:
        return JSONResponse(UNREACHABLE_BODY, status_code=502)

    status = response.status_code
    body = safe_json(response, default={})

    if status in (503, 403):
        return JSONResponse(_WRITES_NOT_ARMED_BODY, status_code=200)
    if status == 409:
        # The bridge nests its 409 discriminator under FastAPI's ``detail``
        # (``{"detail": {"code", "detail", "revision"}}``), but the panel
        # client classifies on a TOP-LEVEL ``code`` (draft-client.js). Unwrap
        # to the panel's top-level ``{"code", "detail", "revision"}`` shape for
        # BOTH stale_draft_revision and draft_revision_already_launched. A 409
        # that isn't the expected nested-detail envelope is relayed as-is.
        nested = unwrap_bridge_conflict_detail(body)
        return JSONResponse(nested if nested is not None else body, status_code=409)
    if status != 200:
        return JSONResponse(
            {"detail": bridge_error_message(body, status)},
            status_code=status,
        )

    run_id = body.get("id") if isinstance(body, dict) else None
    if not run_id:
        return JSONResponse({"detail": "bluesky bridge returned no run id"}, status_code=502)
    result_status = body.get("status", "running") if isinstance(body, dict) else "running"
    return JSONResponse({"run_id": run_id, "status": result_status}, status_code=200)


async def _launch_manual(
    client: httpx.AsyncClient,
    bridge_url: str,
    plan_name: str,
    plan_args: dict[str, Any],
    token: str,
) -> JSONResponse:
    """Compose pending-run create + launch for manual mode, in two bridge calls.

    1. ``POST {bridge}/runs`` records a ``pending`` run and returns its ``id``.
    2. ``POST {bridge}/runs/{run_id}/launch`` (header ``X-Launch-Token``)
       launches that pending run.

    Response mapping mirrors :func:`_launch_draft`:

    - a create failure (>=400) is relayed with the bridge's status and
      ``detail`` message;
    - launch **503 / 403** -> the same inert ``writes_not_armed`` 200 as "no
      token resolved locally" (the bridge is unarmed, or the resolved token
      doesn't match the bridge's);
    - any other non-200 launch (including the bridge's own 409 for an
      already-launched run) is relayed with the bridge's status and ``detail``;
    - **200** -> the launched run projected to this route's ``{"run_id",
      "status"}`` success shape.

    A connection-level failure to either bridge call is this route's existing
    502 ``bluesky bridge unreachable``.
    """
    try:
        create_response = await client.post(
            f"{bridge_url}/runs",
            json={"plan_name": plan_name, "plan_args": plan_args},
        )
    except httpx.HTTPError:
        return JSONResponse(UNREACHABLE_BODY, status_code=502)

    if create_response.status_code >= 400:
        body = safe_json(create_response, default={})
        return JSONResponse(
            {"detail": bridge_error_message(body, create_response.status_code)},
            status_code=create_response.status_code,
        )

    run_body = safe_json(create_response, default={})
    run_id = run_body.get("id") if isinstance(run_body, dict) else None
    if not run_id:
        return JSONResponse({"detail": "bluesky bridge returned no run id"}, status_code=502)

    try:
        launch_response = await client.post(
            f"{bridge_url}/runs/{run_id}/launch",
            headers={"X-Launch-Token": token},
        )
    except httpx.HTTPError:
        return JSONResponse(UNREACHABLE_BODY, status_code=502)

    launch_status = launch_response.status_code
    launch_body = safe_json(launch_response, default={})

    if launch_status in (503, 403):
        # Bridge unarmed, or our resolved token doesn't match the bridge's —
        # for the human and the agent alike this is the same inert state as
        # "no token resolved locally", not a 500.
        return JSONResponse(_WRITES_NOT_ARMED_BODY, status_code=200)
    if launch_status != 200:
        # Any other launch failure (including the bridge's own 409 for an
        # already-launched run) is relayed with the bridge's status and
        # ``detail`` message.
        return JSONResponse(
            {"detail": bridge_error_message(launch_body, launch_status)},
            status_code=launch_status,
        )

    result_status = (
        launch_body.get("status", "running") if isinstance(launch_body, dict) else "running"
    )
    return JSONResponse({"run_id": run_id, "status": result_status}, status_code=200)


@router.post("/runs/launch")
async def launch_run(payload: LaunchRequest, request: Request) -> JSONResponse:
    """Compose pending-run create + launch into the sidecar's one write route.

    Reads the shared ``httpx.AsyncClient`` and bridge base URL off
    ``request.app.state`` (set by the app's lifespan, task 1.1) rather than
    opening a new connection per request.

    In draft mode (``payload.draft_revision`` set), the whole launch is the
    bridge's one ``POST /draft/run`` primitive (:func:`_launch_draft`), taken
    after the arming check -- the launched ``plan_name``/``plan_args`` are the
    bridge's held draft, never ``payload`` itself. In manual mode
    (:func:`_launch_manual`) they come straight from
    ``payload.plan_name``/``payload.plan_args``, exactly as before draft mode
    existed.
    """
    token = resolve_launch_token()
    if not token:
        # No token resolved locally: refuse before ever contacting the
        # bridge (in either mode), so an unarmed deployment never creates an
        # orphan pending run or launch.
        return JSONResponse(_WRITES_NOT_ARMED_BODY, status_code=200)

    client: httpx.AsyncClient = request.app.state.client
    bridge_url: str = request.app.state.bridge_url

    if payload.draft_revision is not None:
        return await _launch_draft(client, bridge_url, payload.draft_revision, token)

    # The validator guarantees plan_name is set in manual mode.
    assert payload.plan_name is not None
    return await _launch_manual(client, bridge_url, payload.plan_name, payload.plan_args, token)
