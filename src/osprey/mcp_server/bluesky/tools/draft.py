"""MCP tools: the agent's side of the shared plan draft.

The bridge holds a single server-side draft (``{plan_name, plan_args,
revision, ...}``) that the agent and the human's plan panel both edit. These
three tools are thin HTTP clients of that draft — they never touch hardware,
never require arming, and never pass through an approval prompt: editing the
draft only stages what a future ``launch_run`` (or the human's Execute click)
might run, it does not run anything itself.

==========================  =================================================
Tool                        Bridge endpoint
==========================  =================================================
get_plan_draft              GET    /draft
set_plan_draft               PATCH  /draft
clear_plan_draft             DELETE /draft
==========================  =================================================

Same conventions as the other tool modules: ``async def``, JSON string return
(``json.dumps`` / ``make_error``), blocking HTTP dispatched via
``anyio.to_thread.run_sync``, and the shared ``bridge_error_message`` helper
from ``bluesky/server_context.py`` for translating a non-2xx bridge response.

Every write this module makes carries a fixed ``client_id: "mcp-agent"`` so
the bridge's SSE frames (and the human's plan panel) can distinguish agent
edits from the human's own, and so the panel's echo-suppression never
swallows an agent edit.
"""

from __future__ import annotations

import json

import anyio

from osprey.mcp_server.bluesky.server import mcp
from osprey.mcp_server.bluesky.server_context import (
    _http_delete_json,
    _http_get_json,
    _http_patch_json,
    bridge_error_message,
)
from osprey.mcp_server.errors import make_error

_CLIENT_ID = "mcp-agent"


# ---------------------------------------------------------------------------
# Tool 1: read the current draft
# ---------------------------------------------------------------------------
@mcp.tool()
async def get_plan_draft() -> str:
    """Read the shared plan draft. Reaches NO hardware.

    The draft is the server-held scratch state the agent and the human's plan
    panel both edit — this only reads it back, it never mutates anything.

    Returns:
        JSON ``{"draft", "revision"}``. ``draft`` is ``null`` when no draft
        exists yet (call set_plan_draft with a ``plan_name`` to create
        one); otherwise ``{"plan_name", "plan_args", "revision", "updated_by",
        "updated_at"}``. ``revision`` is a process-monotonic counter, present
        even when ``draft`` is ``null``.
    """
    status, body = await anyio.to_thread.run_sync(_http_get_json, "/draft")
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    return json.dumps(body)


# ---------------------------------------------------------------------------
# Tool 2: create or edit the draft
# ---------------------------------------------------------------------------
@mcp.tool()
async def set_plan_draft(
    plan_name: str | None = None,
    plan_args_patch: dict | None = None,
    remove: list[str] | None = None,
) -> str:
    """Create or edit the shared plan draft. Reaches NO hardware.

    Every open plan panel reflects this edit within about a second and
    flashes exactly the fields whose values changed — the bridge computes
    ``changed[]`` by comparing values, so re-sending an already-current value
    is a silent no-op (no flash, no revision bump). Setting ``plan_name`` on
    a draft that already names a different plan replaces ``plan_args``
    (with ``plan_args_patch``'s contents, if also given); setting
    ``plan_name`` when no draft exists creates one. This is staging only —
    it never starts a run and never requires arming or approval; a human
    still launches via their own Execute click, or the agent via
    launch_run/create_run_intent afterward.

    Args:
        plan_name: Plan to draft. Required to create a draft that does not
            exist yet. Must be a plan currently known to the bridge (see
            list_plans) — an unrecognized name is rejected.
        plan_args_patch: Top-level values to merge into the draft's
            ``plan_args`` (validated field-by-field against that plan's
            parameter schema; an invalid value is rejected and never reaches
            the panel).
        remove: Keys to delete from the draft's ``plan_args``. Distinct from
            passing ``null`` in ``plan_args_patch``, which is a legal value
            for an Optional field rather than a deletion.

    Returns:
        JSON ``{"revision", "changed", "plan_name"}`` on success —
        ``changed`` lists the field keys whose value actually changed
        (removed keys included), empty on a no-op patch.
    """
    if plan_name is None and plan_args_patch is None and remove is None:
        return make_error(
            "set_plan_draft_no_argument",
            "set_plan_draft called with no argument — nothing to change.",
            [
                "Pass plan_name to create/switch the draft.",
                "Pass plan_args_patch and/or remove to edit an existing draft.",
            ],
        )

    payload: dict = {"client_id": _CLIENT_ID}
    if plan_name is not None:
        payload["plan_name"] = plan_name
    if plan_args_patch is not None:
        payload["plan_args_patch"] = plan_args_patch
    if remove is not None:
        payload["remove"] = remove

    status, body = await anyio.to_thread.run_sync(_http_patch_json, "/draft", payload)
    if status == 409 and isinstance(body, dict) and body.get("code") == "no_draft":
        return make_error(
            "no_draft",
            bridge_error_message(body, status),
            ["no draft; pass plan_name to create one"],
        )
    if status == 422:
        return make_error(
            "unknown_plan",
            bridge_error_message(body, status),
            ["validate the session plan first"],
        )
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    return json.dumps(body)


# ---------------------------------------------------------------------------
# Tool 3: clear the draft
# ---------------------------------------------------------------------------
@mcp.tool()
async def clear_plan_draft() -> str:
    """Clear the shared plan draft. Reaches NO hardware. Idempotent.

    The sole clear path (there is no ``clear`` flag on set_plan_draft).
    Either the agent or the human's discard-draft control can clear the
    draft; calling this when no draft exists is a no-op, not an error.

    Returns:
        JSON ``{"revision", "cleared"}`` — ``cleared`` is ``false`` when no
        draft existed (the no-op case; no revision bump), ``true`` when a
        draft was discarded (revision bumps, never resets).
    """
    status, body = await anyio.to_thread.run_sync(
        _http_delete_json, f"/draft?client_id={_CLIENT_ID}"
    )
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    return json.dumps(body)
