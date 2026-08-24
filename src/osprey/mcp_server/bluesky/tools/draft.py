"""MCP tools: the agent's side of the shared plan draft.

The draft is the ONE shared, live-editable staging surface — a draft of the
next run. The bridge holds a single server-side draft (``{plan_name,
plan_args, revision, ...}``) that the agent and the human's BLUESKY panel both
edit. These three tools are thin HTTP clients of that draft — they never touch
hardware, never require arming, and never pass through an approval prompt:
editing the draft only stages what a future ``queue_add`` (or the human's
Launch plan click) might queue, it does not run anything itself.

==========================  =================================================
Tool                        Bridge endpoint
==========================  =================================================
get_draft                   GET    /draft
set_draft                   PATCH  /draft
clear_draft                 DELETE /draft
==========================  =================================================

Same conventions as the other tool modules: ``async def``, JSON string return
(``json.dumps`` / ``make_error``), blocking HTTP dispatched via
``anyio.to_thread.run_sync``, and the shared ``bridge_error_message`` helper
from ``bluesky/server_context.py`` for translating a non-2xx bridge response.

Every write this module makes carries a fixed ``client_id: "mcp-agent"`` so
the bridge's SSE frames (and the human's BLUESKY panel) can distinguish agent
edits from the human's own, and so the panel's echo-suppression never
swallows an agent edit.

**Drafts are per PLAN LANE, and revisions with them.** The draft is state held
by a bridge process (one singleton draft plus one monotonic revision counter
per process), so a deployment that renders two lanes has two independent
drafts whose revision numbers run independently — revision 3 on the VA lane and
revision 3 on the live lane are different plans. These tools address the ACTIVE
lane, the one serving the target the session is currently on (the default in
``server_context``'s HTTP boundary), which is what keeps the isolation honest:
a draft composed while the session is on one target is edited, read, and
queued on that target's lane, and a session switch moves the agent to the
other lane's draft rather than carrying a revision across.

So every result here names the lane it was read from, and **the launch pin is
``(lane, revision)``, never the revision alone**: pass both to ``queue_add``
and a session that switched in between is refused rather than queueing the
other machine's draft at the same revision number.
"""

from __future__ import annotations

import json
import logging

import anyio

from osprey.mcp_server.bluesky.server import mcp
from osprey.mcp_server.bluesky.server_context import (
    _http_delete_json,
    _http_get_json,
    _http_patch_json,
    addressed_lane_key,
    bridge_error_message,
)
from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import notify_agent_activity

logger = logging.getLogger("osprey.mcp_server.bluesky.tools.draft")

_CLIENT_ID = "mcp-agent"

# Canonical id the build registers the panel under (web.panels.bluesky) and the
# sidecar path segment it is served at.
_DEFAULT_PLANS_PANEL_ID = "bluesky"
_PLANS_PANEL_PATH_SEGMENTS = ("bluesky",)


def _plans_panel_id() -> str:
    """Resolve the web-terminal panel id of the human's BLUESKY panel.

    A successful draft edit is highlighted on that panel's rail entry, so the
    emit must carry the id the web terminal actually knows the panel by: the
    ``web.panels.<id>`` mapping key. The build registers the panel with the
    canonical id ``bluesky`` mounted at the sidecar path ``/bluesky/``; a
    facility that registered it under a different id is found by that fixed
    mount path. Same config fallback chain as the bridge URL resolution
    (``osprey.bluesky_bridge_connection.resolve_bridge_url``): config.yml via
    ``load_osprey_config`` (lazily imported), canonical default when the
    workspace has no matching ``web.panels`` entry — never a bare hardcode.
    """
    from osprey.utils.workspace import load_osprey_config

    panels = load_osprey_config().get("web", {}).get("panels", {})
    if not isinstance(panels, dict):
        return _DEFAULT_PLANS_PANEL_ID
    for segment in _PLANS_PANEL_PATH_SEGMENTS:
        for panel_id, spec in panels.items():
            if not isinstance(spec, dict):
                continue
            if str(spec.get("path", "")).strip("/").split("/")[0] == segment:
                return str(panel_id)
    return _DEFAULT_PLANS_PANEL_ID


def _notify_draft_activity(tool: str, detail: str | None) -> None:
    """Sync body of the fire-and-forget activity emit (worker thread only).

    Reports a successful draft edit to the Web Terminal so the UI can
    highlight the BLUESKY panel (``notify_agent_activity`` is blocking, hence
    dispatched via ``anyio.to_thread.run_sync``; it swallows all exceptions
    itself — a missing web terminal never affects the tool result). Resolving
    the panel id here keeps the config read off the event loop alongside the
    blocking POST.
    """
    notify_agent_activity(tool=tool, kind="panel", panel=_plans_panel_id(), detail=detail)


# ---------------------------------------------------------------------------
# Tool 1: read the current draft
# ---------------------------------------------------------------------------
@mcp.tool()
async def get_draft() -> str:
    """Read the shared plan draft. Reaches NO hardware.

    The draft is the server-held scratch state the agent and the human's
    BLUESKY panel both edit — this only reads it back, it never mutates
    anything.

    Returns:
        JSON ``{"draft", "revision", "lane"}``. ``draft`` is ``null`` when no
        draft exists yet (call set_draft with a ``plan_name`` to create
        one); otherwise ``{"plan_name", "plan_args", "revision", "updated_by",
        "updated_at"}``. ``revision`` is a process-monotonic counter, present
        even when ``draft`` is ``null``. ``lane`` is the plan lane this draft
        belongs to — pass it to ``queue_add`` alongside the revision, because
        the two together are what identify a draft on a deployment with two
        lanes.
    """
    lane = addressed_lane_key()
    status, body = await anyio.to_thread.run_sync(lambda: _http_get_json("/draft", lane=lane))
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    if isinstance(body, dict):
        body["lane"] = lane
    return json.dumps(body)


# ---------------------------------------------------------------------------
# Tool 2: create or edit the draft
# ---------------------------------------------------------------------------
@mcp.tool()
async def set_draft(
    plan_name: str | None = None,
    plan_args_patch: dict | None = None,
    remove: list[str] | None = None,
) -> str:
    """Create or edit the shared plan draft — the staging surface for the next run.

    The draft is the ONE shared, live-editable surface both you and the human
    fill before a run is launched. This edit fills the human's BLUESKY panel
    live (its Plans view): every open panel reflects it within about a second
    and flashes exactly the fields whose values changed — the bridge computes
    ``changed[]`` by comparing values, so re-sending an already-current value is
    a silent no-op (no flash, no revision bump). Setting ``plan_name`` on a
    draft that already names a different plan replaces ``plan_args`` (with
    ``plan_args_patch``'s contents, if also given); setting ``plan_name`` when
    no draft exists creates one.
    Prefer one complete set_draft call (``plan_name`` plus the full
    ``plan_args_patch``) over a trickle of partial edits, so the human sees a
    coherent draft rather than a half-filled form.

    This is staging only — it never starts a run and never requires arming or
    approval. The returned ``revision`` is the launch handle: it identifies this
    exact draft, and ``queue_add(draft_revision)`` queues precisely the draft
    this call produced. A human can instead queue it via their own Launch
    plan click. Nothing runs until a queued item is started.

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
        JSON ``{"revision", "changed", "plan_name", "lane"}`` on success —
        ``changed`` lists the field keys whose value actually changed
        (removed keys included), empty on a no-op patch, and ``lane`` is the
        plan lane this draft belongs to. ``queue_add`` takes both ``revision``
        and ``lane``: on a two-lane deployment the revision alone does not say
        which machine's draft it is.
    """
    if plan_name is None and plan_args_patch is None and remove is None:
        return make_error(
            "set_draft_no_argument",
            "set_draft called with no argument — nothing to change.",
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

    lane = addressed_lane_key()
    status, body = await anyio.to_thread.run_sync(
        lambda: _http_patch_json("/draft", payload, lane=lane)
    )
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

    if isinstance(body, dict):
        body["lane"] = lane

    # Success only (a rejected PATCH must not light up the panel): best-effort
    # activity highlight; must never alter the tool result.
    try:
        detail = body.get("plan_name") if isinstance(body, dict) else None
        await anyio.to_thread.run_sync(_notify_draft_activity, "set_draft", detail)
    except Exception as exc:
        logger.debug("agent-activity emit failed (non-fatal): %s", exc)
    return json.dumps(body)


# ---------------------------------------------------------------------------
# Tool 3: clear the draft
# ---------------------------------------------------------------------------
@mcp.tool()
async def clear_draft() -> str:
    """Clear the shared plan draft. Reaches NO hardware. Idempotent. Destructive.

    Wipes the ONE shared staging surface — the same draft the human may be
    reviewing or filling in their BLUESKY panel right now — back to empty, and
    bumps the revision. Use it deliberately; do not clear a draft just to
    start over when set_draft can replace ``plan_name`` in place. The sole
    clear path (there is no ``clear`` flag on set_draft). Either the agent or
    the human's discard-draft control can clear the draft; calling this when no
    draft exists is a no-op, not an error.

    Returns:
        JSON ``{"revision", "cleared", "lane"}`` — ``cleared`` is ``false``
        when no draft existed (the no-op case; no revision bump), ``true`` when
        a draft was discarded (revision bumps, never resets). ``lane`` is the
        plan lane whose draft this cleared: on a two-lane deployment the other
        lane's draft is untouched.
    """
    lane = addressed_lane_key()
    status, body = await anyio.to_thread.run_sync(
        lambda: _http_delete_json(f"/draft?client_id={_CLIENT_ID}", lane=lane)
    )
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    if isinstance(body, dict):
        body["lane"] = lane

    # Success only: best-effort activity highlight; must never alter the
    # tool result.
    try:
        await anyio.to_thread.run_sync(_notify_draft_activity, "clear_draft", "cleared")
    except Exception as exc:
        logger.debug("agent-activity emit failed (non-fatal): %s", exc)
    return json.dumps(body)
