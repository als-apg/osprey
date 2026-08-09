"""MCP tools: read/allow-listed Bluesky bridge operations.

Each tool is a thin HTTP client of one endpoint of the facility-side Bluesky
bridge. All four are safe to call without operator approval
(``permissions_allow``) — none of them can start motion; ``queue_start`` is
the sole path by which execution begins.

==========================  =================================================
Tool                        Bridge endpoint
==========================  =================================================
get_run                    GET  /runs/{id}
list_plans             GET  /plans
list_runs                   GET  /runs
get_run_data               GET  /runs/{id}/data
==========================  =================================================

The HTTP primitive (``_http_get_json``) and the
``bridge_error_message`` / ``UNKNOWN_RUN_HINTS`` error-envelope helpers live in
``osprey.mcp_server.bluesky.server_context`` so tests can patch the network
boundary and every tool renders identical error shapes. A
connection-level failure there already raises the standard
``bluesky_bridge_unreachable`` error envelope, so the tools below only need to
translate non-2xx bridge responses (404/409/etc.) into ``make_error`` calls.
"""

import json

import anyio

from osprey.mcp_server.bluesky.server import mcp
from osprey.mcp_server.bluesky.server_context import (
    UNKNOWN_RUN_HINTS,
    _http_get_json,
    bridge_error_message,
)
from osprey.mcp_server.errors import make_error


# ---------------------------------------------------------------------------
# Tool 1: get run
# ---------------------------------------------------------------------------
@mcp.tool()
async def get_run(run_id: str) -> str:
    """Get one run's current lifecycle status.

    A run is a queued item seen from the run side: the manager's queue,
    running item and history, keyed by the OSPREY run id. Its lifecycle runs
    ``pending`` -> ``running`` -> ``completed`` | ``stopped`` | ``error``.

    Args:
        run_id: Run id returned by queue_add or list_runs.

    Returns:
        JSON run record. Four keys are always present: ``"id"`` (this run id),
        ``"status"`` (exactly one of "pending", "running", "completed",
        "stopped", "error"), ``"plan_name"``, and ``"plan_args"`` (the plan's
        parameter fields, ``{}`` when it takes none).

        Four more appear only when they are true of this run:
        ``"item_uid"`` (the queue item's handle), ``"run_uid"`` (the
        acquisition's own uid, ABSENT while pending or running because it does
        not exist until the worker starts the plan — read that as "not yet",
        never as "unknown"), ``"error"`` (present if and only if status is
        "error", explaining what the manager reported), and ``"progress"``
        (``{"rows_seen", "expected_points", "fraction", "complete"}``, ABSENT
        when nothing is known — never a fabricated 0%; ``"fraction"`` is null
        whenever the total point count cannot be derived, which is common for
        agent-authored plans, so report it as "N points so far" rather than a
        percentage).

        "stopped" means a human stopped it, by any route. An item that left the
        queue without a cleanly recorded finish reads as "error" by design, so
        an unrecognized ending is never mistaken for a successful scan.

    Refusals:
        - unknown_run: this run id is not in the manager's queue or its
          retained history — typically rotated out of history. Note this does
          NOT mean the run's data is gone: get_run_data can still serve that
          same id from durable storage, so never infer "no data" from this.
    """
    status, body = await anyio.to_thread.run_sync(_http_get_json, f"/runs/{run_id}")
    if status == 404:
        return make_error("unknown_run", bridge_error_message(body, status), UNKNOWN_RUN_HINTS)
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    return json.dumps(body)


# ---------------------------------------------------------------------------
# Tool 3: list plans
# ---------------------------------------------------------------------------
@mcp.tool()
async def list_plans() -> str:
    """List the plans registered on the bridge.

    Each plan entry carries ``metadata`` (the plan's authoring-declared
    ``PLAN_METADATA`` — description/category/required_devices/writes — or
    ``null`` for a built-in that doesn't author one) and ``provenance`` (its
    trust tier: ``shipped``/``preset``/``facility``/``session``/
    ``unreviewed``, ascending ephemerality). Use these to prefer a
    higher-provenance plan and to check ``required_devices``/``writes``
    before staging a plan into the draft (set_draft) for a future
    ``queue_add``.

    Returns:
        JSON ``{"status": "success", "plans": [...]}``, each entry shaped
        like ``{"name", "description", "schema", "metadata", "provenance"}``.
        An empty list means the facility has not injected a plan module (or
        this bridge version does not yet support plan discovery).
    """
    status, body = await anyio.to_thread.run_sync(_http_get_json, "/plans")
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    return json.dumps({"status": "success", "plans": body})


# ---------------------------------------------------------------------------
# Tool 4: list runs
# ---------------------------------------------------------------------------
@mcp.tool()
async def list_runs(limit: int = 20) -> str:
    """List runs the queue knows about — queued, running and recent — newest first.

    Covers what OSPREY enqueued: an item put on the queue by some other route,
    without an OSPREY run id, is absent from this list entirely. queue_list is
    the complete view of what the manager actually holds, so reach for that
    when the question is "what is this machine about to do" rather than "what
    did I start".

    Args:
        limit: Maximum number of runs to return (the bridge clamps this to
            the range [1, 100]).

    Returns:
        JSON ``{"status": "success", "runs": [...]}`` — each entry has the
        same shape as get_run's response.
    """
    status, body = await anyio.to_thread.run_sync(_http_get_json, f"/runs?limit={limit}")
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    return json.dumps({"status": "success", "runs": body})


# ---------------------------------------------------------------------------
# Tool 5: get run data (bounded)
# ---------------------------------------------------------------------------
@mcp.tool()
async def get_run_data(
    run_id: str, max_rows: int = 100, offset: int | None = None, tail: bool = False
) -> str:
    """Read a bounded window of a run's data.

    Row-bounded by design: this never returns an unbounded table. Backed by
    the bridge's in-process live-row buffer (``GET /runs/{id}/data``), so
    reads work with no Tiled server — ``row_count``/``truncated`` describe
    the run's *true* total vs. what this window actually returned.

    Args:
        run_id: Run id returned by queue_add or list_runs.
        max_rows: Maximum number of rows to return (bridge-enforced cap).
        offset: Row offset to start from (``None`` = start from the
            beginning, or from the end if ``tail`` is true).
        tail: When true, return the most recent ``max_rows`` rows instead of
            the earliest ``max_rows`` rows.

    Returns:
        JSON ``{"run_uid", "columns", "rows", "row_count", "truncated"[,
        "partial"]}``. ``partial: true`` means the run is still in progress
        and more rows will arrive; an empty/never-started buffer returns
        ``{"columns": [], "rows": []}``. ``run_uid`` is always present as a key
        but is null when the rows came from the live buffer, and populated when
        they came from durable storage — a null there says which path served
        the read, not that anything is missing.

        Readable for runs get_run no longer knows about: a run rotated out of
        the manager's history still has its data, so a 404 from get_run is
        never a reason to skip reading here.
    """
    params = f"max_rows={max_rows}"
    if offset is not None:
        params += f"&offset={offset}"
    if tail:
        params += "&tail=true"
    status, body = await anyio.to_thread.run_sync(_http_get_json, f"/runs/{run_id}/data?{params}")
    if status == 404:
        return make_error("unknown_run", bridge_error_message(body, status), UNKNOWN_RUN_HINTS)
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    return json.dumps(body)
