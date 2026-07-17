"""MCP tools: read/allow-listed Bluesky bridge operations.

Each tool is a thin HTTP client of one endpoint of the facility-side Bluesky
bridge. All four are safe to call without operator approval
(``permissions_allow``) — none of them can start motion; ``launch_run``
is the sole write path.

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

    A run is the committed record of a launched draft. Its lifecycle runs
    ``pending`` -> ``running`` -> ``completed`` | ``stopped`` | ``error``.

    Args:
        run_id: Run id returned by launch_run or list_runs.

    Returns:
        JSON run record: ``{"id", "status", "tiled_degraded", ["completion"],
        ["launched_by"], ["run_uid"], ["error"]}``. ``status`` is one of "pending",
        "running", "completed", "stopped", "error". ``tiled_degraded`` is True when
        durable persistence to Tiled failed for this run; False when healthy or when
        Tiled is not deployed.
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
    ``launch_run``.

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
    """List this bridge process's tracked runs, newest first.

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
        run_id: Run id returned by launch_run or list_runs.
        max_rows: Maximum number of rows to return (bridge-enforced cap).
        offset: Row offset to start from (``None`` = start from the
            beginning, or from the end if ``tail`` is true).
        tail: When true, return the most recent ``max_rows`` rows instead of
            the earliest ``max_rows`` rows.

    Returns:
        JSON ``{"run_uid", "columns", "rows", "row_count", "truncated"[,
        "partial"]}``. ``partial: true`` means the run is still in progress
        and more rows will arrive; an empty/never-started buffer returns
        ``{"columns": [], "rows": []}``.
    """
    params = f"max_rows={max_rows}"
    if offset is not None:
        params += f"&offset={offset}"
    if tail:
        params += "&tail=true"
    status, body = await anyio.to_thread.run_sync(_http_get_json, f"/runs/{run_id}/data?{params}")
    if status == 404:
        return make_error("unknown_run", bridge_error_message(body, status), UNKNOWN_RUN_HINTS)
    if status == 409:
        return make_error(
            "run_data_not_ready",
            bridge_error_message(body, status),
            [
                "The run has not started yet, so there is no run_uid to read data for.",
                "Check get_run; data becomes readable once the run is launched and running.",
            ],
        )
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    return json.dumps(body)
