"""MCP tools: read/allow-listed Bluesky bridge operations.

Each tool is a thin HTTP client of one endpoint of the facility-side scan
bridge. All five are safe to call without operator approval
(``permissions_allow``) — none of them can start motion; ``launch_scan``
(Task 1.8) is the sole promote path.

==========================  =================================================
Tool                        Bridge endpoint
==========================  =================================================
create_scan_intent          POST /runs
scan_status                 GET  /runs/{id}
list_scan_plans             GET  /plans
list_runs                   GET  /runs
read_scan_data              GET  /runs/{id}/data
==========================  =================================================

The HTTP primitives (``_http_get_json`` / ``_http_post_json``) and the
``bridge_error_message`` / ``UNKNOWN_RUN_HINTS`` error-envelope helpers live in
``osprey.mcp_server.scan.server_context`` so tests can patch the network
boundary and every scan tool renders identical error shapes. A
connection-level failure there already raises the standard
``bluesky_bridge_unreachable`` error envelope, so the tools below only need to
translate non-2xx bridge responses (404/409/etc.) into ``make_error`` calls.
"""

import json

import anyio

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.scan.server import mcp
from osprey.mcp_server.scan.server_context import (
    UNKNOWN_RUN_HINTS,
    _http_get_json,
    _http_post_json,
    bridge_error_message,
)


# ---------------------------------------------------------------------------
# Tool 1: create scan intent
# ---------------------------------------------------------------------------
@mcp.tool()
async def create_scan_intent(plan_name: str, plan_args: dict | None = None) -> str:
    """Record a scan intent on the bridge — validated but NOT started.

    Motion-safe: creating an intent never touches a device or starts the
    RunEngine. It records a request the operator or agent can inspect via
    scan_status before deciding whether to call launch_scan — the sole
    promote path, which requires both an armed bridge token and
    ``control_system.writes_enabled``.

    Args:
        plan_name: Name of a plan registered on the bridge (e.g. "scan",
            "count", "grid_scan" for the v1 bluesky built-ins — see
            list_scan_plans for what this bridge actually supports).
        plan_args: Plan parameters as a JSON-serializable dict (device
            names, points, exposure time, etc.); shape is plan-specific and
            validated by the bridge. Defaults to an empty dict.

    Returns:
        JSON run record, e.g. ``{"id": <run_id>, "status": "intent"}``.
    """
    payload = {"plan_name": plan_name, "plan_args": plan_args or {}}
    status, body = await anyio.to_thread.run_sync(_http_post_json, "/runs", payload)
    if status not in (200, 201):
        return make_error(
            "scan_intent_rejected",
            bridge_error_message(body, status),
            [
                "Check plan_name is registered on the bridge (see list_scan_plans).",
                "Check plan_args matches that plan's parameter schema.",
            ],
        )
    return json.dumps(body)


# ---------------------------------------------------------------------------
# Tool 2: scan status
# ---------------------------------------------------------------------------
@mcp.tool()
async def scan_status(run_id: str) -> str:
    """Get one run's current lifecycle status.

    Args:
        run_id: Run id returned by create_scan_intent or list_runs.

    Returns:
        JSON run record: ``{"id", "status", ["completion"], ["launched_by"],
        ["run_uid"], ["error"]}``. ``status`` is one of "intent", "running",
        "completed", "stopped", "error".
    """
    status, body = await anyio.to_thread.run_sync(_http_get_json, f"/runs/{run_id}")
    if status == 404:
        return make_error("unknown_run", bridge_error_message(body, status), UNKNOWN_RUN_HINTS)
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    return json.dumps(body)


# ---------------------------------------------------------------------------
# Tool 3: list scan plans
# ---------------------------------------------------------------------------
@mcp.tool()
async def list_scan_plans() -> str:
    """List the plans registered on the bridge.

    Returns:
        JSON ``{"status": "success", "plans": [...]}``. An empty list means
        the facility has not injected a plan module (or this bridge version
        does not yet support plan discovery).
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
        same shape as scan_status's response.
    """
    status, body = await anyio.to_thread.run_sync(_http_get_json, f"/runs?limit={limit}")
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    return json.dumps({"status": "success", "runs": body})


# ---------------------------------------------------------------------------
# Tool 5: read scan data (bounded)
# ---------------------------------------------------------------------------
@mcp.tool()
async def read_scan_data(
    run_id: str, max_rows: int = 100, offset: int | None = None, tail: bool = False
) -> str:
    """Read a bounded window of a run's data.

    Row-bounded by design: this never returns an unbounded table. Backed by
    the bridge's in-process live-row buffer (``GET /runs/{id}/data``), so
    reads work with no Tiled server — ``row_count``/``truncated`` describe
    the run's *true* total vs. what this window actually returned.

    Args:
        run_id: Run id returned by create_scan_intent or list_runs.
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
            "scan_data_not_ready",
            bridge_error_message(body, status),
            [
                "The run has not started yet, so there is no run_uid to read data for.",
                "Check scan_status; data becomes readable once the run is promoted and running.",
            ],
        )
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    return json.dumps(body)
