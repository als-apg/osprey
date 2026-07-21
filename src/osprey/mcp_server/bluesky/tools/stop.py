"""MCP tool: stop_run — the always-available halt.

Stopping is the safe direction: unlike launch_run, this tool carries no
writes_enabled gate and no launch token. The bridge's ``POST /runs/{id}/stop``
route is not token-gated either (see ``bluesky_bridge/app.py``) — halting is
always allowed, including on a pending run that was never launched. This tool is
still approval-gated (``permissions_ask``, see the ``bluesky`` ServerDefinition
in task 1.10) so a human sees every stop, but the kill switch (writes-disabled
deny loop) must never block it.
"""

from __future__ import annotations

import json

import anyio

from osprey.mcp_server.bluesky.server import mcp
from osprey.mcp_server.bluesky.server_context import (
    UNKNOWN_RUN_HINTS,
    _http_post_json,
    bridge_error_message,
)
from osprey.mcp_server.errors import make_error


@mcp.tool()
async def stop_run(run_id: str) -> str:
    """Abort a running plan — or stop a pending run that was never launched.

    The always-available halt. Not gated by control_system.writes_enabled —
    halting a run is the safe direction and must always be reachable, so the
    kill switch (writes-disabled deny loop) never blocks it. Still
    approval-gated so a human sees every stop request.

    Args:
        run_id: Run id returned by launch_run or list_runs.

    Returns:
        JSON run record with status "stopped" on success.
    """
    status, body = await anyio.to_thread.run_sync(_http_post_json, f"/runs/{run_id}/stop", {})
    if status == 404:
        return make_error("unknown_run", bridge_error_message(body, status), UNKNOWN_RUN_HINTS)
    if status == 409:
        return make_error(
            "run_stop_conflict",
            bridge_error_message(body, status),
            ["Check get_run for the run's current state."],
        )
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    return json.dumps(body)
