"""MCP tool: launch_scan — the sole promote path from a scan intent to a running scan.

Two safety layers gate this tool, both enforced BEFORE any HTTP call is made:

1. In-tool ``control_system.writes_enabled`` re-check (this module) — the
   authoritative layer. It mirrors ``ControlSystemConnector._writes_enabled``
   (``osprey/connectors/control_system/base.py:100-143``) exactly: same
   config key, same fail-closed except clause. Re-read fresh on every call
   (never cached), so a hook-bypassed invocation — even one carrying a valid
   ``BLUESKY_PROMOTE_TOKEN`` — is still refused whenever writes are disabled.
2. Client-side promote-token presence check — refuses locally, with no
   network call, if this MCP server process was never armed with a token.

Only once both pass does this POST ``/runs/{id}/promote`` with the
``X-Promote-Token`` header. The bridge's own ``security.verify_promote_token``
(server-side, see ``bluesky_bridge/security.py``) is defense-in-depth against a
caller that skips this tool entirely — it is not the primary guard.
"""

from __future__ import annotations

import json

import anyio

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.scan.server import mcp
from osprey.mcp_server.scan.server_context import (
    UNKNOWN_RUN_HINTS,
    _http_post_json,
    bridge_error_message,
    get_server_context,
)


def _writes_enabled() -> bool:
    """Fail-closed re-read of ``control_system.writes_enabled`` straight from config.

    Mirrors ``ControlSystemConnector._writes_enabled`` (same config key, same
    fail-closed except clause) so the scan write gate agrees with every other
    OSPREY write path on one on/off switch. Deliberately NOT cached on the
    ScanContext singleton — the whole point is a fresh read on every call.
    """
    try:
        from osprey.utils.config import get_config_value

        return bool(get_config_value("control_system.writes_enabled", False))
    except (FileNotFoundError, RuntimeError):
        return False


@mcp.tool()
async def launch_scan(run_id: str) -> str:
    """Promote a scan intent into a running scan. The sole write path in this server.

    Two safety layers must pass before any network call is made: this
    deployment's control_system.writes_enabled must re-read true (checked
    fresh on every call, never cached, so a hook-bypassed invocation is still
    refused when writes are disabled), and this MCP server must have been
    armed with a promote token (BLUESKY_PROMOTE_TOKEN). Only then is
    POST /runs/{run_id}/promote sent to the bridge with the X-Promote-Token
    header.

    Args:
        run_id: Run id returned by create_scan_intent or list_runs. Must
            currently be in "intent" status.

    Returns:
        JSON run record with status "running" on success.
    """
    if not _writes_enabled():
        return make_error(
            "writes_disabled",
            "Control-system writes are disabled in this deployment "
            "(control_system.writes_enabled=false in config.yml). launch_scan refused.",
            ["Set control_system.writes_enabled: true in config.yml to enable launch_scan."],
        )

    token = get_server_context().promote_token
    if not token:
        return make_error(
            "scan_promote_unarmed",
            "This scan MCP server has no BLUESKY_PROMOTE_TOKEN configured — "
            "launch_scan is refused client-side before contacting the bridge.",
            [
                "Set BLUESKY_PROMOTE_TOKEN (or scan.promote_token in config.yml) "
                "for this bridge instance."
            ],
        )

    # anyio's run_sync only forwards positional args, and `headers` is
    # keyword-only on `_http_post_json`, hence the lambda.
    status, body = await anyio.to_thread.run_sync(
        lambda: _http_post_json(f"/runs/{run_id}/promote", {}, headers={"X-Promote-Token": token})
    )
    if status == 404:
        return make_error("unknown_run", bridge_error_message(body, status), UNKNOWN_RUN_HINTS)
    if status == 403:
        return make_error(
            "scan_promote_forbidden",
            "The Bluesky bridge rejected the promote token.",
            [
                "Confirm BLUESKY_PROMOTE_TOKEN matches the bridge's configured token for this instance."
            ],
        )
    if status == 409:
        return make_error(
            "scan_promote_conflict",
            bridge_error_message(body, status),
            ["Check scan_status — the run may already be promoted, stopped, or mid-promotion."],
        )
    if status == 503:
        return make_error(
            "scan_promote_unarmed",
            bridge_error_message(body, status),
            [
                "The bridge itself has no BLUESKY_PROMOTE_TOKEN configured; contact the deployment operator."
            ],
        )
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    return json.dumps(body)
