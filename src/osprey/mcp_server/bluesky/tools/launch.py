"""MCP tool: launch_run — the agent's launch-from-draft path, the sole write in this server.

Two safety layers gate this tool, both enforced BEFORE any HTTP call is made:

1. In-tool ``control_system.writes_enabled`` re-check (this module) — the
   authoritative layer. It mirrors ``ControlSystemConnector._writes_enabled``
   (``osprey/connectors/control_system/base.py:100-143``) exactly: same
   config key, same fail-closed except clause. Re-read fresh on every call
   (never cached), so a hook-bypassed invocation — even one carrying a valid
   launch token — is still refused whenever writes are disabled.
2. Client-side launch-token presence check — refuses locally, with no
   network call, if this MCP server process was never armed with a token.

Only once both pass does this POST ``/draft/run`` with the ``X-Launch-Token``
header, pinning the ``draft_revision`` the caller staged with set_draft (and
read back with get_draft). This is the agent analog of the plan panel's Launch
plan button: the panel-visible draft is the only staging surface, and this tool is
the agent's way of launching it. The bridge's own ``security.verify_launch_token``
(server-side) re-verifies the token as defense-in-depth against a caller that
skips this tool entirely, and enforces that the pinned revision is still the
current, never-launched draft.
"""

from __future__ import annotations

import json

import anyio

from osprey.bluesky_bridge_connection import unwrap_bridge_conflict_detail
from osprey.mcp_server.bluesky.server import mcp
from osprey.mcp_server.bluesky.server_context import (
    _http_post_json,
    bridge_error_message,
    get_server_context,
)
from osprey.mcp_server.errors import make_error


def _writes_enabled() -> bool:
    """Fail-closed re-read of ``control_system.writes_enabled`` straight from config.

    Mirrors ``ControlSystemConnector._writes_enabled`` (same config key, same
    fail-closed except clause) so the launch write gate agrees with every other
    OSPREY write path on one on/off switch. Deliberately NOT cached on the
    BridgeContext singleton — the whole point is a fresh read on every call.
    """
    try:
        from osprey.utils.config import get_config_value

        return bool(get_config_value("control_system.writes_enabled", False))
    except (FileNotFoundError, RuntimeError):
        return False


@mcp.tool()
async def launch_run(draft_revision: int) -> str:
    """Launch the shared plan draft at a pinned revision. The sole write path in this server.

    CONSEQUENCE: this is the one write-arming action in this server. It arms
    control-system writes and, after the human approves the prompt, starts a
    real scan on the machine — moving real devices — from whatever the draft
    holds at ``draft_revision``. It is the agent analog of the plan panel's
    Launch plan button: it launches exactly the single, panel-visible draft the
    human can see, so confirm the draft is complete and correct (get_draft)
    before launching. Nothing is queued and nothing is speculative — this run
    starts.

    Two safety layers must pass before any network call is made: this
    deployment's control_system.writes_enabled must re-read true (checked fresh
    on every call, never cached, so a hook-bypassed invocation is still refused
    when writes are disabled), and this MCP server must have been armed with a
    launch token (BLUESKY_LAUNCH_TOKEN). Only then is POST /draft/run sent to
    the bridge with the X-Launch-Token header, and the human still sees an
    approval prompt for the launch.

    Args:
        draft_revision: The draft revision to launch, as returned by get_draft
            or set_draft. The bridge launches the draft snapshot pinned at this
            exact revision.

    Returns:
        JSON run record with status "running" and launched_by "draft" on success.

    Refusals (all returned as error envelopes, nothing launched):
        - writes_disabled: control_system.writes_enabled is false in this
          deployment. Not recoverable by the agent; the operator must enable
          writes in config.yml.
        - run_launch_unarmed: this MCP server (or the bridge) has no launch
          token configured. Not recoverable by the agent; contact the operator.
        - run_launch_forbidden: the bridge rejected the launch token. The
          server's token does not match the bridge's; contact the operator.
        - run_launch_conflict (HTTP 409), two cases carried in ``details.code``:
            * stale_draft_revision — the draft changed after you pinned this
              revision. Re-read get_draft for the fresh revision and re-pin.
            * draft_revision_already_launched — this revision was already
              launched once. Edit the draft with set_draft to mint a new
              revision, then launch that one.
    """
    if not _writes_enabled():
        return make_error(
            "writes_disabled",
            "Control-system writes are disabled in this deployment "
            "(control_system.writes_enabled=false in config.yml). launch_run refused.",
            ["Set control_system.writes_enabled: true in config.yml to enable launch_run."],
        )

    token = get_server_context().launch_token
    if not token:
        return make_error(
            "run_launch_unarmed",
            "This Bluesky MCP server has no BLUESKY_LAUNCH_TOKEN configured — "
            "launch_run is refused client-side before contacting the bridge.",
            [
                "Set BLUESKY_LAUNCH_TOKEN (or bluesky.launch_token in config.yml) "
                "for this bridge instance."
            ],
        )

    # anyio's run_sync only forwards positional args, and `headers` is
    # keyword-only on `_http_post_json`, hence the lambda.
    status, body = await anyio.to_thread.run_sync(
        lambda: _http_post_json(
            "/draft/run",
            {"draft_revision": draft_revision},
            headers={"X-Launch-Token": token},
        )
    )
    if status == 403:
        return make_error(
            "run_launch_forbidden",
            "The Bluesky bridge rejected the launch token.",
            [
                "Confirm BLUESKY_LAUNCH_TOKEN matches the bridge's configured token for this instance."
            ],
        )
    if status == 409:
        return _conflict_error(body, status)
    if status == 503:
        return make_error(
            "run_launch_unarmed",
            bridge_error_message(body, status),
            [
                "The bridge itself has no BLUESKY_LAUNCH_TOKEN configured; contact the deployment operator."
            ],
        )
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    return json.dumps(body)


def _conflict_error(body: object, status: int) -> str:
    """Render a 409 from ``POST /draft/run`` into a ``run_launch_conflict`` envelope.

    The bridge nests a ``{"code", "detail", "revision"}`` payload under FastAPI's
    top-level ``detail`` key. Surface the bridge's ``code`` and fresh ``revision``
    baseline verbatim in the error ``details`` so the agent can tell a stale pin
    (re-read get_draft and re-pin) from an already-consumed revision (edit the
    draft to mint a new one) and re-pin against the returned baseline. Some 409s
    (e.g. a validation-gate rejection) carry a plain-string detail instead — those
    have no code to surface, so fall back to the raw bridge message.
    """
    detail = unwrap_bridge_conflict_detail(body)
    if detail is not None:
        code = detail.get("code")
        revision = detail.get("revision")
        message = detail.get("detail") or f"The Bluesky bridge refused the launch: {code}."
        if code == "draft_revision_already_launched":
            hints = [
                "This draft revision was already launched; edit the draft with set_draft "
                "to create a new revision, then launch that one."
            ]
        else:
            hints = [
                "The draft changed since this revision; re-read it with get_draft and "
                "launch its current revision."
            ]
        return make_error(
            "run_launch_conflict",
            str(message),
            hints,
            details={"code": code, "revision": revision},
        )
    return make_error(
        "run_launch_conflict",
        bridge_error_message(body, status),
        ["Re-read the draft with get_draft before launching again."],
    )
