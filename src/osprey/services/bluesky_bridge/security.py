"""The Bluesky bridge's launch-token gate.

``verify_launch_token`` guards the bridge's arming routes — enqueuing onto an
already-draining queue, and ``POST /queue/start``/``POST /queue/stop``'s
cancel (all in ``queue.py``), the only routes that can set hardware moving.
Fail-closed by design: if ``BLUESKY_LAUNCH_TOKEN`` isn't set in the bridge
process's environment, no route is armed, regardless of what header a caller
sends. Mirrors BELLA's ``runs.require_armed`` / ``launch_intent``
token check.

**What the token is.** Authentication for network callers on a shared
loopback port. The bridge binds to loopback, but loopback is shared: every
process running as the same user on that host can open the port. The token
distinguishes the caller that was handed it (the Bluesky MCP server) from
any other process that can reach the socket. It is not a secret the agent
is unable to obtain — agent-authored Python runs on the host as the same
user and can read the deploy ``.env`` or ``config.yml``'s
``bluesky.launch_token``. Treat the token as caller authentication, never as
the thing that stops a scan from moving hardware.

**Where the security boundary is.** The control-system connector. Every
setpoint a running plan writes goes through
``connector.write_channel_checked`` (``devices/connector.py``), which
re-reads ``control_system.writes_enabled`` and applies the configured limits
on each individual put, raising on refusal. A caller that stole or forged
the token therefore still cannot drive a device past what policy allows. A
writable bridge additionally refuses to start at all when limits checking is
enabled but its database is missing or unparseable
(``validation._assert_limits_readable_if_writable``).

**The approval prompt is best-effort UX defense, on both write paths.** The
human approval prompt on the queue's arming tools and their in-tool
``writes_enabled`` re-read (``osprey/mcp_server/bluesky/tools/queue.py``)
are worth having, but they are not a boundary: code the agent authors and
runs through the python executor can POST these routes directly, or drive
channels directly, without any prompt. Both paths converge on the connector,
and only the connector is enforced.

**The accepted trade.** On a bare-host, writes-enabled deploy the launch
approval prompt is best-effort — exactly as it already is on the Python
execution path. OSPREY accepts that rather than pretending otherwise: the
launch token is minted for every deployed bridge unconditionally, so the two
write paths carry the same posture and no configuration implies a
containment the host does not provide. The connector is the boundary.
"""

from __future__ import annotations

import os
import secrets

from fastapi import HTTPException

_LAUNCH_TOKEN_ENV = "BLUESKY_LAUNCH_TOKEN"


def require_armed() -> str:
    """Return the configured launch token, or raise 503 if none is set.

    Unset means the operator has not armed launch for this bridge instance
    at all — the failure detail names the missing env var but never a token
    value (there isn't one to leak).
    """
    expected = os.environ.get(_LAUNCH_TOKEN_ENV)
    if not expected:
        raise HTTPException(
            status_code=503,
            detail=f"launch disabled: {_LAUNCH_TOKEN_ENV} is not configured",
        )
    return expected


def verify_launch_token(header: str | None) -> None:
    """Raise 503 (unarmed) or 403 (mismatch) unless ``header`` matches the armed token.

    Compares in constant time via :func:`secrets.compare_digest` so a mismatch
    can't be distinguished by timing. The error details never echo the
    expected or received token.
    """
    expected = require_armed()
    if not secrets.compare_digest(header or "", expected):
        raise HTTPException(status_code=403, detail="invalid or missing launch token")
