"""Public names for framework helpers whose own names are private or internal."""

from __future__ import annotations

import os
from typing import Any


def agent_panel_headers() -> dict[str, str]:
    """The headers an agent's call to a panel route carries.

    The ``Authorization`` header the agent's panel token gives, and the
    ``X-Osprey-Session`` header naming the agent's own session, inherited from
    its spawn's ``OSPREY_SESSION_ID`` (absent when that is unset or blank). The
    terminal refuses a write from a held session, or one naming no live
    session, before it reaches the application.

    Stable: part of the application SDK's public surface.
    """
    from osprey.mcp_server.http import _panel_auth_headers
    from osprey.utils.owner_header import SESSION_HEADER

    headers = dict(_panel_auth_headers())
    session = os.environ.get("OSPREY_SESSION_ID", "").strip()
    if session:
        headers[SESSION_HEADER] = session
    return headers


def queue_is_active(status: dict[str, Any]) -> bool:
    """True if a queue manager *status* reports a plan running or about to run.

    Stable: part of the application SDK's public surface. The queue client
    library loads only when this is called.
    """
    from osprey.services.bluesky_bridge.queue_backend import is_queue_active

    return is_queue_active(status)
