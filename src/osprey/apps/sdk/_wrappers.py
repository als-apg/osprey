"""Public names for framework helpers whose own names are private or internal."""

from __future__ import annotations

from typing import Any


def agent_panel_headers() -> dict[str, str]:
    """The ``Authorization`` header the agent's panel token gives a panel route.

    Stable: part of the application SDK's public surface.
    """
    from osprey.mcp_server.http import _panel_auth_headers

    return _panel_auth_headers()


def queue_is_active(status: dict[str, Any]) -> bool:
    """True if a queue manager *status* reports a plan running or about to run.

    Stable: part of the application SDK's public surface. The queue client
    library loads only when this is called.
    """
    from osprey.services.bluesky_bridge.queue_backend import is_queue_active

    return is_queue_active(status)
