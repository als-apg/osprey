"""Public names for framework helpers whose own names are private or internal."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Literal

from osprey.utils.owner_header import CALLER_HEADER, caller_from_header


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


def caller_kind(headers: Mapping[str, str]) -> Literal["agent", "panel"]:
    """Who the terminal proxy says is calling: ``agent`` or ``panel``.

    *headers* is the request's header mapping; the name is matched
    case-insensitively. The proxy strips any inbound ``X-Osprey-Caller`` and
    stamps its own on every request it forwards, so a backend behind it always
    receives one.

    Raises:
        ValueError: The header is absent or names neither kind, so the request
            did not come through the proxy.

    Stable: part of the application SDK's public surface.
    """
    wanted = CALLER_HEADER.lower()
    value = next((v for k, v in headers.items() if k.lower() == wanted), None)
    kind = caller_from_header(value)
    if kind is None:
        raise ValueError(
            f"{CALLER_HEADER} is absent or names neither 'agent' nor 'panel': "
            "the request did not come through the terminal proxy"
        )
    return kind


def queue_is_active(status: dict[str, Any]) -> bool:
    """True if a queue manager *status* reports a plan running or about to run.

    Stable: part of the application SDK's public surface. The queue client
    library loads only when this is called.
    """
    from osprey.services.bluesky_bridge.queue_backend import is_queue_active

    return is_queue_active(status)
