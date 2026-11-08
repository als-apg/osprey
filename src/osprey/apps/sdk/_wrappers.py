"""Public names for framework helpers whose own names are private or internal."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Literal

from osprey.apps import manifest
from osprey.apps.sdk._approval import audit_surface
from osprey.utils.launch_token import LAUNCH_TOKEN_HEADER, verify_launch_token
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


def launch_token_env() -> str:
    """The environment variable the deployment hands this application its launch token in.

    Derived from the application ``OSPREY_APP_NAME`` names, by the same rule the
    build uses, so a backend never spells it: pass it to
    :func:`verify_launch_token` to check a route's caller.

    Raises:
        ValueError: ``OSPREY_APP_NAME`` is unset or is not an application name.

    Stable: part of the application SDK's public surface.
    """
    return manifest.launch_token_env(audit_surface())


def caller_kind(headers: Mapping[str, str]) -> Literal["agent", "panel"]:
    """Who the terminal proxy says is calling: ``agent`` or ``panel``.

    *headers* is the request's header mapping; names are matched
    case-insensitively. The proxy strips any inbound ``X-Osprey-Caller`` and
    stamps its own beside the application's launch token, so the kind is read
    only when the same request carries that token (the one
    :func:`launch_token_env` names). A caller header on a request without the
    token is a claim nobody vouched for.

    Raises:
        ValueError: ``OSPREY_APP_NAME`` or the launch token is not configured,
            or the request carries no token or a different one; or the caller
            header is absent or names neither kind.

    Stable: part of the application SDK's public surface.
    """
    from fastapi import HTTPException

    token_env = launch_token_env()
    lowered = {k.lower(): v for k, v in headers.items()}
    try:
        verify_launch_token(lowered.get(LAUNCH_TOKEN_HEADER.lower()), token_env)
    except HTTPException as exc:
        detail = exc.detail.get("detail") if isinstance(exc.detail, dict) else exc.detail
        raise ValueError(
            f"{CALLER_HEADER} is trusted only beside a valid {LAUNCH_TOKEN_HEADER}: {detail}"
        ) from exc
    kind = caller_from_header(lowered.get(CALLER_HEADER.lower()))
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
