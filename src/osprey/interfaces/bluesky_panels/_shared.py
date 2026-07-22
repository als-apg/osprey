"""Shared constants and helpers for the Bluesky panels sidecar routers.

The three bridge-facing routers (``read_proxy``, ``draft_relay``, ``launch``)
each translate a connection-level bridge failure into the same 502 body and
each parse a bridge response body defensively. Both were inlined per module;
this leaf keeps the one constant and the one safe-parse helper so the routers
stay in agreement.
"""

from __future__ import annotations

from typing import Any

import httpx

# The 502 body every bridge-facing router returns when the bridge cannot be
# reached at all (refused connection, DNS failure, timeout).
UNREACHABLE_BODY: dict[str, str] = {"detail": "bluesky bridge unreachable"}


def safe_json(response: httpx.Response, default: Any = None) -> Any:
    """Parse ``response`` as JSON, returning ``default`` on a non-JSON body.

    Each call site passes the default its downstream logic expects (``None``
    for the passthrough relays, ``{}`` for the launch route, which then indexes
    the result), so the caller's original behavior is preserved exactly.
    """
    try:
        return response.json()
    except ValueError:
        return default
