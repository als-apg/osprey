"""Bluesky MCP Server Context — bridge connection resolution and HTTP boundary.

Holds the resolved facility-side Bluesky bridge connection (base URL and launch
token) for the MCP server process and exposes the module-level HTTP primitives
every tool module uses to talk to the bridge. Centralizing the primitives here
means every tool gets identical ``bluesky_bridge_unreachable`` handling without
repeating a try/except around each HTTP call.

The bridge URL / launch token resolution itself and the ``bridge_error_message``
helper live in the shared leaf ``osprey.bluesky_bridge_connection``, imported by
both this MCP server and the panels sidecar so the two never drift on which
bridge instance or token they use (a safety-relevant bug class). This module
re-exports ``bridge_error_message`` so the tool modules (``read_tools``,
``launch``, ``stop``, ``draft``, ``authoring``) keep importing it from here
alongside the ``UNKNOWN_RUN_HINTS`` helper.

Usage in tools:
    from osprey.mcp_server.bluesky.server_context import _http_get_json, _http_post_json

    status, body = _http_get_json("/runs")
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import httpx

from osprey.bluesky_bridge_connection import (
    DEFAULT_BRIDGE_URL,
    resolve_bridge_url,
    resolve_launch_token,
)
from osprey.bluesky_bridge_connection import (
    # Re-exported so the tool modules keep importing it from server_context.
    bridge_error_message as bridge_error_message,
)
from osprey.mcp_server.errors import make_error

logger = logging.getLogger("osprey.mcp_server.bluesky.server_context")

_TIMEOUT = 15.0  # seconds

_UNREACHABLE_HINTS = [
    "Confirm the facility Bluesky bridge process is running.",
    "Check the BLUESKY_BRIDGE_URL env var or bluesky.bridge_url in config.yml.",
]

# Shared by every tool module (read_tools, launch, stop): the hint
# attached to a 404 from the bridge's in-memory run registry.
UNKNOWN_RUN_HINTS = [
    "The bridge's run registry is in-memory only; a restart forgets prior runs.",
    "List currently-tracked runs with list_runs.",
]


# ---------------------------------------------------------------------------
# BridgeContext
# ---------------------------------------------------------------------------
class BridgeContext:
    """Resolved Bluesky bridge connection details for the current process."""

    def __init__(self) -> None:
        self.bridge_url: str = DEFAULT_BRIDGE_URL
        self.launch_token: str | None = None
        self._initialized = False

    def initialize(self) -> None:
        """Resolve bridge_url and launch_token from env with config.yml fallback.

        Delegates to the shared ``osprey.bluesky_bridge_connection`` resolvers so
        this MCP server and the panels sidecar agree on which bridge instance and
        token they use. Called once during create_server(); subsequent calls are
        no-ops.
        """
        if self._initialized:
            return

        self.bridge_url = resolve_bridge_url()
        self.launch_token = resolve_launch_token()
        self._initialized = True
        logger.info(
            "BridgeContext: initialized (bridge_url=%s, launch_token_set=%s)",
            self.bridge_url,
            self.launch_token is not None,
        )


# ---------------------------------------------------------------------------
# Module-level singleton (mirrors osprey.mcp_server.control_system.server_context)
# ---------------------------------------------------------------------------

_context: BridgeContext | None = None


def get_server_context() -> BridgeContext:
    """Get the BridgeContext singleton.

    Raises RuntimeError if initialize_server_context() hasn't been called.
    """
    if _context is None:
        raise RuntimeError(
            "Bluesky server context not initialized. Call initialize_server_context() first."
        )
    return _context


def initialize_server_context() -> BridgeContext:
    """Create and initialize the BridgeContext singleton."""
    global _context
    _context = BridgeContext()
    _context.initialize()
    return _context


def reset_server_context() -> None:
    """Reset the BridgeContext singleton (for testing)."""
    global _context
    _context = None


# ---------------------------------------------------------------------------
# HTTP boundary (patched in tests)
# ---------------------------------------------------------------------------
def _request_json(
    request: Callable[..., httpx.Response], path: str, **kwargs: Any
) -> tuple[int, Any]:
    """Shared core of the ``_http_*_json`` helpers: dispatch, parse, unreachable handling.

    ``request`` is the ``httpx`` verb function to call. The public wrappers
    below look it up (``httpx.get``/``httpx.post``/...) at call time, so tests
    that patch those module attributes still intercept the request.
    """
    url = f"{get_server_context().bridge_url}{path}"
    try:
        resp = request(url, timeout=_TIMEOUT, **kwargs)
    except httpx.HTTPError as exc:
        make_error(
            "bluesky_bridge_unreachable",
            f"Could not reach the Bluesky bridge: {exc}",
            _UNREACHABLE_HINTS,
        )

    body: dict | list = {}
    try:
        body = resp.json()
    except Exception:
        pass
    return resp.status_code, body


def _http_get_json(path: str) -> tuple[int, dict | list]:
    """GET ``path`` on the Bluesky bridge and return ``(status, parsed_json)``.

    Raises ``ToolError`` via ``make_error("bluesky_bridge_unreachable", ...)`` when
    the bridge cannot be reached at all, so every tool gets identical
    unreachable-bridge handling. HTTP error responses (4xx/5xx) are returned
    to the caller as ``(status, parsed_body)`` so tools can render the
    bridge's own error semantics (404/409/403/503).
    """
    return _request_json(httpx.get, path)


def _http_post_json(
    path: str, payload: dict, *, headers: dict[str, str] | None = None
) -> tuple[int, dict]:
    """POST ``payload`` as JSON to ``path`` on the Bluesky bridge.

    Same unreachable-bridge/error-body contract as :func:`_http_get_json`.
    """
    return _request_json(httpx.post, path, json=payload, headers=headers)


def _http_patch_json(
    path: str, payload: dict, *, headers: dict[str, str] | None = None
) -> tuple[int, dict]:
    """PATCH ``payload`` as JSON to ``path`` on the Bluesky bridge.

    Same unreachable-bridge/error-body contract as :func:`_http_get_json`.
    """
    return _request_json(httpx.patch, path, json=payload, headers=headers)


def _http_delete_json(path: str, *, headers: dict[str, str] | None = None) -> tuple[int, dict]:
    """DELETE ``path`` on the Bluesky bridge.

    Same unreachable-bridge/error-body contract as :func:`_http_get_json`.
    """
    return _request_json(httpx.delete, path, headers=headers)
