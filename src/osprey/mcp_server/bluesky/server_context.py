"""Bluesky MCP Server Context — bridge connection resolution and HTTP boundary.

Resolves the facility-side Bluesky bridge's base URL and promote token (env with
config.yml fallback, mirroring
``osprey.mcp_server.control_system.server_context``), and exposes the
module-level HTTP primitives every tool module uses to talk to the
bridge. Centralizing the primitives here means every tool gets identical
``bluesky_bridge_unreachable`` handling without repeating a try/except around
each HTTP call.

Also home to the ``bridge_error_message`` / ``UNKNOWN_RUN_HINTS`` helpers
shared by every tool module (``read_tools``, ``launch``, ``stop``) for
translating a non-2xx bridge response into a ``make_error`` call, so all three
render the same error envelope for the same bridge failure.

Usage in tools:
    from osprey.mcp_server.bluesky.server_context import _http_get_json, _http_post_json

    status, body = _http_get_json("/runs")
"""

from __future__ import annotations

import logging
import os

import httpx

from osprey.mcp_server.errors import make_error

logger = logging.getLogger("osprey.mcp_server.bluesky.server_context")

_DEFAULT_BRIDGE_URL = "http://127.0.0.1:8090"
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


def bridge_error_message(body: object, status: int) -> str:
    """Extract the bridge's FastAPI ``detail`` message, falling back to the status."""
    if isinstance(body, dict) and body.get("detail"):
        return str(body["detail"])
    return f"Bluesky bridge returned HTTP {status}."


# ---------------------------------------------------------------------------
# BridgeContext
# ---------------------------------------------------------------------------
class BridgeContext:
    """Resolved Bluesky bridge connection details for the current process."""

    def __init__(self) -> None:
        self.bridge_url: str = _DEFAULT_BRIDGE_URL
        self.promote_token: str | None = None
        self._initialized = False

    def initialize(self) -> None:
        """Resolve bridge_url and promote_token from env with config.yml fallback.

        Called once during create_server(). Subsequent calls are no-ops.
        """
        if self._initialized:
            return

        self.bridge_url = self._resolve_bridge_url()
        self.promote_token = self._resolve_promote_token()
        self._initialized = True
        logger.info(
            "BridgeContext: initialized (bridge_url=%s, promote_token_set=%s)",
            self.bridge_url,
            self.promote_token is not None,
        )

    @staticmethod
    def _resolve_bridge_url() -> str:
        """Resolve the Bluesky bridge base URL.

        Resolution order:

        1. ``BLUESKY_BRIDGE_URL`` env var (full URL) — set by the framework server
           definition per bridge instance; wins outright.
        2. ``bluesky.bridge_url`` in config.yml.
        3. ``http://127.0.0.1:8090`` default.
        """
        full = os.environ.get("BLUESKY_BRIDGE_URL")
        if full:
            return full.rstrip("/")

        from osprey.utils.workspace import load_osprey_config

        config = load_osprey_config()
        url = config.get("bluesky", {}).get("bridge_url", _DEFAULT_BRIDGE_URL)
        return str(url).rstrip("/")

    @staticmethod
    def _resolve_promote_token() -> str | None:
        """Resolve the Bluesky bridge promote token.

        Resolution order:

        1. ``BLUESKY_PROMOTE_TOKEN`` env var — minted fail-closed per bridge
           instance by the framework server definition; wins outright.
        2. ``bluesky.promote_token`` in config.yml (local/dev convenience only).
        3. ``None`` — ``launch_run`` refuses client-side when unset.
        """
        token = os.environ.get("BLUESKY_PROMOTE_TOKEN")
        if token:
            return token

        from osprey.utils.workspace import load_osprey_config

        config = load_osprey_config()
        token = config.get("bluesky", {}).get("promote_token")
        return str(token) if token else None


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
def _http_get_json(path: str) -> tuple[int, dict | list]:
    """GET ``path`` on the Bluesky bridge and return ``(status, parsed_json)``.

    Raises ``ToolError`` via ``make_error("bluesky_bridge_unreachable", ...)`` when
    the bridge cannot be reached at all, so every tool gets identical
    unreachable-bridge handling. HTTP error responses (4xx/5xx) are returned
    to the caller as ``(status, parsed_body)`` so tools can render the
    bridge's own error semantics (404/409/403/503).
    """
    url = f"{get_server_context().bridge_url}{path}"
    try:
        resp = httpx.get(url, timeout=_TIMEOUT)
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


def _http_post_json(
    path: str, payload: dict, *, headers: dict[str, str] | None = None
) -> tuple[int, dict]:
    """POST ``payload`` as JSON to ``path`` on the Bluesky bridge.

    Same unreachable-bridge/error-body contract as :func:`_http_get_json`.
    """
    url = f"{get_server_context().bridge_url}{path}"
    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=_TIMEOUT)
    except httpx.HTTPError as exc:
        make_error(
            "bluesky_bridge_unreachable",
            f"Could not reach the Bluesky bridge: {exc}",
            _UNREACHABLE_HINTS,
        )

    body: dict = {}
    try:
        body = resp.json()
    except Exception:
        pass
    return resp.status_code, body
