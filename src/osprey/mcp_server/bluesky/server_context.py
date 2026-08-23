"""Bluesky MCP Server Context — bridge connection resolution and HTTP boundary.

Holds the resolved facility-side Bluesky bridge connection (base URL and launch
token) for the MCP server process and exposes the module-level HTTP primitives
every tool module uses to talk to the bridge. Centralizing the primitives here
means every tool gets identical ``bluesky_bridge_unreachable`` handling without
repeating a try/except around each HTTP call.

The bridge URL / launch token resolution itself and the ``bridge_error_message``
helper live in the shared leaf ``osprey.bluesky_bridge_connection``, imported by
both this MCP server and the bluesky-web sidecar so the two never drift on which
bridge instance or token they use (a safety-relevant bug class). This module
re-exports ``bridge_error_message`` so the tool modules (``read_tools``,
``launch``, ``stop``, ``draft``, ``authoring``) keep importing it from here
alongside the ``UNKNOWN_RUN_HINTS`` helper.

A deployment with two PLAN LANES has two bridges, and every primitive here
takes an optional ``lane`` naming which one a request is addressed to. Omitted,
it means the lane serving the target this SESSION is on — so a read, a draft
edit or a halt follows a session switch without each tool restating the rule,
while the two operations that put hardware in motion (``queue_add``,
``queue_start``) name their lane explicitly and bind the queue item to it. On a
single-lane deployment the whole lane axis short-circuits before any session
state is read, which is what keeps those deployments behaving exactly as they
did before lanes existed.

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
    LANE_ONE,
    UnknownBlueskyLaneError,
    discover_lane_keys,
    resolve_bridge_url,
    resolve_launch_token,
)
from osprey.bluesky_bridge_connection import (
    # Re-exported so the tool modules keep importing it from server_context.
    bridge_error_message as bridge_error_message,
)
from osprey.mcp_server.bluesky.lanes import REASON_UNKNOWN_LANE, resolve_lane_situation
from osprey.mcp_server.errors import make_error

logger = logging.getLogger("osprey.mcp_server.bluesky.server_context")

_TIMEOUT = 15.0  # seconds

_UNREACHABLE_HINTS = [
    "Confirm the facility Bluesky bridge process is running.",
    "Check the BLUESKY_BRIDGE_URL env var — it overrides the configured bridge URL.",
    "To change the configured URL, set bluesky.bridge_url in the build profile "
    "(profile.yml on the host), then rebuild and redeploy.",
]

# Shared by every tool module (read_tools, queue, stop): the hint attached to
# a 404 for a run id the manager no longer holds. Deliberately does NOT say
# the run's data is gone — the queue and its history are durable, and a run
# rotated out of that history can still be read by get_run_data from durable
# storage. A hint claiming otherwise would stop an agent from reading data
# that is right there.
UNKNOWN_RUN_HINTS = [
    "The queue manager no longer holds this run — usually rotated out of its history.",
    "This does not mean the data is gone: try get_run_data, which also reads durable storage.",
    "List the runs the manager still knows about with list_runs.",
]

# What to do about a lane this deployment does not render. Never a retry and
# never a fallback to another lane: the whole point of refusing here is that
# answering from a DIFFERENT lane's bridge would report on a different machine.
_UNKNOWN_LANE_HINTS = [
    "Read the lanes this deployment actually renders with queue_status — it names "
    "every lane, the control target each one drives, and which is active.",
    "A second plan lane is an opt-in deployment change (bluesky.second_lane in the "
    "build profile, then rebuild and redeploy); it is not something to retry into "
    "existence.",
]


# ---------------------------------------------------------------------------
# BridgeContext
# ---------------------------------------------------------------------------
class BridgeContext:
    """Resolved Bluesky bridge connection details for the current process.

    On a deployment with two plan lanes this holds one connection per lane. Lane
    1's URL and token are the plain ``bridge_url``/``launch_token`` attributes —
    the pre-lane names, unchanged, because lane 1 resolves exactly as it always
    did — and any further lane is resolved on first use and cached beside them.
    Caching is correct because both halves are render-time facts: which bridge a
    lane is and which token arms it are fixed when the deployment was built. The
    SESSION's target is not cached anywhere, because that is the one thing that
    moves at run time.
    """

    def __init__(self) -> None:
        self.bridge_url: str = DEFAULT_BRIDGE_URL
        self.launch_token: str | None = None
        #: Every lane this deployment renders, in render order. A one-element
        #: tuple IS the single-lane deployment, and it short-circuits every
        #: lane-addressing decision below.
        self.lane_keys: tuple[str, ...] = (LANE_ONE,)
        self._lane_urls: dict[str, str] = {}
        self._lane_tokens: dict[str, str | None] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Resolve bridge_url and launch_token from env with config.yml fallback.

        Delegates to the shared ``osprey.bluesky_bridge_connection`` resolvers so
        this MCP server and the bluesky-web sidecar agree on which bridge instance and
        token they use. Called once during create_server(); subsequent calls are
        no-ops.
        """
        if self._initialized:
            return

        self.bridge_url = resolve_bridge_url()
        self.launch_token = resolve_launch_token()
        self.lane_keys = discover_lane_keys()
        self._initialized = True
        logger.info(
            "BridgeContext: initialized (bridge_url=%s, launch_token_set=%s, lanes=%s)",
            self.bridge_url,
            self.launch_token is not None,
            ",".join(self.lane_keys),
        )

    @property
    def multi_lane(self) -> bool:
        """Whether this deployment renders more than one plan lane."""
        return len(self.lane_keys) > 1

    def addressed_lane(self, lane: str | None) -> str | None:
        """Which lane a call addresses: the named one, or the ACTIVE one.

        ``None`` in, on a single-lane deployment, is ``None`` out — there is one
        bridge, nothing to choose between, and no session state is read at all.
        That short-circuit is what keeps every single-lane deployment byte-for-byte
        what it was before lanes existed.

        ``None`` in, on a two-lane deployment, resolves the lane serving the
        SESSION's target, so every tool that names no lane follows the session
        rather than pinning itself to lane 1. When no single lane serves it (a
        misrendered pair), reads still have to answer something, and lane 1 —
        the deployment baseline's lane — is that answer; it is logged at WARNING
        rather than taken silently, because a read answered from a lane nobody
        selected is exactly the confusion the lane axis exists to remove and the
        operator needs to hear about the misrender. The two operations that put
        hardware in motion never rely on this fallback at all:
        ``tools/queue.py`` refuses outright in that case.
        """
        if lane is not None:
            return lane
        if not self.multi_lane:
            return None
        active = resolve_lane_situation().active
        if active is None:
            logger.warning(
                "No Bluesky plan lane serves this session's control target; answering "
                "reads from lane %r. The rendered lanes (%s) do not cover it — that is a "
                "deployment misrender, not a session problem.",
                LANE_ONE,
                ",".join(self.lane_keys),
            )
            return LANE_ONE
        return active.key

    def bridge_url_for(self, lane: str | None = None) -> str:
        """The base URL of the bridge serving *lane* (default: the active lane).

        :raises UnknownBlueskyLaneError: *lane* is not a lane this deployment
            renders. Never falls back to another lane's bridge.
        """
        key = self.addressed_lane(lane)
        if key is None or key == LANE_ONE:
            return self.bridge_url
        if key not in self._lane_urls:
            self._lane_urls[key] = resolve_bridge_url(key)
        return self._lane_urls[key]

    def launch_token_for(self, lane: str | None = None) -> str | None:
        """The launch token arming *lane* (default: the active lane).

        Per lane because the token is what ARMS: one token honoured by both
        lanes would let a launch a human approved against one machine be
        replayed against the other.

        :raises UnknownBlueskyLaneError: *lane* is not a lane key.
        """
        key = self.addressed_lane(lane)
        if key is None or key == LANE_ONE:
            return self.launch_token
        if key not in self._lane_tokens:
            self._lane_tokens[key] = resolve_launch_token(key)
        return self._lane_tokens[key]


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


def addressed_lane_key(lane: str | None = None) -> str:
    """The lane a request reaches, as a NAME — never ``None``.

    :meth:`BridgeContext.addressed_lane` answers ``None`` for "the only lane
    there is", which is the right answer for URL resolution and the wrong one
    for anything that has to REPORT which lane it used. Tools that put the lane
    in their result (the draft tools, the queue tools) name it through here, so
    the lane they report is by construction the lane they addressed.

    An uninitialized context answers lane 1 rather than raising: no context is a
    process with no lane axis at all, and a lane LABEL is never a reason for a
    tool to fail before it has even tried its request.
    """
    try:
        context = get_server_context()
    except RuntimeError:
        return LANE_ONE
    return context.addressed_lane(lane) or LANE_ONE


# ---------------------------------------------------------------------------
# HTTP boundary (patched in tests)
# ---------------------------------------------------------------------------
def _request_json(
    request: Callable[..., httpx.Response],
    path: str,
    *,
    lane: str | None = None,
    timeout: float | None = None,
    **kwargs: Any,
) -> tuple[int, Any]:
    """Shared core of the ``_http_*_json`` helpers: dispatch, parse, unreachable handling.

    ``request`` is the ``httpx`` verb function to call. The public wrappers
    below look it up (``httpx.get``/``httpx.post``/...) at call time, so tests
    that patch those module attributes still intercept the request.

    ``lane`` names which PLAN LANE the request is addressed to. Omitted — which
    is every call site that has no lane of its own — it means the ACTIVE lane:
    on a single-lane deployment the only bridge there is, and on a two-lane one
    the bridge serving the target this session is pointed at. That default is
    what makes every read, draft edit and halt follow a session switch without
    each tool restating the rule.

    ``timeout`` overrides :data:`_TIMEOUT` for one call. It exists for the
    handful of bridge routes whose server-side work is a composition of manager
    calls rather than a single one -- the emergency abort behind ``stop_run`` --
    where the shared default would report an unreachable bridge while the bridge
    was still working. It is per-call on purpose: raising ``_TIMEOUT`` globally
    would make every ordinary tool sit on a dead bridge for minutes.
    """
    try:
        base_url = get_server_context().bridge_url_for(lane)
    except UnknownBlueskyLaneError as exc:
        # A structured refusal, never an unhandled error: an agent that named a
        # lane this deployment does not render is entitled to the same
        # machine-readable answer it gets for every other refused request.
        make_error(REASON_UNKNOWN_LANE, str(exc), _UNKNOWN_LANE_HINTS)
    url = f"{base_url}{path}"
    try:
        resp = request(url, timeout=_TIMEOUT if timeout is None else timeout, **kwargs)
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


def _http_get_json(path: str, *, lane: str | None = None) -> tuple[int, dict | list]:
    """GET ``path`` on the Bluesky bridge and return ``(status, parsed_json)``.

    Raises ``ToolError`` via ``make_error("bluesky_bridge_unreachable", ...)`` when
    the bridge cannot be reached at all, so every tool gets identical
    unreachable-bridge handling. HTTP error responses (4xx/5xx) are returned
    to the caller as ``(status, parsed_body)`` so tools can render the
    bridge's own error semantics (404/409/403/503).

    ``lane`` addresses one plan lane; omitted, it is the active lane (see
    :func:`_request_json`).
    """
    return _request_json(httpx.get, path, lane=lane)


def _http_post_json(
    path: str,
    payload: dict,
    *,
    headers: dict[str, str] | None = None,
    lane: str | None = None,
    timeout: float | None = None,
) -> tuple[int, dict]:
    """POST ``payload`` as JSON to ``path`` on the Bluesky bridge.

    Same unreachable-bridge/error-body contract as :func:`_http_get_json`.
    ``timeout`` overrides the shared :data:`_TIMEOUT` for this one request; see
    :func:`_request_json`.
    """
    return _request_json(
        httpx.post, path, lane=lane, json=payload, headers=headers, timeout=timeout
    )


def _http_patch_json(
    path: str, payload: dict, *, headers: dict[str, str] | None = None, lane: str | None = None
) -> tuple[int, dict]:
    """PATCH ``payload`` as JSON to ``path`` on the Bluesky bridge.

    Same unreachable-bridge/error-body contract as :func:`_http_get_json`.
    """
    return _request_json(httpx.patch, path, lane=lane, json=payload, headers=headers)


def _http_delete_json(
    path: str, *, headers: dict[str, str] | None = None, lane: str | None = None
) -> tuple[int, dict]:
    """DELETE ``path`` on the Bluesky bridge.

    Same unreachable-bridge/error-body contract as :func:`_http_get_json`.
    """
    return _request_json(httpx.delete, path, lane=lane, headers=headers)
