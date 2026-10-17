"""The trajectory server's HTTP boundary and the agent's session.

Every request goes to the web terminal's proxy at :data:`BASE_PATH` with the
panel token (:func:`~osprey.mcp_server.http._panel_auth_headers`). The proxy
strips any inbound credential and injects the service's launch token itself,
so nothing here reads or sends one.

The agent holds at most one session on the service. It is opened on the first
write, kept for the server's life, and released when the server exits. A
session the same owner's agent left behind (a server that died without
releasing) is taken over rather than refused; one held by anyone else is
refused with the service's own answer.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, NoReturn

import httpx

from osprey.mcp_server.errors import make_error
from osprey.services.trajectory.constants import AGENT_DOOR, CLIENT_HEADER

logger = logging.getLogger("osprey.mcp_server.trajectory.client")

#: The agent door of the trajectory service, as the terminal proxies it.
BASE_PATH = "/panel/trajectory/api"

#: The service's refusal codes this module acts on.
SESSION_HELD = "session_held"
NO_SESSION = "no_session"
NOT_SESSION_HOLDER = "not_session_holder"
SESSION_TAKEN_OVER = "session_taken_over"

#: Refusals after which the cached session is no longer this agent's.
_SESSION_LOST = frozenset({NO_SESSION, NOT_SESSION_HOLDER, SESSION_TAKEN_OVER})

#: The bound of a request that runs no action: state, step, session calls.
REQUEST_TIMEOUT_S = 15.0

#: The connect bound of every request.
CONNECT_TIMEOUT_S = 5.0

_UNREACHABLE_HINTS = [
    "Confirm the web terminal is running and serves the trajectory panel.",
    "Confirm the trajectory service is running; the terminal proxies to it.",
]

_TOKEN_HINTS = [
    "The terminal did not inject the trajectory service's launch token: the "
    "deployment's terminal holds no TRAJECTORY_LAUNCH_TOKEN, or holds another "
    "than the service. Redeploy so both receive the same token.",
]

_client_id: str | None = None
_session_lock: asyncio.Lock | None = None


def reset_session() -> None:
    """Forget the cached session without releasing it (tests, a lost session)."""
    global _client_id, _session_lock
    _client_id = None
    _session_lock = None


def cached_client_id() -> str | None:
    """The session this agent holds, or ``None``."""
    return _client_id


def _lock() -> asyncio.Lock:
    global _session_lock
    if _session_lock is None:
        _session_lock = asyncio.Lock()
    return _session_lock


def _url(path: str) -> str:
    from osprey.mcp_server.http import web_terminal_url

    return f"{web_terminal_url()}{BASE_PATH}{path}"


async def _send(
    method: str,
    path: str,
    *,
    json: Any = None,
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout,
) -> tuple[int, Any]:
    """Send one request through the proxy; ``(status, parsed body)``.

    The body is the parsed JSON, or the response text when it is not JSON (the
    proxy's own ``502`` is plain text). A request that reaches nobody raises
    ``trajectory_unreachable``.
    """
    from osprey.mcp_server.http import _panel_auth_headers

    sent = {**_panel_auth_headers(), **(headers or {})}
    try:
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
            resp = await client.request(
                method, _url(path), json=json, headers=sent, timeout=timeout
            )
    except httpx.HTTPError as exc:
        make_error(
            "trajectory_unreachable",
            f"Could not reach the trajectory service through the web terminal: {exc!r}",
            _UNREACHABLE_HINTS,
        )
    try:
        body: Any = resp.json()
    except Exception:
        body = resp.text
    return resp.status_code, body


def _short_timeout() -> httpx.Timeout:
    return httpx.Timeout(REQUEST_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)


def action_timeout(snapshot: dict[str, Any]) -> httpx.Timeout:
    """The bound of a shot or an apply submitted after *snapshot*.

    The action's own timeout, the time a stop of it may take, and the reported
    wait behind the actions already queued, all as the service reports them.
    """
    seconds = (
        float(snapshot["action_timeout_s"])
        + float(snapshot["stop_budget_s"])
        + float(snapshot.get("queue_wait_s") or 0.0)
    )
    return httpx.Timeout(seconds, connect=CONNECT_TIMEOUT_S)


def refusal_code(body: Any) -> str | None:
    """The service's refusal code in *body*, or ``None``."""
    if isinstance(body, dict) and isinstance(body.get("detail"), dict):
        code = body["detail"].get("code")
        return str(code) if code else None
    return None


def relay_refusal(status: int, body: Any) -> NoReturn:
    """Raise the service's (or the proxy's) refusal as a tool error.

    A ``{"detail": {"code", "detail", ...}}`` body keeps its code as the error
    type and every further field (``commands_fired``, ``moved``) as details.
    """
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict) and detail.get("code"):
        code = str(detail["code"])
        message = str(detail.get("detail") or code)
        rest = {k: v for k, v in detail.items() if k not in ("code", "detail")}
        hints = _TOKEN_HINTS if code.startswith("launch_token_") else None
        make_error(code, message, hints, details=rest or None)
    if status == 502 and not isinstance(body, dict):
        make_error(
            "trajectory_unreachable",
            str(body) or "the proxy reached no service",
            _UNREACHABLE_HINTS,
        )
    text = detail if isinstance(detail, str) else str(body)
    make_error("trajectory_refused", f"HTTP {status}: {text}")


async def get_json(path: str) -> dict[str, Any]:
    """GET *path*; the body, or the refusal raised."""
    status, body = await _send("GET", path, timeout=_short_timeout())
    if status >= 400 or not isinstance(body, dict):
        relay_refusal(status, body)
    return body


async def get_state() -> dict[str, Any]:
    """The service's state snapshot, as this caller sees it."""
    return await get_json("/state")


async def _open() -> str:
    status, body = await _send("POST", "/session", timeout=_short_timeout())
    if status < 400 and isinstance(body, dict) and body.get("client_id"):
        return str(body["client_id"])
    if refusal_code(body) != SESSION_HELD:
        relay_refusal(status, body)
    snapshot = await get_state()
    holder = snapshot.get("holder") or {}
    if holder.get("kind") != AGENT_DOOR or holder.get("owner") != snapshot.get("caller"):
        relay_refusal(status, body)
    # The same owner's agent holds it: a server of this agent's that exited
    # without releasing. Nobody else's session is ever taken here.
    status, body = await _send("POST", "/takeover", timeout=_short_timeout())
    if status >= 400 or not isinstance(body, dict) or not body.get("client_id"):
        relay_refusal(status, body)
    logger.info("Took over the stale agent session of %s", holder.get("owner"))
    return str(body["client_id"])


async def ensure_session() -> str:
    """The agent's session id, opening (or taking over) the session on first use."""
    global _client_id
    async with _lock():
        if _client_id is None:
            _client_id = await _open()
        return _client_id


async def write(path: str, payload: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    """POST a writing action under the agent's session, bounded by *snapshot*'s budgets."""
    global _client_id
    client_id = await ensure_session()
    status, body = await _send(
        "POST",
        path,
        json=payload,
        headers={CLIENT_HEADER: client_id},
        timeout=action_timeout(snapshot),
    )
    if status >= 400 or not isinstance(body, dict):
        if refusal_code(body) in _SESSION_LOST and _client_id == client_id:
            _client_id = None
        relay_refusal(status, body)
    return body


async def release_session() -> None:
    """Release the agent's session, when it holds one. Never raises."""
    global _client_id
    client_id = _client_id
    if client_id is None:
        return
    _client_id = None
    try:
        status, body = await _send(
            "DELETE", "/session", headers={CLIENT_HEADER: client_id}, timeout=_short_timeout()
        )
        if status >= 400:
            logger.info("The trajectory session was not released: %s %s", status, body)
    except Exception:
        logger.debug("Releasing the trajectory session failed (ignored)", exc_info=True)
