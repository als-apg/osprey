"""The panel tier's trajectory rows: the agent door through the terminal proxy.

The agent's trajectory MCP server reaches the service only through its own
terminal's panel proxy, presenting the panel token, so the agent door's routes
must sit in :data:`web_auth.PANEL_TIER_ROUTES` under the proxy's
``/panel/trajectory`` prefix — and nothing of the service's panel door may. The
rows are held against the service's own routes rather than a second
hand-written list, so a route added to or dropped from the agent door fails
here instead of leaving the agent refused, or the table wider than the door.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from osprey.interfaces import common_middleware, web_auth
from osprey.interfaces.web_auth import WebCredentials, mint_secret
from osprey.services.trajectory.app import create_app
from osprey.utils.owner_header import OWNER_HEADER

#: The path the terminal's panel proxy serves the trajectory service under;
#: ``/panel/trajectory/<rest>`` reaches the service's own ``/<rest>``.
_PROXY_PREFIX = "/panel/trajectory"

#: The agent door's prefix inside the service, and the panel door's.
_AGENT_DOOR = "/api/"
_PANEL_DOOR = "/panel/"

#: The agent door's routes by name: the session, Stop, and the three calls the
#: agent's trajectory tools make.
_AGENT_ROUTES = ("session", "takeover", "state", "stop", "shot", "step", "apply")

#: Routes only the panel door serves, by method and path inside the service.
_PANEL_ONLY = (
    ("POST", "/panel/api/setup"),
    ("POST", "/panel/api/continuous"),
    ("POST", "/panel/api/continuous/heartbeat"),
    ("GET", "/panel/api/references"),
    ("POST", "/panel/api/references"),
    ("GET", "/panel/api/references/golden"),
    ("POST", "/panel/api/references/select"),
    ("GET", "/panel/api/shots"),
    ("GET", "/panel/api/events"),
)


def _service_routes() -> set[tuple[str, str]]:
    """Every ``(method, path)`` the trajectory app declares, lifespan not run.

    Read from the app's OpenAPI document, which lists each included router's
    routes by method and full path; ``app.routes`` holds an included router as
    one opaque entry rather than its routes.
    """
    paths = create_app().openapi()["paths"]
    return {(method.upper(), path) for path, operations in paths.items() for method in operations}


def _trajectory_rows() -> set[tuple[str, str]]:
    """The panel-tier rows that point at the trajectory service."""
    return {
        (method, path)
        for method, path in web_auth.PANEL_TIER_ROUTES
        if path.startswith(_PROXY_PREFIX + "/")
    }


def test_the_rows_are_exactly_the_agent_doors_routes() -> None:
    """One row per agent-door route and method, under the proxy prefix.

    An agent-door route with no row is a tool the agent cannot call; a row with
    no route behind it is panel-tier reach the service never meant to offer.
    """
    service = _service_routes()
    agent_door = {(method, path) for method, path in service if path.startswith(_AGENT_DOOR)}
    assert agent_door, "the trajectory app declares no agent-door routes"
    # The door's routes, so an empty or renamed door cannot pass vacuously.
    assert {path for _, path in agent_door} == {f"/api/{name}" for name in _AGENT_ROUTES}

    assert _trajectory_rows() == {(method, _PROXY_PREFIX + path) for method, path in agent_door}


def test_no_panel_door_path_is_panel_tier() -> None:
    """The panel door and its bundle stay behind operator credentials.

    The panel door is the operator's own browser; the panel token standing in
    for it would let anything holding the token act as that browser.
    """
    panel_prefix = _PROXY_PREFIX + _PANEL_DOOR
    assert not [path for _, path in web_auth.PANEL_TIER_ROUTES if path.startswith(panel_prefix)]

    panel_door = {
        (method, path) for method, path in _service_routes() if path.startswith(_PANEL_DOOR)
    }
    assert panel_door, "the trajectory app declares no panel-door routes"
    for method, path in panel_door:
        proxied = _PROXY_PREFIX + path
        assert web_auth.classify(method, proxied, False) is web_auth.Tier.OPERATOR, (
            f"{method} {proxied} must be operator-only"
        )
    for method in ("GET", "POST", "DELETE"):
        assert web_auth.classify(method, panel_prefix, False) is web_auth.Tier.OPERATOR


def test_health_is_not_panel_tier_through_the_proxy() -> None:
    """``/health`` is the service's own exemption, not the proxy's tier."""
    assert web_auth.classify("GET", _PROXY_PREFIX + "/health", False) is web_auth.Tier.OPERATOR


def _drive(
    method: str, path: str, *, claimed_owner: str | None = None
) -> tuple[list[dict], list[dict]]:
    """Drive one panel-token HTTP request at *path* through the terminal's gate.

    Returns the scopes the downstream app was reached with and the messages the
    middleware sent. The request carries no ``Origin``, as an in-process
    companion's does not.
    """
    credentials = WebCredentials(operator_secret=mint_secret(), panel_token=mint_secret())
    reached: list[dict] = []
    sent: list[dict] = []

    async def downstream(scope, _receive, send) -> None:
        reached.append(scope)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    headers = [(b"authorization", f"Bearer {credentials.panel_token}".encode("latin-1"))]
    if claimed_owner is not None:
        headers.append((OWNER_HEADER.lower().encode("latin-1"), claimed_owner.encode("latin-1")))
    middleware = common_middleware.WebAuthMiddleware(downstream, cookie_name="osprey_session")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "method": method,
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "scheme": "http",
        "headers": headers,
        "client": ("127.0.0.1", 54321),
        "server": ("127.0.0.1", 8080),
        "app": SimpleNamespace(state=SimpleNamespace(web_credentials=credentials)),
    }

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    return reached, sent


def _statuses(sent: list[dict]) -> list[int]:
    return [message["status"] for message in sent if message["type"] == "http.response.start"]


def test_the_panel_token_is_admitted_on_the_agent_doors_session_route() -> None:
    """The gate, not only the table, lets the agent's hop through.

    The panel token names nobody, so the request leaves the gate with no owner
    header even when the caller sent one: the owner on this hop is the one the
    service's gate mints from the terminal secret the proxy injects.
    """
    path = _PROXY_PREFIX + "/api/session"
    reached, sent = _drive("POST", path, claimed_owner="carol")

    assert [scope["path"] for scope in reached] == [path]
    assert _statuses(sent) == [200]
    wire = OWNER_HEADER.lower().encode("latin-1")
    assert not [value for name, value in reached[0]["headers"] if name.lower() == wire]


@pytest.mark.parametrize("method", ["POST", "DELETE"])
def test_the_panel_token_is_refused_on_the_panel_doors_session_route(method: str) -> None:
    """The panel door's session route wants operator credentials, at the gate."""
    reached, sent = _drive(method, _PROXY_PREFIX + "/panel/api/session")

    assert not reached, "the panel door must never be reached on the panel token"
    assert _statuses(sent) == [401]
    body = b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    )
    assert json.loads(body)["detail"] == "this route requires operator credentials"


@pytest.mark.parametrize(
    ("method", "route"),
    [("POST", "stop"), ("POST", "shot"), ("GET", "step"), ("POST", "apply")],
)
def test_the_panel_token_is_admitted_on_the_agent_doors_write_routes(
    method: str, route: str
) -> None:
    """Stop, shot, step and apply reach the agent door on the panel token."""
    path = f"{_PROXY_PREFIX}/api/{route}"
    reached, sent = _drive(method, path)

    assert [scope["path"] for scope in reached] == [path]
    assert _statuses(sent) == [200]


@pytest.mark.parametrize(("method", "path"), _PANEL_ONLY)
def test_the_panel_token_is_refused_on_the_panel_only_routes(method: str, path: str) -> None:
    """Setup, continuous, references, shots and the stream never take the panel token."""
    reached, sent = _drive(method, _PROXY_PREFIX + path)

    assert not reached, f"{method} {path} must never be reached on the panel token"
    assert _statuses(sent) == [401]


@pytest.mark.parametrize("route", ["setup", "continuous", "references", "shots", "events"])
def test_the_agent_door_serves_no_panel_only_route(route: str) -> None:
    """The panel door's own routes have no agent-door twin to reach on the token."""
    agent_paths = {path for _, path in _service_routes() if path.startswith(_AGENT_DOOR)}
    assert not [path for path in agent_paths if path.startswith(f"/api/{route}")]
