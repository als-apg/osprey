"""The dispatcher's MCP transport is bearer-gated, like its dashboard routes.

``manual_fire`` starts an unattended agent run, so the MCP transport carries the
same authority as the dashboard's write routes and must refuse the same callers.
The gate is :class:`~osprey.dispatch.server._DispatcherTokenVerifier`, handed to
``FastMCP(auth=...)``; these rows drive the real ASGI app so the refusals are the
ones a caller on the wire would actually see, not the verifier's return value.

Two properties are load-bearing and easy to break silently:

* A verifier has exactly one way to refuse — answer ``None``. An unset
  ``EVENT_DISPATCHER_TOKEN`` is therefore a 401 on the transport while the custom
  routes keep answering 503; both refuse, and only the route shape can say
  "misconfigured".
* FastMCP installs the bearer backend as APP-level middleware while only the
  transport route requires auth. A success value the ``AccessToken`` model cannot
  build would raise inside that middleware for *every* request presenting a
  bearer, turning the proxied dashboard calls into 500s. The dashboard rows below
  are what notices.

Run: ``python3 -m pytest tests/dispatch/test_mcp_transport_auth.py -q``
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest

from osprey.dispatch import DISPATCHER_MCP_PATH
from tests.conftest import dispatcher_route_registry

# All rows share one process-wide FastMCP singleton (``create_server`` configures
# the module-level instance), so they must not be split across xdist workers
# mid-module.
pytestmark = pytest.mark.xdist_group("dispatch_mcp_transport_auth")

#: The bearer the tests configure as ``EVENT_DISPATCHER_TOKEN``.
_TOKEN = "dispatcher-transport-secret"

#: A dashboard data route: bearer-gated by ``_check_auth``, never by the verifier.
_DASHBOARD_ROUTE = "/dashboard/triggers"

#: Every MCP client advertises both shapes; the transport refuses a request that
#: does not.
_MCP_ACCEPT = "application/json, text/event-stream"

_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "transport-auth-test", "version": "1.0"},
    },
}


@pytest.fixture(scope="module")
def dispatcher_app():
    """The dispatcher's real ASGI app, built once for the module.

    ``dispatcher_route_registry`` owns the module-level FastMCP singleton for the
    duration: the app is built from the routes this module registered, and the
    singleton is left carrying the ones it had before.

    Configuration is read at factory time only; both the verifier and
    ``_check_auth`` re-read ``EVENT_DISPATCHER_TOKEN`` per request, which is what
    lets every row below set or clear the token for itself.

    ``stateless_http``/``json_response`` keep the transport's answers plain JSON
    with no session to carry between calls. They change nothing about the auth
    stack, which is where the assertions live.
    """
    saved_env = {key: os.environ.get(key) for key in ("TRIGGERS_YML", "DISPATCH_TARGET")}
    # No triggers.yml on purpose: the server starts empty, and an explicit
    # target keeps the factory off the deployment-config fallback.
    os.environ["TRIGGERS_YML"] = "/nonexistent/test-triggers.yml"
    os.environ["DISPATCH_TARGET"] = "http://worker.invalid:9000"
    with dispatcher_route_registry() as build:
        try:
            server = build()
        finally:
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        yield server.http_app(
            path=DISPATCHER_MCP_PATH,
            stateless_http=True,
            json_response=True,
        )


@asynccontextmanager
async def _client(app) -> AsyncIterator[httpx.AsyncClient]:
    """Drive the app in-process, with its lifespan running.

    The transport's session manager only exists inside the lifespan, so a request
    made without it fails for a reason that has nothing to do with auth.
    """
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://dispatcher.invalid",
        ) as client:
            yield client


def _mcp_headers(bearer: str | bytes | None = None) -> dict[str, str | bytes]:
    headers: dict[str, str | bytes] = {
        "Accept": _MCP_ACCEPT,
        "Content-Type": "application/json",
    }
    if bearer is not None:
        headers["Authorization"] = bearer
    return headers


# ---------------------------------------------------------------------------
# The transport refuses
# ---------------------------------------------------------------------------


async def test_initialize_without_a_bearer_is_401(dispatcher_app, monkeypatch):
    """An unauthenticated caller never reaches ``initialize``."""
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", _TOKEN)

    async with _client(dispatcher_app) as client:
        response = await client.post(DISPATCHER_MCP_PATH, headers=_mcp_headers(), json=_INITIALIZE)

    assert response.status_code == 401


async def test_initialize_with_the_wrong_bearer_is_401_not_500(dispatcher_app, monkeypatch):
    """A wrong credential is a refusal, never an error the caller can mine."""
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", _TOKEN)

    async with _client(dispatcher_app) as client:
        response = await client.post(
            DISPATCHER_MCP_PATH,
            headers=_mcp_headers(f"Bearer {_TOKEN}-wrong"),
            json=_INITIALIZE,
        )

    assert response.status_code == 401


async def test_initialize_with_a_non_ascii_bearer_is_401_not_500(dispatcher_app, monkeypatch):
    """The comparison takes bytes, so attacker-controlled text cannot raise.

    ``hmac.compare_digest`` refuses two ``str`` arguments unless both are
    ASCII-only; a bearer is whatever the caller sent, and a header byte outside
    ASCII survives Starlette's latin-1 decode, so comparing the decoded strings
    would turn a wrong credential into a 500. The header goes on the wire as
    raw bytes because that is the only way to send one.
    """
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", _TOKEN)

    async with _client(dispatcher_app) as client:
        response = await client.post(
            DISPATCHER_MCP_PATH,
            headers=_mcp_headers("Bearer tökén-ünicode".encode("latin-1")),
            json=_INITIALIZE,
        )

    assert response.status_code == 401


async def test_initialize_with_an_empty_bearer_is_401(dispatcher_app, monkeypatch):
    """An empty credential is not a credential."""
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", _TOKEN)

    async with _client(dispatcher_app) as client:
        response = await client.post(
            DISPATCHER_MCP_PATH, headers=_mcp_headers("Bearer "), json=_INITIALIZE
        )

    assert response.status_code == 401


async def test_initialize_with_the_token_unset_is_401(dispatcher_app, monkeypatch):
    """An unconfigured dispatcher accepts nothing, not everything."""
    monkeypatch.delenv("EVENT_DISPATCHER_TOKEN", raising=False)

    async with _client(dispatcher_app) as client:
        response = await client.post(
            DISPATCHER_MCP_PATH,
            headers=_mcp_headers(f"Bearer {_TOKEN}"),
            json=_INITIALIZE,
        )

    assert response.status_code == 401


async def test_initialize_with_an_empty_token_configured_is_401(dispatcher_app, monkeypatch):
    """An empty ``EVENT_DISPATCHER_TOKEN`` is unset, not a shared secret."""
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "")

    async with _client(dispatcher_app) as client:
        response = await client.post(
            DISPATCHER_MCP_PATH, headers=_mcp_headers("Bearer "), json=_INITIALIZE
        )

    assert response.status_code == 401


async def test_a_secret_that_is_not_utf8_refuses_rather_than_raising(dispatcher_app, monkeypatch):
    """A secret carrying bytes that are not valid UTF-8 refuses; it does not 500.

    ``os.environ`` hands such a value back with one lone surrogate per undecodable
    byte, which the default encoder refuses to turn back into bytes. A raise on
    the comparison path is a 500 on *every* bearer request — an unreachable
    dispatcher instead of a refused caller — so both gates encode the way the
    environment decoded.
    """
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "secret-\udcff")

    async with _client(dispatcher_app) as client:
        transport_response = await client.post(
            DISPATCHER_MCP_PATH,
            headers=_mcp_headers(f"Bearer {_TOKEN}"),
            json=_INITIALIZE,
        )
        dashboard_response = await client.get(
            _DASHBOARD_ROUTE, headers={"Authorization": f"Bearer {_TOKEN}"}
        )

    assert transport_response.status_code == 401
    assert dashboard_response.status_code == 401


# ---------------------------------------------------------------------------
# The transport admits
# ---------------------------------------------------------------------------


async def test_initialize_with_the_right_bearer_succeeds(dispatcher_app, monkeypatch):
    """The configured bearer reaches a real MCP handshake."""
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", _TOKEN)

    async with _client(dispatcher_app) as client:
        response = await client.post(
            DISPATCHER_MCP_PATH,
            headers=_mcp_headers(f"Bearer {_TOKEN}"),
            json=_INITIALIZE,
        )

    assert response.status_code == 200
    body = response.json()
    assert "error" not in body, body
    assert body["result"]["serverInfo"]["name"] == "event_dispatcher"


# ---------------------------------------------------------------------------
# The custom routes are untouched by the verifier
# ---------------------------------------------------------------------------


async def test_dashboard_route_with_the_bearer_still_answers_200(dispatcher_app, monkeypatch):
    """The app-level bearer backend runs on every request and must not break one.

    This is the row that fails if the verifier's success value is missing one of
    ``AccessToken``'s three required fields: the model error would be raised in
    middleware the dashboard also passes through, long before ``_check_auth``.
    """
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", _TOKEN)

    async with _client(dispatcher_app) as client:
        response = await client.get(_DASHBOARD_ROUTE, headers={"Authorization": f"Bearer {_TOKEN}"})

    assert response.status_code == 200
    assert isinstance(response.json(), list)


async def test_dashboard_route_with_a_wrong_bearer_still_answers_401(dispatcher_app, monkeypatch):
    """A verifier that answers ``None`` leaves the route's own gate to refuse."""
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", _TOKEN)

    async with _client(dispatcher_app) as client:
        response = await client.get(
            _DASHBOARD_ROUTE, headers={"Authorization": f"Bearer {_TOKEN}-wrong"}
        )

    assert response.status_code == 401


async def test_dashboard_route_with_a_non_ascii_bearer_still_answers_401(
    dispatcher_app, monkeypatch
):
    """The route's own gate takes bytes too, so a header cannot make it raise.

    This is the route-side twin of the transport row above: a header byte outside
    ASCII survives Starlette's latin-1 decode, and ``compare_digest`` refuses two
    ``str`` arguments unless both are ASCII-only, so comparing the decoded
    strings here would turn a wrong credential into a 500 on every dashboard
    route. The header goes on the wire as raw bytes because that is the only way
    to send one.
    """
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", _TOKEN)

    async with _client(dispatcher_app) as client:
        response = await client.get(
            _DASHBOARD_ROUTE, headers={"Authorization": "Bearer tökén-ünicode".encode("latin-1")}
        )

    assert response.status_code == 401


async def test_unset_token_splits_the_two_gates(dispatcher_app, monkeypatch):
    """With no token configured: 401 on the transport, 503 on the routes.

    The difference is the whole reason both gates exist. A verifier cannot say
    "misconfigured" — its only refusal is ``None``, reported as 401 — while the
    dashboard route can, and the operator-facing 503 is how an unconfigured
    dispatcher is diagnosed.
    """
    monkeypatch.delenv("EVENT_DISPATCHER_TOKEN", raising=False)

    async with _client(dispatcher_app) as client:
        transport_response = await client.post(
            DISPATCHER_MCP_PATH,
            headers=_mcp_headers(f"Bearer {_TOKEN}"),
            json=_INITIALIZE,
        )
        dashboard_response = await client.get(
            _DASHBOARD_ROUTE, headers={"Authorization": f"Bearer {_TOKEN}"}
        )

    assert transport_response.status_code == 401
    assert dashboard_response.status_code == 503


async def test_health_stays_open(dispatcher_app, monkeypatch):
    """The liveness probe was never gated and the verifier must not gate it."""
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", _TOKEN)

    async with _client(dispatcher_app) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# The verifier itself
# ---------------------------------------------------------------------------


async def test_verifier_success_value_carries_the_three_required_fields(monkeypatch):
    """The success value is a fully-built ``AccessToken``, not a stub."""
    from osprey.dispatch.server import _DispatcherTokenVerifier

    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", _TOKEN)

    access = await _DispatcherTokenVerifier().verify_token(_TOKEN)

    assert access is not None
    assert access.token == _TOKEN
    assert access.client_id == "dispatcher-bearer"
    assert access.scopes == []


async def test_verifier_rereads_the_environment_per_call(monkeypatch):
    """A rotated token takes effect without a restart, and the old one stops working."""
    from osprey.dispatch.server import _DispatcherTokenVerifier

    verifier = _DispatcherTokenVerifier()

    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", _TOKEN)
    assert await verifier.verify_token(_TOKEN) is not None

    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", f"{_TOKEN}-rotated")
    assert await verifier.verify_token(_TOKEN) is None
    assert await verifier.verify_token(f"{_TOKEN}-rotated") is not None


async def test_verifier_refuses_when_the_token_is_unset(monkeypatch):
    """No configured token means no caller is authorized, including an empty one."""
    from osprey.dispatch.server import _DispatcherTokenVerifier

    verifier = _DispatcherTokenVerifier()

    monkeypatch.delenv("EVENT_DISPATCHER_TOKEN", raising=False)
    assert await verifier.verify_token("") is None
    assert await verifier.verify_token(_TOKEN) is None

    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "")
    assert await verifier.verify_token("") is None


async def test_server_is_built_with_the_verifier():
    """The transport gate is wired at construction, not merely available."""
    from osprey.dispatch.server import _DispatcherTokenVerifier, mcp

    assert isinstance(mcp.auth, _DispatcherTokenVerifier)
