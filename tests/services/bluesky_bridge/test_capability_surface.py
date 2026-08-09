"""Contract tests for the capability record on the wire.

`queue_backend.py`'s own tests pin how the record is *derived*. These pin how it
is *published*: that the bridge's existing status surface carries it, that every
refusal reaches a consumer as a machine-readable code plus an actionable
sentence, and that the panels sidecar relays both untouched.

Three layers, deliberately:

- **Direct** — `GET /health` on the bridge app, driving the real `QueueBackend`
  against a scripted manager, so the wire shape is asserted against the actual
  derivation rather than a stub of it.
- **Relayed** — the sidecar's `GET /bridge/health` against a mock-transport
  bridge, pinning verbatim passthrough and the 502 the sidecar owes a caller
  when the bridge cannot be reached at all.
- **Both hops at once** — the sidecar's router in front of the *real* bridge
  app over an ASGI transport, so the browse-only refusal (the one whose text an
  operator has to act on) is proven to survive the whole path intact.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.bluesky_panels import read_proxy
from osprey.interfaces.bluesky_panels._shared import UNREACHABLE_BODY
from osprey.services.bluesky_bridge import app as bridge_app_module
from osprey.services.bluesky_bridge import queue_backend as qb
from osprey.services.bluesky_bridge.app import app as bridge_app
from osprey.services.bluesky_bridge.queue_backend import QueueBackend

_BRIDGE_URL = "http://bridge.test"

# The full published vocabulary. Every `reason` this surface can emit must be
# one of these — a code panels and MCP tools have never seen is as useless to
# them as no code at all.
_ALL_REASONS = {
    qb.REASON_EXECUTABLE,
    qb.REASON_BROWSE_ONLY_CONNECTOR,
    qb.REASON_UNSUPPORTED_CONNECTOR,
    qb.REASON_CONFIG_UNREADABLE,
    qb.REASON_MANAGER_NOT_CONFIGURED,
    qb.REASON_MANAGER_UNREACHABLE,
}


class _ScriptedManager:
    """The smallest `REManagerAPI` stand-in `capability()` needs.

    Only `status` is ever called on this path. Pass an exception to have the
    manager raise it instead of answering, which is how "deployed but not
    answering" is staged.
    """

    def __init__(self, status_response: Any = None) -> None:
        self.status_response = status_response
        self.status_calls = 0

    async def status(self, **_kwargs: Any) -> dict[str, Any]:
        self.status_calls += 1
        if isinstance(self.status_response, Exception):
            raise self.status_response
        return self.status_response or {
            "success": True,
            "manager_state": "idle",
            "worker_environment_exists": True,
        }


class _ExplodingBackend:
    """A backend whose capability probe fails in a way nothing anticipated."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def capability(self) -> Any:
        raise self._exc


@pytest.fixture(autouse=True)
def _isolated_backend():
    """Never let one test's injected backend leak into the next.

    `get_queue_backend` memoizes a process-wide singleton, so the override has
    to be cleared on the way out — including when a test fails.
    """
    bridge_app_module.set_queue_backend(None)
    yield
    bridge_app_module.set_queue_backend(None)


@pytest.fixture
def connector(monkeypatch: pytest.MonkeyPatch):
    """Set — or break — the `control_system.type` the bridge resolves.

    Both `_resolve_connector_type` and the shared `_resolve_control_system_type`
    it delegates to import `get_config_value` inside the function body, so
    patching it at its home module covers both.
    """

    def _set(value: str | Exception) -> None:
        def fake_get_config_value(key: str, default: Any = None) -> Any:
            if isinstance(value, Exception):
                raise value
            # Only `control_system.type` is what this fixture is about. Every
            # other key falls through to its default, because the bridge reads
            # more than one (the lifespan's limits guard reads three) and
            # answering all of them with the connector type would stage a
            # nonsense config — a `writes_enabled` of "mock" is truthy.
            if key == "control_system.type":
                return value
            return default

        monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)

    return _set


@pytest.fixture
def bridge():
    """A `TestClient` on the bridge, with a scripted queue backend installed."""

    def _build(manager: Any) -> TestClient:
        bridge_app_module.set_queue_backend(QueueBackend(manager))
        return TestClient(bridge_app)

    return _build


def _capability(response: httpx.Response) -> dict[str, Any]:
    """The capability object off a 200 health response, shape-checked."""
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    capability = body["capability"]
    assert set(capability) == {"can_execute", "reason", "detail"}
    assert isinstance(capability["can_execute"], bool)
    assert capability["reason"] in _ALL_REASONS
    assert isinstance(capability["detail"], str) and capability["detail"]
    return capability


def _sidecar(handler) -> FastAPI:
    """A local app with the read-proxy router mounted onto a faked bridge."""
    app = FastAPI()
    app.include_router(read_proxy.router)
    app.state.bridge_url = _BRIDGE_URL
    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return app


# ---------------------------------------------------------------------------
# Direct: GET /health on the bridge
# ---------------------------------------------------------------------------


def test_health_publishes_capability_alongside_liveness(connector, bridge) -> None:
    connector("virtual_accelerator")

    with bridge(_ScriptedManager()) as client:
        capability = _capability(client.get("/health"))

    assert capability["can_execute"] is True
    assert capability["reason"] == qb.REASON_EXECUTABLE


def test_health_on_mock_refuses_and_names_the_flip_command(connector, bridge) -> None:
    connector("mock")

    with bridge(_ScriptedManager()) as client:
        capability = _capability(client.get("/health"))

    assert capability["can_execute"] is False
    assert capability["reason"] == qb.REASON_BROWSE_ONLY_CONNECTOR
    # The refusal has to tell the operator what to do, not only what is wrong:
    # the exact command, and that it does not take effect until redeploy.
    assert qb.FLIP_COMMAND in capability["detail"]
    assert "redeploy" in capability["detail"].lower()


def test_health_stays_ok_when_the_deployment_cannot_execute(connector, bridge) -> None:
    """A browse-only deployment is a healthy deployment.

    `status` answers for this process; `can_execute` answers for what the
    process can do. Coupling them would make the container healthcheck — which
    reads only the status code — flap on a deployment working exactly as
    configured.
    """
    connector("mock")

    with bridge(_ScriptedManager()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["capability"]["can_execute"] is False


def test_health_fails_closed_on_an_unreadable_config(connector, bridge) -> None:
    connector(FileNotFoundError("no project config mounted"))

    with bridge(_ScriptedManager()) as client:
        capability = _capability(client.get("/health"))

    assert capability["can_execute"] is False
    assert capability["reason"] == qb.REASON_CONFIG_UNREADABLE


def test_health_fails_closed_on_an_unsupported_connector(connector, bridge) -> None:
    connector("doocs")

    with bridge(_ScriptedManager()) as client:
        capability = _capability(client.get("/health"))

    assert capability["can_execute"] is False
    assert capability["reason"] == qb.REASON_UNSUPPORTED_CONNECTOR


def test_health_fails_closed_without_a_configured_manager(connector, bridge) -> None:
    connector("virtual_accelerator")

    with bridge(None) as client:
        capability = _capability(client.get("/health"))

    assert capability["can_execute"] is False
    assert capability["reason"] == qb.REASON_MANAGER_NOT_CONFIGURED


def test_health_fails_closed_when_the_manager_is_silent(connector, bridge) -> None:
    from bluesky_queueserver_api.comm_base import RequestTimeoutError

    connector("virtual_accelerator")
    manager = _ScriptedManager(RequestTimeoutError("timeout", {}))

    with bridge(manager) as client:
        capability = _capability(client.get("/health"))

    assert capability["can_execute"] is False
    assert capability["reason"] == qb.REASON_MANAGER_UNREACHABLE
    assert manager.status_calls == 1


def test_health_fails_closed_when_the_probe_raises_something_unforeseen() -> None:
    """An unanticipated failure still answers 200, and still answers "no".

    The route must never 500 (the container healthcheck reads the status code)
    and must never fall through to an optimistic default. The real error is
    carried in `detail` so it is diagnosable rather than swallowed.
    """
    bridge_app_module.set_queue_backend(_ExplodingBackend(RuntimeError("zmq key is malformed")))

    with TestClient(bridge_app) as client:
        capability = _capability(client.get("/health"))

    assert capability["can_execute"] is False
    assert capability["reason"] == qb.REASON_MANAGER_UNREACHABLE
    assert "zmq key is malformed" in capability["detail"]


def test_health_does_not_probe_the_manager_on_a_browse_only_deployment(connector, bridge) -> None:
    """Mock is answered from config alone — no queue server is ever contacted.

    Ordering matters for the operator: a mock deployment is told to flip the
    connector, not that some queue server it was never meant to have is down.
    """
    connector("mock")
    manager = _ScriptedManager()

    with bridge(manager) as client:
        client.get("/health")

    assert manager.status_calls == 0


# ---------------------------------------------------------------------------
# Relayed: GET /bridge/health on the panels sidecar
# ---------------------------------------------------------------------------


def test_sidecar_relays_the_health_document_verbatim() -> None:
    seen: list[httpx.Request] = []
    document = {
        "status": "ok",
        "capability": {
            "can_execute": False,
            "reason": qb.REASON_BROWSE_ONLY_CONNECTOR,
            "detail": f"... run `{qb.FLIP_COMMAND}` and redeploy.",
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=document)

    with TestClient(_sidecar(handler)) as client:
        response = client.get("/bridge/health")

    assert response.status_code == 200
    assert response.json() == document
    # Relayed from the bridge's own /health — the sidecar invents no path and
    # recomputes no field.
    assert str(seen[0].url) == f"{_BRIDGE_URL}/health"


def test_sidecar_reports_an_unreachable_bridge_as_502() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with TestClient(_sidecar(handler)) as client:
        response = client.get("/bridge/health")

    assert response.status_code == 502
    assert response.json() == UNREACHABLE_BODY


def test_sidecar_health_is_its_own_and_the_bridges_is_one_level_down() -> None:
    """The relay must not shadow the sidecar's container healthcheck.

    Asserted against the real sidecar app, since that is where the collision
    would actually happen: `/health` still answers for the sidecar process
    (and carries no capability, which is not its to report), while the bridge's
    document — capability and all — is served one level down.
    """
    from osprey.interfaces.bluesky_panels.app import app as sidecar_app

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok", "capability": {"from": "bridge"}})

    # Set the state the lifespan would; `TestClient` is used without its
    # context manager here precisely so the lifespan does not replace this
    # mock-transport client with one pointed at a real bridge.
    sidecar_app.state.bridge_url = _BRIDGE_URL
    sidecar_app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = TestClient(sidecar_app)

    own = client.get("/health")
    assert own.status_code == 200
    assert own.json() == {"status": "ok"}

    relayed = client.get("/bridge/health")
    assert relayed.status_code == 200
    assert relayed.json() == {"status": "ok", "capability": {"from": "bridge"}}


# ---------------------------------------------------------------------------
# Both hops: the sidecar's router in front of the real bridge app
# ---------------------------------------------------------------------------


def test_flip_command_survives_the_sidecar_relay_to_the_real_bridge(connector) -> None:
    """The text an operator has to act on reaches the panel intact.

    Neither half is stubbed here: the sidecar forwards over ASGI into the
    actual bridge app, which derives the record from the actual `QueueBackend`.
    """
    connector("mock")
    bridge_app_module.set_queue_backend(QueueBackend(_ScriptedManager()))

    app = FastAPI()
    app.include_router(read_proxy.router)
    app.state.bridge_url = _BRIDGE_URL
    app.state.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=bridge_app))

    with TestClient(app) as client:
        response = client.get("/bridge/health")

    capability = _capability(response)
    assert capability["can_execute"] is False
    assert capability["reason"] == qb.REASON_BROWSE_ONLY_CONNECTOR
    assert qb.FLIP_COMMAND in capability["detail"]
