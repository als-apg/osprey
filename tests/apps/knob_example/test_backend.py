"""The example's backend routes, against a stand-in connector."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from knob_example import STATE_ROUTE, WRITE_ROUTE
from knob_example import server as backend
from knob_example.server import CHANNEL, LAUNCH_TOKEN_ENV, create_app
from starlette.middleware import Middleware

from osprey.apps.sdk import LAUNCH_TOKEN_HEADER, OWNER_STATE_KEY
from osprey.interfaces.common_middleware import WebAuthMiddleware
from osprey_connectors.errors import ChannelWriteBlockedError, ConnectorUnavailableError
from osprey_connectors.posture_store import bound_owner

TOKEN = "launch-token-value"
OWNER = "alice"


class StandInConnector:
    """Holds one value, and records who writes it."""

    def __init__(self, target: str, *, refuse: bool = False, unreachable: bool = False) -> None:
        self.target = target
        self.value = 1.5
        self.refuse = refuse
        self.unreachable = unreachable
        self.writers: list[Any] = []
        self.disconnected = False

    async def read_channel(self, channel: str) -> SimpleNamespace:
        assert channel == CHANNEL
        if self.unreachable:
            raise ConnectorUnavailableError("the transport is gone")
        return SimpleNamespace(value=self.value)

    async def write_channel_checked(self, channel: str, value: float) -> SimpleNamespace:
        assert channel == CHANNEL
        if self.refuse:
            raise ChannelWriteBlockedError(channel, value, "the switch is off")
        self.writers.append(bound_owner())
        self.value = value
        return SimpleNamespace(outcome="confirmed")

    async def disconnect(self) -> None:
        self.disconnected = True


class _GatedAs:
    """Stamp the owner the gate would have stamped, in place of the gate under test elsewhere."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            scope.setdefault("state", {})[OWNER_STATE_KEY] = OWNER
        await self.app(scope, receive, send)


@pytest.fixture
def machines() -> dict[str, StandInConnector]:
    return {}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, machines: dict[str, StandInConnector]):
    monkeypatch.setenv(LAUNCH_TOKEN_ENV, TOKEN)
    monkeypatch.setattr(backend, "_section", lambda: {"type": "virtual_accelerator"}, raising=True)

    async def connector_for(target: str) -> StandInConnector:
        return machines.setdefault(target, StandInConnector(target))

    app: FastAPI = create_app(connector_for)
    app.user_middleware = [
        entry for entry in app.user_middleware if entry.cls is not WebAuthMiddleware
    ] + [Middleware(_GatedAs)]
    with TestClient(app) as test_client:
        yield test_client


def _write(client: TestClient, **overrides: Any) -> Any:
    body = {"value": 2.5, "target": "va", "readback": 1.5, **overrides}
    return client.post(WRITE_ROUTE, json=body, headers={LAUNCH_TOKEN_HEADER: TOKEN})


def test_health_answers(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_the_panel_page_is_served_at_the_entry(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert 'id="knob-state"' in page.text
    assert client.get("/static/panel.js").status_code == 200


def test_the_state_answers_from_the_baseline_when_no_target_is_named(client: TestClient) -> None:
    assert client.get(STATE_ROUTE).json() == {"channel": CHANNEL, "target": "va", "readback": 1.5}


def test_the_state_follows_the_target_it_is_told(
    client: TestClient, machines: dict[str, StandInConnector]
) -> None:
    assert client.get(STATE_ROUTE, params={"target": "standin"}).json()["target"] == "standin"
    assert set(machines) == {"standin"}


def test_a_state_read_that_fails_answers_unavailable(
    client: TestClient, machines: dict[str, StandInConnector]
) -> None:
    machines["va"] = StandInConnector("va", unreachable=True)
    assert client.get(STATE_ROUTE).status_code == 503


def test_a_target_this_deployment_does_not_name_is_refused(machines: dict) -> None:
    async def connector_for(target: str) -> StandInConnector:
        raise ValueError(f"no machine named {target}")

    app = create_app(connector_for)
    app.user_middleware = [e for e in app.user_middleware if e.cls is not WebAuthMiddleware]
    client = TestClient(app, raise_server_exceptions=False)
    assert client.get(STATE_ROUTE, params={"target": "elsewhere"}).status_code == 422
    assert machines == {}


def test_a_write_goes_to_the_approved_target_under_the_gates_owner(
    client: TestClient, machines: dict[str, StandInConnector]
) -> None:
    response = _write(client, target="standin")
    assert response.status_code == 200
    assert response.json() == {"channel": CHANNEL, "target": "standin", "outcome": "confirmed"}
    assert machines["standin"].value == 2.5
    assert machines["standin"].writers == [OWNER]


@pytest.mark.parametrize("headers", [{}, {LAUNCH_TOKEN_HEADER: "wrong"}])
def test_a_write_without_the_launch_token_is_refused(
    client: TestClient, machines: dict[str, StandInConnector], headers: dict[str, str]
) -> None:
    body = {"value": 2.5, "target": "va", "readback": 1.5}
    assert client.post(WRITE_ROUTE, json=body, headers=headers).status_code == 403
    assert machines == {}


def test_no_write_is_served_when_the_deployment_holds_no_launch_token(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(LAUNCH_TOKEN_ENV)
    assert _write(client).status_code == 503


def test_a_write_approved_over_another_readback_is_refused(
    client: TestClient, machines: dict[str, StandInConnector]
) -> None:
    response = _write(client, readback=9.0)
    assert response.status_code == 409
    assert machines["va"].writers == []


def test_a_write_the_connector_refuses_answers_refused(
    client: TestClient, machines: dict[str, StandInConnector]
) -> None:
    machines["va"] = StandInConnector("va", refuse=True)
    response = _write(client)
    assert response.status_code == 403
    assert "the switch is off" in response.text


def test_a_body_naming_an_owner_is_refused(client: TestClient) -> None:
    assert _write(client, owner="mallory").status_code == 422


def test_each_connector_is_closed_on_the_way_out(machines: dict[str, StandInConnector]) -> None:
    async def connector_for(target: str) -> StandInConnector:
        return machines.setdefault(target, StandInConnector(target))

    with TestClient(create_app(connector_for), raise_server_exceptions=False) as test_client:
        test_client.get(STATE_ROUTE, params={"target": "va"})
    assert all(machine.disconnected for machine in machines.values())
