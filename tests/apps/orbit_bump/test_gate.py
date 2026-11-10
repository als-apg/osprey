"""The orbit-bump backend is first-party, so it runs the interfaces gate on its own routes.

The gate is driven with raw ASGI scopes: a test client could carry a credential that turns
a refusal into an admission.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from orbit_bump import APP_NAME
from orbit_bump.server import LAUNCH_TOKEN_ENV, create_app

from osprey.apps.sdk import LAUNCH_TOKEN_HEADER, OWNER_STATE_KEY
from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER, WebAuthMiddleware
from osprey.interfaces.web_auth import WebCredentials, reset_web_credentials
from osprey.profiles.sidecar_panels import SIDECAR_PANELS, ApplicationPanel, PanelRegistry
from osprey.utils.owner_header import OWNER_HEADER
from tests.interfaces._asgi import (
    OPERATOR_SECRET,
    PANEL_TOKEN,
    RecordingApp,
    drive,
    http_scope,
    status_of,
)
from tests.interfaces.web_terminal._proxy_fakes import panel_app

#: The agent's two routes: the ones the tool server reaches through the terminal.
COMPUTE_ROUTE = "/api/compute"
APPLY_ROUTE = "/api/apply"
AGENT_ROUTES = (("POST", COMPUTE_ROUTE), ("POST", APPLY_ROUTE))

JSON_BODY = b'{"plan_id": "p-1", "setpoint_digest": "00"}'
HOST = "localhost:8080"


#: The two gates an apply passes: the terminal's, which the tool server meets with the panel
#: token, and the backend's own, which the terminal's proxy meets with the operator secret.
HOPS = {
    "terminal": (f"/panel/{APP_NAME}{APPLY_ROUTE}", {"authorization": f"Bearer {PANEL_TOKEN}"}),
    "backend": (APPLY_ROUTE, {OPERATOR_SECRET_HEADER: OPERATOR_SECRET}),
}


@pytest.fixture
def app_stub():
    reset_web_credentials()
    credentials = WebCredentials(operator_secret=OPERATOR_SECRET, panel_token=PANEL_TOKEN)
    registry = PanelRegistry(SIDECAR_PANELS, {APP_NAME: ApplicationPanel(APP_NAME, AGENT_ROUTES)})
    yield SimpleNamespace(
        state=SimpleNamespace(web_credentials=credentials, panel_registry=registry)
    )
    reset_web_credentials()


def _post(app_stub, hop: str, content_type: str, body: bytes, **headers: str):
    path, credential = HOPS[hop]
    downstream = RecordingApp()
    gate = WebAuthMiddleware(downstream, cookie_name="osprey_terminal_session_8080")
    scope = http_scope(
        path,
        "POST",
        {"host": HOST, "content-type": content_type, **credential, **headers},
        app=app_stub,
    )
    sent = drive(gate, scope, [{"type": "http.request", "body": body}])
    return status_of(sent), downstream


def test_the_backend_installs_the_gate() -> None:
    assert WebAuthMiddleware in {entry.cls for entry in create_app().user_middleware}


def test_the_launch_token_is_read_from_the_apps_own_variable() -> None:
    assert LAUNCH_TOKEN_ENV == "ORBIT_BUMP_LAUNCH_TOKEN"


BACKEND = "http://127.0.0.1:9300"


@pytest.fixture
def terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FastAPI]:
    """The real terminal app, on a login whose tier arms, with this panel declared."""
    monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
    render = tmp_path / "config.yml"
    render.write_text(yaml.safe_dump({"control_system": {"writes": "allowed"}}), encoding="utf-8")
    panel = {
        "id": APP_NAME,
        "label": "Orbit bump",
        "url": BACKEND,
        "configDefined": True,
        "app": APP_NAME,
        "agentRoutes": [{"method": method, "path": path} for method, path in AGENT_ROUTES],
    }
    for app, _client in panel_app(tmp_path, [panel]):
        app.state.config_path = render
        app.state.web_credentials = WebCredentials(
            operator_secret=OPERATOR_SECRET, panel_token=PANEL_TOKEN
        )
        yield app


@pytest.mark.no_auth_seam
def test_a_json_apply_with_the_panel_token_reaches_the_backend_on_the_operator_secret(
    terminal: FastAPI,
) -> None:
    upstream = AsyncMock(return_value=httpx.Response(200, json={"status": "applied"}))
    terminal.state.proxy_client.request = upstream

    answer = TestClient(terminal, client=("127.0.0.1", 54321)).post(
        f"/panel/{APP_NAME}{APPLY_ROUTE}",
        content=JSON_BODY,
        headers={"content-type": "application/json", "authorization": f"Bearer {PANEL_TOKEN}"},
    )

    assert answer.status_code == 200
    sent = upstream.await_args.kwargs
    assert (sent["method"], sent["url"]) == ("POST", f"{BACKEND}{APPLY_ROUTE}")
    headers = {name.lower(): value for name, value in sent["headers"].items()}
    assert headers[OPERATOR_SECRET_HEADER] == OPERATOR_SECRET
    assert "authorization" not in headers


def test_the_operator_secret_passes_the_backends_gate(app_stub) -> None:
    status, downstream = _post(app_stub, "backend", "application/json", JSON_BODY)
    assert status == 200
    assert downstream.called


def test_the_launch_token_header_reaches_the_backend_through_its_gate(app_stub) -> None:
    status, downstream = _post(
        app_stub, "backend", "application/json", JSON_BODY, **{LAUNCH_TOKEN_HEADER: "token"}
    )
    assert status == 200
    [scope] = downstream.scopes
    assert (LAUNCH_TOKEN_HEADER.lower().encode("latin-1"), b"token") in scope["headers"]


def test_an_inbound_owner_header_is_dropped_by_the_backends_gate(app_stub) -> None:
    status, downstream = _post(
        app_stub, "backend", "application/json", JSON_BODY, **{OWNER_HEADER: "mallory"}
    )
    assert status == 200
    [scope] = downstream.scopes
    assert b"mallory" not in {value for _name, value in scope["headers"]}
    assert scope.get("state", {}).get(OWNER_STATE_KEY) != "mallory"


def test_a_form_body_is_refused_by_the_backends_own_gate(app_stub) -> None:
    status, downstream = _post(app_stub, "backend", "application/x-www-form-urlencoded", b"a=1")
    assert status == 415
    assert not downstream.called


@pytest.mark.parametrize("hop", HOPS)
def test_an_opaque_origin_is_refused(app_stub, hop: str) -> None:
    status, downstream = _post(app_stub, hop, "application/json", JSON_BODY, origin="null")
    assert status == 403
    assert not downstream.called


def test_the_panel_token_does_not_open_the_backend_directly(app_stub) -> None:
    downstream = RecordingApp()
    gate = WebAuthMiddleware(downstream, cookie_name="osprey_terminal_session_8080")
    scope = http_scope(
        APPLY_ROUTE,
        "POST",
        {
            "host": HOST,
            "content-type": "application/json",
            "authorization": f"Bearer {PANEL_TOKEN}",
        },
        app=app_stub,
    )
    assert status_of(drive(gate, scope, [{"type": "http.request", "body": JSON_BODY}])) == 401
    assert not downstream.called
