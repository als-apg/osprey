"""The example's backend is first-party, so it runs the interfaces gate on its own routes.

The gate is driven with raw ASGI scopes: a test client could carry a credential that turns
a refusal into an admission.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from knob_example import APP_NAME, WRITE_ROUTE
from knob_example.server import create_app

from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER, WebAuthMiddleware
from osprey.interfaces.web_auth import (
    WebCredentials,
    reset_web_credentials,
    set_panel_agent_routes,
)
from tests.interfaces._asgi import (
    OPERATOR_SECRET,
    PANEL_TOKEN,
    RecordingApp,
    drive,
    http_scope,
    status_of,
)

JSON_BODY = b'{"value": 1.0, "target": "va", "readback": 0.0}'
HOST = "localhost:8080"


#: The two gates a write passes: the terminal's, which the tool server meets with the panel
#: token, and the backend's own, which the terminal's proxy meets with the operator secret.
HOPS = {
    "terminal": (f"/panel/{APP_NAME}{WRITE_ROUTE}", {"authorization": f"Bearer {PANEL_TOKEN}"}),
    "backend": (WRITE_ROUTE, {OPERATOR_SECRET_HEADER: OPERATOR_SECRET}),
}


@pytest.fixture
def app_stub():
    reset_web_credentials()
    credentials = WebCredentials(operator_secret=OPERATOR_SECRET, panel_token=PANEL_TOKEN)
    set_panel_agent_routes(frozenset({("POST", HOPS["terminal"][0])}))
    yield SimpleNamespace(state=SimpleNamespace(web_credentials=credentials))
    set_panel_agent_routes(frozenset())
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


@pytest.mark.parametrize("hop", HOPS)
def test_a_json_write_passes_the_gate(app_stub, hop: str) -> None:
    status, downstream = _post(app_stub, hop, "application/json", JSON_BODY)
    assert status == 200
    assert downstream.called


def test_a_form_body_is_refused_by_the_backends_own_gate(app_stub) -> None:
    status, downstream = _post(app_stub, "backend", "application/x-www-form-urlencoded", b"value=1")
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
        WRITE_ROUTE,
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
