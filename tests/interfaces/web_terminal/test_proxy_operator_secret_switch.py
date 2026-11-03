"""A sidecar's registry entry decides whether the proxy hands it the operator secret.

The panel proxy re-issues ``X-Osprey-Terminal-Secret`` toward every backend
that is config-declared and on loopback. A sidecar that authenticates with its
own launch credential alone has no use for the operator's proof, so its
registry entry declines it (``SidecarPanel.operator_secret=False``): the secret
is then absent on the HTTP leg and on the WebSocket leg, while the launch
credential the terminal published for it still rides both.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from osprey.interfaces.web_auth import get_web_credentials
from osprey.interfaces.web_terminal.routes import proxy
from osprey.interfaces.web_terminal.routes.proxy import TERMINAL_SECRET_HEADER
from osprey.profiles import web_panels
from osprey.profiles.web_panels import SidecarPanel
from osprey.registry.web import panel_url_state_attr

from ._proxy_fakes import _FakeConnect, _lower, _patch_connect, panel_app

BACKEND_URL = "http://127.0.0.1:9500"
LAUNCH_HEADERS = {"X-Sidecar-Access": "sidecar-launch-credential"}

#: Panel id → whether its registry entry takes the operator secret.
SIDECARS = {"takes-secret": True, "own-credential": False}


@pytest.fixture
def app_and_client(tmp_path, monkeypatch):
    """The whole app with two running sidecars, one of which declines the secret."""
    for panel_id, operator_secret in SIDECARS.items():
        monkeypatch.setitem(
            web_panels.SIDECAR_PANELS,
            panel_id,
            SidecarPanel("a test sidecar", "module:Class", operator_secret=operator_secret),
        )
        monkeypatch.setitem(proxy._PANEL_STATE_MAP, panel_id, panel_url_state_attr(panel_id))
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    for app, client in panel_app(workspace, []):
        for panel_id in SIDECARS:
            # What _launch_sidecar publishes once a sidecar answers.
            setattr(app.state, panel_url_state_attr(panel_id), BACKEND_URL)
            app.state.panel_auth_headers[panel_id] = dict(LAUNCH_HEADERS)
        yield app, client


def _http_headers(app, client, panel_id: str) -> dict[str, str]:
    """The headers the proxy sent upstream for one GET below *panel_id*."""
    captured: dict[str, str] = {}

    async def fake_request(*, method, url, headers, content, follow_redirects=True):  # noqa: ARG001
        captured.update(headers)
        return httpx.Response(200, json={"ok": True}, headers={"content-type": "application/json"})

    app.state.proxy_client.request = AsyncMock(side_effect=fake_request)
    assert client.get(f"/panel/{panel_id}/api/thing").status_code == 200
    return _lower(captured)


def _ws_headers(client, panel_id: str) -> dict[str, str]:
    """The headers the proxy opened the upstream websocket with."""
    fake = _FakeConnect()
    with _patch_connect(fake):
        with client.websocket_connect(f"/panel/{panel_id}/ws"):
            pass
    return _lower(fake.kwargs["additional_headers"] or {})


class TestOperatorSecretSwitch:
    def test_the_secret_is_on_by_default(self):
        assert SidecarPanel("n", "m:C").operator_secret is True

    def test_http_leg_sends_the_secret_to_a_sidecar_that_takes_it(self, app_and_client):
        app, client = app_and_client
        sent = _http_headers(app, client, "takes-secret")

        assert sent[TERMINAL_SECRET_HEADER.lower()] == get_web_credentials(app).operator_secret

    def test_http_leg_withholds_the_secret_from_a_sidecar_that_declines_it(self, app_and_client):
        app, client = app_and_client
        sent = _http_headers(app, client, "own-credential")

        assert TERMINAL_SECRET_HEADER.lower() not in sent
        assert get_web_credentials(app).operator_secret not in " ".join(sent.values())

    def test_ws_leg_sends_the_secret_to_a_sidecar_that_takes_it(self, app_and_client):
        app, client = app_and_client
        sent = _ws_headers(client, "takes-secret")

        assert sent[TERMINAL_SECRET_HEADER.lower()] == get_web_credentials(app).operator_secret

    def test_ws_leg_withholds_the_secret_from_a_sidecar_that_declines_it(self, app_and_client):
        app, client = app_and_client
        sent = _ws_headers(client, "own-credential")

        assert TERMINAL_SECRET_HEADER.lower() not in sent
        assert get_web_credentials(app).operator_secret not in " ".join(sent.values())

    @pytest.mark.parametrize("panel_id", sorted(SIDECARS))
    def test_http_leg_injects_the_launch_credential_either_way(self, app_and_client, panel_id):
        app, client = app_and_client

        assert (
            _http_headers(app, client, panel_id)["x-sidecar-access"]
            == (LAUNCH_HEADERS["X-Sidecar-Access"])
        )

    @pytest.mark.parametrize("panel_id", sorted(SIDECARS))
    def test_ws_leg_injects_the_launch_credential_either_way(self, app_and_client, panel_id):
        _app, client = app_and_client

        assert (
            _ws_headers(client, panel_id)["x-sidecar-access"]
            == (LAUNCH_HEADERS["X-Sidecar-Access"])
        )
