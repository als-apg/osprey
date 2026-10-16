"""The terminal's hop to the trajectory service's agent door.

The trajectory MCP server reaches the service only through its own terminal's
panel proxy (``/panel/trajectory/api/...``). The terminal is the one process
that holds the service's launch token: the app takes the token out of
``os.environ`` while it is assembled, before its web credentials are settled
and its memory guarded -- the SDK hands an agent child ``{**os.environ,
**options.env}``, so anything left there reaches the agent -- and its lifespan
publishes it as the
``trajectory`` panel's own credential, which the proxy injects on the same gate
as the operator secret. Whatever the caller sent in those headers stops here.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import httpx
import pytest

from osprey.interfaces.web_auth import get_web_credentials
from osprey.interfaces.web_terminal.app import (
    TRAJECTORY_LAUNCH_TOKEN_ENV,
    TRAJECTORY_PANEL_ID,
    hold_trajectory_launch_token,
    publish_trajectory_launch_token,
)
from osprey.interfaces.web_terminal.operator_session import build_operator_child_env
from osprey.interfaces.web_terminal.routes.proxy import (
    TERMINAL_SECRET_HEADER,
    _is_stripped_header,
)
from osprey.utils.identity import TERMINAL_USER_ENV
from osprey.utils.launch_token import LAUNCH_TOKEN_HEADER
from osprey.utils.owner_header import OWNER_HEADER

from ._proxy_fakes import _lower, panel_app

LAUNCH_TOKEN = "trajectory-launch-token-0123456789"
BACKEND_URL = "http://127.0.0.1:8095"


def _declared(url: str = BACKEND_URL) -> dict:
    return {
        "id": TRAJECTORY_PANEL_ID,
        "label": "TRAJECTORY",
        "url": url,
        "configDefined": True,
    }


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


def _capture_request(app) -> dict[str, str]:
    captured: dict[str, str] = {}

    # ``httpx.AsyncClient.request``'s signature: the proxy names every field it sends.
    async def fake_request(*, method, url, headers, content, follow_redirects=True):  # noqa: ARG001
        captured.update(headers)
        captured["__url__"] = url
        return httpx.Response(200, json={"ok": True}, headers={"content-type": "application/json"})

    app.state.proxy_client.request = AsyncMock(side_effect=fake_request)
    return captured


class TestLifespanTakesTheToken:
    """The token leaves ``os.environ`` at boot and lives only on the panel map."""

    def test_the_token_leaves_the_environment_before_the_memory_guard(
        self, workspace_dir, monkeypatch
    ):
        """Every credential is out of ``os.environ`` when the process memory is guarded."""
        from osprey.interfaces import web_auth

        seen: list[bool] = []
        monkeypatch.setenv(TRAJECTORY_LAUNCH_TOKEN_ENV, LAUNCH_TOKEN)
        monkeypatch.setattr(
            web_auth,
            "guard_process_memory",
            lambda: seen.append(TRAJECTORY_LAUNCH_TOKEN_ENV in os.environ) or True,
        )

        for app, _client in panel_app(workspace_dir, [_declared()]):
            assert app.state.panel_auth_headers[TRAJECTORY_PANEL_ID] == {
                LAUNCH_TOKEN_HEADER: LAUNCH_TOKEN
            }
        assert seen == [False]

    def test_declared_panel_publishes_the_token(self, workspace_dir, monkeypatch):
        monkeypatch.setenv(TRAJECTORY_LAUNCH_TOKEN_ENV, LAUNCH_TOKEN)

        for app, _client in panel_app(workspace_dir, [_declared()]):
            assert app.state.panel_auth_headers[TRAJECTORY_PANEL_ID] == {
                LAUNCH_TOKEN_HEADER: LAUNCH_TOKEN
            }

    def test_the_agent_child_never_sees_the_token(self, workspace_dir, monkeypatch):
        """Gone from ``os.environ`` and from the SDK child's environment alike."""
        monkeypatch.setenv(TRAJECTORY_LAUNCH_TOKEN_ENV, LAUNCH_TOKEN)

        for app, _client in panel_app(workspace_dir, [_declared()]):
            assert TRAJECTORY_LAUNCH_TOKEN_ENV not in os.environ
            child_env = build_operator_child_env(str(workspace_dir), "session-1", app)
            # The SDK merges the child env over os.environ; check both halves.
            merged = {**os.environ, **child_env}
            assert TRAJECTORY_LAUNCH_TOKEN_ENV not in merged
            assert LAUNCH_TOKEN not in merged.values()

    def test_the_token_is_never_a_web_credential(self, workspace_dir, monkeypatch):
        monkeypatch.setenv(TRAJECTORY_LAUNCH_TOKEN_ENV, LAUNCH_TOKEN)

        for app, _client in panel_app(workspace_dir, [_declared()]):
            credentials = get_web_credentials(app)
            assert LAUNCH_TOKEN not in {
                value for value in vars(credentials).values() if isinstance(value, str)
            }

    def test_undeclared_panel_publishes_nothing_but_still_pops(self, workspace_dir, monkeypatch):
        monkeypatch.setenv(TRAJECTORY_LAUNCH_TOKEN_ENV, LAUNCH_TOKEN)

        for app, _client in panel_app(workspace_dir, []):
            assert TRAJECTORY_PANEL_ID not in app.state.panel_auth_headers
            assert TRAJECTORY_LAUNCH_TOKEN_ENV not in os.environ

    @pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
    def test_blank_token_publishes_nothing(self, workspace_dir, monkeypatch, blank):
        monkeypatch.setenv(TRAJECTORY_LAUNCH_TOKEN_ENV, blank)

        for app, _client in panel_app(workspace_dir, [_declared()]):
            assert TRAJECTORY_PANEL_ID not in app.state.panel_auth_headers
            assert TRAJECTORY_LAUNCH_TOKEN_ENV not in os.environ

    def test_unset_token_publishes_nothing(self, workspace_dir, monkeypatch):
        monkeypatch.delenv(TRAJECTORY_LAUNCH_TOKEN_ENV, raising=False)

        for app, _client in panel_app(workspace_dir, [_declared()]):
            assert TRAJECTORY_PANEL_ID not in app.state.panel_auth_headers

    def test_a_disabled_or_differently_named_panel_is_not_declared(self, monkeypatch):
        """Only a panel carrying the trajectory id counts as the declaration."""

        class _State:
            panel_auth_headers: dict = {}
            custom_panels = [{"id": "trajectory-viewer", "url": BACKEND_URL}]

        class _App:
            state = _State()

        monkeypatch.setenv(TRAJECTORY_LAUNCH_TOKEN_ENV, LAUNCH_TOKEN)
        hold_trajectory_launch_token(_App())
        publish_trajectory_launch_token(_App())
        assert _App.state.panel_auth_headers == {}
        assert TRAJECTORY_LAUNCH_TOKEN_ENV not in os.environ


class TestTheHop:
    """Toward ``/panel/trajectory/api/*`` the proxy vouches; the caller does not."""

    @pytest.fixture
    def booted(self, workspace_dir, monkeypatch):
        monkeypatch.setenv(TRAJECTORY_LAUNCH_TOKEN_ENV, LAUNCH_TOKEN)
        monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
        yield from panel_app(workspace_dir, [_declared()])

    def test_injects_secret_and_token_and_drops_inbound_claims(self, booted):
        app, client = booted
        captured = _capture_request(app)

        resp = client.post(
            f"/panel/{TRAJECTORY_PANEL_ID}/api/runs",
            headers={
                OWNER_HEADER: "mallory",
                "x-launch-token": "forged-token",
                "X_Launch_Token": "forged-underscore-token",
                TERMINAL_SECRET_HEADER: "forged-secret",
            },
            json={"go": True},
        )

        assert resp.status_code == 200
        assert captured["__url__"] == f"{BACKEND_URL}/api/runs"
        forwarded = _lower(captured)
        secret = get_web_credentials(app).operator_secret
        assert secret
        assert forwarded[TERMINAL_SECRET_HEADER.lower()] == secret
        assert forwarded[LAUNCH_TOKEN_HEADER.lower()] == LAUNCH_TOKEN
        assert forwarded[OWNER_HEADER.lower()] == "alice"
        values = " ".join(captured.values())
        for forged in ("forged-token", "forged-underscore-token", "forged-secret", "mallory"):
            assert forged not in values
        assert "x_launch_token" not in forwarded
        launch_names = [k for k in captured if k.lower() == LAUNCH_TOKEN_HEADER.lower()]
        assert launch_names == [LAUNCH_TOKEN_HEADER]

    @pytest.mark.parametrize(
        ("url", "config_defined"),
        [
            pytest.param("http://trajectory.example.com:8095", True, id="declared-off-box"),
            pytest.param(BACKEND_URL, False, id="loopback-registered"),
        ],
    )
    def test_an_unearned_backend_gets_no_token(
        self, workspace_dir, monkeypatch, url, config_defined
    ):
        monkeypatch.setenv(TRAJECTORY_LAUNCH_TOKEN_ENV, LAUNCH_TOKEN)
        panel = _declared(url)
        if not config_defined:
            panel.pop("configDefined")
        for app, client in panel_app(workspace_dir, [panel]):
            # Published or not, the gate is what keeps the token on the box.
            app.state.panel_auth_headers[TRAJECTORY_PANEL_ID] = {LAUNCH_TOKEN_HEADER: LAUNCH_TOKEN}
            captured = _capture_request(app)

            resp = client.get(
                f"/panel/{TRAJECTORY_PANEL_ID}/api/state",
                headers={"x-launch-token": "forged-token"},
            )

            assert resp.status_code == 200
            values = " ".join(captured.values())
            assert LAUNCH_TOKEN not in values
            assert "forged-token" not in values


@pytest.mark.parametrize(
    "name", ["x-launch-token", "X-Launch-Token", "X_LAUNCH_TOKEN", "x_launch-token"]
)
def test_inbound_launch_token_is_a_stripped_header(name):
    assert _is_stripped_header(name)
