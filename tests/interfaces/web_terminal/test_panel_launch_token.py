"""An application panel's launch token: held by the terminal, spent on one hop.

``web.panels.<id>.launch_token_env`` names the variable. The terminal reads it
from its own environment at startup and injects the value as ``X-Launch-Token``
on that panel's ``/panel/<id>`` hop, so the browser never holds it: ``GET
/api/panels`` carries the name only. An inbound ``X-Launch-Token`` is dropped on
every panel hop, armed or not. The grant reaches the terminal through compose
from what the deploy's shell gate cleared, and on a deployment with no roster
through the deploy ``.env``.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from starlette.websockets import WebSocketDisconnect

from osprey.interfaces.web_terminal.app import _load_panel_config
from osprey.interfaces.web_terminal.panel_cap import PANEL_CAP_SEGMENT, mint_cap
from osprey.mcp_env import load_dotenv_from_project
from osprey.utils.launch_token import LAUNCH_TOKEN_HEADER
from tests.interfaces.web_terminal._proxy_fakes import _FakeConnect, _patch_connect, panel_app

#: The panel holding a token, the variable naming it and the value it holds.
PANEL_ID = "knob"
TOKEN_ENV = "KNOB_LAUNCH_TOKEN"
TOKEN = "held-launch-token-value"

#: A value a caller tries to present for itself.
FORGED = "forged-launch-token-value"

_LOGIN_TIER = "osprey.interfaces.web_terminal.routes.proxy._login_tier"


def _panel(*, panel_id: str = PANEL_ID, url: str = "http://127.0.0.1:9300", **extra) -> dict:
    """A config-defined application panel as ``_load_panel_config`` forwards it."""
    return {
        "id": panel_id,
        "label": panel_id.upper(),
        "url": url,
        "configDefined": True,
        "app": panel_id,
        "agentRoutes": [],
        **extra,
    }


#: The same application with no token named: what a render that does not arm it carries.
UNARMED_RENDER_PANEL = _panel()

#: The panel on a render that arms the application.
ARMED_RENDER_PANEL = _panel(launchTokenEnv=TOKEN_ENV)


@pytest.fixture(autouse=True)
def _environment():
    """Each test sets the token its own way; nothing it sets outlives it."""
    with patch.dict(os.environ):
        os.environ.pop(TOKEN_ENV, None)
        os.environ["OSPREY_TERMINAL_USER"] = "alice"
        yield


def _capture(app) -> list[dict[str, str]]:
    """Stub the upstream HTTP client; return the headers of each request it was sent."""
    sent: list[dict[str, str]] = []

    async def fake_request(*, method, url, headers, content, follow_redirects=True):  # noqa: ARG001
        sent.append(dict(headers))
        return httpx.Response(200, json={"ok": True}, headers={"content-type": "application/json"})

    app.state.proxy_client.request = AsyncMock(side_effect=fake_request)
    return sent


def _lowered(headers: dict[str, str]) -> dict[str, str]:
    return {name.lower(): value for name, value in headers.items()}


def _token_on(headers: dict[str, str]) -> str | None:
    return _lowered(headers).get(LAUNCH_TOKEN_HEADER.lower())


@pytest.fixture
def armed(workspace_dir):
    """The token set and named, on a login whose tier arms the application."""
    os.environ[TOKEN_ENV] = TOKEN
    with patch(_LOGIN_TIER, return_value="allowed"):
        yield from panel_app(
            workspace_dir, [dict(ARMED_RENDER_PANEL), _panel(panel_id="other", app="other")]
        )


# ---------------------------------------------------------------------------
# What the terminal holds, and what the browser is told
# ---------------------------------------------------------------------------


def test_the_panel_dict_carries_the_variable_name_and_never_the_value():
    os.environ[TOKEN_ENV] = TOKEN
    with patch(
        "osprey_connectors.workspace.load_osprey_config",
        return_value={
            "web": {
                "panels": {
                    PANEL_ID: {
                        "url": "http://127.0.0.1:9300",
                        "app": PANEL_ID,
                        "launch_token_env": TOKEN_ENV,
                    }
                }
            }
        },
    ):
        _enabled, (panel,), _default = _load_panel_config()

    assert panel["launchTokenEnv"] == TOKEN_ENV
    assert TOKEN not in repr(panel)


def test_a_block_naming_no_token_carries_no_key():
    with patch(
        "osprey_connectors.workspace.load_osprey_config",
        return_value={"web": {"panels": {PANEL_ID: {"url": "http://127.0.0.1:9300"}}}},
    ):
        _enabled, (panel,), _default = _load_panel_config()

    assert "launchTokenEnv" not in panel


def test_the_token_sits_beside_the_other_panel_credentials(armed):
    app, _client = armed
    assert app.state.panel_auth_headers[PANEL_ID] == {LAUNCH_TOKEN_HEADER: TOKEN}


def test_get_api_panels_never_contains_the_value(armed):
    _app, client = armed

    response = client.get("/api/panels")

    assert response.status_code == 200
    assert TOKEN not in response.text
    assert TOKEN_ENV in response.text


@pytest.mark.parametrize("value", [None, "", "   "])
def test_a_variable_holding_nothing_holds_no_credential(workspace_dir, caplog, value):
    """The backend refuses a launch it was not armed for; the operator sees why here."""
    if value is not None:
        os.environ[TOKEN_ENV] = value
    with patch(_LOGIN_TIER, return_value="allowed"):
        for app, client in panel_app(workspace_dir, [dict(ARMED_RENDER_PANEL)]):
            sent = _capture(app)
            assert PANEL_ID not in app.state.panel_auth_headers
            client.get(f"/panel/{PANEL_ID}/api/state")
            assert _token_on(sent[0]) is None
    assert TOKEN_ENV in caplog.text


@pytest.mark.parametrize("delivery", ["compose_environment", "deploy_dotenv"])
def test_the_terminal_reads_the_token_whichever_way_the_deployment_delivers_it(
    workspace_dir, tmp_path, monkeypatch, delivery
):
    """A roster hands it through compose; a roster-less host reads the deploy ``.env``."""
    if delivery == "compose_environment":
        os.environ[TOKEN_ENV] = TOKEN
    else:
        (tmp_path / ".env").write_text(f"{TOKEN_ENV}={TOKEN}\n", encoding="utf-8")
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
        # What `osprey web` runs before the app is served.
        load_dotenv_from_project(include_sensitive=True)

    with patch(_LOGIN_TIER, return_value="allowed"):
        for app, client in panel_app(workspace_dir, [dict(ARMED_RENDER_PANEL)]):
            sent = _capture(app)
            client.get(f"/panel/{PANEL_ID}/api/state")
            assert _token_on(sent[0]) == TOKEN


# ---------------------------------------------------------------------------
# The hop
# ---------------------------------------------------------------------------


def test_the_token_is_injected_on_that_panels_http_hop(armed):
    app, client = armed
    sent = _capture(app)

    client.post(f"/panel/{PANEL_ID}/api/set", json={})

    assert _token_on(sent[0]) == TOKEN


def test_the_token_is_injected_on_that_panels_websocket_hop(armed):
    _app, client = armed
    fake = _FakeConnect()

    with _patch_connect(fake):
        with client.websocket_connect(f"/panel/{PANEL_ID}/ws"):
            pass

    assert _token_on(fake.kwargs["additional_headers"]) == TOKEN


def test_no_other_panel_receives_it(armed):
    app, client = armed
    sent = _capture(app)

    client.get("/panel/other/api/state")

    assert _token_on(sent[0]) is None


def test_a_caller_cannot_replace_the_held_value(armed):
    app, client = armed
    sent = _capture(app)

    client.post(f"/panel/{PANEL_ID}/api/set", json={}, headers={LAUNCH_TOKEN_HEADER: FORGED})

    assert _token_on(sent[0]) == TOKEN


def test_an_off_box_backend_is_not_sent_the_token(workspace_dir):
    """The same gate every credential on this hop answers to."""
    os.environ[TOKEN_ENV] = TOKEN
    panel = _panel(url="http://elsewhere.example.org:9300", launchTokenEnv=TOKEN_ENV)
    with patch(_LOGIN_TIER, return_value="allowed"):
        for app, client in panel_app(workspace_dir, [panel]):
            sent = _capture(app)
            client.get(f"/panel/{PANEL_ID}/api/state")
            assert _token_on(sent[0]) is None


def test_a_cap_path_never_opens_a_projected_panel(armed):
    """A cap serves opaque panels only, and a config-defined panel is never opaque."""
    from osprey.interfaces.web_auth import get_web_credentials

    app, client = armed
    cap = mint_cap(get_web_credentials(app), "alice", PANEL_ID)
    fake = _FakeConnect()

    with _patch_connect(fake), pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/panel/{PANEL_ID}/{PANEL_CAP_SEGMENT}/{cap}/ws"):
            pass

    assert fake.kwargs is None


# ---------------------------------------------------------------------------
# The inbound strip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [LAUNCH_TOKEN_HEADER, LAUNCH_TOKEN_HEADER.lower(), LAUNCH_TOKEN_HEADER.replace("-", "_")],
)
def test_a_caller_supplied_value_never_reaches_the_backend_on_an_unarmed_hop(workspace_dir, name):
    with patch(_LOGIN_TIER, return_value="never"):
        for app, client in panel_app(workspace_dir, [dict(UNARMED_RENDER_PANEL)]):
            sent = _capture(app)
            client.get(f"/panel/{PANEL_ID}/api/state", headers={name: FORGED})
            assert len(sent) == 1
            assert FORGED not in sent[0].values()


def test_a_caller_supplied_value_is_dropped_on_every_panel(workspace_dir):
    """Not an application's panel at all: the header is still never forwarded."""
    plain = {"id": "plain", "label": "P", "url": "http://127.0.0.1:9301", "configDefined": True}
    for app, client in panel_app(workspace_dir, [plain]):
        sent = _capture(app)
        client.get("/panel/plain/api/state", headers={LAUNCH_TOKEN_HEADER: FORGED})
        assert FORGED not in sent[0].values()


# ---------------------------------------------------------------------------
# The token follows arming
# ---------------------------------------------------------------------------


def test_an_asked_login_forwards_the_write_with_the_token(workspace_dir):
    os.environ[TOKEN_ENV] = TOKEN
    with patch(_LOGIN_TIER, return_value="asked"):
        for app, client in panel_app(workspace_dir, [dict(ARMED_RENDER_PANEL)]):
            sent = _capture(app)
            response = client.post(f"/panel/{PANEL_ID}/api/set", json={})
            assert response.status_code == 200
            assert _token_on(sent[0]) == TOKEN


@pytest.mark.parametrize("tier", ["unknown", "never"])
def test_a_login_that_does_not_arm_is_refused_the_write(workspace_dir, tier):
    os.environ[TOKEN_ENV] = TOKEN
    with patch(_LOGIN_TIER, return_value=tier):
        for app, client in panel_app(workspace_dir, [dict(ARMED_RENDER_PANEL)]):
            sent = _capture(app)
            response = client.post(f"/panel/{PANEL_ID}/api/set", json={})
            assert response.status_code == 403
            assert sent == []


@pytest.mark.parametrize("tier", ["unknown", "never"])
def test_a_read_on_a_login_that_does_not_arm_carries_no_token(workspace_dir, tier):
    """The render holds the token, this login's own rung does not arm: the read goes bare."""
    os.environ[TOKEN_ENV] = TOKEN
    with patch(_LOGIN_TIER, return_value=tier):
        for app, client in panel_app(workspace_dir, [dict(ARMED_RENDER_PANEL)]):
            sent = _capture(app)
            response = client.get(f"/panel/{PANEL_ID}/api/state")
            assert response.status_code == 200
            assert _token_on(sent[0]) is None


def test_a_panel_token_on_a_never_login_opens_no_socket(workspace_dir):
    os.environ[TOKEN_ENV] = TOKEN
    fake = _FakeConnect()
    with patch(_LOGIN_TIER, return_value="never"):
        for _app, client in panel_app(workspace_dir, [dict(ARMED_RENDER_PANEL)]):
            with _patch_connect(fake), pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(f"/panel/{PANEL_ID}/ws"):
                    pass
    assert fake.kwargs is None
