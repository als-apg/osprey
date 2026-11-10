"""Projecting an application's panel onto one login's ``web.panels`` block."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest

from osprey.apps.errors import AppManifestError
from osprey.apps.manifest import AppManifest, launch_token_env, parse_manifest
from osprey.apps.projection import project_login
from osprey.apps.rules import AppSelection, LoginSelection

PORT = 18950

DIAL: dict[str, Any] = {
    "name": "dial-demo",
    "containers": {
        "dial-demo": {
            "runtime": "osprey",
            "health": "/health",
            "drives_target": True,
        },
    },
    "panel": {
        "label": "Dial",
        "container": "dial-demo",
        "entry": "/ui/",
        "launch_token": True,
        "agent_routes": [
            ["GET", "/api/state"],
            {"method": "HEAD", "path": "/api/ping"},
            ["POST", "/api/setpoint"],
            ["DELETE", "/api/setpoint"],
        ],
    },
}

SAFE_ROUTES = [
    {"method": "GET", "path": "/api/state"},
    {"method": "HEAD", "path": "/api/ping"},
]
ALL_ROUTES = [
    *SAFE_ROUTES,
    {"method": "POST", "path": "/api/setpoint"},
    {"method": "DELETE", "path": "/api/setpoint"},
]

CONFIG: dict[str, Any] = {"services": {"dial-demo": {"port": PORT}}}

#: What a login that shows no panel and no tool server is projected.
NOTHING: dict[str, Any] = {
    "web": {"panels": {}},
    "web_panels": [],
    "mcp_servers": {},
    "apps": {},
    "dispatch": {"denied_app_tools": []},
}


def _manifest(data: dict[str, Any] = DIAL, **panel: Any) -> AppManifest:
    data = copy.deepcopy(data)
    data["panel"].update(panel)
    return parse_manifest(data, app=data["name"])


def _login(
    tier: str = "allowed",
    *,
    shown: bool = True,
    app: str = "dial-demo",
    audience: str = "person",
) -> LoginSelection:
    return LoginSelection(
        audience=audience,  # type: ignore[arg-type]
        apps={app: AppSelection(shown=shown)},
        tier=tier,  # type: ignore[arg-type]
        shared_cards=("control-room",) if audience == "shared" else (),
    )


def _block(result: dict[str, Any], panel_id: str = "dial-demo") -> dict[str, Any]:
    return result["web"]["panels"][panel_id]


def test_a_shown_panel_is_projected_with_its_address_entry_and_application() -> None:
    result = project_login([_manifest()], CONFIG, _login())
    block = _block(result)
    assert block["label"] == "Dial"
    assert block["url"] == f"http://127.0.0.1:{PORT}"
    assert block["path"] == "/ui/"
    assert block["health_endpoint"] == "/health"
    assert block["app"] == "dial-demo"
    assert "entry" not in block
    assert result["web_panels"] == ["dial-demo"]


def test_the_port_is_read_from_the_projected_service_never_recomputed() -> None:
    config = {"services": {"dial-demo": {"port": 19999}}}
    block = _block(project_login([_manifest()], config, _login()))
    assert block["url"] == "http://127.0.0.1:19999"


def test_a_shown_panel_whose_service_is_not_projected_is_refused() -> None:
    with pytest.raises(AppManifestError) as info:
        project_login([_manifest()], {"services": {}}, _login())
    assert info.value.violations[0].app == "dial-demo"
    assert "services.dial-demo.port" in str(info.value)


def test_the_panel_id_names_the_block_and_the_selection() -> None:
    result = project_login([_manifest(id="dial")], CONFIG, _login())
    assert list(result["web"]["panels"]) == ["dial"]
    assert result["web_panels"] == ["dial"]


def test_a_container_without_health_projects_no_health_endpoint() -> None:
    data = copy.deepcopy(DIAL)
    del data["containers"]["dial-demo"]["health"]
    block = _block(project_login([parse_manifest(data, app="dial-demo")], CONFIG, _login()))
    assert "health_endpoint" not in block


@pytest.mark.parametrize(
    "selection",
    [
        _login(shown=False),
        _login(app="another-app"),
    ],
    ids=["hidden", "not-in-apps-block"],
)
def test_a_panel_the_login_does_not_show_is_not_projected(selection: LoginSelection) -> None:
    assert project_login([_manifest()], CONFIG, selection) == NOTHING


def test_an_application_without_a_panel_projects_nothing() -> None:
    data = copy.deepcopy(DIAL)
    del data["panel"]
    result = project_login([parse_manifest(data, app="dial-demo")], CONFIG, _login())
    assert result == NOTHING


@pytest.mark.parametrize("tier", ["asked", "allowed"])
def test_an_arming_login_gets_write_routes_and_the_launch_token(tier: str) -> None:
    block = _block(project_login([_manifest()], CONFIG, _login(tier)))
    assert block["agent_routes"] == ALL_ROUTES
    assert block["launch_token_env"] == launch_token_env("dial-demo") == "DIAL_DEMO_LAUNCH_TOKEN"
    assert "arm_writes" not in block


def test_a_never_login_gets_safe_routes_only_and_no_launch_token() -> None:
    block = _block(project_login([_manifest()], CONFIG, _login("never")))
    assert block["agent_routes"] == SAFE_ROUTES
    assert "launch_token_env" not in block
    assert "arm_writes" not in block


@pytest.mark.parametrize("tier", ["never", "asked", "allowed"])
def test_a_shared_audience_never_gets_write_routes_or_the_launch_token(tier: str) -> None:
    block = _block(project_login([_manifest()], CONFIG, _login(tier, audience="shared")))
    assert block["agent_routes"] == SAFE_ROUTES
    assert "launch_token_env" not in block
    assert "arm_writes" not in block


def test_an_arming_login_gets_no_launch_token_the_manifest_does_not_ask_for() -> None:
    block = _block(project_login([_manifest(launch_token=False)], CONFIG, _login()))
    assert "launch_token_env" not in block
    assert block["agent_routes"] == ALL_ROUTES


def test_read_timeout_is_projected_only_when_the_manifest_sets_it() -> None:
    assert "read_timeout_s" not in _block(project_login([_manifest()], CONFIG, _login()))
    timed = _block(project_login([_manifest(read_timeout_s=120)], CONFIG, _login()))
    assert timed["read_timeout_s"] == 120
    unbounded = _block(project_login([_manifest(read_timeout_s=None)], CONFIG, _login()))
    assert "read_timeout_s" in unbounded and unbounded["read_timeout_s"] is None


def test_the_projection_is_pure() -> None:
    config = copy.deepcopy(CONFIG)
    project_login([_manifest()], config, _login())
    assert config == CONFIG


def test_the_projected_block_is_a_first_party_panel_of_the_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The terminal reads the block as config-defined: framed as the deployment's own."""
    from osprey.interfaces.web_terminal import app as terminal
    from osprey.interfaces.web_terminal.panel_cap import opaque_panel_kind

    block = _block(project_login([_manifest(read_timeout_s=30)], CONFIG, _login()))
    rendered = {"web": {"panels": {"dial-demo": {**block, "enabled": True}}}}
    monkeypatch.setattr("osprey_connectors.workspace.load_osprey_config", lambda *a, **k: rendered)

    _, custom, _ = terminal._load_panel_config()
    (panel,) = custom
    assert panel["configDefined"] is True
    assert panel["app"] == "dial-demo"
    assert panel["agentRoutes"] == ALL_ROUTES
    assert panel["readTimeoutS"] == 30

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(custom_panels=custom)))
    assert opaque_panel_kind(request, "dial-demo") is None  # type: ignore[arg-type]
