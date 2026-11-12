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


TOOLS_ONLY: dict[str, Any] = {
    "name": "dial-tools",
    "tools": {"module": "dial_tools.tools", "read": ["read_dial"]},
}


def test_an_application_with_no_container_still_projects_its_tool_server() -> None:
    manifest = parse_manifest(copy.deepcopy(TOOLS_ONLY), app="dial-tools")
    result = project_login([manifest], {}, _login(app="dial-tools"))
    assert result["mcp_servers"]["dial-tools"]["args"] == ["-m", "dial_tools.tools"]
    assert result["apps"]["dial-tools"]["tools"]["read"] == ["read_dial"]
    assert result["web"]["panels"] == {}


def test_an_application_with_no_container_hidden_by_the_login_projects_nothing() -> None:
    manifest = parse_manifest(copy.deepcopy(TOOLS_ONLY), app="dial-tools")
    result = project_login([manifest], {}, _login(app="dial-tools", shown=False))
    assert result == NOTHING


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


#: The three presentation facts a panel declares for its sidebar row.
PRESENTATION = {
    "icon": "sliders",
    "description": "Turn the demo dial and watch it settle",
    "help": "how-to/apps/build-your-own-app.html",
}


def test_the_panel_presentation_is_projected_when_the_manifest_declares_it() -> None:
    block = _block(project_login([_manifest(**PRESENTATION)], CONFIG, _login()))
    assert {key: block[key] for key in PRESENTATION} == PRESENTATION


def test_an_undeclared_presentation_key_stays_absent() -> None:
    block = _block(project_login([_manifest()], CONFIG, _login()))
    assert not set(PRESENTATION) & set(block)
    only_icon = _block(project_login([_manifest(icon="sliders")], CONFIG, _login()))
    assert only_icon["icon"] == "sliders"
    assert "description" not in only_icon and "help" not in only_icon


def test_a_help_page_may_be_an_absolute_https_url() -> None:
    url = "https://example.org/dial/manual.html#start"
    assert _block(project_login([_manifest(help=url)], CONFIG, _login()))["help"] == url


@pytest.mark.parametrize(
    "page",
    [
        "../secrets.html",
        "how-to/../../outside.html",
        "how-to/%2e%2e/%2E%2E/outside.html",
        "%2e./outside.html",
        "/how-to/apps/index.html",
        "http://example.org/dial.html",
        "javascript:alert(1)",
        "",
    ],
)
def test_a_help_page_outside_the_docs_root_or_not_https_is_refused(page: str) -> None:
    with pytest.raises(AppManifestError) as excinfo:
        _manifest(help=page)
    assert excinfo.value.field == "panel.help"


@pytest.mark.parametrize("key", ["icon", "description"])
def test_an_empty_icon_or_description_is_refused(key: str) -> None:
    with pytest.raises(AppManifestError) as excinfo:
        _manifest(**{key: ""})
    assert excinfo.value.field == f"panel.{key}"
