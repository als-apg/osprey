"""The bundled example's declaration: what it asks of the framework, and what each tier gets."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml
from knob_example import APP_NAME, STATE_ROUTE, WRITE_ROUTE
from knob_example import server as backend
from knob_example.tools import build_server

from osprey.apps.manifest import AppManifest, launch_token_env, parse_manifest
from osprey.apps.projection import project_login
from osprey.apps.rules import AppSelection, LoginSelection, Reserved, fixed_names, validate
from osprey.profiles import presets

APP_DIR = Path(presets.__file__).parent / "apps" / APP_NAME
CONFIG = {"services": {APP_NAME: {"port": 19000}}}


def _manifest() -> AppManifest:
    return parse_manifest(yaml.safe_load((APP_DIR / "app.yml").read_text()), app=APP_NAME)


def _login(tier: str, *, shown: bool = True) -> LoginSelection:
    return LoginSelection(
        audience="person",
        apps={APP_NAME: AppSelection(shown=shown)},
        tier=tier,  # type: ignore[arg-type]
        shared_cards=(),
    )


def test_every_app_scoped_identifier_is_the_applications_name() -> None:
    manifest = _manifest()
    assert manifest.name == APP_NAME
    assert list(manifest.containers) == [APP_NAME]
    assert manifest.containers[APP_NAME].state == f"var/{APP_NAME}"
    assert manifest.panel is not None
    assert manifest.panel.id == APP_NAME
    assert manifest.panel.container == APP_NAME
    assert manifest.tools is not None
    assert manifest.tools.module.split(".")[0] == APP_NAME.replace("-", "_")


def test_the_declaration_passes_the_deployment_rules() -> None:
    reserved = Reserved(
        services=frozenset(), panels=frozenset(), servers=frozenset(), names=fixed_names()
    )
    logins = {"readwrite": _login("asked"), "admin": _login("allowed"), "readonly": _login("never")}
    validate([_manifest()], logins=logins, reserved=reserved)


def test_the_example_needs_no_network_to_build() -> None:
    assert _manifest().vendor == []
    assert not (APP_DIR / "panel" / "vendor").exists()


def test_the_backend_reads_the_token_the_deployment_hands_it() -> None:
    assert backend.LAUNCH_TOKEN_ENV == launch_token_env(APP_NAME)


def test_the_backend_serves_the_routes_the_declaration_names() -> None:
    manifest = _manifest()
    assert manifest.panel is not None
    assert [(route.method, route.path) for route in manifest.panel.agent_routes] == [
        ("GET", STATE_ROUTE),
        ("POST", WRITE_ROUTE),
    ]
    assert manifest.tools is not None
    assert [entry.state_path for entry in manifest.tools.approval] == [STATE_ROUTE]


def test_the_tool_server_registers_exactly_the_declared_tools() -> None:
    manifest = _manifest()
    assert manifest.tools is not None
    declared = {*manifest.tools.read, *(entry.tool for entry in manifest.tools.approval)}
    assert {tool.name for tool in asyncio.run(build_server().list_tools())} == declared


def test_the_write_is_bound_to_the_target_and_the_readback() -> None:
    manifest = _manifest()
    assert manifest.tools is not None
    [approval] = manifest.tools.approval
    assert approval.tool == "set_corrector"
    assert {"approved_target", "approved_readback"} <= set(approval.require.all_of)


@pytest.mark.parametrize("tier", ["asked", "allowed"])
def test_a_writing_tier_gets_the_write_route_and_the_launch_token(tier: str) -> None:
    result = project_login([_manifest()], CONFIG, _login(tier))
    block = result["web"]["panels"][APP_NAME]
    assert {"method": "POST", "path": WRITE_ROUTE} in block["agent_routes"]
    assert block["launch_token_env"] == launch_token_env(APP_NAME)
    assert result["mcp_servers"][APP_NAME]["writes"] == ["set_corrector"]


def test_a_never_tier_shows_the_panel_with_the_state_route_only() -> None:
    result = project_login([_manifest()], CONFIG, _login("never"))
    block = result["web"]["panels"][APP_NAME]
    assert block["agent_routes"] == [{"method": "GET", "path": STATE_ROUTE}]
    assert "launch_token_env" not in block
    assert result["mcp_servers"][APP_NAME]["permissions"]["allow"] == ["read_corrector"]


def test_the_write_tool_binds_fetched_state_so_a_job_cannot_run_it() -> None:
    result = project_login([_manifest()], CONFIG, _login("allowed"))
    assert result["dispatch"]["denied_app_tools"] == [f"mcp__{APP_NAME}__set_corrector"]
