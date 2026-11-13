"""The application's declaration: what it asks of the framework, and what each tier gets."""

from __future__ import annotations

import asyncio
import inspect
import re
from pathlib import Path
from typing import Any

import orbit_bump.tools as tools
import pytest
import yaml
from orbit_bump import APP_NAME
from orbit_bump import server as backend

from osprey.apps.manifest import AppManifest, launch_token_env, parse_manifest
from osprey.apps.projection import project_login
from osprey.apps.rules import AppSelection, LoginSelection, Reserved, fixed_names, validate
from osprey.apps.sdk import launch_token_env as sdk_launch_token_env
from osprey.mcp_server.startup import application_server
from osprey.profiles import presets
from tests.hooks.conftest import import_hook

APP_DIR = Path(presets.__file__).parent / "apps" / APP_NAME
CONFIG = {"services": {APP_NAME: {"port": 19000}}}
READ_ROUTES = [
    {"method": "GET", "path": backend.STATE_ROUTE},
    {"method": "GET", "path": backend.APPROVAL_ROUTE},
]


def _manifest() -> AppManifest:
    return parse_manifest(yaml.safe_load((APP_DIR / "app.yml").read_text()), app=APP_NAME)


def _login(tier: str, *, shown: bool = True) -> LoginSelection:
    return LoginSelection(
        audience="person",
        apps={APP_NAME: AppSelection(shown=shown)},
        tier=tier,  # type: ignore[arg-type]
        shared_cards=(),
    )


def _entry() -> dict[str, Any]:
    """The approval entry the render writes for ``apply_bump``."""
    result = project_login([_manifest()], CONFIG, _login("asked"))
    entry: dict[str, Any] = result["apps"][APP_NAME]["approval"]["apply_bump"]
    return entry


def test_every_app_scoped_identifier_is_the_applications_name() -> None:
    manifest = _manifest()
    assert manifest.name == APP_NAME
    assert list(manifest.containers) == [APP_NAME]
    container = manifest.containers[APP_NAME]
    assert container.state == f"var/{APP_NAME}"
    assert container.runtime == "osprey"
    assert container.drives_target and container.control_identity
    assert manifest.panel is not None
    assert manifest.panel.id == APP_NAME
    assert manifest.panel.container == APP_NAME
    assert manifest.panel.launch_token
    assert manifest.panel.roster_grant is not None
    assert manifest.panel.roster_grant.single_user_alias
    assert manifest.tools is not None
    assert manifest.tools.module.split(".")[0] == APP_NAME.replace("-", "_")
    assert manifest.skills == [APP_NAME]


def test_the_declaration_passes_the_deployment_rules() -> None:
    reserved = Reserved(
        services=frozenset(), panels=frozenset(), servers=frozenset(), names=fixed_names()
    )
    logins = {"readwrite": _login("asked"), "admin": _login("allowed"), "readonly": _login("never")}
    validate([_manifest()], logins=logins, reserved=reserved)


def test_the_application_needs_no_network_to_build() -> None:
    assert _manifest().vendor == []
    assert not (APP_DIR / "panel" / "vendor").exists()


def test_the_backend_reads_the_token_the_deployment_hands_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OSPREY_APP_NAME", APP_NAME)
    assert sdk_launch_token_env() == launch_token_env(APP_NAME)


def test_the_backend_serves_the_routes_the_declaration_names() -> None:
    manifest = _manifest()
    assert manifest.panel is not None
    assert [(route.method, route.path) for route in manifest.panel.agent_routes] == [
        ("GET", backend.STATE_ROUTE),
        ("GET", backend.APPROVAL_ROUTE),
        ("POST", backend.RECOMPUTE_ROUTE),
        ("POST", backend.DISCARD_ROUTE),
        ("POST", backend.COMPUTE_ROUTE),
        ("POST", backend.APPLY_ROUTE),
    ]
    # The draft is the operator's input: the panel posts it, the agent only reads it.
    assert backend.DRAFT_ROUTE not in {route.path for route in manifest.panel.agent_routes}
    assert manifest.tools is not None
    assert [entry.state_path for entry in manifest.tools.approval] == [backend.APPROVAL_ROUTE]


def test_the_tool_server_names_the_backends_routes_and_settle_bound() -> None:
    assert tools.STATE_ROUTE == backend.STATE_ROUTE
    assert tools.APPROVAL_ROUTE == backend.APPROVAL_ROUTE
    assert tools.RECOMPUTE_ROUTE == backend.RECOMPUTE_ROUTE
    assert tools.DISCARD_ROUTE == backend.DISCARD_ROUTE
    assert tools.COMPUTE_ROUTE == backend.COMPUTE_ROUTE
    assert tools.APPLY_ROUTE == backend.APPLY_ROUTE
    assert tools.SETTLE_TIMEOUT_S == backend.SETTLE_TIMEOUT_S


def test_the_apply_bounds_cover_the_writes_and_the_settle() -> None:
    manifest = _manifest()
    assert manifest.panel is not None
    assert tools.APPLY_TIMEOUT_S > tools.WRITE_BUDGET_S + backend.SETTLE_TIMEOUT_S
    # The panel door holds the apply at least as long as the tool waits for it.
    assert manifest.panel.read_timeout_s is not None
    assert manifest.panel.read_timeout_s >= tools.APPLY_TIMEOUT_S


def test_the_tool_server_registers_exactly_the_declared_tools() -> None:
    manifest = _manifest()
    assert manifest.tools is not None
    # Only apply writes a channel; the others read, plan, drop a plan or rebuild a model.
    assert manifest.tools.read == ["bump_state", "plan_bump", "discard_bump", "recompute_orm"]
    declared = {*manifest.tools.read, *(entry.tool for entry in manifest.tools.approval)}
    assert declared == {*manifest.tools.read, "apply_bump"}
    server = application_server(APP_NAME)
    tools.register(server)
    assert {tool.name for tool in asyncio.run(server.list_tools())} == declared


def test_the_apply_is_bound_to_the_target_the_plan_and_its_setpoints() -> None:
    manifest = _manifest()
    assert manifest.tools is not None
    [approval] = manifest.tools.approval
    assert approval.tool == "apply_bump"
    assert approval.display_lines_key == "display_lines"
    assert [(item.field, item.key) for item in approval.bind.state_fields] == [
        ("approved_plan_id", "plan_id"),
        ("approved_setpoints", "setpoint_digest"),
    ]
    assert approval.bind.input_digests == []
    assert approval.require.all_of == ["approved_target", "approved_plan_id", "approved_setpoints"]
    assert "{{ state.plan_id }}" in approval.prompt
    assert "{{ input.plan_id }}" in approval.prompt


def test_the_hook_binds_both_flat_keys_of_the_approval_state() -> None:
    approval = import_hook("osprey_approval")
    entry = _entry()
    state = {"plan_id": "p-7", "setpoint_digest": "d" * 64, "display_lines": ["HC 1: 0 -> 1"]}
    binding = approval._app_binding(entry, {"plan_id": "p-7"}, state, "va")
    assert binding == {
        "approved_target": "va",
        "approved_plan_id": "p-7",
        "approved_setpoints": "d" * 64,
    }
    assert approval._app_requirement_unmet(entry, binding) is None


def test_the_hook_cannot_ask_when_no_plan_is_ready() -> None:
    approval = import_hook("osprey_approval")
    entry = _entry()
    binding = approval._app_binding(entry, {"plan_id": "p-7"}, {}, "va")
    assert approval._app_requirement_unmet(entry, binding) is not None


@pytest.mark.parametrize("tier", ["asked", "allowed"])
def test_a_writing_tier_gets_every_route_and_the_launch_token(tier: str) -> None:
    result = project_login([_manifest()], CONFIG, _login(tier))
    block = result["web"]["panels"][APP_NAME]
    for route in (
        backend.RECOMPUTE_ROUTE,
        backend.DISCARD_ROUTE,
        backend.COMPUTE_ROUTE,
        backend.APPLY_ROUTE,
    ):
        assert {"method": "POST", "path": route} in block["agent_routes"]
    assert block["launch_token_env"] == launch_token_env(APP_NAME)
    assert block["read_timeout_s"] == _manifest().panel.read_timeout_s  # type: ignore[union-attr]
    assert result["mcp_servers"][APP_NAME]["writes"] == ["apply_bump"]


def test_a_never_tier_shows_the_panel_with_the_read_routes_only() -> None:
    result = project_login([_manifest()], CONFIG, _login("never"))
    block = result["web"]["panels"][APP_NAME]
    assert block["agent_routes"] == READ_ROUTES
    assert "launch_token_env" not in block
    assert result["mcp_servers"][APP_NAME]["permissions"]["allow"] == [
        "bump_state",
        "plan_bump",
        "discard_bump",
        "recompute_orm",
    ]


def test_a_job_can_run_no_tool() -> None:
    result = project_login([_manifest()], CONFIG, _login("allowed"))
    assert sorted(result["dispatch"]["denied_app_tools"]) == [
        f"mcp__{APP_NAME}__{name}"
        for name in ("apply_bump", "bump_state", "discard_bump", "plan_bump", "recompute_orm")
    ]


def test_tool_texts_name_no_protocol_and_no_facility_prefix() -> None:
    manifest = _manifest()
    assert manifest.tools is not None and manifest.tools.errors is not None
    texts = [
        inspect.getsource(tools),
        manifest.summary or "",
        manifest.tools.approval[0].prompt,
        manifest.tools.errors.unreachable or "",
        manifest.tools.errors.refused or "",
    ]
    for text in texts:
        assert not re.search(r"\b(EPICS|PV|PVs|Channel Access|pyepics)\b", text), text
        assert not re.search(r"\b(SR|HCM|VCM|MAG|DIAG)\b", text), text
        assert "SR:" not in text
