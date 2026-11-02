"""An application's tool server is a facility server.

:func:`~osprey.apps.projection.project_login` projects one server per shown
application, named for it, whose ``writes:`` are the application's approval
tools; the registry, the kill switch and the rendered ``hook_config.json`` then
treat it exactly as any facility server that declares ``writes:``. The tests
pin the projected keys, their path through the profile schema, the loader and
the persisted config, and the two renders: an ``allowed`` render gates the
application's write tools, a ``never`` render denies them statically.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

import pytest
import yaml

from osprey.apps.manifest import AppManifest, parse_manifest
from osprey.apps.projection import project_login
from osprey.apps.rules import AppSelection, LoginSelection
from osprey.cli.build_profile_load import _parse_profile
from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager
from osprey.registry.mcp import (
    _APPROVAL,
    _APPROVAL_PREFLIGHT,
    _WRITES_CHECK,
    _custom_server_from_spec,
    facility_write_tools,
    resolve_servers,
)
from osprey_connectors.errors import BuildProfileError
from tests._preset_data import create_project_as_built

PORT = 18951

KNOB: dict[str, Any] = {
    "name": "knob",
    "containers": {"knob": {"runtime": "osprey", "health": "/health"}},
    "panel": {
        "label": "Knob",
        "container": "knob",
        "agent_routes": [["GET", "/api/state"], ["POST", "/api/set"]],
    },
    "tools": {
        "module": "knob_app.tools",
        "shown_when": "panel",
        "read": ["knob_read"],
        "approval": [
            {
                "tool": "knob_set",
                "prompt": "Set the knob to {value}",
                "state_path": "/api/state",
                "display_lines_key": "lines",
                "bind": {
                    "state_fields": [{"field": "approved_state_sha", "key": "sha"}],
                    "input_digests": [{"field": "approved_digest", "over": ["value"]}],
                },
                "require": {
                    "all_of": ["approved_target", "approved_digest"],
                    "one_of": [["approved_state_sha"]],
                },
            },
            {
                "tool": "knob_zero",
                "prompt": "Zero the knob",
                "require": {"all_of": ["approved_target"]},
            },
        ],
        "errors": {"unreachable": "The knob panel is down.", "refused": "Reopen the panel."},
    },
}

#: An application with no panel whose every tool is denied in jobs.
DIAL: dict[str, Any] = {
    "name": "dial",
    "containers": {"dial": {"runtime": "osprey"}},
    "tools": {
        "module": "dial_app.tools",
        "deny_in_jobs": True,
        "read": ["dial_read"],
        "approval": [{"tool": "dial_turn", "prompt": "Turn the dial"}],
    },
}

CONFIG: dict[str, Any] = {"services": {"knob": {"port": PORT}}}

KNOB_WRITES = ["mcp__knob__knob_set", "mcp__knob__knob_zero"]


def _manifest(data: dict[str, Any]) -> AppManifest:
    return parse_manifest(copy.deepcopy(data), app=data["name"])


def _login(tier: str = "allowed", **shown: bool) -> LoginSelection:
    return LoginSelection(
        audience="person",
        apps={name: AppSelection(shown=value) for name, value in shown.items()},
        tier=tier,  # type: ignore[arg-type]
        shared_cards=(),
    )


def _project(tier: str = "allowed", manifests=None, **shown: bool) -> dict[str, Any]:
    manifests = manifests if manifests is not None else [_manifest(KNOB)]
    return project_login(manifests, CONFIG, _login(tier, **(shown or {"knob": True})))


# ---------------------------------------------------------------------------
# Projection: the server
# ---------------------------------------------------------------------------


def test_a_shown_application_is_projected_as_one_server_named_for_it() -> None:
    server = _project()["mcp_servers"]["knob"]
    assert server == {
        "command": "{current_python_env}",
        "args": ["-m", "knob_app.tools"],
        "env": {"OSPREY_APP_NAME": "knob"},
        "permissions": {"allow": ["knob_read"]},
        "writes": ["knob_set", "knob_zero"],
        "app": "knob",
        "condition": "panel:knob",
    }


@pytest.mark.parametrize("key", ["mcp_servers", "apps", "dispatch"])
def test_the_tools_projection_is_tier_blind(key) -> None:
    assert _project("never")[key] == _project("allowed")[key]


def test_the_server_names_its_application_by_the_sdk_variable() -> None:
    from osprey.apps.projection import APP_NAME_ENV
    from osprey.apps.sdk._approval import APP_NAME_ENV_VAR

    assert APP_NAME_ENV == APP_NAME_ENV_VAR


def test_an_always_shown_server_carries_no_condition() -> None:
    server = _project(manifests=[_manifest(DIAL)], dial=True)["mcp_servers"]["dial"]
    assert "condition" not in server
    assert server["writes"] == ["dial_turn"]


@pytest.mark.parametrize("shown", [{"knob": False}, {"other": True}])
def test_an_application_the_login_does_not_show_projects_no_server(shown) -> None:
    result = _project(**shown)
    assert result["mcp_servers"] == {}
    assert result["apps"] == {}
    assert result["dispatch"] == {"denied_app_tools": []}


# ---------------------------------------------------------------------------
# Projection: rendered config keys
# ---------------------------------------------------------------------------


def test_each_approval_tool_gets_its_full_entry_under_its_short_name() -> None:
    approval = _project()["apps"]["knob"]["approval"]
    assert list(approval) == ["knob_set", "knob_zero"]
    assert approval["knob_set"] == {
        "prompt": "Set the knob to {value}",
        "panel_id": "knob",
        "state_path": "/api/state",
        "display_lines_key": "lines",
        "bind": KNOB["tools"]["approval"][0]["bind"],
        "require": KNOB["tools"]["approval"][0]["require"],
        "errors": KNOB["tools"]["errors"],
    }
    assert approval["knob_zero"]["state_path"] is None
    assert approval["knob_zero"]["bind"] == {"state_fields": [], "input_digests": []}
    assert approval["knob_zero"]["require"] == {"all_of": ["approved_target"], "one_of": []}


def test_no_projected_one_of_group_holds_a_digest_field() -> None:
    for entry in _project()["apps"]["knob"]["approval"].values():
        digests = {d["field"] for d in entry["bind"]["input_digests"]}
        assert all(not digests & set(group) for group in entry["require"]["one_of"])


def test_the_declared_read_surface_and_errors_are_written_beside_the_approvals() -> None:
    tools = _project()["apps"]["knob"]["tools"]
    assert tools == {"read": ["knob_read"], "errors": KNOB["tools"]["errors"]}
    dial = _project(manifests=[_manifest(DIAL)], dial=True)["apps"]["dial"]["tools"]
    assert dial == {"read": ["dial_read"], "errors": {"unreachable": None, "refused": None}}


def test_the_jobs_deny_list_is_always_written() -> None:
    no_tools = copy.deepcopy(KNOB)
    del no_tools["tools"]
    result = _project(manifests=[_manifest(no_tools)])
    assert result["dispatch"] == {"denied_app_tools": []}


def test_the_jobs_deny_list_golden() -> None:
    """A state-binding tool, and every tool of a ``deny_in_jobs`` application."""
    result = _project(manifests=[_manifest(KNOB), _manifest(DIAL)], knob=True, dial=True)
    assert result["dispatch"]["denied_app_tools"] == [
        "mcp__knob__knob_set",
        "mcp__dial__dial_read",
        "mcp__dial__dial_turn",
    ]


# ---------------------------------------------------------------------------
# Profile schema, loader and persistence carry `condition` and `app`
# ---------------------------------------------------------------------------


def _profile_servers(**servers: dict) -> dict:
    return _parse_profile({"mcp_servers": servers}).mcp_servers


def test_the_loader_admits_the_projected_server() -> None:
    server = _profile_servers(knob=_project()["mcp_servers"]["knob"])["knob"]
    assert server.condition == "panel:knob"
    assert server.app == "knob"
    assert server.writes == ["knob_set", "knob_zero"]


@pytest.mark.parametrize(
    ("key", "value"),
    [("condition", ""), ("condition", 3), ("app", "Knob"), ("app", ""), ("app", ["knob"])],
)
def test_the_loader_refuses_a_malformed_condition_or_app(key, value) -> None:
    spec = {"command": "python", "args": ["-m", "x"], key: value}
    with pytest.raises(BuildProfileError, match=f"MCP server 'knob' {key}"):
        _profile_servers(knob=spec)


def test_persistence_writes_condition_and_app_beside_writes(tmp_path) -> None:
    from osprey.cli.build_persistence import _persist_mcp_servers

    (tmp_path / "config.yml").write_text("project_name: p\n")
    servers = _profile_servers(knob=_project()["mcp_servers"]["knob"])
    _persist_mcp_servers(tmp_path, servers)
    spec = yaml.safe_load((tmp_path / "config.yml").read_text())["claude_code"]["servers"]["knob"]
    assert spec["condition"] == "panel:knob"
    assert spec["app"] == "knob"
    assert spec["writes"] == ["knob_set", "knob_zero"]
    assert spec["env"] == {"OSPREY_APP_NAME": "knob"}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _ctx(**extra) -> dict:
    return {"project_root": "/tmp/p", "current_python_env": "/usr/bin/python3", **extra}


def _spec() -> dict:
    return _project()["mcp_servers"]["knob"]


def test_the_registry_carries_app_and_condition() -> None:
    sdef = _custom_server_from_spec("knob", _spec())
    assert sdef is not None
    assert sdef.app == "knob"
    assert sdef.condition == "panel:knob"


def test_an_application_write_buys_the_preflight_approval_budget() -> None:
    sdef = _custom_server_from_spec("knob", _spec())
    assert sdef is not None
    assert [r.matcher for r in sdef.hooks_pre] == KNOB_WRITES
    for rule in sdef.hooks_pre:
        assert rule.hooks == [_WRITES_CHECK, _APPROVAL_PREFLIGHT]


def test_a_facility_write_keeps_the_default_approval_budget() -> None:
    spec = {k: v for k, v in _spec().items() if k not in ("app", "condition")}
    sdef = _custom_server_from_spec("knob", spec)
    assert sdef is not None
    assert all(rule.hooks == [_WRITES_CHECK, _APPROVAL] for rule in sdef.hooks_pre)


def test_a_custom_server_condition_is_evaluated_like_a_framework_one() -> None:
    def knob(ctx: dict) -> dict:
        servers = resolve_servers({"servers": {"knob": _spec()}}, ctx)
        return next(s for s in servers if s["name"] == "knob")

    assert knob(_ctx())["enabled"] is False
    assert knob(_ctx(**{"panel:knob": True}))["enabled"] is True
    off = resolve_servers({"servers": {"knob": _spec()}}, _ctx())
    assert facility_write_tools(off) == []


def test_an_application_server_is_not_warned_about_a_missing_live_target(caplog) -> None:
    """An application's writes reach the session's control target, not ``live``."""
    section = {"type": "mock"}
    with caplog.at_level(logging.WARNING, logger="osprey.registry.mcp"):
        resolve_servers(
            {"servers": {"knob": _spec()}},
            _ctx(control_system_section=section, **{"panel:knob": True}),
        )
    assert not [r for r in caplog.records if "no session can arm them" in r.getMessage()]


# ---------------------------------------------------------------------------
# config_derived_context
# ---------------------------------------------------------------------------


def _rendered_config(tier: str = "allowed") -> dict:
    projected = _project(tier)
    return {
        "apps": projected["apps"],
        "dispatch": projected["dispatch"],
        "web": {"panels": {"knob": {"enabled": True}, "hidden": {"enabled": False}}},
    }


def test_a_shown_panel_sets_its_condition_key(tmp_path) -> None:
    ctx = claude_code.config_derived_context(_rendered_config(), tmp_path)
    assert ctx["panel:knob"] is True
    assert "panel:hidden" not in ctx


def test_a_malformed_panel_block_shows_no_panel(tmp_path) -> None:
    config = _rendered_config()
    config["web"]["panels"].update({"blank": None, "odd": "yes", "bare": True})
    ctx = claude_code.config_derived_context(config, tmp_path)
    assert "panel:blank" not in ctx
    assert "panel:odd" not in ctx
    assert ctx["panel:bare"] is True


def test_app_approvals_is_derived_from_the_rendered_leaves_by_full_name(tmp_path) -> None:
    config = _rendered_config()
    ctx = claude_code.config_derived_context(config, tmp_path)
    assert list(ctx["app_approvals"]) == KNOB_WRITES
    leaf = config["apps"]["knob"]["approval"]["knob_set"]
    assert ctx["app_approvals"]["mcp__knob__knob_set"] == {"app": "knob", **leaf}


def test_app_error_types_classes_both_application_errors(tmp_path) -> None:
    ctx = claude_code.config_derived_context(_rendered_config(), tmp_path)
    assert ctx["app_error_types"] == {
        "knob.unreachable": {"class": "Connection", "text": "The knob panel is down."},
        "knob.refused": {"class": "Safety", "text": "Reopen the panel."},
    }


def test_a_render_with_no_application_derives_empty_mappings(tmp_path) -> None:
    ctx = claude_code.config_derived_context({}, tmp_path)
    assert ctx["app_approvals"] == {}
    assert ctx["app_error_types"] == {}


# ---------------------------------------------------------------------------
# The two renders
# ---------------------------------------------------------------------------

_PROJECTS = 0


def _render(tmp_path, tier: str):
    """A built project carrying the projected application, and its context."""
    global _PROJECTS
    _PROJECTS += 1
    manager = TemplateManager()
    project_dir = create_project_as_built(
        manager,
        project_name=f"app-servers-{_PROJECTS}",
        output_dir=tmp_path,
        context={"channel_finder_mode": "hierarchical"},
    )
    from osprey.cli.build_persistence import _persist_mcp_servers

    _persist_mcp_servers(project_dir, _profile_servers(knob=_spec()))
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    config["control_system"]["writes"] = tier
    rendered = _rendered_config(tier)
    config["apps"] = rendered["apps"]
    config["dispatch"] = rendered["dispatch"]
    config.setdefault("web", {}).setdefault("panels", {})["knob"] = {"enabled": True}
    (project_dir / "config.yml").write_text(yaml.dump(config))
    ctx = claude_code.build_claude_code_context(
        manager.template_root, manager.jinja_env, project_dir, config
    )
    return manager, project_dir, ctx


def _hook_config(project_dir) -> dict:
    return json.loads((project_dir / ".claude" / "hooks" / "hook_config.json").read_text())


def test_an_allowed_render_gates_the_application_writes(tmp_path) -> None:
    manager, project_dir, ctx = _render(tmp_path, "allowed")
    assert [t for t in ctx["facility_write_tools"] if t.startswith("mcp__knob__")] == KNOB_WRITES
    assert not set(KNOB_WRITES) & set(claude_code._rendered_deny_list(ctx))
    manager.regenerate_claude_code(project_dir)
    hook_config = _hook_config(project_dir)
    assert set(KNOB_WRITES) <= set(hook_config["write_tools"])
    assert set(KNOB_WRITES) <= set(hook_config["facility_write_tools"])
    assert not set(KNOB_WRITES) & set(hook_config["control_system_write_tools"])
    assert list(hook_config["app_approvals"]) == KNOB_WRITES
    assert set(hook_config["app_error_types"]) == {"knob.unreachable", "knob.refused"}


def test_a_never_render_denies_the_application_writes_statically(tmp_path) -> None:
    _, _, ctx = _render(tmp_path, "never")
    deny = claude_code._rendered_deny_list(ctx)
    for tool in KNOB_WRITES:
        assert tool in ctx["killswitch_deny"]
        assert tool in deny
    assert "mcp__knob__knob_read" not in deny
    assert not set(KNOB_WRITES) & set(ctx.get("control_system_write_tools") or [])


def test_build_and_regenerate_render_the_same_hook_config(tmp_path) -> None:
    manager, project_dir, _ = _render(tmp_path, "allowed")
    manager.regenerate_claude_code(project_dir)
    first = (project_dir / ".claude" / "hooks" / "hook_config.json").read_bytes()
    manager.regenerate_claude_code(project_dir)
    assert (project_dir / ".claude" / "hooks" / "hook_config.json").read_bytes() == first
