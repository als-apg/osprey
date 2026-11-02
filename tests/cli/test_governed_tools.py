"""The governed set: the tools the write switch's position decides for.

One producer (:func:`osprey.registry.mcp.governed_write_tools`) and its readers:
the render context and ``hook_config.json``, the hook's ``is_governed``,
:func:`~osprey.cli.build_posture_check.ask_capable` and the floor under a render
that serves a governed tool with no approval hook. Registry-level mechanics
(clones, disabled servers, ordering) are in ``tests/registry/test_mcp.py``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import osprey.templates.claude_code.claude.hooks.osprey_hook_log as hook_log
from osprey.cli.build_posture_check import (
    GUARDED_TOOLS,
    STAGE1_TOOLS,
    ask_capable,
    check_approval_hook_selected,
)
from osprey.cli.build_profile import resolve_build_profile
from osprey.cli.templates import claude_code
from osprey.cli.templates.claude_code import build_claude_code_context
from osprey.cli.templates.manager import TemplateManager
from osprey.facility.errors import FacilityBuildError
from osprey.registry.mcp import governed_write_tools, resolve_servers

#: The presets the project ships.
PRESETS = ("hello-world", "control-assistant")

GOVERNED = (
    "mcp__controls__channel_write",
    "mcp__python__execute",
    "mcp__python__execute_file",
    "mcp__python__pyaml_measure",
    "mcp__bluesky__queue_add",
    "mcp__bluesky__queue_start",
)

NOT_GOVERNED = (
    "mcp__phoebus__phoebus_drive",
    "mcp__controls__control_target_set",
    "mcp__osprey_workspace__setup_patch",
    "mcp__bluesky__queue_stop",
    "mcp__bluesky__queue_remove",
    "mcp__event_dispatcher__manual_fire",
    "mcp__ariel__entry_create",
)

#: Switches every server off that serves a governed tool by default, so a test
#: of what is left is not answered by ``execute`` first.
NO_DEFAULT_GOVERNED = {"controls": {"enabled": False}, "python": {"enabled": False}}


def _preset_servers(preset: str) -> dict[str, Any]:
    """The preset's ``claude_code.servers`` overrides, with every opt-in server on."""
    profile, _ = resolve_build_profile(None, preset=preset)
    servers: dict[str, Any] = {}
    prefix = "claude_code.servers."
    for key, value in profile.config.items():
        if key.startswith(prefix):
            name, _, leaf = key[len(prefix) :].partition(".")
            servers.setdefault(name, {})[leaf] = value
    for name in ("bluesky", "phoebus", "ariel"):
        servers[name] = {"enabled": True}
    return servers


def _facility(**extra: Any) -> dict[str, Any]:
    """A custom server spec."""
    return {"command": "python3", "args": ["-m", "plc"], **extra}


def _rendered(approval: Any = None, **servers: Any) -> dict[str, Any]:
    """A render stating *approval* with *servers* overriding the registry."""
    return {
        "approval": approval if approval is not None else {"enabled": False},
        "claude_code": {"servers": servers},
    }


class TestShippedPresets:
    """The set both presets ship, with the opt-in servers forced on to be seen."""

    @pytest.fixture(params=PRESETS)
    def governed(self, request: pytest.FixtureRequest) -> list[str]:
        servers = resolve_servers(
            {"servers": _preset_servers(request.param)},
            {"event_dispatcher_wired": True, "phoebus_agent_access": "read_write"},
        )
        return governed_write_tools(servers)

    @pytest.mark.parametrize("tool", GOVERNED)
    def test_a_governed_tool(self, governed: list[str], tool: str) -> None:
        assert tool in governed

    @pytest.mark.parametrize("tool", NOT_GOVERNED)
    def test_a_tool_that_keeps_its_policy(self, governed: list[str], tool: str) -> None:
        assert tool not in governed

    def test_the_guarded_and_stage1_tools_are_governed_but_the_drive(
        self, governed: list[str]
    ) -> None:
        shorts = {tool.rsplit("__", 1)[1] for tool in governed}
        assert {*GUARDED_TOOLS, *STAGE1_TOOLS} - shorts == {"phoebus_drive"}

    def test_nothing_else_is_governed(self, governed: list[str]) -> None:
        """Nothing outside the shipped six: a new member is a decision, not an accident."""
        assert sorted(governed) == sorted(GOVERNED)


class TestFacilityTools:
    """A declared write tool is governed exactly when its rule carries both hooks."""

    def test_a_declared_write_tool_has_both_hooks_and_is_governed(self) -> None:
        servers = resolve_servers({"servers": {"plc": _facility(writes=["set_output"])}}, {})
        assert "mcp__plc__set_output" in governed_write_tools(servers)

    def test_writes_check_alone_is_not_governed(self) -> None:
        spec = _facility(hooks={"pre_tool_use": ["writes_check"]})
        servers = resolve_servers({"servers": {"plc": spec}}, {})
        assert not any("plc" in tool for tool in governed_write_tools(servers))

    def test_approval_alone_is_not_governed(self) -> None:
        spec = _facility(hooks={"pre_tool_use": ["approval"]})
        servers = resolve_servers({"servers": {"plc": spec}}, {})
        assert not any("plc" in tool for tool in governed_write_tools(servers))

    def test_a_whole_server_pair_yields_the_wildcard(self) -> None:
        spec = _facility(hooks={"pre_tool_use": ["writes_check", "approval"]})
        servers = resolve_servers({"servers": {"plc": spec}}, {})
        assert "mcp__plc__.*" in governed_write_tools(servers)


class TestRender:
    """The render carries the set, and ships it in ``hook_config.json``."""

    @pytest.fixture(autouse=True)
    def _no_gate_lint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """These configs select no hooks; that lint has its own tests."""
        monkeypatch.setattr(claude_code, "_lint_write_gates_are_selected", lambda *_: None)

    @staticmethod
    def _hook_config(ctx: dict[str, Any]) -> dict[str, Any]:
        env = TemplateManager().jinja_env
        template = env.get_template("claude_code/claude/hooks/hook_config.json.j2")
        return json.loads(template.render(**ctx))

    def test_the_context_carries_the_producers_list(self, tmp_path: Any) -> None:
        manager = TemplateManager()
        config = {
            "control_system": {"type": "mock", "writes": "allowed"},
            "claude_code": {"servers": {"plc": _facility(writes=["set_output"])}},
        }
        ctx = build_claude_code_context(manager.template_root, manager.jinja_env, tmp_path, config)
        assert ctx["governed_tools"] == governed_write_tools(ctx["servers"])
        assert "mcp__plc__set_output" in ctx["governed_tools"]
        assert self._hook_config(ctx)["governed_tools"] == ctx["governed_tools"]

    def test_the_key_renders_on_a_read_only_tier(self, tmp_path: Any) -> None:
        """Hooks read it under every posture, so it is not under the kill-switch branch."""
        manager = TemplateManager()
        config = {"control_system": {"type": "mock", "writes": "never"}}
        ctx = build_claude_code_context(manager.template_root, manager.jinja_env, tmp_path, config)
        assert "mcp__python__execute" in self._hook_config(ctx)["governed_tools"]

    @pytest.mark.parametrize("ctx", [{}, {"governed_tools": None}], ids=["absent", "none"])
    def test_an_absent_list_still_renders_the_key(self, ctx: dict[str, Any]) -> None:
        assert self._hook_config({"servers": [], **ctx})["governed_tools"] == []

    def test_a_policy_keeping_tool_still_leaves_ask_on_an_armed_render(self, tmp_path: Any) -> None:
        """The unask rule is both hooks, so ``phoebus_drive`` keeps today's render."""
        manager = TemplateManager()
        config = {
            "control_system": {"type": "mock", "writes": "allowed"},
            "phoebus": {"agent_access": "read_write"},
            "claude_code": {"servers": {"phoebus": {"enabled": True}}},
        }
        ctx = build_claude_code_context(manager.template_root, manager.jinja_env, tmp_path, config)
        remove_ask = ctx["facility_permissions"]["remove_ask"]
        assert "mcp__phoebus__phoebus_drive" in remove_ask
        assert "mcp__phoebus__phoebus_drive" not in ctx["governed_tools"]
        assert set(ctx["governed_tools"]) <= set(remove_ask)


class TestIsGoverned:
    """The hook's reader of the rendered list."""

    @pytest.fixture
    def hook_config(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
        def _write(content: str) -> None:
            path = tmp_path / "hook_config.json"
            path.write_text(content)
            monkeypatch.setenv("OSPREY_HOOK_CONFIG", str(path))
            monkeypatch.setattr(hook_log, "_hook_config_cache", None)

        return _write

    def test_an_exact_entry(self, hook_config: Any) -> None:
        hook_config('{"governed_tools": ["mcp__python__execute"]}')
        assert hook_log.is_governed("mcp__python__execute")
        assert not hook_log.is_governed("mcp__python__execute_file")

    def test_a_whole_server_entry_matches_as_a_wildcard(self, hook_config: Any) -> None:
        hook_config('{"governed_tools": ["mcp__plc__.*"]}')
        assert hook_log.is_governed("mcp__plc__set_output")
        assert not hook_log.is_governed("mcp__plc2__set_output")

    def test_a_malformed_entry_is_skipped(self, hook_config: Any) -> None:
        hook_config('{"governed_tools": [7, "mcp__a__b"]}')
        assert hook_log.is_governed("mcp__a__b")

    @pytest.mark.parametrize("content", ["{}", "not json"], ids=["key-absent", "unreadable"])
    def test_a_degraded_config_governs_nothing(self, hook_config: Any, content: str) -> None:
        """Documented choice: the policy path asks, and the server refuses what it lets by."""
        hook_config(content)
        assert not hook_log.is_governed("mcp__python__execute")


class TestAskCapable:
    """A governed tool asks whatever ``approval.*`` says."""

    @pytest.mark.parametrize(
        "approval",
        [
            {"enabled": False},
            {"enabled": True, "default_policy": "skip"},
            {"enabled": "false"},
            "disabled",
        ],
        ids=["disabled", "skip", "non-bool-switch", "non-mapping"],
    )
    @pytest.mark.parametrize("tool", ["channel_write", *GUARDED_TOOLS])
    def test_a_governed_tool_regardless_of_approval(self, tool: str, approval: Any) -> None:
        assert ask_capable(_rendered(approval), tool) is True

    def test_a_policy_keeping_tool_still_reads_its_policy(self) -> None:
        rendered = _rendered(phoebus={"enabled": True})
        assert ask_capable(rendered, "phoebus_drive") is False

    def test_a_clones_tool_counts_by_its_short_name(self) -> None:
        rendered = _rendered(python={"enabled": False}, pyva={"extends": "python"})
        assert ask_capable(rendered, "execute") is True

    def test_a_declared_facility_tool_asks(self) -> None:
        rendered = _rendered(plc=_facility(writes=["set_output"]))
        assert ask_capable(rendered, "set_output") is True

    def test_a_tool_on_a_disabled_server_reads_its_policy(self) -> None:
        rendered = _rendered(python={"enabled": False})
        assert ask_capable(rendered, "execute") is False

    def test_a_section_the_registry_refuses_fails_closed(self) -> None:
        """A run-time check answers rather than raises, and it answers ask."""
        spec = _facility(writes=["go"], hooks={"pre_tool_use": ["writes_check"]})
        assert ask_capable(_rendered(plc=spec), "phoebus_drive") is True

    def test_an_empty_render_resolves_the_registry_defaults(self) -> None:
        """``channel_write`` passes ``{}`` for a config it cannot read."""
        assert ask_capable({}, "channel_write") is True


class TestFloor:
    """A render serving a governed tool selects the hook, even with approval off."""

    @staticmethod
    def _stop(rendered: dict[str, Any]) -> str:
        with pytest.raises(FacilityBuildError) as stop:
            check_approval_hook_selected(rendered, ["error-guidance"])
        assert stop.value.kind == "profile-invalid"
        assert stop.value.record_id == "approval.enabled"
        return stop.value.format_message()

    def test_a_framework_tool_under_approval_off(self) -> None:
        message = self._stop(_rendered())
        assert "governed tool `execute` is put to a human by the write switch" in message

    def test_the_fix_no_longer_recommends_turning_approval_off(self) -> None:
        message = self._stop(_rendered())
        assert "add `approval` to `hooks:` in profile.yml" in message
        assert "approval.enabled: false" not in message

    def test_a_declared_facility_tool_under_approval_off(self) -> None:
        message = self._stop(_rendered(**NO_DEFAULT_GOVERNED, plc=_facility(writes=["go"])))
        assert "governed tool `mcp__plc__go`" in message

    def test_a_whole_server_pair_is_named_by_its_matcher(self) -> None:
        spec = _facility(hooks={"pre_tool_use": ["writes_check", "approval"]})
        message = self._stop(_rendered(**NO_DEFAULT_GOVERNED, plc=spec))
        assert "governed tool `mcp__plc__.*`" in message

    def test_a_clones_tool_on_a_disabled_template_is_named(self) -> None:
        rendered = _rendered(**NO_DEFAULT_GOVERNED, pybis={"extends": "python"})
        assert "governed tool `execute`" in self._stop(rendered)

    def test_a_writes_check_only_tool_asks_nobody_here(self) -> None:
        spec = _facility(hooks={"pre_tool_use": ["writes_check"]})
        check_approval_hook_selected(_rendered(**NO_DEFAULT_GOVERNED, plc=spec), ["error-guidance"])

    def test_a_render_with_the_hook_passes(self) -> None:
        check_approval_hook_selected(_rendered(), ["approval"])

    def test_no_governed_tool_served_passes_under_approval_off(self) -> None:
        check_approval_hook_selected(_rendered(**NO_DEFAULT_GOVERNED), ["error-guidance"])
