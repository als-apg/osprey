"""C-FW: a custom MCP server's declared ``writes:`` list in the registry.

A server that declares which of its tools write gets one hook rule per write
tool (writes-check AND approval), instead of the whole-server rule a server
with no list keeps. These tests pin that split, the window rules for
undeclared write-looking tools on a declaring server, the build error and
warnings around the ``writes_check`` preset, the live-target warning, and
:func:`facility_write_tools`, the one producer of the facility write set.
"""

from __future__ import annotations

import logging

import pytest

from osprey.cli.build_profile_load import RETIRED_WRITE_KEYS_REFUSED_FROM
from osprey.registry.mcp import (
    _APPROVAL,
    _WRITES_CHECK,
    _custom_server_from_spec,
    facility_write_tools,
    resolve_servers,
)
from osprey_connectors.workspace import DEFAULT_AGENT_DATA_BASE_DIR

WRITES_HOOK = "osprey_writes_check.py"
APPROVAL_HOOK = "osprey_approval.py"


def _ctx(**overrides):
    ctx = {
        "project_root": "/tmp/test-project",
        "current_python_env": "/usr/bin/python3",
        "agent_data_root": DEFAULT_AGENT_DATA_BASE_DIR,
    }
    ctx.update(overrides)
    return ctx


def _spec(**extra):
    spec = {"command": "python", "args": ["-m", "my_plc_server"]}
    spec.update(extra)
    return spec


def _server(cfg_servers: dict, name: str = "my-plc", **ctx_overrides) -> dict:
    servers = resolve_servers({"servers": cfg_servers}, _ctx(**ctx_overrides))
    return next(s for s in servers if s["name"] == name)


def _hook_names(rule: dict) -> list[str]:
    return [h["command"] for h in rule["hooks"]]


def _gates_writes(rule: dict) -> bool:
    return any(WRITES_HOOK in c for c in _hook_names(rule))


def _gates_approval(rule: dict) -> bool:
    return any(APPROVAL_HOOK in c for c in _hook_names(rule))


# ---------------------------------------------------------------------------
# Declaring server: one rule per declared tool
# ---------------------------------------------------------------------------


class TestDeclaredWrites:
    def test_each_declared_tool_gets_its_own_rule(self):
        plc = _server({"my-plc": _spec(writes=["set_output", "reset_counter_value"])})
        matchers = [r["matcher"] for r in plc["hooks_pre"]]
        assert matchers == ["mcp__my-plc__set_output", "mcp__my-plc__reset_counter_value"]
        for rule in plc["hooks_pre"]:
            assert _gates_writes(rule) and _gates_approval(rule)

    def test_rule_reuses_the_hook_singletons(self):
        """Consumers classify by HookEntry identity, so copies would go unseen."""
        sdef = _custom_server_from_spec("my-plc", _spec(writes=["set_output"]))
        assert sdef is not None
        (rule,) = sdef.hooks_pre
        assert rule.hooks[0] is _WRITES_CHECK
        assert rule.hooks[1] is _APPROVAL

    def test_no_whole_server_rule_on_a_declaring_server(self):
        plc = _server({"my-plc": _spec(writes=["set_output"])})
        assert all(r["matcher"] != "mcp__my-plc__.*" for r in plc["hooks_pre"])

    def test_writes_exposed_on_definition_and_dict(self):
        sdef = _custom_server_from_spec("my-plc", _spec(writes=["set_output", "pulse"]))
        assert sdef is not None
        assert sdef.writes == ("set_output", "pulse")
        plc = _server({"my-plc": _spec(writes=["set_output", "pulse"])})
        assert plc["writes"] == ["set_output", "pulse"]

    def test_empty_list_declares_no_writes(self):
        """``writes: []`` is a declaration (nothing writes), not an absent list."""
        sdef = _custom_server_from_spec("my-plc", _spec(writes=[]))
        assert sdef is not None
        assert sdef.writes == ()
        plc = _server({"my-plc": _spec(writes=[])})
        assert plc["writes"] == []
        assert plc["hooks_pre"] == []

    def test_duplicate_declarations_render_once(self):
        plc = _server({"my-plc": _spec(writes=["set_output", "set_output"])})
        assert [r["matcher"] for r in plc["hooks_pre"]] == ["mcp__my-plc__set_output"]

    def test_other_presets_keep_their_whole_server_rule(self):
        """``approval`` on a declaring server still covers every tool."""
        plc = _server(
            {
                "my-plc": _spec(
                    writes=["set_output"],
                    hooks={"pre_tool_use": ["approval"]},
                )
            }
        )
        per_tool = [r for r in plc["hooks_pre"] if r["matcher"] == "mcp__my-plc__set_output"]
        whole = [r for r in plc["hooks_pre"] if r["matcher"] == "mcp__my-plc__.*"]
        assert len(per_tool) == 1 and _gates_writes(per_tool[0])
        assert len(whole) == 1
        assert _gates_approval(whole[0]) and not _gates_writes(whole[0])


# ---------------------------------------------------------------------------
# Warning window (BX-1)
# ---------------------------------------------------------------------------


class TestWarningWindow:
    def test_undeclared_ask_tool_gets_the_same_rule(self):
        plc = _server(
            {
                "my-plc": _spec(
                    writes=["set_output"],
                    permissions={"ask": ["set_output", "move_stage"]},
                )
            }
        )
        rules = {r["matcher"]: r for r in plc["hooks_pre"]}
        assert set(rules) == {"mcp__my-plc__set_output", "mcp__my-plc__move_stage"}
        assert _gates_writes(rules["mcp__my-plc__move_stage"])
        assert _gates_approval(rules["mcp__my-plc__move_stage"])

    def test_undeclared_destructive_name_gets_the_same_rule(self):
        plc = _server(
            {
                "my-plc": _spec(
                    writes=["set_output"],
                    permissions={"allow": ["read_status", "Purge_Buffer"]},
                )
            }
        )
        matchers = [r["matcher"] for r in plc["hooks_pre"]]
        assert "mcp__my-plc__Purge_Buffer" in matchers
        assert "mcp__my-plc__read_status" not in matchers

    def test_fully_qualified_permission_entries_are_recognised(self):
        plc = _server(
            {
                "my-plc": _spec(
                    writes=["set_output"],
                    permissions={"ask": ["mcp__my-plc__set_output", "mcp__my-plc__jog"]},
                )
            }
        )
        matchers = [r["matcher"] for r in plc["hooks_pre"]]
        assert matchers == ["mcp__my-plc__set_output", "mcp__my-plc__jog"]

    def test_window_tools_are_not_added_to_the_declaration(self):
        plc = _server({"my-plc": _spec(writes=["set_output"], permissions={"ask": ["move_stage"]})})
        assert plc["writes"] == ["set_output"]

    def test_listless_server_gets_no_window(self):
        """A server with no list behaves as today: one whole-server rule or none."""
        plc = _server(
            {"my-plc": _spec(permissions={"ask": ["move_stage"], "allow": ["delete_all"]})}
        )
        assert plc["hooks_pre"] == []
        assert plc["writes"] is None


# ---------------------------------------------------------------------------
# Hook-preset validation
# ---------------------------------------------------------------------------


class TestPresetChecks:
    def test_writes_check_beside_writes_is_an_error(self):
        with pytest.raises(ValueError, match="my-plc"):
            _server(
                {
                    "my-plc": _spec(
                        writes=["set_output"],
                        hooks={"pre_tool_use": ["writes_check"]},
                    )
                }
            )

    def test_writes_check_beside_empty_writes_is_an_error(self):
        with pytest.raises(ValueError, match="writes_check"):
            _custom_server_from_spec(
                "my-plc", _spec(writes=[], hooks={"pre_tool_use": ["writes_check"]})
            )

    def test_writes_check_alone_warns_and_keeps_whole_server_rule(self, caplog):
        with caplog.at_level(logging.WARNING, logger="osprey.registry.mcp"):
            plc = _server({"my-plc": _spec(hooks={"pre_tool_use": ["writes_check"]})})
        assert [r["matcher"] for r in plc["hooks_pre"]] == ["mcp__my-plc__.*"]
        assert _gates_writes(plc["hooks_pre"][0])
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        hit = [m for m in msgs if "declare `writes:`" in m and "my-plc" in m]
        assert hit, msgs
        assert RETIRED_WRITE_KEYS_REFUSED_FROM in hit[0]

    def test_unknown_preset_warns_with_the_sunset(self, caplog):
        with caplog.at_level(logging.WARNING, logger="osprey.registry.mcp"):
            plc = _server({"my-plc": _spec(hooks={"pre_tool_use": ["approval", "bogus"]})})
        assert len(plc["hooks_pre"]) == 1
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        hit = [m for m in msgs if "bogus" in m and "my-plc" in m]
        assert hit, msgs
        assert RETIRED_WRITE_KEYS_REFUSED_FROM in hit[0]

    def test_declaring_server_without_writes_check_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger="osprey.registry.mcp"):
            _server({"my-plc": _spec(writes=["set_output"])})
        assert not [r for r in caplog.records if "declare `writes:`" in r.getMessage()]


# ---------------------------------------------------------------------------
# Live-target warning
# ---------------------------------------------------------------------------


_LIVE_EPICS = {"type": "epics", "connector": {"epics": {"gateways": {"read_only": {}}}}}


class TestLiveTarget:
    def _warnings(self, caplog) -> list[str]:
        return [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING and "live" in r.getMessage()
        ]

    def test_writes_without_live_target_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="osprey.registry.mcp"):
            _server(
                {"my-plc": _spec(writes=["set_output"])},
                control_system_section={"type": "mock"},
            )
        assert any("my-plc" in m for m in self._warnings(caplog))

    def test_writes_with_live_target_is_quiet(self, caplog):
        with caplog.at_level(logging.WARNING, logger="osprey.registry.mcp"):
            _server(
                {"my-plc": _spec(writes=["set_output"])},
                control_system_section=_LIVE_EPICS,
            )
        assert not [m for m in self._warnings(caplog) if "my-plc" in m]

    def test_empty_writes_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger="osprey.registry.mcp"):
            _server({"my-plc": _spec(writes=[])}, control_system_section={"type": "mock"})
        assert not [m for m in self._warnings(caplog) if "my-plc" in m]

    def test_no_section_in_ctx_does_not_guess(self, caplog):
        """A caller that never passed the section has not said there is no target."""
        with caplog.at_level(logging.WARNING, logger="osprey.registry.mcp"):
            _server({"my-plc": _spec(writes=["set_output"])})
        assert not [m for m in self._warnings(caplog) if "my-plc" in m]


# ---------------------------------------------------------------------------
# facility_write_tools
# ---------------------------------------------------------------------------


class TestFacilityWriteTools:
    def test_declared_window_and_whole_server(self):
        cfg = {
            "my-plc": _spec(writes=["set_output"], permissions={"ask": ["move_stage"]}),
            "legacy": _spec(hooks={"pre_tool_use": ["approval", "writes_check"]}),
            "approver": _spec(hooks={"pre_tool_use": ["approval"]}),
            "quiet": _spec(writes=[]),
        }
        servers = resolve_servers({"servers": cfg}, _ctx())
        assert facility_write_tools(servers) == [
            "mcp__my-plc__set_output",
            "mcp__my-plc__move_stage",
            "mcp__legacy__.*",
        ]

    def test_disabled_and_framework_servers_are_skipped(self):
        cfg = {
            "my-plc": _spec(writes=["set_output"], enabled=False),
            "other": _spec(writes=["go"]),
        }
        servers = resolve_servers({"servers": cfg}, _ctx())
        assert facility_write_tools(servers) == ["mcp__other__go"]

    def test_no_custom_servers_yields_empty(self):
        assert facility_write_tools(resolve_servers({}, _ctx())) == []
