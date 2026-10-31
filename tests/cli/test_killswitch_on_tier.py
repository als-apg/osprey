"""The writes kill switch reads the tier, ``control_system.writes``, alone.

``never`` statically and unliftably denies the write-only tools — the
writes-check-gated framework matchers, the declared facility tools exactly, and
a list-less writes_check facility server whole as ``mcp__<srv>`` — while mixed
read/write tools and ``control_system.write_tools`` entries keep their reads and
are refused per call. ``allowed`` denies nothing; a write tool leaves ``ask``
only when its rule carries both writes-check and an approval hook.

Also pins the build refusals of ``_lint_write_gates_are_selected`` on both tiers
and that every rendered deny entry is valid Claude Code permission syntax.
"""

from __future__ import annotations

import json
import re

import pytest
import yaml

from osprey import bluesky_tool_names as bsky
from osprey.cli.templates import claude_code
from osprey.cli.templates import manifest as manifest_mod
from osprey.cli.templates.manager import TemplateManager
from osprey.registry.mcp import _WRITES_CHECK, FRAMEWORK_SERVERS, is_mixed_matcher
from osprey_connectors.errors import BuildProfileError
from tests._preset_data import create_project_as_built

_PROJECT_COUNTER = 0

#: A facility server that declares its one write, asks for it, and reads freely.
_PLC = {
    "command": "node",
    "args": ["plc.js"],
    "writes": ["set_value"],
    "permissions": {"ask": ["set_value"], "allow": ["get_value"]},
}
#: A list-less facility server gated whole by writes_check and approval.
_SITECTL = {
    "command": "node",
    "args": ["sitectl.js"],
    "permissions": {"ask": ["set_point"]},
    "hooks": {"pre_tool_use": ["writes_check", "approval"]},
}
#: A list-less facility server gated by writes_check alone, no approval.
_RAWCTL = {
    "command": "node",
    "args": ["rawctl.js"],
    "permissions": {"ask": ["poke"]},
    "hooks": {"pre_tool_use": ["writes_check"]},
}
_FACILITY = {"bluesky": {"enabled": True}, "plc": _PLC, "sitectl": _SITECTL, "rawctl": _RAWCTL}

#: Claude Code permission rule syntax: a whole MCP server, one MCP tool (either
#: may end in the ``*`` glob the deny floor uses), or a built-in tool name with
#: an optional parenthesised specifier. Never a regex.
_PERMISSION_RULE = re.compile(
    r"^(mcp__[A-Za-z0-9_-]+\*?(__[A-Za-z0-9_-]+\*?)?|[A-Z][A-Za-z]*(\(.+\))?)$"
)


def _project(tmp_path, *, writes, servers=None, write_tools=None, drop_hooks=(), extra=None):
    """Build a project, set its tier and servers, and return ``(manager, dir, config)``."""
    global _PROJECT_COUNTER
    _PROJECT_COUNTER += 1
    manager = TemplateManager()
    project_dir = create_project_as_built(
        manager,
        facility_views=True,
        project_name=f"killswitch-tier-{_PROJECT_COUNTER}",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    config["control_system"]["writes"] = writes
    if write_tools is not None:
        config["control_system"]["write_tools"] = write_tools
    if servers is not None:
        config.setdefault("claude_code", {})["servers"] = servers
    for key, value in (extra or {}).items():
        config.setdefault("claude_code", {})[key] = value
    (project_dir / "config.yml").write_text(yaml.dump(config))
    if drop_hooks:
        manifest_path = project_dir / ".osprey-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        # The fixture's manifest records no selection, so the render falls back
        # to the preset's; spell that selection out, minus the dropped hooks.
        artifacts = dict(manifest_mod.load_template_manifest("control-assistant")["artifacts"])
        artifacts["hooks"] = [h for h in artifacts["hooks"] if h not in drop_hooks]
        manifest["artifacts"] = artifacts
        manifest_path.write_text(json.dumps(manifest))
    return manager, project_dir, config


def _ctx(tmp_path, **kwargs) -> dict:
    manager, project_dir, config = _project(tmp_path, **kwargs)
    return claude_code.build_claude_code_context(
        manager.template_root, manager.jinja_env, project_dir, config
    )


def _framework_write_only(ctx: dict) -> list[str]:
    """Every enabled framework server's non-mixed writes-check matcher, clone-anchored."""
    matchers = []
    for srv in ctx["servers"]:
        if not srv["enabled"] or srv.get("is_custom"):
            continue
        template = srv.get("extends_of") or srv["name"]
        template_def = FRAMEWORK_SERVERS.get(template)
        if template_def is None:
            continue
        for rule in template_def.hooks_pre:
            if _WRITES_CHECK not in rule.hooks:
                continue
            matcher = rule.matcher.replace(f"mcp__{template}__", f"mcp__{srv['name']}__", 1)
            if not is_mixed_matcher(template, matcher):
                matchers.append(matcher)
    return matchers


# ---------------------------------------------------------------------------
# never
# ---------------------------------------------------------------------------


def test_never_denies_every_framework_write_only_tool(tmp_path):
    ctx = _ctx(tmp_path, writes="never", servers={"bluesky": {"enabled": True}})
    expected = _framework_write_only(ctx)
    assert "mcp__controls__channel_write" in expected
    deny = claude_code._rendered_deny_list(ctx)
    for matcher in expected:
        assert matcher in deny, f"{matcher} must be denied on a never render"
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" in ctx["killswitch_deny"]


def test_never_keeps_mixed_tools_out_of_the_deny_and_out_of_ask(tmp_path):
    ctx = _ctx(tmp_path, writes="never")
    perms = ctx["facility_permissions"]
    for tool in ctx["mixed_read_write_tools"]:
        assert tool not in claude_code._rendered_deny_list(ctx)
        assert tool in perms["remove_ask"]
    # The agent that hard-requires execute keeps it, off the prompt path.
    assert "mcp__python__execute" in perms["allow"]


def test_never_denies_declared_facility_tools_exactly(tmp_path):
    ctx = _ctx(tmp_path, writes="never", servers=_FACILITY)
    deny = ctx["killswitch_deny"]
    assert "mcp__plc__set_value" in deny
    # The declaring server's reads stay reachable, and it is never denied whole.
    assert "mcp__plc__get_value" not in deny
    assert "mcp__plc" not in deny


def test_never_denies_a_list_less_writes_check_server_whole(tmp_path):
    ctx = _ctx(tmp_path, writes="never", servers=_FACILITY)
    deny = ctx["killswitch_deny"]
    assert "mcp__sitectl" in deny
    assert "mcp__rawctl" in deny
    assert not any(entry.endswith(".*") for entry in deny)


#: A facility server with one asked tool and no hooks at all: nothing but the
#: static ``ask`` gates it.
_DUMB = {"command": "node", "args": ["dumb.js"], "permissions": {"ask": ["poke"]}}


def _settings(tmp_path, **kwargs) -> dict:
    manager, project_dir, _config = _project(tmp_path, **kwargs)
    manager.regenerate_claude_code(project_dir)
    return json.loads((project_dir / ".claude" / "settings.json").read_text())


def _writes_check_matchers(settings: dict) -> list[str]:
    """Every rendered PreToolUse matcher whose hooks run osprey_writes_check.py."""
    return [
        rule["matcher"]
        for rule in settings.get("hooks", {}).get("PreToolUse", [])
        if any("osprey_writes_check.py" in h.get("command", "") for h in rule.get("hooks", []))
    ]


def _gated_per_call(settings: dict, tool: str) -> bool:
    return any(re.fullmatch(m, tool) for m in _writes_check_matchers(settings))


def test_never_keeps_control_system_write_tools_out_of_the_deny(tmp_path):
    ctx = _ctx(tmp_path, writes="never", write_tools=["mcp__controls__tune_knob"])
    assert "mcp__controls__tune_knob" not in claude_code._rendered_deny_list(ctx)


def test_never_keeps_an_uncovered_write_tools_entry_in_ask(tmp_path):
    """No writes-check rule matches the entry, so its static ask is its only gate."""
    settings = _settings(
        tmp_path,
        writes="never",
        servers={"dumb": _DUMB},
        write_tools=["mcp__dumb__poke"],
        extra={"permissions": {"allow": ["mcp__dumb"]}},
    )
    perms = settings["permissions"]
    assert not _gated_per_call(settings, "mcp__dumb__poke")
    assert "mcp__dumb__poke" in perms["ask"]
    assert "mcp__dumb__poke" not in perms["deny"]


def test_never_keeps_an_uncovered_framework_named_write_tools_entry_in_ask(tmp_path):
    ctx = _ctx(tmp_path, writes="never", write_tools=["mcp__controls__tune_knob"])
    assert "mcp__controls__tune_knob" not in ctx["facility_permissions"]["remove_ask"]


def test_never_unasks_a_write_tools_entry_only_when_writes_check_fires_on_it(tmp_path):
    """Every entry pulled from ask is matched by a wired writes-check rule."""
    write_tools = ["mcp__python__execute", "mcp__dumb__poke", "mcp__controls__tune_knob"]
    manager, project_dir, config = _project(
        tmp_path, writes="never", servers={"dumb": _DUMB}, write_tools=write_tools
    )
    ctx = claude_code.build_claude_code_context(
        manager.template_root, manager.jinja_env, project_dir, config
    )
    manager.regenerate_claude_code(project_dir)
    settings = json.loads((project_dir / ".claude" / "settings.json").read_text())
    remove_ask = ctx["facility_permissions"]["remove_ask"]
    # The covered entry leaves ask and is gated per call by the rendered hook.
    assert "mcp__python__execute" in remove_ask
    for tool in write_tools:
        if tool in remove_ask:
            assert _gated_per_call(settings, tool), f"{tool} left ask with no writes-check"
        else:
            assert not _gated_per_call(settings, tool)


def test_allowed_keeps_an_uncovered_write_tools_entry_in_ask(tmp_path):
    settings = _settings(
        tmp_path, writes="allowed", servers={"dumb": _DUMB}, write_tools=["mcp__dumb__poke"]
    )
    assert "mcp__dumb__poke" in settings["permissions"]["ask"]


def test_never_deny_cannot_be_lifted_by_remove_deny(tmp_path):
    lifted = ["mcp__controls__channel_write", "mcp__plc__set_value", "mcp__sitectl"]
    ctx = _ctx(
        tmp_path,
        writes="never",
        servers=_FACILITY,
        extra={"permissions": {"deny": list(lifted), "remove_deny": list(lifted)}},
    )
    deny = claude_code._rendered_deny_list(ctx)
    for matcher in lifted:
        assert matcher in deny
        assert deny.count(matcher) == 1


@pytest.mark.parametrize("value", ["Allowed", "true", True, 1, None, "yes"])
def test_only_the_literal_allowed_arms_writes(tmp_path, value):
    ctx = _ctx(tmp_path, writes=value)
    assert "mcp__controls__channel_write" in ctx["killswitch_deny"]


# ---------------------------------------------------------------------------
# allowed
# ---------------------------------------------------------------------------


def test_allowed_denies_nothing(tmp_path):
    ctx = _ctx(tmp_path, writes="allowed", servers=_FACILITY)
    assert ctx["killswitch_deny"] == []


def test_allowed_pulls_writes_check_and_approval_tools_from_ask(tmp_path):
    ctx = _ctx(tmp_path, writes="allowed", servers=_FACILITY)
    remove_ask = ctx["facility_permissions"]["remove_ask"]
    assert "mcp__controls__channel_write" in remove_ask
    # The Bluesky arming pair carries the preflight approval variant.
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" in remove_ask
    assert "mcp__python__execute" in remove_ask
    assert "mcp__python__execute" in ctx["facility_permissions"]["allow"]
    assert "mcp__plc__set_value" in remove_ask
    assert "mcp__sitectl__set_point" in remove_ask


def test_allowed_keeps_tools_missing_either_hook_in_ask(tmp_path):
    ctx = _ctx(tmp_path, writes="allowed", servers=_FACILITY)
    remove_ask = ctx["facility_permissions"]["remove_ask"]
    allow = ctx["facility_permissions"]["allow"]
    # writes-check alone, no approval.
    assert "mcp__rawctl__poke" not in remove_ask
    # approval alone, no writes-check.
    for tool in (bsky.QUEUE_STOP, bsky.QUEUE_REMOVE, bsky.STOP_RUN, "write_plan"):
        assert f"mcp__bluesky__{tool}" not in remove_ask
    # No pure-write tool ever reaches allow.
    for matcher in _framework_write_only(ctx):
        assert matcher not in allow


# ---------------------------------------------------------------------------
# Permission syntax
# ---------------------------------------------------------------------------


def test_rendered_deny_is_valid_permission_syntax(tmp_path):
    manager, project_dir, _config = _project(tmp_path, writes="never", servers=_FACILITY)
    manager.regenerate_claude_code(project_dir)
    settings = json.loads((project_dir / ".claude" / "settings.json").read_text())
    deny = settings["permissions"]["deny"]
    assert "mcp__sitectl" in deny and "mcp__plc__set_value" in deny
    bad = [entry for entry in deny if not _PERMISSION_RULE.match(entry)]
    assert bad == [], f"not Claude Code permission syntax: {bad}"


def test_the_permission_validator_rejects_a_regex():
    """The validator above would catch the regex form it exists to keep out."""
    assert not _PERMISSION_RULE.match("mcp__sitectl__.*")
    assert _PERMISSION_RULE.match("mcp__sitectl")
    assert _PERMISSION_RULE.match("Edit(//etc/hosts)")
    assert _PERMISSION_RULE.match("mcp__plugin_*")


# ---------------------------------------------------------------------------
# _lint_write_gates_are_selected
# ---------------------------------------------------------------------------


def test_never_without_writes_check_refuses_a_mixed_tool(tmp_path):
    with pytest.raises(BuildProfileError, match="writes-check"):
        _ctx(tmp_path, writes="never", drop_hooks=("writes-check",))


def test_never_without_writes_check_refuses_a_write_tools_entry(tmp_path):
    """A never render cannot leave a write_tools entry callable with no gate."""
    with pytest.raises(BuildProfileError, match="control_system.write_tools"):
        _ctx(
            tmp_path,
            writes="never",
            write_tools=["mcp__controls__tune_knob"],
            servers={"python": {"enabled": False}},
            drop_hooks=("writes-check",),
        )


def test_never_without_writes_check_still_builds_for_facility_write_tools(tmp_path):
    """Facility write tools are denied statically on never; no per-call gate is needed."""
    ctx = _ctx(
        tmp_path,
        writes="never",
        servers={"python": {"enabled": False}, "plc": _PLC},
        drop_hooks=("writes-check",),
    )
    assert "mcp__plc__set_value" in ctx["killswitch_deny"]


def test_allowed_without_approval_is_refused_naming_the_tier_key(tmp_path):
    with pytest.raises(BuildProfileError, match=r"control_system\.writes: never"):
        _ctx(tmp_path, writes="allowed", drop_hooks=("approval",))
