"""Tests for the bluesky ServerDefinition in FRAMEWORK_SERVERS.

Covers: module/env resolution, permissions_allow/permissions_ask contents,
hooks_pre structure (launch_run carries writes-check + approval, stop_run
carries approval only — the kill switch must never block stopping a run),
hooks_post error guidance, opt-in default_enabled, and that the registry-driven
read-only disallow floor (``read_only_disallowed_tools``) automatically picks
up both ask-gated bluesky tools without any bluesky-specific code there.

Also covers the authoring tools added in task 2.3 — ``write_plan`` and
``validate_plan`` — which reach no hardware either way (write only
emits a file, validate only dry-runs mock devices in an EPICS_CA_*-scrubbed
subprocess), so both carry ``_APPROVAL`` only and must never be kill-switched.
"""

from pathlib import Path

from osprey.registry.mcp import FRAMEWORK_SERVERS, resolve_servers


def _base_ctx(**overrides):
    ctx = {
        "project_root": "/tmp/test-project",
        "current_python_env": "/usr/bin/python3",
    }
    ctx.update(overrides)
    return ctx


def _resolve_bluesky(cfg=None, ctx=None):
    servers = resolve_servers(cfg or {}, ctx or _base_ctx())
    matches = [s for s in servers if s["name"] == "bluesky"]
    assert len(matches) == 1
    return matches[0]


# ---------------------------------------------------------------------------
# Module / env resolution
# ---------------------------------------------------------------------------


def test_bluesky_server_module():
    bluesky = _resolve_bluesky()
    assert bluesky["args"] == ["-m", "osprey.mcp_server.bluesky"]


def test_bluesky_server_env():
    bluesky = _resolve_bluesky()
    assert bluesky["env"]["OSPREY_CONFIG"] == "/tmp/test-project/config.yml"
    assert bluesky["env"]["CONFIG_FILE"] == "/tmp/test-project/config.yml"
    # Shell variable references pass through untouched for runtime expansion.
    assert bluesky["env"]["BLUESKY_BRIDGE_URL"] == "${BLUESKY_BRIDGE_URL:-http://127.0.0.1:8090}"
    assert bluesky["env"]["BLUESKY_PROMOTE_TOKEN"] == "${BLUESKY_PROMOTE_TOKEN:-}"


def test_bluesky_server_disabled_by_default():
    """Opt-in like phoebus — running it requires a live facility Bluesky bridge."""
    bluesky = _resolve_bluesky()
    assert bluesky["enabled"] is False


def test_bluesky_server_enabled_via_config_override():
    bluesky = _resolve_bluesky(cfg={"servers": {"bluesky": {"enabled": True}}})
    assert bluesky["enabled"] is True


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------


def test_bluesky_permissions_allow():
    bluesky = _resolve_bluesky()
    assert bluesky["permissions_allow"] == [
        "create_run_intent",
        "run_status",
        "list_plans",
        "list_runs",
        "read_run_data",
        "get_plan_draft",
        "set_plan_draft",
        "clear_plan_draft",
    ]


def test_bluesky_permissions_ask():
    bluesky = _resolve_bluesky()
    assert bluesky["permissions_ask"] == [
        "launch_run",
        "stop_run",
        "write_plan",
        "validate_plan",
    ]


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def test_bluesky_hooks_pre_structure():
    bluesky = _resolve_bluesky()
    by_matcher = {r["matcher"]: r for r in bluesky["hooks_pre"]}
    assert set(by_matcher) == {
        "mcp__bluesky__launch_run",
        "mcp__bluesky__stop_run",
        "mcp__bluesky__write_plan",
        "mcp__bluesky__validate_plan",
    }

    launch = by_matcher["mcp__bluesky__launch_run"]
    launch_commands = [h["command"] for h in launch["hooks"]]
    assert len(launch["hooks"]) == 2
    assert any("osprey_writes_check.py" in c for c in launch_commands)
    assert any("osprey_approval.py" in c for c in launch_commands)

    stop = by_matcher["mcp__bluesky__stop_run"]
    stop_commands = [h["command"] for h in stop["hooks"]]
    # stop_run must NEVER be writes-check-gated — the kill switch must not
    # block stopping a run (the safe direction).
    assert len(stop["hooks"]) == 1
    assert "osprey_approval.py" in stop_commands[0]
    assert not any("osprey_writes_check.py" in c for c in stop_commands)

    # write_plan/validate_plan (task 2.3) reach no hardware
    # either way — write only emits a file, validate only dry-runs mock
    # devices in an EPICS_CA_*-scrubbed subprocess — so both are
    # approval-gated but NEVER kill-switchable, exactly like stop_run.
    for tool in ("write_plan", "validate_plan"):
        rule = by_matcher[f"mcp__bluesky__{tool}"]
        commands = [h["command"] for h in rule["hooks"]]
        assert len(rule["hooks"]) == 1
        assert "osprey_approval.py" in commands[0]
        assert not any("osprey_writes_check.py" in c for c in commands)


def test_bluesky_authoring_tools_never_writes_check_gated():
    """Regression: assert directly on the dataclass HookRule/HookEntry objects
    (not the resolved dict form) that write_plan and
    validate_plan carry _APPROVAL but never _WRITES_CHECK — the
    safety posture that keeps them permitted when control_system.writes_enabled
    is off, since neither reaches hardware."""
    from osprey.registry.mcp import FRAMEWORK_SERVERS

    bluesky_def = FRAMEWORK_SERVERS["bluesky"]
    by_matcher = {rule.matcher: rule for rule in bluesky_def.hooks_pre}

    for tool in ("write_plan", "validate_plan"):
        rule = by_matcher[f"mcp__bluesky__{tool}"]
        assert tool in bluesky_def.permissions_ask
        assert len(rule.hooks) == 1
        assert "osprey_approval.py" in rule.hooks[0].command
        assert not any("osprey_writes_check.py" in h.command for h in rule.hooks)

    # launch_run stays kill-switchable; stop_run stays approval-only —
    # unchanged by the new authoring tools.
    launch_rule = by_matcher["mcp__bluesky__launch_run"]
    assert any("osprey_writes_check.py" in h.command for h in launch_rule.hooks)
    assert any("osprey_approval.py" in h.command for h in launch_rule.hooks)

    stop_rule = by_matcher["mcp__bluesky__stop_run"]
    assert len(stop_rule.hooks) == 1
    assert "osprey_approval.py" in stop_rule.hooks[0].command
    assert not any("osprey_writes_check.py" in h.command for h in stop_rule.hooks)

    # The read tiers are unchanged by task 2.3's authoring additions (task
    # 2.1's draft tools are silent-allow too, added separately below).
    assert bluesky_def.permissions_allow == [
        "create_run_intent",
        "run_status",
        "list_plans",
        "list_runs",
        "read_run_data",
        "get_plan_draft",
        "set_plan_draft",
        "clear_plan_draft",
    ]


def test_bluesky_hooks_post_error_guidance():
    bluesky = _resolve_bluesky()
    assert len(bluesky["hooks_post"]) == 1
    rule = bluesky["hooks_post"][0]
    assert rule["matcher"] == "mcp__bluesky__.*"
    assert any("osprey_error_guidance.py" in h["command"] for h in rule["hooks"])


# ---------------------------------------------------------------------------
# read_only_disallowed_tools auto-derivation (headless kill-switch floor)
# ---------------------------------------------------------------------------


def test_read_only_disallowed_tools_covers_bluesky_ask_tools(tmp_path: Path):
    """launch_run and stop_run must both be blocked under bypassPermissions —
    purely from the registry's permissions_ask entry, no bluesky-specific code."""
    from osprey.agent_runner.write_tools import read_only_disallowed_tools

    result = set(read_only_disallowed_tools(tmp_path))
    assert "mcp__bluesky__launch_run" in result
    assert "mcp__bluesky__stop_run" in result


def test_hook_config_template_derives_write_tools_and_approval_prefix():
    """hook_config.json.j2's own derivation logic: launch_run lands in
    write_tools (writes-check-gated), and 'mcp__bluesky__' lands in
    approval_prefixes (both ask tools carry the approval hook)."""
    import jinja2

    template_path = (
        Path(__file__).resolve().parents[2]
        / "src/osprey/templates/claude_code/claude/hooks/hook_config.json.j2"
    )
    env = jinja2.Environment()
    template = env.from_string(template_path.read_text(encoding="utf-8"))
    bluesky = _resolve_bluesky(cfg={"servers": {"bluesky": {"enabled": True}}})
    rendered = template.render(servers=[bluesky], control_system_write_tools=[])

    import json

    data = json.loads(rendered)
    assert "mcp__bluesky__launch_run" in data["write_tools"]
    assert "mcp__bluesky__stop_run" not in data["write_tools"]
    assert "mcp__bluesky__" in data["approval_prefixes"]


def test_bluesky_can_be_extended_like_other_framework_servers():
    """Drift guard: bluesky has no `condition`, so build_extended_server() (used
    for a second bridge instance, one per BLUESKY_BRIDGE_URL/BLUESKY_PROMOTE_TOKEN
    per the plan) must accept it exactly like phoebus's extends path."""
    from osprey.registry.mcp import build_extended_server

    clone = build_extended_server("bluesky2", {"extends": "bluesky"})
    assert clone is not None
    assert clone.permissions_ask == [
        "launch_run",
        "stop_run",
        "write_plan",
        "validate_plan",
    ]
    matchers = {r.matcher for r in clone.hooks_pre}
    assert matchers == {
        "mcp__bluesky2__launch_run",
        "mcp__bluesky2__stop_run",
        "mcp__bluesky2__write_plan",
        "mcp__bluesky2__validate_plan",
    }


def test_bluesky_present_in_framework_servers():
    assert "bluesky" in FRAMEWORK_SERVERS
    assert FRAMEWORK_SERVERS["bluesky"].module == "osprey.mcp_server.bluesky"
