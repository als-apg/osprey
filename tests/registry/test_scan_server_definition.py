"""Tests for the scan ServerDefinition in FRAMEWORK_SERVERS.

Covers: module/env resolution, permissions_allow/permissions_ask contents,
hooks_pre structure (launch_scan carries writes-check + approval, stop_scan
carries approval only — the kill switch must never block stopping a scan),
hooks_post error guidance, opt-in default_enabled, and that the registry-driven
read-only disallow floor (``read_only_disallowed_tools``) automatically picks
up both ask-gated scan tools without any scan-specific code there.
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


def _resolve_scan(cfg=None, ctx=None):
    servers = resolve_servers(cfg or {}, ctx or _base_ctx())
    matches = [s for s in servers if s["name"] == "scan"]
    assert len(matches) == 1
    return matches[0]


# ---------------------------------------------------------------------------
# Module / env resolution
# ---------------------------------------------------------------------------


def test_scan_server_module():
    scan = _resolve_scan()
    assert scan["args"] == ["-m", "osprey.mcp_server.scan"]


def test_scan_server_env():
    scan = _resolve_scan()
    assert scan["env"]["OSPREY_CONFIG"] == "/tmp/test-project/config.yml"
    assert scan["env"]["CONFIG_FILE"] == "/tmp/test-project/config.yml"
    # Shell variable references pass through untouched for runtime expansion.
    assert scan["env"]["BLUESKY_BRIDGE_URL"] == "${BLUESKY_BRIDGE_URL:-http://127.0.0.1:8090}"
    assert scan["env"]["BLUESKY_PROMOTE_TOKEN"] == "${BLUESKY_PROMOTE_TOKEN:-}"


def test_scan_server_disabled_by_default():
    """Opt-in like phoebus — running it requires a live facility Bluesky bridge."""
    scan = _resolve_scan()
    assert scan["enabled"] is False


def test_scan_server_enabled_via_config_override():
    scan = _resolve_scan(cfg={"servers": {"scan": {"enabled": True}}})
    assert scan["enabled"] is True


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------


def test_scan_permissions_allow():
    scan = _resolve_scan()
    assert scan["permissions_allow"] == [
        "create_scan_intent",
        "scan_status",
        "list_scan_plans",
        "list_runs",
        "read_scan_data",
    ]


def test_scan_permissions_ask():
    scan = _resolve_scan()
    assert scan["permissions_ask"] == ["launch_scan", "stop_scan"]


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def test_scan_hooks_pre_structure():
    scan = _resolve_scan()
    by_matcher = {r["matcher"]: r for r in scan["hooks_pre"]}
    assert set(by_matcher) == {"mcp__scan__launch_scan", "mcp__scan__stop_scan"}

    launch = by_matcher["mcp__scan__launch_scan"]
    launch_commands = [h["command"] for h in launch["hooks"]]
    assert len(launch["hooks"]) == 2
    assert any("osprey_writes_check.py" in c for c in launch_commands)
    assert any("osprey_approval.py" in c for c in launch_commands)

    stop = by_matcher["mcp__scan__stop_scan"]
    stop_commands = [h["command"] for h in stop["hooks"]]
    # stop_scan must NEVER be writes-check-gated — the kill switch must not
    # block stopping a scan (the safe direction).
    assert len(stop["hooks"]) == 1
    assert "osprey_approval.py" in stop_commands[0]
    assert not any("osprey_writes_check.py" in c for c in stop_commands)


def test_scan_hooks_post_error_guidance():
    scan = _resolve_scan()
    assert len(scan["hooks_post"]) == 1
    rule = scan["hooks_post"][0]
    assert rule["matcher"] == "mcp__scan__.*"
    assert any("osprey_error_guidance.py" in h["command"] for h in rule["hooks"])


# ---------------------------------------------------------------------------
# read_only_disallowed_tools auto-derivation (headless kill-switch floor)
# ---------------------------------------------------------------------------


def test_read_only_disallowed_tools_covers_scan_ask_tools(tmp_path: Path):
    """launch_scan and stop_scan must both be blocked under bypassPermissions —
    purely from the registry's permissions_ask entry, no scan-specific code."""
    from osprey.agent_runner.write_tools import read_only_disallowed_tools

    result = set(read_only_disallowed_tools(tmp_path))
    assert "mcp__scan__launch_scan" in result
    assert "mcp__scan__stop_scan" in result


def test_hook_config_template_derives_write_tools_and_approval_prefix():
    """hook_config.json.j2's own derivation logic: launch_scan lands in
    write_tools (writes-check-gated), and 'mcp__scan__' lands in
    approval_prefixes (both ask tools carry the approval hook)."""
    import jinja2

    template_path = (
        Path(__file__).resolve().parents[2]
        / "src/osprey/templates/claude_code/claude/hooks/hook_config.json.j2"
    )
    env = jinja2.Environment()
    template = env.from_string(template_path.read_text(encoding="utf-8"))
    scan = _resolve_scan(cfg={"servers": {"scan": {"enabled": True}}})
    rendered = template.render(servers=[scan], control_system_write_tools=[])

    import json

    data = json.loads(rendered)
    assert "mcp__scan__launch_scan" in data["write_tools"]
    assert "mcp__scan__stop_scan" not in data["write_tools"]
    assert "mcp__scan__" in data["approval_prefixes"]


def test_scan_can_be_extended_like_other_framework_servers():
    """Drift guard: scan has no `condition`, so build_extended_server() (used
    for a second bridge instance, one per BLUESKY_BRIDGE_URL/BLUESKY_PROMOTE_TOKEN
    per the plan) must accept it exactly like phoebus's extends path."""
    from osprey.registry.mcp import build_extended_server

    clone = build_extended_server("scan2", {"extends": "scan"})
    assert clone is not None
    assert clone.permissions_ask == ["launch_scan", "stop_scan"]
    matchers = {r.matcher for r in clone.hooks_pre}
    assert matchers == {"mcp__scan2__launch_scan", "mcp__scan2__stop_scan"}


def test_scan_present_in_framework_servers():
    assert "scan" in FRAMEWORK_SERVERS
    assert FRAMEWORK_SERVERS["scan"].module == "osprey.mcp_server.scan"
