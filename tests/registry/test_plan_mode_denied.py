"""The plan-mode deny list: every tool that acts, derived from the registry.

In plan mode the session may read and think, never act. Which tools act is the
registry's own side-effect classification (``server_side_effect_tools``), the
one the headless read-only floor reads too; ``plan_mode_allowed`` names the few
side-effecting tools that stay callable while planning, and ``audited_reads``
the approval-gated tools that only read. These tests assert relations to the
registry, never a copied list.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from osprey.agent_runner.tool_names import FOREIGN_MCP_NAMESPACES, READ_ONLY_DENIED_BUILTINS
from osprey.agent_runner.write_tools import read_only_disallowed_tools
from osprey.registry.mcp import (
    _APPROVAL,
    _APPROVAL_PREFLIGHT,
    FRAMEWORK_SERVERS,
    _server_plan_mode_denied,
    build_extended_server,
    framework_write_tools,
    plan_mode_denied_tools,
    resolve_servers,
    server_side_effect_tools,
)

#: The servers a framework profile only gets by asking for them.
_OPT_IN_SERVERS = sorted(
    name for name, sdef in FRAMEWORK_SERVERS.items() if not sdef.default_enabled
)

#: A render context in which every conditional framework server renders.
_ALL_ON_CTX = {
    "project_root": "/tmp/test-project",
    "current_python_env": "/usr/bin/python3",
    "channel_finder_pipeline": "hierarchical",
    "graphdb_configured": True,
    "event_dispatcher_wired": True,
    "phoebus_agent_access": "read_write",
    "pyaml_view_present": True,
}


def _all_on_config(extra: dict | None = None) -> dict:
    servers = {name: {"enabled": True} for name in _OPT_IN_SERVERS}
    servers.update(extra or {})
    return {"servers": servers}


def _resolve(extra: dict | None = None) -> list[dict]:
    return resolve_servers(_all_on_config(extra), dict(_ALL_ON_CTX))


def _by_name(servers: list[dict]) -> dict[str, dict]:
    return {server["name"]: server for server in servers}


def _short(sdef_name: str, full: str) -> str:
    return full[len(f"mcp__{sdef_name}__") :]


def _registry_expected() -> list[str]:
    """Every framework side-effect tool minus its server's exemptions and reads."""
    out: list[str] = []
    for sdef in FRAMEWORK_SERVERS.values():
        allowed = {f"mcp__{sdef.name}__{t}" for t in (*sdef.plan_mode_allowed, *sdef.audited_reads)}
        for name in server_side_effect_tools(sdef):
            if name not in allowed and name not in out:
                out.append(name)
    return out


# ── Exemptions ───────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(FRAMEWORK_SERVERS))
def test_every_exemption_names_a_side_effecting_tool(name):
    """A tool that stays callable in plan mode must be one the classifier marks.

    An exemption naming a read, or a tool the server no longer has, is stale.
    """
    sdef = FRAMEWORK_SERVERS[name]
    side_effect = {_short(sdef.name, t) for t in server_side_effect_tools(sdef)}

    assert set(sdef.plan_mode_allowed) <= side_effect


# ── Coverage: a new approval-gated tool is classified or the test fails ─


def _approval_matchers(sdef) -> list[str]:
    return [
        rule.matcher
        for rule in sdef.hooks_pre
        if any(hook is _APPROVAL or hook is _APPROVAL_PREFLIGHT for hook in rule.hooks)
    ]


def test_every_approval_gated_tool_is_denied_exempted_or_a_read():
    """A new approval-gated tool fails here until the registry places it."""
    resolved = _by_name(_resolve())
    unclassified: list[str] = []
    for sdef in FRAMEWORK_SERVERS.values():
        denied = set(resolved[sdef.name]["plan_mode_denied"])
        for matcher in _approval_matchers(sdef):
            short = _short(sdef.name, matcher)
            if matcher in denied:
                continue
            if short in sdef.plan_mode_allowed or short in sdef.permissions_allow:
                continue
            if short in sdef.audited_reads:
                continue
            unclassified.append(matcher)

    assert unclassified == []


# ── Audited reads ────────────────────────────────────────────────────


def _audited_read_faults(sdef) -> list[str]:
    """Every ``audited_reads`` name that is not an approval-gated read of *sdef*."""
    gated = {_short(sdef.name, m) for m in _approval_matchers(sdef)}
    acting = {_short(sdef.name, t) for t in server_side_effect_tools(sdef)}
    return [tool for tool in sdef.audited_reads if tool not in gated or tool in acting]


@pytest.mark.parametrize("name", sorted(FRAMEWORK_SERVERS))
def test_every_audited_read_is_an_approval_gated_read(name):
    """An audited read carries the approval hook and is not one the classifier marks."""
    assert _audited_read_faults(FRAMEWORK_SERVERS[name]) == []


@pytest.mark.parametrize(
    "bogus",
    [
        pytest.param("channel_write", id="side-effecting"),
        pytest.param("channel_limits", id="no-approval-rule"),
    ],
)
def test_a_bogus_audited_read_is_rejected(bogus):
    sdef = dataclasses.replace(FRAMEWORK_SERVERS["controls"], audited_reads=[bogus])

    assert _audited_read_faults(sdef) == [bogus]


def test_the_controls_server_names_its_two_audited_reads():
    controls = FRAMEWORK_SERVERS["controls"]
    denied = set(_by_name(_resolve())["controls"]["plan_mode_denied"])

    assert sorted(controls.audited_reads) == ["archiver_read", "channel_read"]
    assert denied.isdisjoint(f"mcp__controls__{t}" for t in controls.audited_reads)


def test_the_deny_derivation_never_denies_an_audited_read():
    """Even a tool the classifier marks stays callable once the registry names it a read."""
    controls = FRAMEWORK_SERVERS["controls"]
    sdef = dataclasses.replace(controls, audited_reads=[*controls.audited_reads, "channel_write"])
    assert "mcp__controls__channel_write" in _server_plan_mode_denied(controls)

    denied = _server_plan_mode_denied(sdef)

    assert "mcp__controls__channel_write" not in denied


def test_the_coverage_check_sees_both_approval_entries():
    """The bluesky arming pair carries the pre-flight approval entry, not the plain one."""
    matchers = _approval_matchers(FRAMEWORK_SERVERS["bluesky"])
    preflight = [
        rule.matcher
        for rule in FRAMEWORK_SERVERS["bluesky"].hooks_pre
        if any(hook is _APPROVAL_PREFLIGHT for hook in rule.hooks)
    ]

    assert preflight
    assert set(preflight) <= set(matchers)


# ── The render-wide list ─────────────────────────────────────────────


def test_an_all_on_render_denies_the_registry_side_effects_minus_exemptions():
    denied = plan_mode_denied_tools(_resolve(), [])

    assert sorted(denied) == sorted(_registry_expected())
    assert len(denied) == len(set(denied))


def test_every_framework_write_tool_is_denied():
    denied = set(plan_mode_denied_tools(_resolve(), []))

    assert set(framework_write_tools()) <= denied


def test_plan_mode_never_denies_what_the_read_only_floor_allows(tmp_path: Path):
    """The two consumers of one classifier cannot disagree beyond the exemptions."""
    config = {"claude_code": _all_on_config()}
    (tmp_path / "config.yml").write_text(yaml.safe_dump(config))
    floor = set(read_only_disallowed_tools(tmp_path))
    floor -= set(READ_ONLY_DENIED_BUILTINS) | set(FOREIGN_MCP_NAMESPACES)

    assert set(plan_mode_denied_tools(_resolve(), [])) <= floor


def test_each_exemption_is_absent_from_the_render():
    denied = set(plan_mode_denied_tools(_resolve(), []))
    for sdef in FRAMEWORK_SERVERS.values():
        for tool in sdef.plan_mode_allowed:
            assert f"mcp__{sdef.name}__{tool}" not in denied


# ── Clones, custom servers, disabled servers, extras ─────────────────


def test_an_extends_clone_inherits_the_classification_under_its_own_name():
    bluesky = FRAMEWORK_SERVERS["bluesky"]
    acting = next(
        _short("bluesky", t)
        for t in server_side_effect_tools(bluesky)
        if _short("bluesky", t) not in bluesky.plan_mode_allowed
    )
    exempt = bluesky.plan_mode_allowed[0]

    denied = plan_mode_denied_tools(_resolve({"bluesky2": {"extends": "bluesky"}}), [])

    assert f"mcp__bluesky2__{acting}" in denied
    assert f"mcp__bluesky2__{exempt}" not in denied


def test_an_extends_clone_carries_the_audited_reads_under_its_own_name():
    reads = FRAMEWORK_SERVERS["controls"].audited_reads
    assert reads, "the comparison below is vacuous"

    assert build_extended_server("controls2", {"extends": "controls"}).audited_reads == reads
    clone = _by_name(_resolve({"controls2": {"extends": "controls"}}))["controls2"]

    gated = {rule["matcher"] for rule in clone["hooks_pre"]}
    for tool in reads:
        assert f"mcp__controls2__{tool}" in gated
        assert f"mcp__controls2__{tool}" not in clone["plan_mode_denied"]


def test_a_custom_servers_ask_tools_are_denied_and_its_reads_are_not():
    custom = {
        "sitecam": {
            "command": "node",
            "args": ["sitecam.js"],
            "permissions": {"allow": ["snapshot"], "ask": ["pan"]},
        }
    }

    denied = plan_mode_denied_tools(_resolve(custom), [])

    assert "mcp__sitecam__pan" in denied
    assert "mcp__sitecam__snapshot" not in denied


def test_a_custom_writes_check_server_is_denied_whole():
    custom = {
        "sitectl": {
            "command": "node",
            "args": ["sitectl.js"],
            "hooks": {"pre_tool_use": ["writes_check"]},
        }
    }

    denied = plan_mode_denied_tools(_resolve(custom), [])

    assert "mcp__sitectl__.*" in denied


def test_a_disabled_server_contributes_nothing():
    servers = _resolve({"bluesky": {"enabled": False}})
    bluesky = _by_name(servers)["bluesky"]
    assert bluesky["plan_mode_denied"], "the comparison below is vacuous"

    denied = set(plan_mode_denied_tools(servers, []))

    assert denied.isdisjoint(bluesky["plan_mode_denied"])


def test_extra_write_tools_are_appended_once():
    servers = _resolve()
    base = plan_mode_denied_tools(servers, [])
    already = base[0]

    denied = plan_mode_denied_tools(servers, ["mcp__sitectl__set_point", already])

    assert denied == [*base, "mcp__sitectl__set_point"]
