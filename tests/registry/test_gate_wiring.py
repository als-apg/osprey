"""Whole-gate drift net for the Bluesky MCP safety surface.

Every later rename (Tasks 3.1-3.4 delete/rename tools as atomic changes to a
single ``osprey.bluesky_tool_names`` constant) runs under this test. It proves
that the ENTIRE gate surface resolves from those constants — so a rename that
updates the constant carries through every gate by construction, and a rename
that detaches a tool from its kill switch, destructive-marker floor, approval
hook, or allow/ask list fails HERE.

The assertions are written *against the constants* (iterate the constants,
check each surface), never against a second hardcoded name list: the failure
this test exists to catch is drift — a name present in one surface but not
another. A second literal list here would itself be one more surface to drift.

Surfaces spanned:

* the *registered* Bluesky MCP server (its actual ``@mcp.tool()`` names) and
  the "bluesky" ServerDefinition's rendered permission lists;
* the registry hook matchers + ``permissions_allow`` / ``permissions_ask``;
* ``agent_runner.write_tools`` destructive-marker classification;
* the rendered standalone hook template *sources* (deployed hooks run in a
  separate process/venv and cannot import OSPREY — so a constant they depend on
  is pinned by substring, the replica-drift-guard pattern from
  ``tests/agent_runner/test_write_tools.py``).

Registry structures are read via their public/registered forms (``resolve_servers``
and the live FastMCP registration), never by re-parsing ``registry/mcp.py``
source — route-safety style.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import osprey
from osprey import bluesky_tool_names as bsky
from osprey.registry.mcp import resolve_agents, resolve_servers

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HOOKS_DIR = Path(osprey.__file__).parent / "templates" / "claude_code" / "claude" / "hooks"


def _resolve_bluesky() -> dict:
    """Return the rendered "bluesky" server dict (its public/registered form)."""
    servers = resolve_servers(
        {"servers": {"bluesky": {"enabled": True}}},
        {"project_root": "/tmp/test-project", "current_python_env": "/usr/bin/python3"},
    )
    matches = [s for s in servers if s["name"] == "bluesky"]
    assert len(matches) == 1, "expected exactly one resolved bluesky server"
    return matches[0]


def _registered_server_tool_names() -> set[str]:
    """The tool names the live Bluesky FastMCP server actually registers.

    Importing the tool modules registers them via ``@mcp.tool()``; introspect
    the singleton directly (the ``test_python_server_registers_only_execute_tools``
    precedent) rather than running ``create_server()``, which does heavy
    config/workspace startup.
    """
    from osprey.mcp_server.bluesky import server as bsky_server
    from osprey.mcp_server.bluesky.tools import (  # noqa: F401 — registers tools
        authoring,
        draft,
        launch,
        read_tools,
        stop,
    )

    tools = asyncio.run(bsky_server.mcp._list_tools())
    return {getattr(t, "name", t) for t in tools}


def _hook_source(filename: str) -> str:
    path = _HOOKS_DIR / filename
    assert path.exists(), f"hook template source not found: {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# (a) Registered surface == the constants' full name set (no extras/omissions)
# ---------------------------------------------------------------------------


def test_registered_server_tools_equal_constant_name_set() -> None:
    """The live server's registered tool names equal ``ALL_TOOLS`` exactly.

    Catches a tool added to a ``tools/`` module but not the constants (extra),
    or a constant with no registered tool (omission) — either direction detaches
    the constant-driven gate wiring from the real tool surface.
    """
    registered = _registered_server_tool_names()
    expected = set(bsky.ALL_TOOLS)
    assert registered == expected, (
        f"registered Bluesky tools != constants ALL_TOOLS — "
        f"extras (registered, no constant): {sorted(registered - expected)}; "
        f"omissions (constant, not registered): {sorted(expected - registered)}"
    )


def test_permission_surface_equals_constant_name_set() -> None:
    """Every allow/ask entry is a constant value, and their union is ALL_TOOLS.

    Reads the rendered ServerDefinition (registered form), not the source.
    """
    bluesky = _resolve_bluesky()
    allow = bluesky["permissions_allow"]
    ask = bluesky["permissions_ask"]

    for name in (*allow, *ask):
        assert name in bsky.ALL_TOOLS, (
            f"{name!r} sits in a bluesky permission list but is not a "
            f"bluesky_tool_names constant value — a literal has drifted in"
        )
    assert set(allow) | set(ask) == set(bsky.ALL_TOOLS), (
        "allow ∪ ask must cover every constant with no extras/omissions"
    )
    assert set(allow).isdisjoint(ask), "no tool may be both silent-allow and ask-gated"


# ---------------------------------------------------------------------------
# (b) allow / ask / hook matchers / destructive-marker check == constant values
# ---------------------------------------------------------------------------


def test_permissions_allow_is_exactly_read_plus_draft_constants() -> None:
    """Silent-allow == the read + draft constant groups (touch no hardware)."""
    allow = _resolve_bluesky()["permissions_allow"]
    for tool in (*bsky.READ_TOOLS, *bsky.DRAFT_TOOLS):
        assert tool in allow, f"{tool!r} (read/draft, silent-allow) missing from permissions_allow"
    assert set(allow) == set(bsky.READ_TOOLS) | set(bsky.DRAFT_TOOLS)


def test_permissions_ask_is_exactly_authoring_plus_run_control_constants() -> None:
    """Approval-gated == the authoring + run-control constant groups."""
    ask = _resolve_bluesky()["permissions_ask"]
    for tool in (*bsky.AUTHORING_TOOLS, *bsky.RUN_CONTROL_TOOLS):
        assert tool in ask, f"{tool!r} (authoring/run-control) missing from permissions_ask"
    assert set(ask) == set(bsky.AUTHORING_TOOLS) | set(bsky.RUN_CONTROL_TOOLS)


def test_hook_matchers_resolve_from_constants() -> None:
    """Every pre-hook matcher equals ``matcher(<constant>)`` — no free string.

    And the matcher set is exactly the ask-gated tools (authoring + run-control):
    the silent-allow read/draft tools carry no pre-hook.
    """
    bluesky = _resolve_bluesky()
    matchers = {r["matcher"] for r in bluesky["hooks_pre"]}

    constant_matchers = {bsky.matcher(t) for t in bsky.ALL_TOOLS}
    for m in matchers:
        assert m in constant_matchers, (
            f"hook matcher {m!r} does not equal bsky.matcher(<constant>) — "
            f"a raw matcher string has drifted from the constants"
        )
    expected = {bsky.matcher(t) for t in (*bsky.AUTHORING_TOOLS, *bsky.RUN_CONTROL_TOOLS)}
    assert matchers == expected


def test_write_tools_destructive_markers_is_shared_constant() -> None:
    """The headless read-only floor's marker vocabulary IS the shared constant.

    Identity, not equality: a rename or narrowing of ``DESTRUCTIVE_MARKERS``
    must not leave ``write_tools`` pointing at a private copy.
    """
    from osprey.agent_runner import write_tools

    assert write_tools._DESTRUCTIVE_MARKERS is bsky.DESTRUCTIVE_MARKERS


# ---------------------------------------------------------------------------
# (c) Rendered hook template SOURCES carry the load-bearing constant literals
#
# Deployed hooks run standalone (separate process/venv) and cannot import
# OSPREY, so a tool name they depend on lives as a string literal in the
# rendered source — pinned here by substring so a constant rename that skips
# the template fails loudly.
#
# Inspecting the two sources: only ``osprey_approval.py`` carries a Bluesky
# tool literal — ``launch_run`` — because it special-cases that one name to
# render plan/provenance/validation detail at approval time. ``stop_run`` /
# ``write_plan`` / ``validate_plan`` are handled by the hook's GENERIC per-tool
# policy dispatch (keyed on the short name extracted from an approval prefix),
# so they carry no literal. ``osprey_writes_check.py`` carries NO Bluesky
# literal at all: its write-tool set is data-driven from ``hook_config.json``
# (rendered from the registry HookRule), so launch_run's kill-switch leg is
# rename-safe without touching that source. Those are the load-bearing facts
# this section pins.
# ---------------------------------------------------------------------------


def test_approval_template_source_carries_launch_run_constant() -> None:
    """The standalone approval hook still literals ``launch_run``.

    ``osprey_approval.py`` branches on ``short_name == "launch_run"`` (and lists
    it in its frontmatter) to fetch and render plan source / provenance /
    validation status for the human approver — the documented backstop for a
    plan body that slips past the automated validator. A rename of the
    ``LAUNCH_RUN`` constant that leaves this deployed source untouched silently
    detaches that enrichment; pin the value so the rename must update it too.
    """
    src = _hook_source("osprey_approval.py")
    assert bsky.LAUNCH_RUN in src, (
        f"osprey_approval.py no longer contains {bsky.LAUNCH_RUN!r} — the "
        f"standalone approval hook special-cases this name to render launch "
        f"detail; a LAUNCH_RUN rename that skips this template detaches it"
    )


def test_writes_check_template_carries_no_bluesky_tool_literal() -> None:
    """The kill switch stays data-driven — no Bluesky tool name is hardcoded.

    ``launch_run``'s writes-check gating flows registry HookRule →
    ``hook_config.json`` → this hook's runtime ``write_tools`` load, never a
    literal here. Pinning the ABSENCE documents why a Bluesky tool rename never
    needs to touch this standalone source (and flags anyone who reintroduces a
    literal that would then silently drift on the next rename).
    """
    src = _hook_source("osprey_writes_check.py")
    present = [t for t in bsky.ALL_TOOLS if t in src]
    assert present == [], (
        f"osprey_writes_check.py hardcodes Bluesky tool name(s) {present} — the "
        f"writes kill switch must stay data-driven (write_tools loaded from "
        f"hook_config.json rendered off the registry); keep the gate in the "
        f"registry HookRule, not this standalone hook source"
    )


# ---------------------------------------------------------------------------
# Safety-semantics mapping from the constants (FR7 skeleton)
# ---------------------------------------------------------------------------


def test_clear_draft_matches_destructive_marker_get_set_do_not() -> None:
    """``clear_draft`` is destructive-classified; get/set draft are not.

    This is what keeps ``clear_draft`` blocked under the headless read-only
    floor despite sitting in ``permissions_allow`` — by design, via the shared
    marker vocabulary, not a Bluesky-specific rule.
    """
    from osprey.agent_runner.write_tools import _is_destructive

    assert _is_destructive(bsky.CLEAR_DRAFT)
    assert not _is_destructive(bsky.GET_DRAFT)
    assert not _is_destructive(bsky.SET_DRAFT)
    # The specific marker that classifies it is a member of the shared vocab.
    assert any(m in bsky.CLEAR_DRAFT for m in bsky.DESTRUCTIVE_MARKERS)


def test_draft_tools_are_silent_allow() -> None:
    """get/set/clear draft are silent-allow (in allow, not ask, no pre-hook).

    (``clear_draft`` is still blocked headless-read-only via the marker
    floor above — that is a separate, orthogonal mechanism from the interactive
    allow/ask/hook posture asserted here.)
    """
    bluesky = _resolve_bluesky()
    allow = bluesky["permissions_allow"]
    ask = bluesky["permissions_ask"]
    gated = {r["matcher"] for r in bluesky["hooks_pre"]}

    for tool in bsky.DRAFT_TOOLS:
        assert tool in allow, f"{tool!r} draft tool must be silent-allow"
        assert tool not in ask
        assert bsky.matcher(tool) not in gated, f"{tool!r} draft tool must carry no pre-hook"


def test_launch_run_carries_both_gates_stop_run_approval_only() -> None:
    """``launch_run`` = writes-check + approval; ``stop_run`` = approval only.

    Pins how the rules distinguish the two directions: the kill switch
    (writes-check) gates starting a scan but must NEVER gate stopping one (the
    safe direction), so ``stop_run`` carries approval alone.
    """
    by_matcher = {r["matcher"]: r for r in _resolve_bluesky()["hooks_pre"]}

    launch = by_matcher[bsky.matcher(bsky.LAUNCH_RUN)]
    launch_cmds = [h["command"] for h in launch["hooks"]]
    assert any("osprey_writes_check.py" in c for c in launch_cmds), (
        "launch_run must be kill-switched"
    )
    assert any("osprey_approval.py" in c for c in launch_cmds), "launch_run must be approval-gated"

    stop = by_matcher[bsky.matcher(bsky.STOP_RUN)]
    stop_cmds = [h["command"] for h in stop["hooks"]]
    assert any("osprey_approval.py" in c for c in stop_cmds), "stop_run must be approval-gated"
    assert not any("osprey_writes_check.py" in c for c in stop_cmds), (
        "stop_run must NEVER be writes-check/kill-switch gated — the kill switch "
        "must not be able to block stopping a run"
    )


# ---------------------------------------------------------------------------
# Health server gate wiring (opt-in, read-only allow/ask split)
# ---------------------------------------------------------------------------

_HEALTH_CTX = {"project_root": "/tmp/test-project", "current_python_env": "/usr/bin/python3"}


def _render_settings(claude_code_config: dict) -> dict:
    """Render settings.json.j2 end-to-end and return the parsed permissions block.

    Exercises the real allow/ask wiring: the template prefixes each server's
    permissions_allow / permissions_ask tools as ``mcp__<name>__<tool>`` and only
    emits entries for ENABLED servers.
    """
    from osprey.cli.templates.manager import TemplateManager

    ctx = dict(_HEALTH_CTX)
    ctx["facility_permissions"] = {}
    ctx["servers"] = resolve_servers(claude_code_config, ctx)
    ctx["agents"] = resolve_agents(claude_code_config, ctx, resolved_servers=ctx["servers"])
    tm = TemplateManager()
    template = tm.jinja_env.get_template("claude_code/claude/settings.json.j2")
    return json.loads(template.render(**ctx))


def test_health_allow_ask_split_renders_prefixed_tools() -> None:
    """With health enabled, the allow/ask split renders as prefixed tool names.

    health_check → permissions.allow (silent, read-only); health_check_full →
    permissions.ask (approval-gated). The ``mcp__health__`` prefix is applied by
    the settings template, not stored in the registry.
    """
    data = _render_settings({"servers": {"health": {"enabled": True}}})
    allow = set(data["permissions"]["allow"])
    ask = set(data["permissions"]["ask"])

    assert "mcp__health__health_check" in allow
    assert "mcp__health__health_check_full" in ask
    # The split is exclusive: neither tool leaks into the other list.
    assert "mcp__health__health_check_full" not in allow
    assert "mcp__health__health_check" not in ask


def test_health_absent_from_rendered_settings_unless_enabled() -> None:
    """Opt-in: health tools appear in the rendered gate ONLY when enabled.

    The server ships default_enabled=False, so a default config emits no
    ``mcp__health__*`` permission entry; setting claude_code.servers.health.enabled
    = true is what surfaces the tools.
    """
    default = _render_settings({})
    default_perms = default["permissions"]["allow"] + default["permissions"]["ask"]
    assert not any(p.startswith("mcp__health__") for p in default_perms), (
        "health tools must be absent from the default (opt-out) rendered settings"
    )

    enabled = _render_settings({"servers": {"health": {"enabled": True}}})
    enabled_perms = enabled["permissions"]["allow"] + enabled["permissions"]["ask"]
    assert any(p.startswith("mcp__health__") for p in enabled_perms), (
        "health tools must surface once the server is opted in"
    )


def test_health_carries_no_pretooluse_hook() -> None:
    """Read-only posture: the enabled health server contributes no PreToolUse rule.

    No _WRITES_CHECK / approval hook is wired for either tool — every connector
    touch is config-declared and read-only, so there is nothing to gate.
    """
    data = _render_settings({"servers": {"health": {"enabled": True}}})
    pre_matchers = [r["matcher"] for r in data["hooks"]["PreToolUse"]]
    assert not any(m.startswith("mcp__health__") for m in pre_matchers)
