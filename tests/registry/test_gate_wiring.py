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

import ast
import asyncio
import json
import re
from pathlib import Path

import yaml

import osprey
from osprey import bluesky_tool_names as bsky
from osprey.registry.mcp import FRAMEWORK_SERVERS, HOOK_PRESETS, resolve_agents, resolve_servers

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
        queue,
        read_tools,
        stop,
    )

    tools = asyncio.run(bsky_server.mcp.list_tools())
    return {getattr(t, "name", t) for t in tools}


def _hook_source(filename: str) -> str:
    path = _HOOKS_DIR / filename
    assert path.exists(), f"hook template source not found: {path}"
    return path.read_text(encoding="utf-8")


def _hook_docstring_and_body(filename: str) -> tuple[str, str]:
    """A hook template split at its module docstring: (docstring, everything after).

    The docstring carries the frontmatter header — what the hook is called,
    which event it answers and which tools it gates — and the body is the code
    that runs. Guards that pin a *header* against the registry read the first;
    guards that forbid a literal in the *code* read the second.
    """
    src = _hook_source(filename)
    start = src.find('"""')
    end = src.find('"""', start + 3)
    assert start >= 0 and end > start, f"{filename} has no module docstring"
    return src[start + 3 : end], src[end + 3 :]


def _hook_header_tools(filename: str) -> set[str]:
    """The short tool names a hook's frontmatter ``tools:`` line declares."""
    docstring, _ = _hook_docstring_and_body(filename)
    parts = docstring.split("---")
    assert len(parts) >= 3, f"{filename} has no frontmatter block"
    meta = yaml.safe_load(parts[1])
    return {tool.strip() for tool in str(meta["tools"]).split(",") if tool.strip()}


#: The hook script inside a wired command, whatever precedes or follows it.
_HOOK_SCRIPT = re.compile(r"/\.claude/hooks/(?P<script>[A-Za-z0-9_.-]+\.py)")


def _script_of(command: str) -> str:
    """The hook script a wired command runs, however the command is spelled.

    A command names an interpreter before the script path and may pass the
    script arguments after it — the approval hook is handed its pre-flight
    budget that way — so the script is read out of the path, never off either
    end of the string.
    """
    match = _HOOK_SCRIPT.search(command)
    assert match, f"no hook script in command: {command!r}"
    return match.group("script")


def _short_name(matcher: str) -> str:
    """``mcp__<server>__<tool>`` → ``<tool>``; any other matcher as it is."""
    if matcher.startswith("mcp__") and matcher.count("__") >= 2:
        return matcher.split("__", 2)[2]
    return matcher


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


def test_permissions_allow_is_exactly_read_plus_draft_plus_queue_read_constants() -> None:
    """Silent-allow == the read + draft + queue-read constant groups (touch no hardware)."""
    allow = _resolve_bluesky()["permissions_allow"]
    silent = (*bsky.READ_TOOLS, *bsky.DRAFT_TOOLS, *bsky.QUEUE_READ_TOOLS)
    for tool in silent:
        assert tool in allow, f"{tool!r} (read/draft/queue-read) missing from permissions_allow"
    assert set(allow) == set(silent)


def test_permissions_ask_is_exactly_authoring_plus_queue_control_plus_run_control() -> None:
    """Approval-gated == the authoring + queue-control + run-control constant groups."""
    ask = _resolve_bluesky()["permissions_ask"]
    gated = (*bsky.AUTHORING_TOOLS, *bsky.QUEUE_CONTROL_TOOLS, *bsky.RUN_CONTROL_TOOLS)
    for tool in gated:
        assert tool in ask, f"{tool!r} (authoring/queue-control/run-control) missing from ask"
    assert set(ask) == set(gated)


def test_hook_matchers_resolve_from_constants() -> None:
    """Every pre-hook matcher equals ``matcher(<constant>)`` — no free string.

    And the matcher set is exactly the ask-gated tools (authoring +
    queue-control + run-control): the silent-allow read/draft/queue-read tools
    carry no pre-hook.
    """
    bluesky = _resolve_bluesky()
    matchers = {r["matcher"] for r in bluesky["hooks_pre"]}

    constant_matchers = {bsky.matcher(t) for t in bsky.ALL_TOOLS}
    for m in matchers:
        assert m in constant_matchers, (
            f"hook matcher {m!r} does not equal bsky.matcher(<constant>) — "
            f"a raw matcher string has drifted from the constants"
        )
    expected = {
        bsky.matcher(t)
        for t in (*bsky.AUTHORING_TOOLS, *bsky.QUEUE_CONTROL_TOOLS, *bsky.RUN_CONTROL_TOOLS)
    }
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
# Inspecting the two sources: only ``osprey_approval.py`` carries Bluesky tool
# literals — the three queue-control names — because it special-cases each to
# render what would be queued/run/resumed at approval time. ``stop_run`` /
# ``write_plan`` / ``validate_plan`` are handled by the hook's GENERIC per-tool
# policy dispatch (keyed on the short name extracted from an approval prefix),
# so they carry no literal. ``osprey_writes_check.py`` carries NO Bluesky
# literal at all: its write-tool set is data-driven from ``hook_config.json``
# (rendered from the registry HookRule), so the arming tools' kill-switch leg is
# rename-safe without touching that source. The one Bluesky literal on the
# hooks' side is ``FALLBACK_WRITE_TOOLS`` in ``osprey_hook_log.py`` — the
# degraded floor refused when ``hook_config.json`` cannot be read at all, which
# is by definition the one set that cannot be data-driven — and the floor is
# kept fresh by ``tests/registry/test_mixed_floor_driftguard.py``, which pins it
# against ``registry.mcp.framework_write_tools()``. Those are the load-bearing
# facts this section pins.
# ---------------------------------------------------------------------------


def test_approval_template_source_carries_the_queue_control_constants() -> None:
    """The standalone approval hook still literals every queue-control name.

    ``osprey_approval.py`` dispatches on these short names (and lists them in
    its frontmatter) to fetch and render the draft, the queue contents, the
    plan's provenance/validation/source, and whether the queue is already
    draining — the documented backstop for a plan body that slips past the
    automated validator, and the only place a human learns that an enqueue is
    really an execution. A rename of any of these constants that leaves this
    deployed source untouched silently detaches that enrichment; pin the values
    so the rename must update it too.
    """
    src = _hook_source("osprey_approval.py")
    missing = [tool for tool in bsky.QUEUE_CONTROL_TOOLS if tool not in src]
    assert not missing, (
        f"osprey_approval.py no longer contains {missing} — the standalone "
        f"approval hook special-cases these names to render queue/plan detail; "
        f"a rename that skips this template detaches the enrichment"
    )


def test_hook_headers_list_exactly_the_tools_the_registry_attaches() -> None:
    """Every safety hook's frontmatter ``tools:`` line equals its registry matchers.

    The header is what an operator reads to learn what a hook gates, and the
    registry's ``HookRule``s are what actually attach it — so the two are
    pinned to each other, as sets, across every ``FRAMEWORK_SERVERS`` entry.
    A tool gated in the registry and missing from the header under-reports the
    hook; a tool named in the header and gated nowhere over-reports it. This
    is also what lets a header carry a Bluesky tool name at all: a rename that
    updates the constant moves the registry's matcher, and this guard then
    names the header that still spells the old name.

    A hook is identified by the script it runs, not by the ``HookEntry`` it was
    wired from: one script can be wired through several entries (the approval
    hook carries a wider pre-flight budget on the arming pair), and every one
    of them attaches the same header to the tool it gates.
    """
    for key, entry in HOOK_PRESETS.items():
        filename = _script_of(entry.command)
        attached = {
            _short_name(rule.matcher)
            for template in FRAMEWORK_SERVERS.values()
            for rule in template.hooks_pre
            if any(_script_of(hook.command) == filename for hook in rule.hooks)
        }
        declared = _hook_header_tools(filename)
        assert declared == attached, (
            f"{filename} header tools: {sorted(declared)} != registry matchers for the "
            f"{key!r} hook {sorted(attached)} — under-reported: "
            f"{sorted(attached - declared)}, over-reported: {sorted(declared - attached)}"
        )


def test_writes_check_template_carries_no_bluesky_tool_literal() -> None:
    """The kill switch stays data-driven — no Bluesky tool name is hardcoded.

    Both of the hook's tool-keyed decisions arrive as rendered data, never as a
    literal in its code: which tools it gates at all flows registry HookRule →
    ``hook_config.json`` → its runtime ``write_tools`` load, and which of them
    skip the per-target stage because a plan lane addresses them flows
    ``QUEUE_CONTROL_TOOLS`` → ``hook_config.json`` → its ``lane_addressed_tools``
    load. Pinning the ABSENCE flags anyone who reintroduces a literal that
    would then silently drift on the next rename. The frontmatter header is
    the one place the names may appear: it is pinned against the registry by
    ``test_hook_headers_list_exactly_the_tools_the_registry_attaches``, so a
    rename that leaves it stale fails there rather than drifting.
    """
    _, body = _hook_docstring_and_body("osprey_writes_check.py")
    present = [t for t in bsky.ALL_TOOLS if t in body]
    assert present == [], (
        f"osprey_writes_check.py hardcodes Bluesky tool name(s) {present} — the "
        f"writes kill switch must stay data-driven (write_tools loaded from "
        f"hook_config.json rendered off the registry); keep the gate in the "
        f"registry HookRule, not this standalone hook source"
    )


def test_hooks_write_floor_refuses_the_arming_pair_when_the_render_is_unreadable() -> None:
    """The one literal that must be there.

    A degraded render (missing/unreadable/malformed ``hook_config.json``) is
    exactly when a deployment that enabled Bluesky has nothing left to refuse
    ``queue_add``/``queue_start`` — arming and starting a plan queue, both
    control-system writes. The floor lives in the hooks' shared
    ``osprey_hook_log.py`` (both ``osprey_writes_check`` and ``osprey_approval``
    read it through ``write_tools()``) and is kept in step with the registry by
    ``tests/registry/test_mixed_floor_driftguard.py``, so this literal cannot
    go stale on a rename the way an ungated one would.
    """
    src = _hook_source("osprey_hook_log.py")
    tree = ast.parse(src)
    floor = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "FALLBACK_WRITE_TOOLS" for t in n.targets)
    )
    for tool in (bsky.QUEUE_ADD, bsky.QUEUE_START):
        assert f"mcp__{bsky.SERVER_NAME}__{tool}" in floor


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


def _hook_commands(by_matcher: dict, tool: str) -> list[str]:
    return [h["command"] for h in by_matcher[bsky.matcher(tool)]["hooks"]]


def test_arming_tools_membership_is_pinned_literally() -> None:
    """``ARMING_TOOLS`` names exactly the two tools that arm hardware motion.

    Every other assertion about the kill switch iterates this tuple, so they
    all agree with it by construction and none of them can notice a tool
    LEAVING it — drop ``queue_add`` here and the writes-check simply stops
    being required for it, silently. This literal is the one place that
    breaks that circularity.

    A tool removed from this tuple loses the kill switch. Only remove one if
    it genuinely can no longer put hardware in motion.
    """
    assert set(bsky.ARMING_TOOLS) == {"queue_add", "queue_start"}


def test_kill_switch_gates_exactly_the_arming_tools() -> None:
    """``_WRITES_CHECK`` is attached to every ARMING_TOOL and to nothing else.

    Both halves matter and neither implies the other. An arming tool WITHOUT
    the writes-check registers with an approval prompt and no kill switch —
    it looks gated and is not, which is precisely how a new write path slips
    past the deny loop. A non-arming tool WITH it hands the kill switch veto
    power over the safe direction.
    """
    by_matcher = {r["matcher"]: r for r in _resolve_bluesky()["hooks_pre"]}

    for tool in bsky.ARMING_TOOLS:
        cmds = _hook_commands(by_matcher, tool)
        assert any("osprey_writes_check.py" in c for c in cmds), (
            f"{tool!r} arms hardware motion and MUST carry the writes-check kill switch"
        )
        assert any("osprey_approval.py" in c for c in cmds), f"{tool!r} must be approval-gated"

    kill_switched = {
        rule["matcher"]
        for rule in _resolve_bluesky()["hooks_pre"]
        if any("osprey_writes_check.py" in h["command"] for h in rule["hooks"])
    }
    assert kill_switched == {bsky.matcher(t) for t in bsky.ARMING_TOOLS}, (
        "the writes-check kill switch must gate exactly bsky.ARMING_TOOLS"
    )


def test_stop_tools_are_approval_only_never_kill_switched() -> None:
    """``queue_stop``, ``queue_remove`` and ``stop_run`` = approval only, in both directions.

    Halting is the safe direction, so the kill switch must never be able to
    block it: attaching the writes-check to ``queue_stop`` would make a plain
    stop fail exactly when writes are disabled — the moment an operator is most
    likely to want the queue halted. ``queue_stop``'s one arming case
    (``cancel=true``, which withdraws a pending halt) is gated in-tool and again
    at the bridge instead, so the arming half is covered without taking the
    halting half hostage. ``queue_remove`` discards pending work and is the
    sole way past the interrupted-item start refusal — a kill switch that
    blocked it would trap a wedged queue exactly when writes are disabled.
    """
    by_matcher = {r["matcher"]: r for r in _resolve_bluesky()["hooks_pre"]}

    for tool in (bsky.QUEUE_STOP, bsky.QUEUE_REMOVE, bsky.STOP_RUN):
        cmds = _hook_commands(by_matcher, tool)
        assert any("osprey_approval.py" in c for c in cmds), f"{tool!r} must be approval-gated"
        assert not any("osprey_writes_check.py" in c for c in cmds), (
            f"{tool!r} must NEVER be writes-check/kill-switch gated — the kill switch "
            f"must not be able to block halting"
        )


def test_queue_read_tools_are_silent_allow() -> None:
    """``queue_list`` / ``queue_status`` are reads: allow, not ask, no pre-hook.

    ``queue_status`` in particular is the question an agent should ask BEFORE
    composing anything; prompting for it would train operators to click through
    prompts that never precede motion.
    """
    bluesky = _resolve_bluesky()
    gated = {r["matcher"] for r in bluesky["hooks_pre"]}

    for tool in bsky.QUEUE_READ_TOOLS:
        assert tool in bluesky["permissions_allow"], f"{tool!r} (read) must be silent-allow"
        assert tool not in bluesky["permissions_ask"]
        assert bsky.matcher(tool) not in gated, f"{tool!r} (read) must carry no pre-hook"


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


# ---------------------------------------------------------------------------
# Approval budget: the preview deadline and the harness timeout are one number
# ---------------------------------------------------------------------------

#: Every approval hook the framework wires, as (server, event, matcher, timeout).
#:
#: A snapshot rather than a derivation: the claim is that ONLY the Bluesky
#: arming pair buys the wider pre-flight budget, and an expectation derived
#: from the registry would move with the code it exists to pin. A new
#: approval-gated tool adds a row here deliberately; a timeout that drifts
#: onto another prompt fails here.
_APPROVAL_RULE_TABLE: frozenset[tuple[str, str, str, int]] = frozenset(
    {
        ("controls", "pre", "mcp__controls__channel_write", 5),
        ("controls", "pre", "mcp__controls__control_target_set", 5),
        ("controls", "pre", "mcp__controls__channel_read", 5),
        ("controls", "pre", "mcp__controls__archiver_read", 5),
        ("phoebus", "pre", "mcp__phoebus__phoebus_drive", 5),
        ("python", "pre", "mcp__python__execute", 5),
        ("python", "pre", "mcp__python__execute_file", 5),
        ("osprey_workspace", "pre", "mcp__osprey_workspace__setup_patch", 5),
        ("osprey_workspace", "pre", "mcp__osprey_workspace__add_panel_to_rail", 5),
        ("osprey_workspace", "pre", "mcp__osprey_workspace__remove_panel_from_rail", 5),
        ("osprey_workspace", "pre", "mcp__osprey_workspace__register_panel", 5),
        ("ariel", "pre", "mcp__ariel__entry_create", 5),
        ("ariel", "pre", "mcp__ariel__entry_publish", 5),
        ("osprey_facility_knowledge", "pre", "mcp__osprey_facility_knowledge__draft_concept", 5),
        ("bluesky", "pre", "mcp__bluesky__queue_add", 30),
        ("bluesky", "pre", "mcp__bluesky__queue_start", 30),
        ("bluesky", "pre", "mcp__bluesky__queue_stop", 5),
        ("bluesky", "pre", "mcp__bluesky__queue_remove", 5),
        ("bluesky", "pre", "mcp__bluesky__stop_run", 5),
        ("bluesky", "pre", "mcp__bluesky__write_plan", 5),
        ("bluesky", "pre", "mcp__bluesky__validate_plan", 5),
        ("event_dispatcher", "pre", "mcp__event_dispatcher__manual_fire", 5),
    }
)


def _approval_hooks() -> list[tuple[str, str, str, int, str]]:
    """Every wired approval hook as (server, event, matcher, timeout, command)."""
    found: list[tuple[str, str, str, int, str]] = []
    for name, definition in FRAMEWORK_SERVERS.items():
        for event, rules in (("pre", definition.hooks_pre), ("post", definition.hooks_post)):
            for rule in rules:
                for hook in rule.hooks:
                    if "osprey_approval.py" in hook.command:
                        found.append((name, event, rule.matcher, hook.timeout, hook.command))
    return found


def test_approval_rule_table_is_pinned() -> None:
    """The approval surface and its timeouts are exactly the pinned table.

    The wider budget is a safety-relevant asymmetry: a hook the harness kills
    mid-render answers without the preview the operator was promised, while a
    prompt that keeps a long timeout for no preview widens the window in which
    a hung hook stalls a tool call. Pinning the whole table — not only the two
    wide rows — is what makes either drift fail here rather than in a control
    room.
    """
    assert {(s, e, m, t) for s, e, m, t, _ in _approval_hooks()} == _APPROVAL_RULE_TABLE


def test_approval_budget_flag_equals_its_timeout() -> None:
    """Each approval entry passes the hook the number the harness kills it at.

    The hook plans its pre-flight preview against ``--budget``; the harness
    enforces ``timeout``. Two numbers free to disagree are a preview budget the
    hook can never spend, so both come from one argument to ``_approval_entry``.
    """
    for server, _event, matcher, timeout, command in _approval_hooks():
        assert f"--budget {timeout}" in command, (
            f"{server}:{matcher} passes the approval hook a budget that does not "
            f"match its {timeout}s timeout: {command!r}"
        )


def test_only_the_bluesky_arming_pair_carries_the_preflight_budget() -> None:
    """``queue_add`` and ``queue_start`` get 30 s; every other prompt keeps 5 s.

    Those two are the prompts that render a queue preview over the bridge
    before an operator can answer — the arming pair, so the preview is what the
    approval is FOR. Every other approval prompt answers from data the hook
    already holds and needs no budget beyond the default.
    """
    wide = {matcher for _s, _e, matcher, timeout, _c in _approval_hooks() if timeout != 5}
    assert wide == {bsky.matcher(tool) for tool in bsky.ARMING_TOOLS}

    for _s, _e, matcher, timeout, command in _approval_hooks():
        if matcher in wide:
            assert timeout == 30, f"{matcher} pre-flight budget must be 30s"
            assert "--budget 30" in command, f"{matcher} must hand the hook a 30s budget"


# ---------------------------------------------------------------------------
# Event dispatcher gate wiring: an approval prefix with no env block behind it
# ---------------------------------------------------------------------------

#: A render of a deployment that declares the EVENTS panel, which is what puts
#: the dispatcher entry in the agent's ``.mcp.json`` at all.
_DISPATCHER_CTX = {
    "project_root": "/tmp/test-project",
    "current_python_env": "/usr/bin/python3",
    "event_dispatcher_wired": True,
}


def _render_hook_config(ctx: dict) -> dict:
    """Render ``hook_config.json.j2`` against *ctx* and return the parsed contract.

    The rendered file — not the registry it is rendered from — is what the
    standalone hooks read at run time, so the assertions below go through it.
    """
    from osprey.cli.templates.manager import TemplateManager

    ctx = dict(ctx)
    ctx["servers"] = resolve_servers({}, ctx)
    ctx.setdefault("control_system_write_tools", [])
    tm = TemplateManager()
    template = tm.jinja_env.get_template("claude_code/claude/hooks/hook_config.json.j2")
    return json.loads(template.render(**ctx))


def test_event_dispatcher_approval_prefix_derives_from_its_matcher_alone() -> None:
    """The dispatcher's approval prefix survives an entry that renders no env.

    Every other approval-gated server is a stdio launch whose ``.mcp.json``
    block carries ``env``; this one is a URL entry, rendered as
    ``{type, url, headers}`` and nothing else. The approval hook decides
    whether a call is gated at all by matching it against
    ``approval_prefixes``, and that list is built from ``hooks_pre`` matchers —
    so the prefix has to arrive from the matcher, with no env block for
    anything to be read out of.
    """
    config = _render_hook_config(_DISPATCHER_CTX)

    assert "mcp__event_dispatcher__" in config["approval_prefixes"]

    dispatcher = [
        s for s in resolve_servers({}, _DISPATCHER_CTX) if s["name"] == "event_dispatcher"
    ][0]
    assert dispatcher["url"], (
        "the dispatcher entry must stay a URL server — the claim this test pins "
        "is that its approval prefix needs no env block"
    )


def test_manual_fire_is_approval_gated_and_never_kill_switched() -> None:
    """Firing a job prompts, but the writes kill switch has no say over it.

    ``manual_fire`` starts a job; it does not itself touch the control system.
    The writes the job then makes are what meet the operator's chip, at the
    tools that make them. Putting the firing call in ``write_tools`` would
    instead leave a writes-off session unable to start even a read-only job,
    and would claim a gate that is enforced a layer further in.
    """
    config = _render_hook_config(_DISPATCHER_CTX)

    assert "manual_fire" not in config["write_tools"]
    assert not any("event_dispatcher" in tool for tool in config["write_tools"])


def test_event_dispatcher_absent_from_the_contract_without_the_events_panel() -> None:
    """No EVENTS panel → no dispatcher prefix anywhere in the rendered contract.

    The entry is conditioned on the panel declaration, because the panel is
    what puts the proxy hop — and the panel token that authenticates to it —
    in front of the agent. A deployment without it gets neither the server nor
    its gate.
    """
    ctx = {k: v for k, v in _DISPATCHER_CTX.items() if k != "event_dispatcher_wired"}
    config = _render_hook_config(ctx)

    prefixes = config["server_prefixes"] + config["approval_prefixes"]
    assert not any(p.startswith("mcp__event_dispatcher__") for p in prefixes)
