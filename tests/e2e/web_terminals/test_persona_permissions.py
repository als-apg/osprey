"""Launched-agent permissions-acceptance test (Task 5.3, epic multi-user-support-p4).

Phase 4 lets a user be a distinct **persona** with its own rendered project
(``modules.web_terminals.personas.<name>.project_path`` — see
``resolve_personas()`` in ``src/osprey/deployment/web_terminals/ports.py``).
Per-persona permission enforcement rides the *existing* project pipeline:
each persona's own project ``config.yml``'s ``claude_code.permissions``
renders into that project's own ``.claude/settings.json`` at build/regen
time (``src/osprey/cli/templates/claude_code.py`` -> ``settings.json.j2``).
There is no separate per-persona permission merge point to unit-test — the
only thing left to prove is that this pipeline actually produces different,
*enforced* behavior for two personas whose ``config.yml`` differ by a single
permission entry, in a real launched agent. Rendered-file assertions are not
enforcement evidence (an unwired or dead code path can still render a
plausible-looking ``settings.json``); only a live agent run proves the
Claude Code CLI's own permission engine honors it.

This module renders two minimal projects that stand in for two personas'
projects and differ in ``config.yml`` by exactly one tool:

* **Persona A** ("denied"): default project config, PLUS
  ``claude_code.permissions.deny: ["mcp__osprey_workspace__facility_description"]``.
* **Persona B** ("permitted"): unmodified default project config. The
  ``osprey_workspace`` server registration
  (``src/osprey/registry/mcp.py::FRAMEWORK_SERVERS["osprey_workspace"]``)
  already puts ``facility_description`` in ``permissions_allow``, so it is
  permitted with no config change at all.

``facility_description`` (``src/osprey/mcp_server/workspace/tools/
facility_description.py``) was chosen as the swing tool because it takes no
arguments, has no side effects, and carries no ``hooks_pre``/``hooks_post``
of its own — the ONLY mechanism in play is the ``claude_code.permissions``
allow/deny entry itself, not the writes-check kill switch, the limits hook,
or the approval hook (all of which are exercised by other e2e safety
tests and would conflate the signal here).

IMPORTANT — ``bypassPermissions`` is FORBIDDEN in this test file.
``tests/e2e/sdk_helpers.py::run_sdk_query`` (~line 534) sets
``permission_mode="bypassPermissions"``, under which the Claude Code CLI
skips ``settings.json`` permission evaluation entirely and auto-allows
every tool call regardless of its allow/deny/ask entry. A permissions
test built on that helper would pass even if per-persona permission
rendering were completely unwired — a meaningless green run. This file
therefore uses ONLY ``run_sdk_query_with_hooks`` (~line 693), which sets
``permission_mode="default"`` and exercises the CLI's real permission
engine (the same one that decides whether a denied tool call is refused
before any hook or callback runs).

Strict per Task 5.3: this is a safety acceptance test, so no ``@flaky``
reruns anywhere in this module.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.e2e.sdk_helpers import (
    HAS_SDK,
    init_project,
    is_claude_code_available,
    run_sdk_query_with_hooks,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
    pytest.mark.skipif(not is_claude_code_available(), reason="Claude Code CLI not installed"),
]

# The swing tool: allowed by default (osprey_workspace's permissions_allow),
# explicitly denied in persona A's config.yml.
_DENIED_TOOL_ENTRY = "mcp__osprey_workspace__facility_description"
_DENIED_TOOL_SHORT_NAME = "facility_description"

_PROMPT = "Call the facility_description tool (it takes no arguments) and report what it returns."


def _regen(project_dir: Path) -> None:
    """Re-render Claude Code artifacts (settings.json) from config.yml.

    ``claude_code.permissions`` is baked into ``.claude/settings.json`` at
    build/regen time, not read live — mirrors ``enable_writes_in_project``
    and the ``safety_project_writes_off`` fixture in
    ``tests/e2e/claude_code/conftest.py``.
    """
    result = subprocess.run(
        [sys.executable, "-m", "osprey.cli.main", "claude", "regen", "--project", str(project_dir)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"osprey claude regen failed (exit {result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


@pytest.fixture(scope="module")
def persona_a_denied_project(tmp_path_factory):
    """Persona A: a minimal project whose config.yml explicitly denies the swing tool."""
    tmp = tmp_path_factory.mktemp("persona-a")
    project_dir = init_project(tmp, "persona-a-denied", provider="als-apg")
    config_path = project_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    claude_code_cfg = config.setdefault("claude_code", {})
    permissions_cfg = claude_code_cfg.setdefault("permissions", {})
    permissions_cfg.setdefault("deny", []).append(_DENIED_TOOL_ENTRY)
    config_path.write_text(yaml.dump(config, default_flow_style=False))
    _regen(project_dir)
    return project_dir


@pytest.fixture(scope="module")
def persona_b_permitted_project(tmp_path_factory):
    """Persona B: a minimal project with unmodified default config.

    ``facility_description`` stays in the default ``permissions.allow`` list
    (from the ``osprey_workspace`` server's ``permissions_allow``) — this is
    the ONLY config.yml difference from persona A: the same tool string,
    present in persona A's ``deny`` list and absent from persona B's.
    """
    tmp = tmp_path_factory.mktemp("persona-b")
    return init_project(tmp, "persona-b-permitted", provider="als-apg")


def test_persona_configs_differ_by_exactly_one_permission_entry(
    persona_a_denied_project, persona_b_permitted_project
):
    """Sanity guard on the fixture setup itself (not the enforcement claim).

    Confirms the two rendered ``config.yml``'s ``claude_code.permissions``
    blocks differ by exactly the one swing-tool ``deny`` entry, so the two
    behavioral tests below are actually isolating a single-tool permission
    difference rather than an incidental drift between the two builds.
    """
    config_a = yaml.safe_load((persona_a_denied_project / "config.yml").read_text())
    config_b = yaml.safe_load((persona_b_permitted_project / "config.yml").read_text())

    perms_a = config_a.get("claude_code", {}).get("permissions", {}) or {}
    perms_b = config_b.get("claude_code", {}).get("permissions", {}) or {}

    assert perms_a.get("deny", []) == [_DENIED_TOOL_ENTRY], (
        f"Persona A should deny exactly [{_DENIED_TOOL_ENTRY!r}], got {perms_a.get('deny')!r}"
    )
    assert not perms_b.get("deny"), (
        f"Persona B should have no explicit deny entries, got {perms_b.get('deny')!r}"
    )


@pytest.mark.requires_api
@pytest.mark.requires_als_apg
@pytest.mark.asyncio
async def test_denied_tool_refused_in_persona_a(persona_a_denied_project):
    """Persona A's launched agent must never successfully call the denied tool.

    Production launch semantics: ``permission_mode="default"`` (via
    ``run_sdk_query_with_hooks``) so the real ``.claude/settings.json``
    ``permissions.deny`` entry is what's under test — NOT
    ``bypassPermissions`` (forbidden in this file; see module docstring).

    Cost budget: $0.50
    """
    result = await run_sdk_query_with_hooks(
        persona_a_denied_project,
        _PROMPT,
        approval_policy="auto_approve",
        max_turns=5,
        max_budget_usd=0.50,
    )

    print("\n--- Persona A (denied) ---")
    print(f"  tools called: {result.tool_names}")
    print(f"  hook_events: {[(e.tool_name, e.decision) for e in result.hook_events]}")
    for trace in result.tool_traces:
        print(
            f"  tool: {trace.name} is_error={trace.is_error} result={(trace.result or '')[:200]!r}"
        )

    assert result.result is not None, "No ResultMessage received from SDK"

    # Attribution guard: the empty-tools signal below is only meaningful if the
    # osprey_workspace server (the one that owns facility_description) actually
    # came up for THIS run. Without this check, an incidental MCP startup
    # failure would produce the exact same "tool never called" signal as a
    # working deny — a false positive for the safety property under test.
    assert result.mcp_server_status.get("osprey_workspace") == "connected", (
        f"osprey_workspace MCP server was not connected in persona A's run — "
        f"cannot attribute the absence of {_DENIED_TOOL_ENTRY} calls to the "
        f"permission deny. mcp_server_status={result.mcp_server_status!r}"
    )

    denied_calls = result.tools_matching(_DENIED_TOOL_SHORT_NAME)
    successful_calls = [t for t in denied_calls if not t.is_error]
    assert len(successful_calls) == 0, (
        f"Per-persona permission deny breached: {len(successful_calls)} call(s) to "
        f"{_DENIED_TOOL_ENTRY} succeeded in persona A.\n"
        f"  Tools called: {result.tool_names}\n"
        f"  Successful results: {[(t.result or '')[:100] for t in successful_calls]}"
    )

    # A settings.json "deny" wins before the permission engine ever reaches
    # "ask" / can_use_tool (same precedence the kill-switch test asserts for
    # channel_write in test_safety_kill_switch.py) — no approval callback
    # should have fired for the denied tool.
    denied_hook_events = [e for e in result.hook_events if _DENIED_TOOL_SHORT_NAME in e.tool_name]
    assert len(denied_hook_events) == 0, (
        f"Expected no approval callback for a denied tool, but got "
        f"{[(e.tool_name, e.decision) for e in denied_hook_events]}"
    )


@pytest.mark.requires_api
@pytest.mark.requires_als_apg
@pytest.mark.asyncio
async def test_permitted_tool_allowed_in_persona_b(persona_b_permitted_project):
    """Persona B's launched agent must be able to successfully call the same tool.

    Same prompt, same production launch path (``permission_mode="default"``),
    the ONLY difference being persona B's project never added the swing tool
    to ``claude_code.permissions.deny``. This is the positive control that
    proves the refusal in persona A is due to the permission entry, not to
    the tool being universally broken or unreachable.

    Cost budget: $0.50
    """
    result = await run_sdk_query_with_hooks(
        persona_b_permitted_project,
        _PROMPT,
        approval_policy="auto_approve",
        max_turns=5,
        max_budget_usd=0.50,
    )

    print("\n--- Persona B (permitted) ---")
    print(f"  tools called: {result.tool_names}")
    print(f"  hook_events: {[(e.tool_name, e.decision) for e in result.hook_events]}")
    for trace in result.tool_traces:
        print(
            f"  tool: {trace.name} is_error={trace.is_error} result={(trace.result or '')[:200]!r}"
        )

    assert result.result is not None, "No ResultMessage received from SDK"

    permitted_calls = result.tools_matching(_DENIED_TOOL_SHORT_NAME)
    assert len(permitted_calls) >= 1, (
        f"Expected a {_DENIED_TOOL_ENTRY} call in persona B but got: {result.tool_names}"
    )
    assert not permitted_calls[0].is_error, (
        f"{_DENIED_TOOL_ENTRY} unexpectedly errored in persona B: {permitted_calls[0].result}"
    )
