"""Launched-agent permissions-acceptance test (Task 5.3, epic multi-user-support-p4).

Phase 4 lets a user be a distinct **persona** with its own rendered project
(``modules.web_terminals.personas.<name>.project_path`` — see
``resolve_personas()`` in ``src/osprey/deployment/web_terminals/personas.py``).
Per-persona permission enforcement rides the *existing* render pipeline:
each persona project's own ``config.yml``'s ``claude_code.permissions``
renders into that project's own ``.claude/settings.json`` at build time
(``src/osprey/cli/templates/claude_code.py`` -> ``settings.json.j2``).
There is no separate per-persona permission merge point to unit-test — the
only thing left to prove is that this pipeline actually produces different,
*enforced* behavior for two personas whose ``config.yml`` differ by a single
permission entry, in a real launched agent. Rendered-file assertions are not
enforcement evidence (an unwired or dead code path can still render a
plausible-looking ``settings.json``); only a live agent run proves the
Claude Code CLI's own permission engine honors it.

This module builds two minimal deployment repos that stand in for two
personas' projects and differ in the render's ``config.yml`` by exactly one
tool:

* **Persona A** ("denied"): default rendered config, PLUS
  ``claude_code.permissions.deny: ["mcp__osprey_workspace__facility_description"]``.
* **Persona B** ("permitted"): unmodified default rendered config. The
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

from pathlib import Path

import pytest
import yaml

from tests.e2e.sdk_helpers import (
    HAS_SDK,
    init_project,
    is_claude_code_available,
    render_dir,
    run_sdk_query_with_hooks,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.harness_benchmark,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
    pytest.mark.skipif(not is_claude_code_available(), reason="Claude Code CLI not installed"),
]

# The swing tool: allowed by default (osprey_workspace's permissions_allow),
# explicitly denied in persona A's config.yml.
_DENIED_TOOL_ENTRY = "mcp__osprey_workspace__facility_description"
_DENIED_TOOL_SHORT_NAME = "facility_description"

_PROMPT = "Call the facility_description tool (it takes no arguments) and report what it returns."


def _deny_swing_tool(repo: Path) -> None:
    """Add the swing tool to the render's deny list and re-render from it.

    ``claude_code.permissions`` is baked into ``.claude/settings.json`` when the
    Claude Code artifacts are rendered, not read live, so flipping the config
    alone leaves ``settings.json`` stale. ``regen_if_drift`` re-renders them from
    the edited config — the same step ``osprey build`` performs, and the same one
    ``sdk_helpers.enable_writes_in_project`` uses for the kill switch.

    The edit lands in the RENDER (``<repo>/build/config.yml``), which is what the
    agent's ``.claude/`` tree is generated from and what ``CONFIG_FILE`` points
    the MCP servers at. Nothing here rebuilds the repo from ``profile.yml``: a
    full rebuild would also re-render the ARIEL DSN override
    ``init_project`` applies to the same file.
    """
    from osprey.cli.templates.manager import TemplateManager

    render = render_dir(repo)
    config_path = render / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    claude_code_cfg = config.setdefault("claude_code", {})
    permissions_cfg = claude_code_cfg.setdefault("permissions", {})
    permissions_cfg.setdefault("deny", []).append(_DENIED_TOOL_ENTRY)
    config_path.write_text(yaml.dump(config, default_flow_style=False))

    regenerated = TemplateManager().regen_if_drift(render)
    assert regenerated, (
        f"no Claude Code artifact was re-rendered after adding {_DENIED_TOOL_ENTRY} to "
        f"{config_path} — settings.json would still carry the pre-edit permissions, and "
        "the denial under test would never reach the CLI's permission engine"
    )


@pytest.fixture(scope="module")
def persona_a_denied_repo(tmp_path_factory):
    """Persona A: a minimal deployment repo whose render denies the swing tool."""
    tmp = tmp_path_factory.mktemp("persona-a")
    repo = init_project(tmp, "persona-a-denied", provider="als-apg")
    _deny_swing_tool(repo)
    return repo


@pytest.fixture(scope="module")
def persona_b_permitted_repo(tmp_path_factory):
    """Persona B: a minimal deployment repo with unmodified default config.

    ``facility_description`` stays in the default ``permissions.allow`` list
    (from the ``osprey_workspace`` server's ``permissions_allow``) — this is
    the ONLY config.yml difference from persona A: the same tool string,
    present in persona A's ``deny`` list and absent from persona B's.
    """
    tmp = tmp_path_factory.mktemp("persona-b")
    return init_project(tmp, "persona-b-permitted", provider="als-apg")


def test_persona_configs_differ_by_exactly_one_permission_entry(
    persona_a_denied_repo, persona_b_permitted_repo
):
    """Sanity guard on the fixture setup itself (not the enforcement claim).

    Confirms the two rendered ``config.yml``'s ``claude_code.permissions``
    blocks differ by exactly the one swing-tool ``deny`` entry, so the two
    behavioral tests below are actually isolating a single-tool permission
    difference rather than an incidental drift between the two builds.
    """
    config_a = yaml.safe_load((render_dir(persona_a_denied_repo) / "config.yml").read_text())
    config_b = yaml.safe_load((render_dir(persona_b_permitted_repo) / "config.yml").read_text())

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
async def test_denied_tool_refused_in_persona_a(persona_a_denied_repo):
    """Persona A's launched agent must never successfully call the denied tool.

    Production launch semantics: ``permission_mode="default"`` (via
    ``run_sdk_query_with_hooks``) so the real ``.claude/settings.json``
    ``permissions.deny`` entry is what's under test — NOT
    ``bypassPermissions`` (forbidden in this file; see module docstring).

    Cost budget: $0.50
    """
    result = await run_sdk_query_with_hooks(
        persona_a_denied_repo,
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
async def test_permitted_tool_allowed_in_persona_b(persona_b_permitted_repo):
    """Persona B's launched agent must be able to successfully call the same tool.

    Same prompt, same production launch path (``permission_mode="default"``),
    the ONLY difference being persona B's project never added the swing tool
    to ``claude_code.permissions.deny``. This is the positive control that
    proves the refusal in persona A is due to the permission entry, not to
    the tool being universally broken or unreachable.

    Cost budget: $0.50
    """
    result = await run_sdk_query_with_hooks(
        persona_b_permitted_repo,
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
