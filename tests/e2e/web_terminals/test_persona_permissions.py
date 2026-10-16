"""Persona permissions: the config diff and the permitted-tool positive control.

Phase 4 lets a user be a distinct **persona** with its own rendered project
(``modules.web_terminals.personas.<name>.project_path`` — see
``resolve_personas()`` in ``src/osprey/deployment/web_terminals/personas.py``).
Per-persona permission enforcement rides the *existing* render pipeline:
each persona project's own ``config.yml``'s ``claude_code.permissions``
renders into that project's own ``.claude/settings.json`` at build time
(``src/osprey/cli/templates/claude_code.py`` -> ``settings.json.j2``).
There is no separate per-persona permission merge point to unit-test.
Rendered-file assertions are not enforcement evidence (an unwired or dead
code path can still render a plausible-looking ``settings.json``); only a
launched agent shows the Claude Code CLI's own permission engine honoring it.

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

It holds two tests: a check that the two rendered configs differ by exactly
that one ``deny`` entry, and a live run showing persona B's agent can call
the tool. Persona A's refusal is enforced and asserted in
test_persona_permissions_scripted.py, where a scripted call reaches the
permission engine on every run.

``facility_description`` (``src/osprey/mcp_server/workspace/tools/
facility_description.py``) is the swing tool because it takes no
arguments, has no side effects, and carries no ``hooks_pre``/``hooks_post``
of its own — the ONLY mechanism in play is the ``claude_code.permissions``
allow/deny entry itself, not the writes-check kill switch, the limits hook,
or the approval hook (all of which are exercised by other e2e safety
tests and would conflate the signal here).

IMPORTANT — ``bypassPermissions`` is FORBIDDEN in this test file.
``tests/e2e/sdk_helpers.py::run_sdk_query`` sets
``permission_mode="bypassPermissions"``, under which the Claude Code CLI
skips ``settings.json`` permission evaluation entirely and auto-allows
every tool call regardless of its allow/deny/ask entry. A permissions
test built on that helper would pass even if per-persona permission
rendering were completely unwired — a meaningless green run. This file
therefore uses ONLY ``run_sdk_query_with_hooks``, which sets
``permission_mode="default"`` and exercises the CLI's real permission
engine.

This is a safety acceptance test, so no ``@flaky`` reruns anywhere in this
module.
"""

from __future__ import annotations

import pytest
import yaml

from tests.e2e.sdk_helpers import (
    HAS_SDK,
    is_claude_code_available,
    render_dir,
    run_sdk_query_with_hooks,
)
from tests.e2e.web_terminals.persona_projects import DENIED_TOOL_ENTRY, DENIED_TOOL_SHORT_NAME

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.harness_benchmark,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
    pytest.mark.skipif(not is_claude_code_available(), reason="Claude Code CLI not installed"),
]

_PROMPT = "Call the facility_description tool (it takes no arguments) and report what it returns."


def test_persona_configs_differ_by_exactly_one_permission_entry(
    persona_a_denied_repo, persona_b_permitted_repo
):
    """Sanity guard on the fixture setup itself (not the enforcement claim).

    Confirms the two rendered ``config.yml``'s ``claude_code.permissions``
    blocks differ by exactly the one swing-tool ``deny`` entry, so the permitted-tool
    test here and the scripted deny test isolate a single-tool permission
    difference rather than an incidental drift between the two builds.

    Stated as a difference rather than as persona A's whole ``deny`` list: a
    build also renders the tier's own deny floor, which both personas carry
    identically and neither test is about. Pinning the absolute list would
    make this fixture guard fail every time that floor gains a member, which
    says nothing about whether the two builds still differ by one tool.
    """
    config_a = yaml.safe_load((render_dir(persona_a_denied_repo) / "config.yml").read_text())
    config_b = yaml.safe_load((render_dir(persona_b_permitted_repo) / "config.yml").read_text())

    perms_a = config_a.get("claude_code", {}).get("permissions", {}) or {}
    perms_b = config_b.get("claude_code", {}).get("permissions", {}) or {}

    deny_a = set(perms_a.get("deny", []) or [])
    deny_b = set(perms_b.get("deny", []) or [])

    assert deny_a - deny_b == {DENIED_TOOL_ENTRY}, (
        f"Persona A should deny exactly [{DENIED_TOOL_ENTRY!r}] beyond what persona B "
        f"denies, got {sorted(deny_a - deny_b)!r} (A={sorted(deny_a)!r}, B={sorted(deny_b)!r})"
    )
    assert not deny_b - deny_a, (
        f"Persona B denies something persona A does not: {sorted(deny_b - deny_a)!r}"
    )


@pytest.mark.requires_api
@pytest.mark.requires_als_apg
@pytest.mark.asyncio
async def test_permitted_tool_allowed_in_persona_b(persona_b_permitted_repo):
    """Persona B's launched agent must be able to successfully call the same tool.

    Same prompt, same production launch path (``permission_mode="default"``),
    the ONLY difference being persona B's project never added the swing tool
    to ``claude_code.permissions.deny``. This is the positive control that
    proves that persona A's refusal is due to the permission entry, not to
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

    permitted_calls = result.tools_matching(DENIED_TOOL_SHORT_NAME)
    assert len(permitted_calls) >= 1, (
        f"Expected a {DENIED_TOOL_ENTRY} call in persona B but got: {result.tool_names}"
    )
    assert not permitted_calls[0].is_error, (
        f"{DENIED_TOOL_ENTRY} unexpectedly errored in persona B: {permitted_calls[0].result}"
    )
