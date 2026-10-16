"""A persona's denied tool is gone from its agent, reached by a scripted call.

Two renders differ by one ``permissions.deny`` entry. In persona A the tool is
not offered and a call that names it anyway is rejected as unknown; in persona
B the same scripted call succeeds. The model is scripted
(:mod:`tests.e2e.scripted_model`), so the call is made on every run; the Claude
Code permission engine and the MCP server it guards are real.
"""

from __future__ import annotations

import pytest

from tests.e2e.scripted_model import ScriptedCall
from tests.e2e.sdk_helpers import run_sdk_query_with_hooks
from tests.e2e.web_terminals.persona_projects import DENIED_TOOL_ENTRY

pytestmark = [pytest.mark.model_free, pytest.mark.harness_benchmark]

_WORKSPACE_PREFIX = "mcp__osprey_workspace__"


async def _call_the_swing_tool(repo, scripted_model):
    model = scripted_model([[ScriptedCall(DENIED_TOOL_ENTRY, {})]])
    result = await run_sdk_query_with_hooks(
        repo, "Carry out the scripted steps.", max_turns=3, max_budget_usd=0.50
    )
    assert result.result is not None, "No ResultMessage received from SDK"
    model.assert_script_consumed()
    assert result.mcp_server_status.get("osprey_workspace") == "connected", (
        f"the osprey_workspace server did not connect: {result.mcp_server_status!r}"
    )
    (trace,) = result.tool_traces
    return model, result, trace


async def test_denied_tool_is_removed_in_persona_a(persona_a_denied_repo, scripted_model):
    """Persona A's agent is not offered the denied tool and cannot call it.

    Other tools of the same server are checked as offered, so the absence is
    the deny entry's doing and not a server that failed to start.
    """
    model, result, trace = await _call_the_swing_tool(persona_a_denied_repo, scripted_model)
    assert any(name.startswith(_WORKSPACE_PREFIX) for name in model.offered_tools)
    assert DENIED_TOOL_ENTRY not in model.offered_tools
    assert trace.is_error and "No such tool available" in (trace.result or ""), (
        f"a forced {DENIED_TOOL_ENTRY} was not rejected as an unknown tool: {trace.result}"
    )
    events = [e for e in result.hook_events if e.tool_name == DENIED_TOOL_ENTRY]
    assert events == [], "an approval prompt was surfaced for a denied tool"


async def test_the_same_call_succeeds_in_persona_b(persona_b_permitted_repo, scripted_model):
    """Positive control: without the deny entry the tool is offered and the call works."""
    model, _, trace = await _call_the_swing_tool(persona_b_permitted_repo, scripted_model)
    assert DENIED_TOOL_ENTRY in model.offered_tools
    assert not trace.is_error, f"{DENIED_TOOL_ENTRY} failed in persona B: {trace.result}"
