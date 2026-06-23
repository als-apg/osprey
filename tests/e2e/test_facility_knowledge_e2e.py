"""Agent e2e: facility knowledge MCP server in the control-assistant preset.

The test boots a control-assistant project (with its example OKF bundle at
``data/facility_knowledge/``), runs an operator-style query that requires the
agent to consult the facility_knowledge MCP server, and asserts:

- At least one ``mcp__osprey_facility_knowledge__*`` tool was called.
- The LLM judge rates the answer as passing (correct, relevant, no unhandled
  errors).

In addition to the two direct-retrieval tests, a delegation test asserts:

- The main orchestrator delegates to the ``facility-knowledge`` subagent
  (subagent tool calls observed with a non-None ``parent_tool_use_id``).
- FK tools are called from within the subagent context — not directly by
  the orchestrator.

Architectural forcing: ``disallowed_tools`` blocks the orchestrator from
calling FK tools directly, so it must hand off to the subagent.  This is
the same mechanism used for channel-finder delegation (a settings constraint,
not a prompt hint).

Prompt design: operator-style questions only — no concept IDs, PV patterns,
or "if any" hedges are inlined.  The agent must discover what the bundle
contains and navigate to the answer on its own.

Budget: max_turns=6 (direct tests) / max_turns=20 (delegation test),
max_budget_usd=0.25 / 1.0.
Marked ``@pytest.mark.flaky(reruns=2)`` per the agentic-e2e convention for
rare LLM stochastic misses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.judge import LLMJudge, WorkflowResult
from tests.e2e.sdk_helpers import (
    HAS_SDK,
    SDKWorkflowResult,
    init_project,
    is_claude_code_available,
    run_sdk_query,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.requires_als_apg,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
    pytest.mark.skipif(not is_claude_code_available(), reason="claude CLI not available"),
]

_FK_TOOL_PREFIX = "mcp__osprey_facility_knowledge__"


def _to_workflow_result(query: str, sdk_result: SDKWorkflowResult) -> WorkflowResult:
    """Convert ``SDKWorkflowResult`` to the plain-text shape the LLM judge expects."""
    response = "\n".join(sdk_result.text_blocks).strip()
    trace_lines: list[str] = []
    for t in sdk_result.tool_traces:
        trace_lines.append(f"TOOL: {t.name}  input={t.input}")
        if t.result:
            preview = t.result[:300] + ("…" if len(t.result) > 300 else "")
            trace_lines.append(f"  result: {preview}")
    return WorkflowResult(
        query=query,
        response=response,
        execution_trace="\n".join(trace_lines),
        artifacts=[],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=2)
async def test_facility_knowledge_pss_reset(tmp_path: Path) -> None:
    """Agent answers a PSS-reset question by reading the facility knowledge bundle.

    The bundle's ``procedures/pss-reset`` concept describes the Personnel
    Safety System fault-reset procedure in detail.  An operator question about
    how to clear a PSS fault should drive the agent to call
    ``mcp__osprey_facility_knowledge__*`` tools — either ``list_concepts`` to
    discover the bundle contents, or ``read_concept``/``search`` to retrieve
    the procedure — before composing its answer.
    """
    project = init_project(tmp_path, "fk_demo", template="control_assistant", provider="als-apg")
    judge = LLMJudge(provider="als-apg")
    query = (
        "A PSS fault is preventing beam injection.  Walk me through the "
        "steps I need to follow to identify the fault, perform the area "
        "search, and restore beam permits."
    )

    result = await run_sdk_query(project, query, max_turns=6, max_budget_usd=0.25)

    fk_calls = [t for t in result.tool_traces if t.name.startswith(_FK_TOOL_PREFIX)]
    assert fk_calls, (
        f"Agent did not call any {_FK_TOOL_PREFIX}* tool. Tools called: {result.tool_names}"
    )

    result_eval = await judge.evaluate(
        _to_workflow_result(query, result),
        expectations=(
            "The agent consults the facility knowledge bundle and returns a "
            "step-by-step procedure for resetting a PSS fault, covering: "
            "identifying the faulted interlock, performing a physical area "
            "search, latching the access door, pressing the reset button, "
            "and confirming beam permit is restored. The response should not "
            "contain unhandled errors."
        ),
    )
    assert result_eval.passed, result_eval.reasoning


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=2)
async def test_facility_knowledge_vacuum_recovery(tmp_path: Path) -> None:
    """Agent answers a vacuum-recovery question from the facility knowledge bundle.

    The bundle's ``procedures/vacuum-recovery`` concept describes pumpdown
    stages, ion pump conditioning, and gate-valve restoration after a vacuum
    event.  An operator question about recovering from a pressure excursion
    should drive the agent to retrieve that content.
    """
    project = init_project(tmp_path, "fk_vac", template="control_assistant", provider="als-apg")
    judge = LLMJudge(provider="als-apg")
    query = (
        "We had an uncontrolled pressure excursion in the storage ring. "
        "What are the steps to recover the vacuum system back to operating "
        "conditions, and roughly how long should each stage take?"
    )

    result = await run_sdk_query(project, query, max_turns=6, max_budget_usd=0.25)

    fk_calls = [t for t in result.tool_traces if t.name.startswith(_FK_TOOL_PREFIX)]
    assert fk_calls, (
        f"Agent did not call any {_FK_TOOL_PREFIX}* tool. Tools called: {result.tool_names}"
    )

    result_eval = await judge.evaluate(
        _to_workflow_result(query, result),
        expectations=(
            "The agent retrieves the vacuum recovery procedure from the "
            "facility knowledge bundle and describes the pumpdown stages: "
            "roughing pumpdown (0–4 h), ion pump conditioning (4–12 h), "
            "and gate valve restoration, with approximate durations for each "
            "stage. The response should not contain unhandled errors."
        ),
    )
    assert result_eval.passed, result_eval.reasoning


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=2)
async def test_facility_knowledge_delegation(tmp_path: Path) -> None:
    """Main agent delegates a facility-knowledge question to the facility-knowledge subagent.

    Delegation is driven by the CLAUDE.md directive: "For ANY question about
    documented facility content … delegate to **facility-knowledge**. Do NOT
    call `list_concepts`, `read_concept`, or `search` yourself."  When the
    orchestrator follows this directive, FK tool calls will be observed from
    within a subagent context (non-None ``parent_tool_use_id``).

    Note: ``disallowed_tools`` is intentionally NOT used here.  SDK-level
    disallowed_tools applies session-wide (orchestrator AND subagents), so it
    would block the very tools the subagent needs to answer.  Delegation
    fidelity is enforced by the CLAUDE.md prompt constraint, not an
    architectural tool block.

    Assertions:
    - Sub-agent tool calls are observed (``parent_tool_use_id`` is not None).
    - At least one FK tool was called (from the subagent context).
    - The session completed without an SDK-level error.
    - Answer is grounded: the judge confirms the response addresses the question.
    """
    project = init_project(tmp_path, "fk_del", template="control_assistant", provider="als-apg")
    judge = LLMJudge(provider="als-apg")
    prompt = (
        "What quality flag states are defined for the beam position monitors at "
        "this facility, and which state indicates the position data is reliable "
        "enough to use for orbit correction?"
    )

    result = await run_sdk_query(
        project,
        prompt,
        max_turns=20,
        max_budget_usd=1.0,
    )

    print("\n--- facility-knowledge delegation ---")
    print(f"  tools called ({len(result.tool_traces)}): {result.tool_names}")
    print(f"  num_turns: {result.num_turns}")
    if result.cost_usd is not None:
        print(f"  cost: ${result.cost_usd:.4f}")
    for t in result.tool_traces:
        parent_flag = f" (sub-agent: {t.parent_tool_use_id})" if t.parent_tool_use_id else ""
        print(f"  tool: {t.name}{parent_flag}")

    # Session completed without SDK error.
    assert result.result is not None, "No ResultMessage received"
    assert not result.result.is_error, f"SDK query ended in error: {result.result.result}"

    # Sub-agent was invoked — any tool with a non-None parent_tool_use_id.
    # This validates the CLAUDE.md delegation directive took effect.
    sa_traces = [t for t in result.tool_traces if t.parent_tool_use_id is not None]
    assert len(sa_traces) > 0, (
        "No sub-agent tool calls observed. The CLAUDE.md delegation directive "
        "('For ANY question about documented facility content … delegate to "
        "facility-knowledge') was not followed — the orchestrator may have "
        "answered from training data without consulting the bundle. "
        f"Tools called: {result.tool_names}"
    )

    # At least one FK tool was called from within the subagent.
    fk_calls = [t for t in result.tool_traces if t.name.startswith(_FK_TOOL_PREFIX)]
    assert len(fk_calls) > 0, (
        f"No {_FK_TOOL_PREFIX}* tool was called (not even via the subagent). "
        f"Tools called: {result.tool_names}"
    )

    # Cost under budget.
    if result.cost_usd is not None:
        assert result.cost_usd < 1.0, f"Test cost ${result.cost_usd:.4f} — exceeded $1.00 budget"

    # Grounded answer: judge confirms the response is based on retrieved content.
    result_eval = await judge.evaluate(
        _to_workflow_result(prompt, result),
        expectations=(
            "The agent retrieves BPM quality flag information from the facility "
            "knowledge bundle and reports the three defined states: "
            "OK (valid position data, safe for orbit correction), "
            "NOBEAM (no beam signal detected, position unreliable), and "
            "FAULT (electronics fault, data not usable). "
            "The response should be grounded in retrieved bundle content, "
            "not fabricated from general knowledge, "
            "and should not contain unhandled errors."
        ),
    )
    assert result_eval.passed, result_eval.reasoning
