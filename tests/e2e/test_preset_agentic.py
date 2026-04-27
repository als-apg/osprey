"""Tier 3 agentic per-preset smoke tests.

One canonical workflow per preset. Each test:
  1. Builds a project from the preset (with --skip-deps for speed; Tier 1
     covers the real-build path).
  2. Asks an agent (via the Claude Agent SDK) to perform the canonical task.
  3. Asserts the right MCP tool appears in the tool trace (the contract that
     would catch a "preset config drift means the agent never reaches for
     the documented tool" failure).
  4. Optionally has the LLM judge evaluate the response.

Budget per test: max_turns=4, max_budget_usd=0.25, target wall-clock <5 min.
Marked advisory in CI (continue-on-error) — coverage of the "is the tool
wired up at all?" contract comes from Tier 0+1.
"""

from __future__ import annotations

import json
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
    pytest.mark.requires_cborg,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
    pytest.mark.skipif(not is_claude_code_available(), reason="claude CLI not available"),
]


def _to_workflow_result(query: str, sdk_result: SDKWorkflowResult) -> WorkflowResult:
    """Convert ``SDKWorkflowResult`` (rich tool traces) to the plain-text shape
    the LLM judge expects."""
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


def _channel_finder_server_name(project_dir: Path) -> str | None:
    """Return the channel-finder backend module short name (or None if absent)."""
    cfg = json.loads((project_dir / ".mcp.json").read_text(encoding="utf-8"))
    for name, entry in cfg.get("mcpServers", {}).items():
        if "channel-finder" in name or "channel_finder" in name:
            args = entry.get("args") or []
            for arg in args:
                if isinstance(arg, str) and "channel_finder_" in arg:
                    return name
            return name
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hello_world_canonical_flow(tmp_path: Path, llm_judge: LLMJudge) -> None:
    """Hello-world preset: agent reads the example mock channel.

    Asserts the agent reached for ``mcp__controls__channel_read``. If the
    preset wiring breaks, this fires within 5 min instead of the nested
    Claude Code session hanging on missing tools.
    """
    project = init_project(
        tmp_path, "hello_demo", template="hello_world", provider="cborg"
    )
    query = "Read the example channel and tell me its current value."
    result = await run_sdk_query(project, query, max_turns=4, max_budget_usd=0.25)

    assert any(
        t.name == "mcp__controls__channel_read" for t in result.tool_traces
    ), (
        f"agent did not call mcp__controls__channel_read. "
        f"Tools called: {result.tool_names}"
    )

    eval = await llm_judge.evaluate(
        _to_workflow_result(query, result),
        expectations=(
            "The agent reads a control-system channel using the controls MCP "
            "server and reports a numeric value back to the user. The response "
            "should not contain unhandled errors."
        ),
    )
    assert eval.passed, eval.reasoning


@pytest.mark.asyncio
async def test_control_assistant_channel_finder_flow(
    tmp_path: Path, llm_judge: LLMJudge
) -> None:
    """Control-assistant preset: agent uses the channel-finder pipeline.

    Soft assertion (the channel-finder backend differs by preset config) —
    we only require that *some* ``mcp__channel-finder__*`` tool was used.
    """
    project = init_project(
        tmp_path, "ca_demo", template="control_assistant", provider="cborg"
    )
    cf_server = _channel_finder_server_name(project)
    if cf_server is None:
        pytest.skip("control-assistant preset has no channel-finder server")

    query = "Help me find the address of a beam-position-monitor channel for sector 5."
    result = await run_sdk_query(project, query, max_turns=4, max_budget_usd=0.25)

    cf_tool_calls = [t for t in result.tool_traces if t.name.startswith("mcp__channel-finder__")]
    assert cf_tool_calls, (
        f"agent did not call any mcp__channel-finder__* tool. "
        f"Tools called: {result.tool_names}"
    )

    eval = await llm_judge.evaluate(
        _to_workflow_result(query, result),
        expectations=(
            "The agent uses the channel-finder MCP server to search for or "
            "narrow down beam-position-monitor channels. It should engage "
            "with the channel-finder pipeline and report progress back to "
            "the user — even if no exact match is found."
        ),
    )
    assert eval.passed, eval.reasoning
