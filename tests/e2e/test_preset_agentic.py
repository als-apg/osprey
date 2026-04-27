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
    run_sdk_query_with_hooks,
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


async def _assert_approval_hook_fires(
    project: Path,
    query: str,
    expected_write_tool: str,
    *,
    max_turns: int = 6,
    max_budget_usd: float = 0.30,
) -> None:
    """Run a write-style query and verify the approval hook recorded an event.

    Uses ``run_sdk_query_with_hooks`` with ``permission_mode="default"`` and
    ``approval_policy="auto_approve"`` so hooks fire (vs. ``run_sdk_query``
    which uses ``bypassPermissions`` and silently elides them). Asserts:

    - at least one ``HookEvent`` was recorded (otherwise hooks are absent)
    - one of those events corresponds to ``expected_write_tool``
    - decision was ``"allow"`` under the auto-approve policy
    """
    result = await run_sdk_query_with_hooks(
        project,
        query,
        approval_policy="auto_approve",
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
    )
    assert result.hook_events, (
        "no hook events recorded — approval hook did not fire on a write query. "
        "Either the agent never reached for a write tool, or hooks are not "
        "wired into the project's .claude/settings.json."
    )
    matching = [e for e in result.hook_events if e.tool_name == expected_write_tool]
    assert matching, (
        f"approval hook fired but not for {expected_write_tool}. "
        f"Recorded: {[(e.tool_name, e.decision) for e in result.hook_events]}"
    )
    assert all(e.decision == "allow" for e in matching), (
        f"auto_approve policy should allow all writes, got: "
        f"{[(e.tool_name, e.decision, e.reason) for e in matching]}"
    )


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


@pytest.mark.asyncio
async def test_hello_world_write_triggers_approval_hook(tmp_path: Path) -> None:
    """Hello-world write probe: agent sets a mock channel; approval hook fires."""
    project = init_project(
        tmp_path, "hw_write", template="hello_world", provider="cborg"
    )
    await _assert_approval_hook_fires(
        project,
        "Set the example channel to 1.5.",
        expected_write_tool="mcp__controls__channel_write",
    )


@pytest.mark.asyncio
async def test_control_assistant_write_triggers_approval_hook(tmp_path: Path) -> None:
    """Control-assistant write probe: same contract on the heavier preset."""
    project = init_project(
        tmp_path, "ca_write", template="control_assistant", provider="cborg"
    )
    await _assert_approval_hook_fires(
        project,
        "Set the BPM offset channel for sector 5 to 0.0.",
        expected_write_tool="mcp__controls__channel_write",
    )
