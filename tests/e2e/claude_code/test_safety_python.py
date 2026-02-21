"""E2E safety tests for execute tool via Claude Code SDK.

Scenario 8: Safe python execution succeeds.

Scenarios 6-7 (caput/Tango write pattern detection) removed -- Claude correctly
refuses to run hardware writes through execute at the prompt level.
Write pattern detection is defense-in-depth tested in unit tests.

Uses run_sdk_query_with_hooks to exercise the full hook chain rather than
bypassPermissions mode.
"""

from __future__ import annotations

import pytest

from tests.e2e.sdk_helpers import run_sdk_query_with_hooks


def _combined_text(result) -> str:
    """Combine all text blocks and tool results into a single searchable string."""
    parts = list(result.text_blocks)
    for trace in result.tool_traces:
        if trace.result:
            parts.append(trace.result)
    return " ".join(parts).lower()


# ---------------------------------------------------------------------------
# Scenario 8: Safe Python execution
# ---------------------------------------------------------------------------


@pytest.mark.requires_api
@pytest.mark.requires_anthropic
@pytest.mark.asyncio
async def test_safe_python_execution_succeeds(safety_project):
    """Scenario 8: Safe math code should execute without safety errors.

    Simple math operations have no write patterns, so the code pattern
    detector should not flag them.

    Cost budget: $0.25
    """
    prompt = (
        "Use the execute tool to run this code: "
        "import math; print(math.pi * 2). Report the output."
    )

    result = await run_sdk_query_with_hooks(
        safety_project,
        prompt,
        approval_policy="auto_approve",
        max_turns=5,
        max_budget_usd=0.25,
    )

    # -- Debug output --
    print("\n--- Scenario 8: safe python execution ---")
    print(f"  tools called: {result.tool_names}")
    print(f"  num_turns: {result.num_turns}")
    print(f"  cost: ${result.cost_usd:.4f}" if result.cost_usd else "  cost: N/A")
    for trace in result.tool_traces:
        print(f"  tool: {trace.name}")
        print(f"    is_error: {trace.is_error}")
        result_preview = (trace.result or "")[:300]
        print(f"    result preview: {result_preview}")

    # -- Assertions --
    assert result.result is not None, "No ResultMessage received from SDK"

    py_calls = result.tools_matching("execute")
    assert len(py_calls) >= 1, (
        f"Expected execute call but got: {result.tool_names}"
    )

    # The tool call should not have errored
    assert not py_calls[0].is_error, (
        f"execute returned error: {py_calls[0].result}"
    )

    # Output should contain 6.283 (2*pi)
    combined = _combined_text(result)
    assert "6.283" in combined, (
        f"Expected '6.283' in output.\n  Combined: {combined[:500]}"
    )
