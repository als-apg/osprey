"""E2E safety tests for python_execute via Claude Code SDK.

Scenarios 6-8: Write pattern detection (caput, Tango) and safe execution.
"""

from __future__ import annotations

import pytest

from tests.e2e.sdk_helpers import run_sdk_query

# Keywords that indicate a safety error from the python executor
SAFETY_KEYWORDS = ["safety_error", "safety error", "write pattern", "detected",
                   "caput", "prohibited", "blocked", "not allowed", "denied"]


def _combined_text(result) -> str:
    """Combine all text blocks and tool results into a single searchable string."""
    parts = list(result.text_blocks)
    for trace in result.tool_traces:
        if trace.result:
            parts.append(trace.result)
    return " ".join(parts).lower()


# ---------------------------------------------------------------------------
# Scenario 6: Python with caput (EPICS write pattern)
# ---------------------------------------------------------------------------


@pytest.mark.requires_api
@pytest.mark.requires_anthropic
@pytest.mark.asyncio
async def test_python_caput_pattern_detected(safety_project):
    """Scenario 6: Code containing caput() should trigger safety error.

    The python_execute tool's code pattern detector scans for EPICS write
    patterns (caput, caput_many, etc.) and blocks them in readonly mode.

    Cost budget: $0.25
    """
    prompt = (
        "Use the python_execute tool with execution_mode 'readonly' to run "
        "this code: caput('SR:BEAM', 100). Report the result."
    )

    result = await run_sdk_query(
        safety_project,
        prompt,
        max_turns=5,
        max_budget_usd=0.25,
    )

    # -- Debug output --
    print("\n--- Scenario 6: caput pattern detection ---")
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

    # python_execute should have been called
    py_calls = result.tools_matching("python_execute")
    assert len(py_calls) >= 1, (
        f"Expected python_execute call but got: {result.tool_names}"
    )

    # The result should contain a safety error
    combined = _combined_text(result)
    safety_triggered = (
        any(t.is_error for t in py_calls)
        or any(kw in combined for kw in SAFETY_KEYWORDS)
    )
    assert safety_triggered, (
        f"Expected safety error for caput pattern.\n"
        f"  Tools: {result.tool_names}\n"
        f"  Text: {combined[:500]}"
    )


# ---------------------------------------------------------------------------
# Scenario 7: Python with Tango write pattern
# ---------------------------------------------------------------------------


@pytest.mark.requires_api
@pytest.mark.requires_anthropic
@pytest.mark.asyncio
async def test_python_tango_write_pattern_detected(safety_project):
    """Scenario 7: Code containing Tango write_attribute should trigger safety error.

    The python_execute tool's code pattern detector scans for Tango write
    patterns (write_attribute, command_inout) and blocks them in readonly mode.

    Cost budget: $0.25
    """
    prompt = (
        "Use the python_execute tool with execution_mode 'readonly' to run "
        "this code: device.write_attribute('MOTOR:POS', 100). Report the result."
    )

    result = await run_sdk_query(
        safety_project,
        prompt,
        max_turns=5,
        max_budget_usd=0.25,
    )

    # -- Debug output --
    print("\n--- Scenario 7: Tango write pattern detection ---")
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

    py_calls = result.tools_matching("python_execute")
    assert len(py_calls) >= 1, (
        f"Expected python_execute call but got: {result.tool_names}"
    )

    combined = _combined_text(result)
    safety_triggered = (
        any(t.is_error for t in py_calls)
        or any(kw in combined for kw in SAFETY_KEYWORDS)
    )
    assert safety_triggered, (
        f"Expected safety error for Tango write pattern.\n"
        f"  Tools: {result.tool_names}\n"
        f"  Text: {combined[:500]}"
    )


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
        "Use the python_execute tool to run this code: "
        "import math; print(math.pi * 2). Report the output."
    )

    result = await run_sdk_query(
        safety_project,
        prompt,
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

    py_calls = result.tools_matching("python_execute")
    assert len(py_calls) >= 1, (
        f"Expected python_execute call but got: {result.tool_names}"
    )

    # The tool call should not have errored
    assert not py_calls[0].is_error, (
        f"python_execute returned error: {py_calls[0].result}"
    )

    # Output should contain 6.283 (2*pi)
    combined = _combined_text(result)
    assert "6.283" in combined, (
        f"Expected '6.283' in output.\n  Combined: {combined[:500]}"
    )
