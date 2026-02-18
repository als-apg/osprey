"""E2E safety tests for the writes-disabled kill switch via Claude Code SDK.

Scenarios 9-10: Master kill switch (writes_enabled: false) blocks all writes.
"""

from __future__ import annotations

import pytest

from tests.e2e.sdk_helpers import run_sdk_query

# Keywords that indicate the writes_check hook denied the operation
WRITES_DISABLED_KEYWORDS = ["writes", "disabled", "denied", "blocked", "not allowed",
                            "not enabled", "kill switch", "write operations",
                            "cannot write", "refused"]


def _combined_text(result) -> str:
    """Combine all text blocks and tool results into a single searchable string."""
    parts = list(result.text_blocks)
    for trace in result.tool_traces:
        if trace.result:
            parts.append(trace.result)
    return " ".join(parts).lower()


# ---------------------------------------------------------------------------
# Scenario 9: Writes disabled — channel_write
# ---------------------------------------------------------------------------


@pytest.mark.requires_api
@pytest.mark.requires_anthropic
@pytest.mark.asyncio
async def test_channel_write_denied_when_writes_disabled(safety_project_writes_off):
    """Scenario 9: channel_write should be blocked when writes_enabled=false.

    The writes_check hook reads config.yml and denies any channel_write
    or python_execute(write) when writes_enabled is false.

    Cost budget: $0.50
    """
    prompt = (
        "Use the channel_write tool to write the value 5.0 to the channel "
        "'MAG:HCM01:CURRENT:SP'. Report the result."
    )

    result = await run_sdk_query(
        safety_project_writes_off,
        prompt,
        max_turns=5,
        max_budget_usd=0.50,
    )

    # -- Debug output --
    print("\n--- Scenario 9: writes disabled (channel_write) ---")
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

    combined = _combined_text(result)
    write_calls = result.tools_matching("channel_write")

    denied = (
        any(t.is_error for t in write_calls)
        or any(kw in combined for kw in WRITES_DISABLED_KEYWORDS)
    )
    assert denied, (
        f"Expected channel_write to be denied by kill switch.\n"
        f"  Tools: {result.tool_names}\n"
        f"  Text: {combined[:500]}"
    )


# ---------------------------------------------------------------------------
# Scenario 10: Writes disabled — python_execute in write mode
# ---------------------------------------------------------------------------


@pytest.mark.requires_api
@pytest.mark.requires_anthropic
@pytest.mark.asyncio
async def test_python_write_denied_when_writes_disabled(safety_project_writes_off):
    """Scenario 10: python_execute with write mode should be blocked.

    The writes_check hook also blocks python_execute when execution_mode
    is 'write' or 'readwrite' and writes_enabled is false.

    Cost budget: $0.50
    """
    prompt = (
        "Use the python_execute tool with execution_mode 'write' to run "
        "this code: caput('X', 1). Report the result."
    )

    result = await run_sdk_query(
        safety_project_writes_off,
        prompt,
        max_turns=5,
        max_budget_usd=0.50,
    )

    # -- Debug output --
    print("\n--- Scenario 10: writes disabled (python_execute write) ---")
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

    combined = _combined_text(result)
    py_calls = result.tools_matching("python_execute")

    denied = (
        any(t.is_error for t in py_calls)
        or any(kw in combined for kw in WRITES_DISABLED_KEYWORDS)
    )
    assert denied, (
        f"Expected python_execute write to be denied by kill switch.\n"
        f"  Tools: {result.tool_names}\n"
        f"  Text: {combined[:500]}"
    )
