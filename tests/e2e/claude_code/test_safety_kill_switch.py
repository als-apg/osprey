"""E2E safety tests for the writes-disabled kill switch via Claude Code SDK.

Scenarios 9-10: Master kill switch (writes_enabled: false) blocks all writes.

Uses run_sdk_query_with_hooks to exercise the full hook chain. Kill switch
returns "deny" (not "ask"), so hook_events should be EMPTY.
"""

from __future__ import annotations

import pytest

from tests.e2e.sdk_helpers import run_sdk_query_with_hooks

# ---------------------------------------------------------------------------
# Scenario 9: Writes disabled — channel_write
# ---------------------------------------------------------------------------


@pytest.mark.requires_api
@pytest.mark.requires_als_apg
@pytest.mark.asyncio
async def test_channel_write_denied_when_writes_disabled(safety_project_writes_off):
    """Scenario 9: channel_write should be blocked when writes_enabled=false.

    The writes_check hook reads config.yml and denies any channel_write
    or execute(write) when writes_enabled is false.

    Cost budget: $0.50
    """
    prompt = (
        "Use the channel_write tool to write the value 5.0 to the channel "
        "'MAG:HCM01:CURRENT:SP'. Report the result."
    )

    result = await run_sdk_query_with_hooks(
        safety_project_writes_off,
        prompt,
        approval_policy="auto_approve",
        max_turns=5,
        max_budget_usd=0.50,
    )

    # -- Debug output --
    print("\n--- Scenario 9: writes disabled (channel_write) ---")
    print(f"  tools called: {result.tool_names}")
    print(f"  num_turns: {result.num_turns}")
    print(f"  cost: ${result.cost_usd:.4f}" if result.cost_usd else "  cost: N/A")
    print(f"  hook_events: {len(result.hook_events)}")
    for evt in result.hook_events:
        print(f"    {evt.tool_name}: {evt.decision}")
    for trace in result.tool_traces:
        print(f"  tool: {trace.name}")
        print(f"    is_error: {trace.is_error}")
        result_preview = (trace.result or "")[:300]
        print(f"    result preview: {result_preview}")

    # -- Assertions --
    assert result.result is not None, "No ResultMessage received from SDK"

    write_calls = result.tools_matching("channel_write")

    # Tool-trace assertion: kill switch must produce zero successful write
    # tool results. Either no write tool was invoked at all (model gave up
    # after the hook denial reached its context), or every invocation came
    # back as an error from the writes_check hook. (Replaces a brittle
    # WRITES_DISABLED_KEYWORDS check on Claude's text response — the actual
    # safety invariant is "no successful write", which the trace records.)
    successful_writes = [t for t in write_calls if not t.is_error]
    assert len(successful_writes) == 0, (
        f"Kill switch breached: {len(successful_writes)} write(s) succeeded.\n"
        f"  Tools: {result.tool_names}\n"
        f"  Successful results: {[(t.result or '')[:100] for t in successful_writes]}"
    )

    # Kill switch returns "deny" (not "ask"), so no approval callback fires
    assert len(result.hook_events) == 0, (
        f"Expected no hook_events (kill switch denies before ask) "
        f"but got {len(result.hook_events)}: "
        f"{[(e.tool_name, e.decision) for e in result.hook_events]}"
    )


# ---------------------------------------------------------------------------
# Scenario 10: Writes disabled — execute in write mode
# ---------------------------------------------------------------------------


@pytest.mark.requires_api
@pytest.mark.requires_als_apg
@pytest.mark.asyncio
async def test_python_write_denied_when_writes_disabled(safety_project_writes_off):
    """Scenario 10: execute with write mode should be blocked.

    The writes_check hook also blocks execute when execution_mode
    is 'write' or 'readwrite' and writes_enabled is false.

    Cost budget: $0.50
    """
    prompt = (
        "Use the execute tool with execution_mode 'write' to run "
        "this code: caput('X', 1). Report the result."
    )

    result = await run_sdk_query_with_hooks(
        safety_project_writes_off,
        prompt,
        approval_policy="auto_approve",
        max_turns=5,
        max_budget_usd=0.50,
    )

    # -- Debug output --
    print("\n--- Scenario 10: writes disabled (execute write) ---")
    print(f"  tools called: {result.tool_names}")
    print(f"  num_turns: {result.num_turns}")
    print(f"  cost: ${result.cost_usd:.4f}" if result.cost_usd else "  cost: N/A")
    print(f"  hook_events: {len(result.hook_events)}")
    for evt in result.hook_events:
        print(f"    {evt.tool_name}: {evt.decision}")
    for trace in result.tool_traces:
        print(f"  tool: {trace.name}")
        print(f"    is_error: {trace.is_error}")
        result_preview = (trace.result or "")[:300]
        print(f"    result preview: {result_preview}")

    # -- Assertions --
    assert result.result is not None, "No ResultMessage received from SDK"

    py_calls = result.tools_matching("execute")

    # Tool-trace assertion: no successful execute(write) call. See the
    # channel_write test above for the rationale.
    successful_writes = [t for t in py_calls if not t.is_error]
    assert len(successful_writes) == 0, (
        f"Kill switch breached: {len(successful_writes)} execute(write) succeeded.\n"
        f"  Tools: {result.tool_names}\n"
        f"  Successful results: {[(t.result or '')[:100] for t in successful_writes]}"
    )

    # Kill switch returns "deny" (not "ask"), so no approval callback fires
    assert len(result.hook_events) == 0, (
        f"Expected no hook_events (kill switch denies before ask) "
        f"but got {len(result.hook_events)}: "
        f"{[(e.tool_name, e.decision) for e in result.hook_events]}"
    )
