"""E2E safety tests for read operations via Claude Code SDK.

Scenario 1: Verify that channel_read succeeds for a valid channel.

Uses run_sdk_query_with_hooks to exercise the full hook chain. Reads in
selective mode don't trigger approval, so hook_events should be EMPTY.
"""

from __future__ import annotations

import pytest

from tests.e2e.sdk_helpers import run_sdk_query_with_hooks


@pytest.mark.requires_api
@pytest.mark.requires_anthropic
@pytest.mark.asyncio
async def test_channel_read_succeeds(safety_project):
    """Scenario 1: Reading a channel should succeed without error.

    Uses channel_read on SR:BEAM:CURRENT (unlisted in limits DB, so no
    restrictions apply). The mock connector returns synthetic data.

    Cost budget: $0.25
    """
    prompt = (
        "Use the channel_read tool to read the channel 'SR:BEAM:CURRENT'. "
        "Report the value you get back."
    )

    result = await run_sdk_query_with_hooks(
        safety_project,
        prompt,
        approval_policy="auto_approve",
        max_turns=5,
        max_budget_usd=0.25,
    )

    # -- Debug output --
    print("\n--- Scenario 1: channel_read ---")
    print(f"  tools called: {result.tool_names}")
    print(f"  num_turns: {result.num_turns}")
    print(f"  cost: ${result.cost_usd:.4f}" if result.cost_usd else "  cost: N/A")
    print(f"  hook_events: {len(result.hook_events)}")
    for evt in result.hook_events:
        print(f"    {evt.tool_name}: {evt.decision}")
    for trace in result.tool_traces:
        print(f"  tool: {trace.name}")
        print(f"    is_error: {trace.is_error}")
        result_preview = (trace.result or "")[:200]
        print(f"    result preview: {result_preview}")

    # -- Assertions --
    assert result.result is not None, "No ResultMessage received from SDK"
    assert not result.result.is_error, f"SDK query ended in error: {result.result.result}"

    # channel_read should have been called
    read_calls = result.tools_matching("channel_read")
    assert len(read_calls) >= 1, (
        f"Expected channel_read call but got: {result.tool_names}"
    )

    # The tool call should not have errored
    assert not read_calls[0].is_error, (
        f"channel_read returned error: {read_calls[0].result}"
    )

    # Reads in selective mode don't trigger approval → hook_events should be EMPTY
    assert len(result.hook_events) == 0, (
        f"Expected no hook_events for reads in selective mode "
        f"but got {len(result.hook_events)}: "
        f"{[(e.tool_name, e.decision) for e in result.hook_events]}"
    )
