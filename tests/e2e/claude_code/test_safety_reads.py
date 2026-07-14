"""E2E safety tests for read operations via Claude Code SDK.

Scenario 1: Verify that channel_read succeeds for a valid channel.

Uses run_sdk_query_with_hooks to exercise the full hook chain. The
``safety_project`` fixture builds the DEFAULT control_assistant project, which
ships ``approval.tools.channel_read: skip``. Under a ``skip`` policy the
approval hook auto-approves the read (no human prompt) by emitting
``permissionDecision: 'allow'``. The safety-relevant invariant is therefore
"every channel_read event is an allow, never a deny/ask" — NOT "no events at
all". See the assertion comment for why an ``allow`` hook decision still
surfaces through can_use_tool while a ``deny`` does not.
"""

from __future__ import annotations

import pytest

from tests.e2e.sdk_helpers import run_sdk_query_with_hooks


@pytest.mark.requires_api
@pytest.mark.requires_als_apg
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
    assert len(read_calls) >= 1, f"Expected channel_read call but got: {result.tool_names}"

    # The tool call should not have errored
    assert not read_calls[0].is_error, f"channel_read returned error: {read_calls[0].result}"

    # Under the default config's `channel_read: skip` policy the approval hook
    # auto-approves the read by emitting permissionDecision='allow' (no human
    # prompt). With claude-agent-sdk 0.2.93 / CLI 2.1.x an `allow` hook decision
    # is STILL routed through can_use_tool and recorded as a hook event — only a
    # hook `deny` suppresses the callback (see test_safety_kill_switch.py
    # scenarios 9/10, which legitimately assert zero hook_events for denies, and
    # scenario 2d/2e in test_safety_approval_e2e.py). So asserting len==0 here
    # was wrong: it conflated allow-suppresses-callback with
    # deny-suppresses-callback. The real safety invariant is that a skip-policy
    # read is auto-approved — never DENIED and never human-blocked — i.e. every
    # recorded channel_read event must be an `allow`.
    read_events = [e for e in result.hook_events if "channel_read" in e.tool_name]
    assert all(e.decision == "allow" for e in read_events), (
        f"channel_read under skip policy must auto-approve (allow), got: "
        f"{[(e.tool_name, e.decision) for e in read_events]}"
    )
