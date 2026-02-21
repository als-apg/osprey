"""E2E approval flow tests via Claude Code SDK with hooks enabled.

Tests that the approval hook correctly fires (or doesn't fire) based on
approval mode, and that approval/denial decisions propagate to Claude.

4 scenarios:
  2a: channel_write triggers approval callback (selective + auto-approve)
  2c: Approval denial propagates to Claude
  2d: all_capabilities mode asks for everything (including reads)
  2e: Limits deny blocks before approval callback fires
"""

from __future__ import annotations

import pytest

from tests.e2e.sdk_helpers import combined_text, run_sdk_query_with_hooks

# Keywords that indicate a hook denied the tool call
DENY_KEYWORDS = [
    "deny", "denied", "violation", "blocked", "exceed", "limit",
    "not writable", "read-only", "read only", "cannot write",
    "not allowed", "refused",
]


# ---------------------------------------------------------------------------
# 2a: channel_write triggers approval callback (selective + auto-approve)
# ---------------------------------------------------------------------------


@pytest.mark.requires_api
@pytest.mark.requires_anthropic
@pytest.mark.asyncio
async def test_channel_write_triggers_approval_callback(safety_project_selective):
    """Scenario 2a: channel_write should trigger the approval hook callback.

    In selective mode with auto-approve, the approval hook returns "ask",
    the SDK callback approves it, and the write succeeds.

    Cost budget: $0.50
    """
    prompt = (
        "Use the channel_write tool to write the value 50.0 to the channel "
        "'DIAGNOSTICS:TEMPERATURE:SP'. Report the result."
    )

    result = await run_sdk_query_with_hooks(
        safety_project_selective,
        prompt,
        approval_policy="auto_approve",
        max_turns=5,
        max_budget_usd=0.50,
    )

    # -- Debug output --
    print("\n--- 2a: channel_write approval callback ---")
    print(f"  tools called: {result.tool_names}")
    print(f"  hook_events: {len(result.hook_events)}")
    for evt in result.hook_events:
        print(f"    {evt.tool_name}: {evt.decision}")

    # -- Assertions --
    assert result.result is not None, "No ResultMessage received from SDK"

    # The approval hook should have fired for channel_write
    write_events = [e for e in result.hook_events if "channel_write" in e.tool_name]
    assert len(write_events) >= 1, (
        f"Expected approval callback for channel_write but got: "
        f"{[e.tool_name for e in result.hook_events]}"
    )
    assert write_events[0].decision == "allow"

    # The write should have succeeded
    write_calls = result.tools_matching("channel_write")
    assert len(write_calls) >= 1, (
        f"Expected channel_write call but got: {result.tool_names}"
    )
    assert not write_calls[0].is_error, (
        f"channel_write returned error: {write_calls[0].result}"
    )


# ---------------------------------------------------------------------------
# 2b: (removed) python_execute with caput — caput is intentionally blocked
# at the prompt level; all hardware writes go through channel_write.
# The write pattern detection in the approval hook is defense-in-depth only.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2c: Approval denial propagates to Claude
# ---------------------------------------------------------------------------


@pytest.mark.requires_api
@pytest.mark.requires_anthropic
@pytest.mark.asyncio
async def test_approval_denial_propagates_to_claude(safety_project_selective):
    """Scenario 2c: Denying approval should propagate to Claude's response.

    With auto-deny, the approval hook returns "ask", the SDK callback denies
    it, and Claude should report that the operation was denied.

    Cost budget: $0.50
    """
    prompt = (
        "Use the channel_write tool to write the value 50.0 to the channel "
        "'DIAGNOSTICS:TEMPERATURE:SP'. Report the result."
    )

    result = await run_sdk_query_with_hooks(
        safety_project_selective,
        prompt,
        approval_policy="auto_deny",
        max_turns=5,
        max_budget_usd=0.50,
    )

    # -- Debug output --
    print("\n--- 2c: approval denial propagation ---")
    print(f"  tools called: {result.tool_names}")
    print(f"  hook_events: {len(result.hook_events)}")
    for evt in result.hook_events:
        print(f"    {evt.tool_name}: {evt.decision}")
    print(f"  text blocks: {len(result.text_blocks)}")
    for tb in result.text_blocks[:3]:
        print(f"    {tb[:200]}")

    # -- Assertions --
    assert result.result is not None, "No ResultMessage received from SDK"

    # The approval hook should have fired and been denied
    write_events = [e for e in result.hook_events if "channel_write" in e.tool_name]
    assert len(write_events) >= 1, (
        f"Expected approval callback for channel_write but got: "
        f"{[e.tool_name for e in result.hook_events]}"
    )
    assert write_events[0].decision == "deny"

    # Claude should reference the denial in its response
    combined = combined_text(result)
    assert any(kw in combined for kw in DENY_KEYWORDS), (
        f"Expected Claude to reference denial in response.\n"
        f"  Text: {combined[:500]}"
    )


# ---------------------------------------------------------------------------
# 2d: all_capabilities mode asks for everything (including reads)
# ---------------------------------------------------------------------------


@pytest.mark.requires_api
@pytest.mark.requires_anthropic
@pytest.mark.asyncio
async def test_all_capabilities_asks_for_reads(safety_project_all_capabilities):
    """Scenario 2d: all_capabilities mode should ask for read operations too.

    In all_capabilities mode, even channel_read triggers the approval hook.

    Cost budget: $0.50
    """
    prompt = (
        "Use the channel_read tool to read the channel 'SR:BEAM:CURRENT'. "
        "Report the value."
    )

    result = await run_sdk_query_with_hooks(
        safety_project_all_capabilities,
        prompt,
        approval_policy="auto_approve",
        max_turns=5,
        max_budget_usd=0.50,
    )

    # -- Debug output --
    print("\n--- 2d: all_capabilities reads ---")
    print(f"  tools called: {result.tool_names}")
    print(f"  hook_events: {len(result.hook_events)}")
    for evt in result.hook_events:
        print(f"    {evt.tool_name}: {evt.decision}")

    # -- Assertions --
    assert result.result is not None, "No ResultMessage received from SDK"

    # The approval hook should have fired for channel_read
    read_events = [e for e in result.hook_events if "channel_read" in e.tool_name]
    assert len(read_events) >= 1, (
        f"Expected approval callback for channel_read in all_capabilities mode "
        f"but got: {[e.tool_name for e in result.hook_events]}"
    )
    assert read_events[0].decision == "allow", (
        f"Expected auto_approve to produce 'allow' decision "
        f"but got: {read_events[0].decision}"
    )


# ---------------------------------------------------------------------------
# 2e: Limits deny blocks before approval callback fires
# ---------------------------------------------------------------------------


@pytest.mark.requires_api
@pytest.mark.requires_anthropic
@pytest.mark.asyncio
async def test_limits_deny_blocks_before_approval(safety_project_selective):
    """Scenario 2e: Limits violation should deny before approval hook fires.

    Writing 999.0 exceeds DIAGNOSTICS:TEMPERATURE:SP limits (max=100).
    The limits hook returns "deny" directly, so the approval hook never
    reaches "ask" and hook_events should be EMPTY.

    Cost budget: $0.50
    """
    prompt = (
        "Use the channel_write tool to write the value 999.0 to the channel "
        "'DIAGNOSTICS:TEMPERATURE:SP'. Report the result."
    )

    result = await run_sdk_query_with_hooks(
        safety_project_selective,
        prompt,
        approval_policy="auto_approve",
        max_turns=5,
        max_budget_usd=0.50,
    )

    # -- Debug output --
    print("\n--- 2e: limits deny before approval ---")
    print(f"  tools called: {result.tool_names}")
    print(f"  hook_events: {len(result.hook_events)}")
    for evt in result.hook_events:
        print(f"    {evt.tool_name}: {evt.decision}")

    # -- Assertions --
    assert result.result is not None, "No ResultMessage received from SDK"

    # Limits hook should have denied before approval callback fired.
    # hook_events records only "ask" callbacks — limits returns "deny" directly.
    assert len(result.hook_events) == 0, (
        f"Expected no hook_events (limits should deny before ask) "
        f"but got {len(result.hook_events)}: "
        f"{[(e.tool_name, e.decision) for e in result.hook_events]}"
    )

    # Claude should report the denial
    combined = combined_text(result)
    assert any(kw in combined for kw in DENY_KEYWORDS), (
        f"Expected Claude to report limits violation.\n"
        f"  Text: {combined[:500]}"
    )
