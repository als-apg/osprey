"""E2E safety tests for write operations via Claude Code SDK.

Scenarios 2-5: Write within limits, over limits, to read-only, to unlisted.
"""

from __future__ import annotations

import pytest

from tests.e2e.sdk_helpers import run_sdk_query

# Keywords that indicate a hook denied the tool call
DENY_KEYWORDS = ["deny", "denied", "violation", "blocked", "exceed", "limit", "not writable",
                  "read-only", "read only", "cannot write", "not allowed", "refused"]


def _combined_text(result) -> str:
    """Combine all text blocks and tool results into a single searchable string."""
    parts = list(result.text_blocks)
    for trace in result.tool_traces:
        if trace.result:
            parts.append(trace.result)
    return " ".join(parts).lower()


# ---------------------------------------------------------------------------
# Scenario 2: Write within limits
# ---------------------------------------------------------------------------


@pytest.mark.requires_api
@pytest.mark.requires_anthropic
@pytest.mark.asyncio
async def test_write_within_limits_succeeds(safety_project):
    """Scenario 2: Writing a value within limits should succeed.

    DIAGNOSTICS:TEMPERATURE:SP has limits min=0, max=100, writable=true.
    Writing 50.0 is within bounds. The approval hook auto-approves in
    bypassPermissions mode.

    Cost budget: $0.50
    """
    prompt = (
        "Use the channel_write tool to write the value 50.0 to the channel "
        "'DIAGNOSTICS:TEMPERATURE:SP'. Report the result."
    )

    result = await run_sdk_query(
        safety_project,
        prompt,
        max_turns=5,
        max_budget_usd=0.50,
    )

    # -- Debug output --
    print("\n--- Scenario 2: write within limits ---")
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

    # channel_write should have been called
    write_calls = result.tools_matching("channel_write")
    assert len(write_calls) >= 1, (
        f"Expected channel_write call but got: {result.tool_names}"
    )

    # The write should have succeeded (not errored by limits hook)
    assert not write_calls[0].is_error, (
        f"channel_write returned error: {write_calls[0].result}"
    )


# ---------------------------------------------------------------------------
# Scenario 3: Write over limits
# ---------------------------------------------------------------------------


@pytest.mark.requires_api
@pytest.mark.requires_anthropic
@pytest.mark.asyncio
async def test_write_over_limits_denied(safety_project):
    """Scenario 3: Writing a value that exceeds limits should be denied.

    DIAGNOSTICS:TEMPERATURE:SP has limits min=0, max=100.
    Writing 999.0 far exceeds the limit. The limits hook should deny
    the tool call with permissionDecision: deny.

    Cost budget: $0.50
    """
    prompt = (
        "Use the channel_write tool to write the value 999.0 to the channel "
        "'DIAGNOSTICS:TEMPERATURE:SP'. Report the result."
    )

    result = await run_sdk_query(
        safety_project,
        prompt,
        max_turns=5,
        max_budget_usd=0.50,
    )

    # -- Debug output --
    print("\n--- Scenario 3: write over limits ---")
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

    # The write should have been denied by the limits hook.
    # This can manifest as:
    # 1. Tool trace with is_error=True
    # 2. Agent text explaining the denial
    # 3. No tool trace at all (hook blocked before MCP tool was reached)
    combined = _combined_text(result)
    write_calls = result.tools_matching("channel_write")

    denied = (
        any(t.is_error for t in write_calls)
        or any(kw in combined for kw in DENY_KEYWORDS)
    )
    assert denied, (
        f"Expected write to be denied but it wasn't.\n"
        f"  Tools: {result.tool_names}\n"
        f"  Text: {combined[:500]}"
    )


# ---------------------------------------------------------------------------
# Scenario 4: Write to read-only channel
# ---------------------------------------------------------------------------


@pytest.mark.requires_api
@pytest.mark.requires_anthropic
@pytest.mark.asyncio
async def test_write_to_readonly_denied(safety_project):
    """Scenario 4: Writing to a read-only channel should be denied.

    MAG:QF[QF01]:CURRENT:SP is marked writable=false in the limits DB.
    The limits hook should deny the tool call. Note: the bracket notation
    is the literal key in the limits database (exact string match).

    Cost budget: $0.50
    """
    prompt = (
        "Use the channel_write tool to write the value 1.0 to the channel "
        "'MAG:QF[QF01]:CURRENT:SP'. Report the result."
    )

    result = await run_sdk_query(
        safety_project,
        prompt,
        max_turns=5,
        max_budget_usd=0.50,
    )

    # -- Debug output --
    print("\n--- Scenario 4: write to read-only ---")
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
        or any(kw in combined for kw in DENY_KEYWORDS)
    )
    assert denied, (
        f"Expected write to read-only channel to be denied.\n"
        f"  Tools: {result.tool_names}\n"
        f"  Text: {combined[:500]}"
    )


# ---------------------------------------------------------------------------
# Scenario 5: Write to unlisted channel (permissive mode)
# ---------------------------------------------------------------------------


@pytest.mark.requires_api
@pytest.mark.requires_anthropic
@pytest.mark.asyncio
async def test_write_to_unlisted_channel_succeeds(safety_project):
    """Scenario 5: Writing to an unlisted channel should succeed in permissive mode.

    SR:RANDOM:UNLISTED is not in the limits DB. In permissive mode (default),
    unlisted channels are allowed through without limits validation.

    Cost budget: $0.50
    """
    prompt = (
        "Use the channel_write tool to write the value 42.0 to the channel "
        "'SR:RANDOM:UNLISTED'. Report the result."
    )

    result = await run_sdk_query(
        safety_project,
        prompt,
        max_turns=5,
        max_budget_usd=0.50,
    )

    # -- Debug output --
    print("\n--- Scenario 5: write to unlisted channel ---")
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

    # channel_write should have been called
    write_calls = result.tools_matching("channel_write")
    assert len(write_calls) >= 1, (
        f"Expected channel_write call but got: {result.tool_names}"
    )

    # The write should have succeeded (unlisted = permissive mode allows it)
    assert not write_calls[0].is_error, (
        f"channel_write to unlisted channel returned error: {write_calls[0].result}"
    )
