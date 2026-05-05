"""E2E error guidance tests via Claude Code SDK with hooks enabled.

Tests that the error_guidance PostToolUse hook fires on tool errors and that
Claude follows the error-handling protocol (report clearly, don't retry).

The error_guidance hook fires on PostToolUse when the tool response contains
an error envelope ({"error": true, ...}).  We trigger this via the execute tool
with code that raises exceptions — the sandbox executor returns a structured
error, which fires the hook reliably in mock mode.

2 scenarios:
  3a: Execution error  -> error_guidance hook fires, Claude reports the error
  3b: Connection error -> error_guidance prevents retry behavior
"""

from __future__ import annotations

import pytest

from tests.e2e.sdk_helpers import combined_text, run_sdk_query_with_hooks

# Keywords indicating Claude reported an error (expected behavior)
ERROR_REPORT_KEYWORDS = [
    "error",
    "exception",
    "failed",
    "traceback",
    "import",
    "modulenotfounderror",
    "no module",
    "cannot import",
    "connectionerror",
    "not reachable",
]

# ---------------------------------------------------------------------------
# 3a: Execution error -> error_guidance hook fires
# ---------------------------------------------------------------------------


@pytest.mark.requires_api
@pytest.mark.requires_als_apg
@pytest.mark.asyncio
async def test_execution_error_triggers_guidance(safety_project):
    """Scenario 3a: execute tool error -> error guidance hook fires.

    Code that raises an exception triggers the error_guidance PostToolUse hook,
    which injects additionalContext pointing Claude to the error-handling protocol.
    Claude should report the error clearly without retrying.

    Cost budget: $0.50
    """
    prompt = (
        "Use the execute tool in read mode to run this exact code:\n"
        "import nonexistent_module_that_does_not_exist_xyz\n"
        "Report what happens."
    )

    result = await run_sdk_query_with_hooks(
        safety_project,
        prompt,
        approval_policy="auto_approve",
        max_turns=5,
        max_budget_usd=0.50,
    )

    # -- Debug output --
    print("\n--- 3a: execution error triggers guidance ---")
    print(f"  tools called: {result.tool_names}")
    print(f"  num_turns: {result.num_turns}")
    for trace in result.tool_traces:
        print(f"  tool: {trace.name}")
        print(f"    is_error: {trace.is_error}")
        result_preview = (trace.result or "")[:300]
        print(f"    result preview: {result_preview}")
    for tb in result.text_blocks[:3]:
        print(f"  text: {tb[:200]}")

    # -- Assertions --
    assert result.result is not None, "No ResultMessage received from SDK"

    # execute should have been called
    py_calls = result.tools_matching("execute")
    assert len(py_calls) >= 1, f"Expected execute call but got: {result.tool_names}"

    # The tool should have returned an error (either via SDK is_error or via
    # the MCP ``{"error": true}`` envelope wrapped in a successful response).
    assert py_calls[0].failed, (
        f"Expected execute to report an error but failed=False.\n"
        f"  Result: {(py_calls[0].result or '')[:300]}"
    )

    # Claude should report the error
    combined = combined_text(result)
    assert any(kw in combined for kw in ERROR_REPORT_KEYWORDS), (
        f"Expected Claude to report an error.\n  Text: {combined[:500]}"
    )

    # Tool-trace assertion: the agent must NOT invoke the failed tool again
    # after seeing the error. (Replaces a brittle keyword check on Claude's
    # text response — "let me retry"/"trying again"/etc. — which the
    # error-guidance hook surfaced reliably for some phrasings and missed
    # others. The behavioral invariant we actually care about is "no second
    # execute call", which the tool trace records exactly.)
    assert len(py_calls) == 1, (
        f"Agent invoked execute {len(py_calls)} times; expected exactly 1 "
        f"(no retry after the first error). Tool sequence: {result.tool_names}"
    )


# ---------------------------------------------------------------------------
# 3b: Error guidance should prevent retry behavior
# ---------------------------------------------------------------------------


@pytest.mark.requires_api
@pytest.mark.requires_als_apg
@pytest.mark.asyncio
async def test_error_response_no_retry_protocol(safety_project):
    """Scenario 3b: Error guidance should prevent retry behavior.

    When the execute tool returns an error, the error_guidance hook injects
    protocol guidance. Claude should report the failure, not attempt workarounds.

    Cost budget: $0.50
    """
    prompt = (
        "Use the execute tool in read mode to run this code:\n"
        "raise ConnectionError('EPICS channel NONEXISTENT:PV not reachable')\n"
        "Report the result."
    )

    result = await run_sdk_query_with_hooks(
        safety_project,
        prompt,
        approval_policy="auto_approve",
        max_turns=5,
        max_budget_usd=0.50,
    )

    # -- Debug output --
    print("\n--- 3b: error response no-retry protocol ---")
    print(f"  tools called: {result.tool_names}")
    print(f"  num_turns: {result.num_turns}")
    for trace in result.tool_traces:
        print(f"  tool: {trace.name}")
        print(f"    is_error: {trace.is_error}")
        result_preview = (trace.result or "")[:300]
        print(f"    result preview: {result_preview}")
    for tb in result.text_blocks[:3]:
        print(f"  text: {tb[:200]}")

    # -- Assertions --
    assert result.result is not None, "No ResultMessage received from SDK"

    # execute should have been called
    py_calls = result.tools_matching("execute")
    assert len(py_calls) >= 1, f"Expected execute call but got: {result.tool_names}"

    # The tool should have returned an error (either via SDK is_error or via
    # the MCP ``{"error": true}`` envelope wrapped in a successful response).
    assert py_calls[0].failed, (
        f"Expected execute to report an error but failed=False.\n"
        f"  Result: {(py_calls[0].result or '')[:300]}"
    )

    # Claude should report the error
    combined = combined_text(result)
    assert any(kw in combined for kw in ERROR_REPORT_KEYWORDS), (
        f"Expected Claude to report the connection error.\n  Text: {combined[:500]}"
    )

    # Tool-trace assertion: no retry of the failing execute call.
    # See test_execution_error_triggers_guidance for the rationale.
    assert len(py_calls) == 1, (
        f"Agent invoked execute {len(py_calls)} times; expected exactly 1 "
        f"(no retry after the first error). Tool sequence: {result.tool_names}"
    )
