"""E2E tests for Channel Finder feedback capture hook.

Verifies that the PostToolUse feedback capture hook fires during real
SDK-driven queries and silently writes results to pending_reviews.json
without contaminating agent output.
"""

from __future__ import annotations

import json

import pytest

from tests.e2e.sdk_helpers import (
    agent_data_dir,
    combined_text,
    init_project,
    render_dir,
    run_sdk_query_with_hooks,
)

pytestmark = pytest.mark.harness_benchmark


@pytest.fixture(scope="module")
def feedback_project(tmp_path_factory):
    """Module-scoped deployment repo with channel_finder_mode=hierarchical."""
    tmp = tmp_path_factory.mktemp("feedback-capture")
    return init_project(
        tmp, "feedback-capture", provider="als-apg", channel_finder_mode="hierarchical"
    )


def _channel_finder_report(result, project) -> str:
    """Every channel-finder call with its answer, and the capture hook's own log
    lines, so a missing store says whether the search came back empty or the
    hook never wrote."""
    lines = ["  channel-finder calls:"]
    for trace in result.tool_traces:
        if "channel-finder" not in trace.name:
            continue
        lines.append(f"    {trace.name} input={json.dumps(trace.input, default=str)[:400]}")
        lines.append(f"      error={trace.is_error} result={(trace.result or '')[:600]}")
    hook_log = render_dir(project) / ".claude" / "hooks" / "hook_debug.jsonl"
    if hook_log.is_file():
        captured = [
            line
            for line in hook_log.read_text().splitlines()
            if "cf-feedback-capture" in line and '"skip-tool"' not in line
        ]
        lines.append(f"  cf-feedback-capture log ({len(captured)} lines):")
        lines.extend(f"    {line[:400]}" for line in captured)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test 1: Hook captures search results to pending_reviews.json
# ---------------------------------------------------------------------------


@pytest.mark.requires_api
@pytest.mark.requires_als_apg
@pytest.mark.asyncio
async def test_feedback_hook_captures_search_results(feedback_project):
    """The feedback capture hook should write results to pending_reviews.json.

    After a channel-finder search that returns results, the PostToolUse hook
    should silently create var/agent_data/feedback/pending_reviews.json with valid
    items — the state zone the channel-finder feedback app reads back.

    The query names one device, so the answer is a handful of addresses: an
    answer past the CLI's MCP output cap reaches the hook as a "saved to file"
    notice rather than the search result. The Python executor is withheld, so
    the addresses come from ``build_channels`` and not from a list the agent
    assembles itself.

    Cost budget: $0.50
    """
    prompt = (
        "Use the channel finder to find the X and Y position readback channels of "
        "BPM 01 in the booster ring (BR). Report the channel addresses it returns."
    )

    result = await run_sdk_query_with_hooks(
        feedback_project,
        prompt,
        approval_policy="auto_approve",
        max_turns=15,
        max_budget_usd=0.50,
        disallowed_tools=["mcp__python__execute"],
    )

    # -- Debug output --
    print("\n--- feedback capture: search results ---")
    print(f"  tools called: {result.tool_names}")
    print(f"  text blocks: {len(result.text_blocks)}")
    search_report = _channel_finder_report(result, feedback_project)
    print(search_report)

    # -- Assertions --
    assert result.result is not None, "No ResultMessage received from SDK"

    # A tool the hook captures should have been called
    build_calls = result.tools_matching("build_channels")
    assert len(build_calls) >= 1, (
        f"Expected a build_channels call but got: {result.tool_names}\n{search_report}"
    )

    # pending_reviews.json should exist with captured items
    store_path = agent_data_dir(feedback_project) / "feedback" / "pending_reviews.json"
    assert store_path.exists(), (
        f"Expected {store_path} to exist after channel-finder search\n{search_report}"
    )

    data = json.loads(store_path.read_text())
    assert "items" in data, f"Expected 'items' key in pending_reviews.json: {data.keys()}"
    assert len(data["items"]) >= 1, "Expected at least one captured item"

    # Validate item structure
    for item_id, item in data["items"].items():
        assert "id" in item or item_id, "Item should have an id"
        assert "tool_name" in item, f"Item missing tool_name: {item.keys()}"
        assert "channel_count" in item, f"Item missing channel_count: {item.keys()}"
        assert item["channel_count"] > 0, f"Expected channel_count > 0, got {item['channel_count']}"
        assert "captured_at" in item, f"Item missing captured_at: {item.keys()}"


# ---------------------------------------------------------------------------
# Test 2: Hook is completely silent (no stdout contamination)
# ---------------------------------------------------------------------------


@pytest.mark.requires_api
@pytest.mark.requires_als_apg
@pytest.mark.asyncio
async def test_feedback_hook_is_silent(feedback_project):
    """The feedback capture hook must not contaminate agent output.

    The hook should be completely silent — no system messages or text blocks
    should contain feedback-related keywords from the hook itself.

    Cost budget: $0.50
    """
    prompt = "Use the channel finder to search for quadrupole channels. Report what you found."

    result = await run_sdk_query_with_hooks(
        feedback_project,
        prompt,
        approval_policy="auto_approve",
        max_turns=15,
        max_budget_usd=0.50,
    )

    # -- Debug output --
    print("\n--- feedback capture: silence check ---")
    print(f"  tools called: {result.tool_names}")
    print(f"  system messages: {len(result.system_messages)}")
    print(f"  text blocks: {len(result.text_blocks)}")

    # -- Assertions --
    assert result.result is not None, "No ResultMessage received from SDK"

    # Hook-internal keywords that should never appear in agent output
    hook_keywords = [
        "pending_reviews",
        "pending_review",
        "feedback_capture",
        "feedback capture hook",
        "osprey_cf_feedback",
    ]

    # Check system messages
    for msg in result.system_messages:
        msg_text = str(msg).lower()
        for kw in hook_keywords:
            assert kw not in msg_text, (
                f"Hook keyword '{kw}' leaked into system message: {msg_text[:300]}"
            )

    # Check text blocks
    combined = combined_text(result)
    for kw in hook_keywords:
        assert kw not in combined, (
            f"Hook keyword '{kw}' leaked into agent text output: {combined[:300]}"
        )
