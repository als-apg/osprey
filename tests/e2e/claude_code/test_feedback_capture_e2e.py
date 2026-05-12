"""E2E tests for Channel Finder feedback capture hook.

Verifies that the PostToolUse feedback capture hook fires during real
SDK-driven queries and silently writes results to pending_reviews.json
without contaminating agent output.
"""

from __future__ import annotations

import json

import pytest

from tests.e2e.sdk_helpers import combined_text, init_project, run_sdk_query_with_hooks


@pytest.fixture(scope="module")
def feedback_project(tmp_path_factory):
    """Module-scoped project with channel_finder_mode=hierarchical."""
    tmp = tmp_path_factory.mktemp("feedback-capture")
    return init_project(
        tmp, "feedback-capture", provider="als-apg", channel_finder_mode="hierarchical"
    )


# ---------------------------------------------------------------------------
# Test 1: Hook captures search results to pending_reviews.json
# ---------------------------------------------------------------------------


@pytest.mark.requires_api
@pytest.mark.requires_als_apg
@pytest.mark.asyncio
async def test_feedback_hook_captures_search_results(feedback_project):
    """The feedback capture hook should write results to pending_reviews.json.

    After a channel-finder search that returns results, the PostToolUse hook
    should silently create data/feedback/pending_reviews.json with valid items.

    Cost budget: $0.50
    """
    prompt = (
        "Use the channel finder to search for BPM channels. Report how many channels were found."
    )

    result = await run_sdk_query_with_hooks(
        feedback_project,
        prompt,
        approval_policy="auto_approve",
        max_turns=15,
        max_budget_usd=0.50,
    )

    # -- Debug output --
    print("\n--- feedback capture: search results ---")
    print(f"  tools called: {result.tool_names}")
    print(f"  text blocks: {len(result.text_blocks)}")

    # -- Assertions --
    assert result.result is not None, "No ResultMessage received from SDK"

    # Channel-finder tool should have been called
    cf_calls = result.tools_matching("channel-finder")
    assert len(cf_calls) >= 1, f"Expected channel-finder tool call but got: {result.tool_names}"

    # pending_reviews.json should exist with captured items
    store_path = feedback_project / "data" / "feedback" / "pending_reviews.json"
    assert store_path.exists(), f"Expected {store_path} to exist after channel-finder search"

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
