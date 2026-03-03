"""Tests for view_examples tool."""

import json
from unittest.mock import PropertyMock, patch

import pytest

from osprey.mcp_server.channel_finder_hierarchical.server_context import (
    initialize_cf_hier_context,
)
from osprey.services.channel_finder.feedback.store import FeedbackStore
from tests.mcp_server.channel_finder_hierarchical.conftest import get_tool_fn


def _setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    initialize_cf_hier_context()


def _make_store_with_entries(tmp_path):
    """Create a FeedbackStore with some test entries."""
    store = FeedbackStore(tmp_path / "feedback.json")
    store.record_success(
        query="show me magnets",
        facility="ALS",
        selections={"system": "MAG", "device": "QF1"},
        channel_count=42,
    )
    store.record_failure(
        query="show me magnets",
        facility="ALS",
        partial_selections={"system": "RF"},
        reason="no options at family level",
    )
    store.record_success(
        query="find BPM positions",
        facility="ALS",
        selections={"system": "BPM", "field": "X"},
        channel_count=10,
    )
    return store


@pytest.mark.unit
def test_view_examples_no_store(tmp_path, monkeypatch):
    """Registry has no feedback_store, returns empty."""
    _setup(tmp_path, monkeypatch)
    with patch(
        "osprey.mcp_server.channel_finder_hierarchical.server_context."
        "ChannelFinderHierContext.feedback_store",
        new_callable=PropertyMock,
        return_value=None,
    ):
        from osprey.mcp_server.channel_finder_hierarchical.tools.view_examples import (
            view_examples,
        )

        fn = get_tool_fn(view_examples)
        result = json.loads(fn())
    assert result["examples"] == []
    assert "No feedback data" in result["message"]


@pytest.mark.unit
def test_view_examples_no_keywords_lists_all(tmp_path, monkeypatch):
    """Store with entries, no keywords — returns all summaries."""
    _setup(tmp_path, monkeypatch)
    store = _make_store_with_entries(tmp_path)

    with patch(
        "osprey.mcp_server.channel_finder_hierarchical.server_context."
        "ChannelFinderHierContext.feedback_store",
        new_callable=PropertyMock,
        return_value=store,
    ):
        from osprey.mcp_server.channel_finder_hierarchical.tools.view_examples import (
            view_examples,
        )

        fn = get_tool_fn(view_examples)
        result = json.loads(fn())

    assert "all_examples" in result
    assert len(result["all_examples"]) == 2
    # Should contain GOOD/BAD formatted entries
    all_text = "\n".join(result["all_examples"])
    assert "GOOD" in all_text
    assert "BAD" in all_text


@pytest.mark.unit
def test_view_examples_with_keywords_returns_matches(tmp_path, monkeypatch):
    """Keyword search surfaces relevant entries."""
    _setup(tmp_path, monkeypatch)
    store = _make_store_with_entries(tmp_path)

    with patch(
        "osprey.mcp_server.channel_finder_hierarchical.server_context."
        "ChannelFinderHierContext.feedback_store",
        new_callable=PropertyMock,
        return_value=store,
    ):
        from osprey.mcp_server.channel_finder_hierarchical.tools.view_examples import (
            view_examples,
        )

        fn = get_tool_fn(view_examples)
        result = json.loads(fn(keywords="magnets, corrector"))

    assert "keyword_matches" in result
    assert len(result["keyword_matches"]) >= 1
    # The magnets entry should match
    assert any("magnets" in m for m in result["keyword_matches"])
    # All examples should still be present
    assert len(result["all_examples"]) == 2


@pytest.mark.unit
def test_view_examples_with_exact_query_returns_hints(tmp_path, monkeypatch):
    """Exact query match returns hints in exact_match field."""
    _setup(tmp_path, monkeypatch)
    store = _make_store_with_entries(tmp_path)

    with (
        patch(
            "osprey.mcp_server.channel_finder_hierarchical.server_context."
            "ChannelFinderHierContext.feedback_store",
            new_callable=PropertyMock,
            return_value=store,
        ),
        patch(
            "osprey.mcp_server.channel_finder_hierarchical.server_context."
            "ChannelFinderHierContext.facility_name",
            new_callable=PropertyMock,
            return_value="ALS",
        ),
    ):
        from osprey.mcp_server.channel_finder_hierarchical.tools.view_examples import (
            view_examples,
        )

        fn = get_tool_fn(view_examples)
        result = json.loads(fn(keywords="show me magnets"))

    assert "exact_match" in result
    assert len(result["exact_match"]) >= 1
    assert "GOOD" in result["exact_match"][0]


@pytest.mark.unit
def test_view_examples_with_keywords_no_match(tmp_path, monkeypatch):
    """Unrelated keywords return empty matches but still all summaries."""
    _setup(tmp_path, monkeypatch)
    store = _make_store_with_entries(tmp_path)

    with patch(
        "osprey.mcp_server.channel_finder_hierarchical.server_context."
        "ChannelFinderHierContext.feedback_store",
        new_callable=PropertyMock,
        return_value=store,
    ):
        from osprey.mcp_server.channel_finder_hierarchical.tools.view_examples import (
            view_examples,
        )

        fn = get_tool_fn(view_examples)
        result = json.loads(fn(keywords="temperature, cryogenics"))

    assert "keyword_matches" not in result
    assert len(result["all_examples"]) == 2
