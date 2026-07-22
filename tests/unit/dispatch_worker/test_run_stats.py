"""Tests for the dispatch-worker per-run execution-stats map.

Contracts:
  - ``increment_tool_calls`` lazily creates a run's entry and accumulates;
  - ``get_run_stats`` returns the *live* entry (latest count) or a zeroed
    default without creating an entry;
  - ``pop_run_stats`` removes and returns the entry, or a zeroed default when
    the run made no tool calls.
"""

from __future__ import annotations

import pytest

from osprey.mcp_server.dispatch_worker import run_stats


@pytest.fixture(autouse=True)
def _clear_map():
    """Clear the module-level map around each test (serial-safe isolation)."""
    run_stats._run_stats.clear()
    yield
    run_stats._run_stats.clear()


@pytest.mark.unit
def test_increment_creates_entry_lazily():
    run_stats.increment_tool_calls("run-1")
    run_stats.increment_tool_calls("run-1")
    assert run_stats.get_run_stats("run-1") == {"num_tool_calls": 2}


@pytest.mark.unit
def test_get_run_stats_default_does_not_create_entry():
    """Reading an unknown run yields a zeroed default and leaves the map empty."""
    assert run_stats.get_run_stats("ghost") == {"num_tool_calls": 0}
    assert "ghost" not in run_stats._run_stats


@pytest.mark.unit
def test_get_run_stats_returns_live_entry():
    """The returned dict is the live entry, so later increments are visible."""
    run_stats.increment_tool_calls("run-2")
    stats = run_stats.get_run_stats("run-2")
    run_stats.increment_tool_calls("run-2")
    assert stats["num_tool_calls"] == 2


@pytest.mark.unit
def test_runs_are_independent():
    run_stats.increment_tool_calls("a")
    run_stats.increment_tool_calls("b")
    run_stats.increment_tool_calls("b")
    assert run_stats.get_run_stats("a") == {"num_tool_calls": 1}
    assert run_stats.get_run_stats("b") == {"num_tool_calls": 2}


@pytest.mark.unit
def test_pop_removes_and_returns():
    run_stats.increment_tool_calls("run-3")
    popped = run_stats.pop_run_stats("run-3")
    assert popped == {"num_tool_calls": 1}
    assert "run-3" not in run_stats._run_stats


@pytest.mark.unit
def test_pop_missing_returns_zeroed_default():
    """A run that made no tool calls pops cleanly to a zeroed default."""
    assert run_stats.pop_run_stats("never-ran") == {"num_tool_calls": 0}
