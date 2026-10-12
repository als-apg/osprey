"""The shared picture-trace assertions, exercised on hand-built traces.

``tests/e2e/ariel_picture.py`` grades agentic runs by what the tool trace
shows: the seeded picture was viewed, and viewed inside a ``logbook-search``
subagent turn, with no ARIEL tool erring. Those assertions decide a benchmark's
verdict, so they are pinned here without an agent or a stack.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tests.e2e.ariel_picture import (
    LOGBOOK_SUBAGENT,
    VIEW_TOOL,
    assert_no_ariel_errors,
    assert_viewed_in_logbook_turn,
)

ATTACHMENT_ID = "att-1"


def _trace(name: str, **fields: Any) -> SimpleNamespace:
    base = {
        "input": {},
        "tool_use_id": None,
        "parent_tool_use_id": None,
        "result": None,
        "is_error": False,
    }
    return SimpleNamespace(name=name, **{**base, **fields})


def _delegation(subagent: str, tool_use_id: str, result: str = "") -> SimpleNamespace:
    return _trace(
        "Agent", input={"subagent_type": subagent}, tool_use_id=tool_use_id, result=result
    )


def _view(parent: str | None, attachment_id: str = ATTACHMENT_ID) -> SimpleNamespace:
    return _trace(VIEW_TOOL, input={"attachment_id": attachment_id}, parent_tool_use_id=parent)


def _check(traces: list[Any]) -> None:
    assert_viewed_in_logbook_turn(traces, ATTACHMENT_ID, [t.name for t in traces])


def test_view_under_the_delegating_tool_use_passes() -> None:
    _check([_delegation(LOGBOOK_SUBAGENT, "toolu_1"), _view("toolu_1")])


def test_view_under_the_reported_agent_id_passes() -> None:
    traces = [
        _delegation(LOGBOOK_SUBAGENT, "toolu_1", result="agentId: a1b2-c3"),
        _delegation("data-visualizer", "toolu_2"),
        _view("a1b2-c3"),
    ]
    _check(traces)


def test_view_of_another_attachment_fails() -> None:
    traces = [_delegation(LOGBOOK_SUBAGENT, "toolu_1"), _view("toolu_1", "att-other")]
    with pytest.raises(AssertionError, match="never called with the seeded id"):
        _check(traces)


def test_view_without_a_logbook_delegation_fails() -> None:
    traces = [_delegation("data-visualizer", "toolu_2"), _view("toolu_2")]
    with pytest.raises(AssertionError, match="never delegated"):
        _check(traces)


def test_view_in_another_subagents_turn_fails() -> None:
    traces = [
        _delegation(LOGBOOK_SUBAGENT, "toolu_1"),
        _delegation("data-visualizer", "toolu_2"),
        _view("toolu_2"),
    ]
    with pytest.raises(AssertionError, match="ran outside"):
        _check(traces)


def test_view_by_the_main_agent_fails() -> None:
    traces = [_delegation(LOGBOOK_SUBAGENT, "toolu_1"), _view(None)]
    with pytest.raises(AssertionError, match="ran outside"):
        _check(traces)


def test_clean_ariel_results_pass() -> None:
    assert_no_ariel_errors([_trace("mcp__ariel__keyword_search", result='{"entries": []}')])


def test_an_ariel_error_envelope_fails() -> None:
    erred = _trace("mcp__ariel__attachment_view", result='{"error_type": "not_ready"}')
    with pytest.raises(AssertionError, match="mcp__ariel__attachment_view"):
        assert_no_ariel_errors([erred])


def test_an_absent_qmd_sidecar_is_not_an_error() -> None:
    absent = _trace(
        "mcp__ariel__hybrid_search",
        result='{"error_type": "service_unavailable", "message": "qmd sidecar unreachable"}',
    )
    assert_no_ariel_errors([absent])


def test_errors_from_other_servers_are_not_counted() -> None:
    assert_no_ariel_errors([_trace("mcp__controls__archiver_read", is_error=True)])
