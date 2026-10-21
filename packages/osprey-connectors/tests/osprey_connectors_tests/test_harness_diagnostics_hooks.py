"""The harness's CI diagnostics hooks write one record per event.

In a run that collects both suites, the xdist controller relays each worker's
``pytest_runtest_logstart`` and ``logfinish`` through the unfiltered hook relay,
so both conftests' copies of the hooks are called for the same event, one after
the other. These tests drive the hooks the way that relay does and read what
reaches the recorder.
"""

from __future__ import annotations

import pytest

from osprey_connectors_tests._harness import _guards


class _Recorder:
    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def record(self, event: str, *, nodeid: str) -> None:
        self.records.append((event, nodeid))


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """A recorder standing in for the CI one, with no record written yet."""
    fake = _Recorder()
    monkeypatch.setattr(_guards, "_CI_DIAGNOSTICS", fake)
    monkeypatch.setattr(_guards, "_CI_LAST_RECORDED", None)
    return fake


def test_an_event_relayed_to_both_conftests_is_recorded_once(recorder: _Recorder) -> None:
    for hook, nodeid in (
        (_guards.pytest_runtest_logstart, "a.py::t"),
        (_guards.pytest_runtest_logstart, "a.py::t"),
        (_guards.pytest_runtest_logstart, "b.py::t"),
        (_guards.pytest_runtest_logstart, "b.py::t"),
        (_guards.pytest_runtest_logfinish, "a.py::t"),
        (_guards.pytest_runtest_logfinish, "a.py::t"),
    ):
        hook(nodeid)
    assert recorder.records == [
        ("start", "a.py::t"),
        ("start", "b.py::t"),
        ("finish", "a.py::t"),
    ]


def test_a_rerun_logs_a_fresh_start(recorder: _Recorder) -> None:
    """pytest-rerunfailures logs ``start`` again for the rerun; a rerun that
    wedges must still leave an unmatched ``start`` on record."""
    _guards.pytest_runtest_logstart("a.py::t")
    _guards.pytest_runtest_logfinish("a.py::t")
    _guards.pytest_runtest_logstart("a.py::t")
    assert recorder.records == [
        ("start", "a.py::t"),
        ("finish", "a.py::t"),
        ("start", "a.py::t"),
    ]


def test_nothing_is_recorded_without_a_recorder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_guards, "_CI_DIAGNOSTICS", None)
    monkeypatch.setattr(_guards, "_CI_LAST_RECORDED", None)
    _guards.pytest_runtest_logstart("a.py::t")
    assert _guards._CI_LAST_RECORDED is None
