"""How long an e2e query waited on the model (tests/e2e/sdk_helpers.py).

The benchmark dashboard's "in model" column is the share of a query's span
spent waiting on the model. It is measured from the session transcripts, as
the union of each model request's interval, because the SDK's summed request
time counts concurrent requests twice and can exceed the query's wall time.
"""

from __future__ import annotations

from tests.e2e.sdk_helpers import model_wait


def _at(second: float) -> str:
    return f"2026-11-06T12:00:{second:06.3f}+00:00"


def _entry(kind: str, second: float, request: str | None = None) -> dict:
    entry = {"type": kind, "timestamp": _at(second)}
    if request:
        entry["requestId"] = request
    return entry


MAIN = [
    _entry("user", 0),  # the prompt
    _entry("assistant", 2, "r1"),  # 0-3 s: the model answers with a tool call
    _entry("assistant", 3, "r1"),
    _entry("user", 9),  # the tool (a subagent) returns at 9 s
    _entry("assistant", 10, "r2"),  # 9-10 s: the closing answer
]
SUBAGENT = [
    _entry("user", 3),
    _entry("assistant", 6, "s1"),  # 3-6 s: the subagent's request
    _entry("user", 7),
    _entry("assistant", 8, "s2"),  # 7-8 s
]


def test_the_wait_is_the_union_of_request_intervals_over_the_span():
    assert model_wait([MAIN, SUBAGENT]) == (8_000, 10_000)


def test_concurrent_requests_count_once():
    """Two subagents waiting at the same time add no more wait than one, so the
    share never exceeds the span."""
    wait, span = model_wait([MAIN, SUBAGENT, list(SUBAGENT)])
    assert (wait, span) == (8_000, 10_000)
    assert wait <= span


def test_a_request_outside_the_main_span_is_clipped():
    late = [_entry("user", 8.5), _entry("assistant", 14, "late")]
    assert model_wait([MAIN, late]) == (4_500, 10_000)


def test_no_timestamps_measure_nothing():
    assert model_wait([]) is None
    assert model_wait([[{"type": "user"}]]) is None
