"""``read_cypher``'s guidance when a result fills the query's own LIMIT.

The server-side row cap flags truncation only when the *store* had more rows
than the cap (``truncated = len(fetched) > cap``). A query that carries its own
``LIMIT n`` and matches more than ``n`` rows comes back with exactly ``n`` rows
and ``truncated: false`` — a clipped answer labeled complete. The agent then
has no signal that "all of X" was cut short, and a benchmark run showed it
presenting the clipped list as a facility census.

These tests pin the honest contract: a result whose row count equals the
query's own literal ``LIMIT`` carries ``guidance`` saying completeness is
unverified; every other untruncated result stays clean.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from osprey.mcp_server.graph.server_context import QueryResult
from osprey.mcp_server.graph.tools import read_cypher as read_cypher_module

_ROW = {"pv": "SR:DIAG:BPM:01:GOLDEN:X"}


class _FakeContext:
    """Just enough of GraphContext for the tool's success path."""

    query_max_rows = 200

    def __init__(self, rows: list[dict[str, Any]], truncated: bool) -> None:
        self._result = QueryResult(rows=rows, truncated=truncated)

    def run_read(self, query: str, params: dict[str, Any] | None = None) -> QueryResult:
        return self._result

    def is_empty(self) -> bool:
        return False


def _call(monkeypatch: pytest.MonkeyPatch, query: str, rows: int, truncated: bool = False) -> dict:
    ctx = _FakeContext([dict(_ROW) for _ in range(rows)], truncated)
    monkeypatch.setattr(read_cypher_module, "get_server_context", lambda: ctx)
    fn = getattr(read_cypher_module.read_cypher, "fn", read_cypher_module.read_cypher)
    return json.loads(fn(query))


def test_row_count_equal_to_the_querys_own_limit_carries_guidance(monkeypatch) -> None:
    payload = _call(monkeypatch, "MATCH (b:ChannelBinding) RETURN b.fullPv AS pv LIMIT 3", rows=3)

    assert payload["truncated"] is False
    guidance = " ".join(payload.get("guidance", []))
    assert "LIMIT 3" in guidance, payload
    assert "count" in guidance.lower(), "the remedy must name a count() check"


def test_fewer_rows_than_the_limit_stays_clean(monkeypatch) -> None:
    payload = _call(monkeypatch, "MATCH (b:ChannelBinding) RETURN b.fullPv AS pv LIMIT 10", rows=3)

    assert "guidance" not in payload, payload


def test_a_query_without_a_limit_stays_clean(monkeypatch) -> None:
    payload = _call(monkeypatch, "MATCH (b:ChannelBinding) RETURN b.fullPv AS pv", rows=3)

    assert "guidance" not in payload, payload


def test_the_final_limit_is_the_one_that_counts(monkeypatch) -> None:
    """An inner subquery LIMIT must not trigger the hint on a coincidental count."""
    query = (
        "CALL { MATCH (c:Class) RETURN c LIMIT 1 }\n"
        "MATCH (b:ChannelBinding) RETURN b.fullPv AS pv LIMIT 50"
    )
    payload = _call(monkeypatch, query, rows=1)

    assert "guidance" not in payload, payload


def test_limit_one_is_an_intentional_single_row_and_stays_clean(monkeypatch) -> None:
    """Aggregates and existence probes end in LIMIT 1 and always fill it."""
    payload = _call(monkeypatch, "MATCH (d:Resource) RETURN count(d) AS n LIMIT 1", rows=1)

    assert "guidance" not in payload, payload


def test_server_cap_truncation_keeps_the_cap_guidance(monkeypatch) -> None:
    """When the store itself truncates, the cap message wins — no double hint."""
    payload = _call(
        monkeypatch,
        "MATCH (b:ChannelBinding) RETURN b.fullPv AS pv LIMIT 200",
        rows=200,
        truncated=True,
    )

    guidance = payload.get("guidance", [])
    assert len(guidance) == 1, payload
    assert "query_max_rows" in guidance[0], payload
