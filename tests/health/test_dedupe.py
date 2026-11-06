"""One shared resource, one row: duplicate probes fold and name their consumers."""

from __future__ import annotations

from osprey.health.dedupe import fold_shared
from osprey.health.models import CheckResult, Status


def _row(name: str, category: str, status: Status = Status.OK, message: str = "m") -> CheckResult:
    return CheckResult(name, category, status, message)


def test_the_graph_store_probes_fold_into_the_canonical_row():
    rows = [
        _row("a", "configuration"),
        _row("channel_finder_store", "channel_finder"),
        _row("graphdb_connection", "graphdb", message="Graph store reachable"),
        _row("reach.graphdb", "reach", message="graph MCP server: reachable at localhost:1"),
    ]
    folded = fold_shared(rows)

    assert [r.name for r in folded] == ["a", "graphdb_connection"]
    kept = folded[1]
    assert kept.message == "Graph store reachable"
    assert kept.used_by == ("graph store service", "channel finder", "graph MCP server")


def test_the_worst_probe_is_kept_when_they_disagree():
    rows = [
        _row("graphdb_connection", "graphdb"),
        _row("reach.graphdb", "reach", Status.WARNING, "graph MCP server: unreachable"),
    ]
    (kept,) = fold_shared(rows)

    assert kept.status is Status.WARNING
    assert kept.message == "graph MCP server: unreachable"
    assert kept.used_by == ("graph store service", "graph MCP server")


def test_a_single_present_row_is_untouched():
    rows = [_row("graphdb_connection", "graphdb"), _row("x", "y")]
    assert fold_shared(rows) == rows
    assert fold_shared(rows)[0].used_by == ()


def test_used_by_reaches_the_wire():
    (kept,) = fold_shared(
        [_row("qmd_okf", "qmd"), _row("reach.qmd-okf", "reach", message="OKF: up")]
    )
    assert kept.to_dict()["used_by"] == ["qmd index", "OKF"]
