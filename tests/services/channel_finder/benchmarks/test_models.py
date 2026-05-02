"""Tests for cross-paradigm benchmark data models."""

from __future__ import annotations

import json

import pytest

from osprey.services.channel_finder.benchmarks.models import (
    BenchmarkRun,
    BenchmarkSuite,
    QueryResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_query_result(
    query_id: int = 0,
    *,
    precision: float = 0.8,
    recall: float = 0.6,
    f1: float = 0.685,
    cost_usd: float = 0.001,
    latency_s: float = 1.5,
) -> QueryResult:
    """Helper to build a QueryResult with sensible defaults."""
    return QueryResult(
        query_id=query_id,
        user_query=f"Find PV for magnet {query_id}",
        expected=["PV:A", "PV:B"],
        predicted=["PV:A", "PV:C"],
        precision=precision,
        recall=recall,
        f1=f1,
        cost_usd=cost_usd,
        latency_s=latency_s,
        eval_meta={"note": "test"},
    )


@pytest.fixture()
def sample_query_result() -> QueryResult:
    return _make_query_result()


@pytest.fixture()
def sample_run() -> BenchmarkRun:
    results = [
        _make_query_result(0, precision=1.0, recall=1.0, f1=1.0, cost_usd=0.01, latency_s=2.0),
        _make_query_result(1, precision=0.5, recall=0.5, f1=0.5, cost_usd=0.02, latency_s=4.0),
    ]
    return BenchmarkRun.from_query_results(
        paradigm="in_context",
        tier=1,
        model="anthropic/claude-haiku",
        results=results,
        channel_count=205,
    )


@pytest.fixture()
def sample_suite() -> BenchmarkSuite:
    """Suite with a sparse set of runs for lookup / table tests."""
    runs: list[BenchmarkRun] = []
    paradigms = ["in_context", "hierarchical", "middle_layer"]
    for p_idx, paradigm in enumerate(paradigms):
        for tier in (1, 2, 3):
            f1 = round(0.1 * (p_idx * 3 + tier), 3)
            qr = _make_query_result(0, f1=f1)
            run = BenchmarkRun.from_query_results(
                paradigm=paradigm,
                tier=tier,
                model="test-model",
                results=[qr],
                channel_count=tier * 200,
            )
            runs.append(run)
    return BenchmarkSuite(
        runs=runs,
        metadata={"description": "unit test suite"},
    )


# ---------------------------------------------------------------------------
# QueryResult
# ---------------------------------------------------------------------------


class TestQueryResult:
    def test_creation(self, sample_query_result: QueryResult) -> None:
        qr = sample_query_result
        assert qr.query_id == 0
        assert qr.user_query == "Find PV for magnet 0"
        assert qr.expected == ["PV:A", "PV:B"]
        assert qr.predicted == ["PV:A", "PV:C"]
        assert qr.precision == 0.8
        assert qr.recall == 0.6
        assert qr.f1 == 0.685
        assert qr.cost_usd == 0.001
        assert qr.num_turns == 1
        assert qr.latency_s == 1.5
        assert qr.eval_meta == {"note": "test"}

    def test_defaults(self) -> None:
        qr = QueryResult(
            query_id=0,
            user_query="q",
            expected=[],
            predicted=[],
            precision=0.0,
            recall=0.0,
            f1=0.0,
        )
        assert qr.cost_usd == 0.0
        assert qr.num_turns == 1
        assert qr.latency_s == 0.0
        assert qr.eval_meta == {}

    def test_dict_round_trip(self, sample_query_result: QueryResult) -> None:
        d = sample_query_result.to_dict()
        restored = QueryResult.from_dict(d)
        assert restored == sample_query_result

    def test_json_file_round_trip(
        self,
        sample_query_result: QueryResult,
        tmp_path,
    ) -> None:
        path = tmp_path / "qr.json"
        sample_query_result.to_json(path)

        # Verify valid JSON on disk
        raw = json.loads(path.read_text())
        assert raw["query_id"] == 0

        restored = QueryResult.from_json(path)
        assert restored == sample_query_result

    def test_from_dict_with_missing_optional_fields(self) -> None:
        """from_dict should tolerate absent optional fields."""
        data = {
            "query_id": 5,
            "user_query": "q",
            "expected": [],
            "predicted": [],
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        qr = QueryResult.from_dict(data)
        assert qr.cost_usd == 0.0
        assert qr.num_turns == 1
        assert qr.latency_s == 0.0
        assert qr.eval_meta == {}


# ---------------------------------------------------------------------------
# BenchmarkRun
# ---------------------------------------------------------------------------


class TestBenchmarkRun:
    def test_from_query_results_aggregates(self, sample_run: BenchmarkRun) -> None:
        run = sample_run
        assert run.paradigm == "in_context"
        assert run.tier == 1
        assert run.model == "anthropic/claude-haiku"
        assert run.channel_count == 205
        assert len(run.query_results) == 2

        # Aggregates: mean of [1.0, 0.5] = 0.75
        assert run.aggregate_f1 == pytest.approx(0.75)
        assert run.aggregate_precision == pytest.approx(0.75)
        assert run.aggregate_recall == pytest.approx(0.75)

        # Cost: sum of [0.01, 0.02] = 0.03
        assert run.total_cost_usd == pytest.approx(0.03)

        # Latency: mean of [2.0, 4.0] = 3.0
        assert run.avg_latency_s == pytest.approx(3.0)

        # Timestamp is an ISO string
        assert "T" in run.timestamp

    def test_from_query_results_empty(self) -> None:
        run = BenchmarkRun.from_query_results(
            paradigm="hierarchical",
            tier=2,
            model="m",
            results=[],
        )
        assert run.aggregate_f1 == 0.0
        assert run.aggregate_precision == 0.0
        assert run.aggregate_recall == 0.0
        assert run.total_cost_usd == 0.0
        assert run.avg_latency_s == 0.0

    def test_dict_round_trip(self, sample_run: BenchmarkRun) -> None:
        d = sample_run.to_dict()
        restored = BenchmarkRun.from_dict(d)
        assert restored == sample_run

    def test_json_file_round_trip(
        self,
        sample_run: BenchmarkRun,
        tmp_path,
    ) -> None:
        path = tmp_path / "run.json"
        sample_run.to_json(path)
        restored = BenchmarkRun.from_json(path)
        assert restored == sample_run

    def test_from_dict_with_missing_optional_fields(self) -> None:
        data = {
            "paradigm": "middle_layer",
            "tier": 3,
            "model": "m",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "query_results": [],
            "aggregate_f1": 0.0,
            "aggregate_precision": 0.0,
            "aggregate_recall": 0.0,
        }
        run = BenchmarkRun.from_dict(data)
        assert run.total_cost_usd == 0.0
        assert run.avg_latency_s == 0.0
        assert run.channel_count == 0
        assert run.num_failed == 0

    def test_num_failed_round_trip(self) -> None:
        """num_failed survives to_dict/from_dict round-trip."""
        run = BenchmarkRun(
            paradigm="in_context",
            tier=1,
            model="m",
            timestamp="2026-01-01T00:00:00+00:00",
            query_results=[],
            aggregate_f1=0.0,
            aggregate_precision=0.0,
            aggregate_recall=0.0,
            num_failed=3,
        )
        d = run.to_dict()
        assert d["num_failed"] == 3
        restored = BenchmarkRun.from_dict(d)
        assert restored.num_failed == 3

    def test_repeat_idx_round_trip(self) -> None:
        """repeat_idx survives to_dict/from_dict round-trip."""
        qr = _make_query_result(0)
        run = BenchmarkRun.from_query_results(
            paradigm="in_context",
            tier=1,
            model="m",
            results=[qr],
            repeat_idx=4,
        )
        assert run.repeat_idx == 4
        restored = BenchmarkRun.from_dict(run.to_dict())
        assert restored.repeat_idx == 4

    def test_repeat_idx_defaults_to_zero_when_absent(self) -> None:
        """Legacy suite JSONs without repeat_idx load cleanly with default 0."""
        data = {
            "paradigm": "in_context",
            "tier": 1,
            "model": "m",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "query_results": [],
            "aggregate_f1": 0.0,
            "aggregate_precision": 0.0,
            "aggregate_recall": 0.0,
        }
        run = BenchmarkRun.from_dict(data)
        assert run.repeat_idx == 0


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------


class TestBenchmarkSuite:
    def test_get_run_found(self, sample_suite: BenchmarkSuite) -> None:
        run = sample_suite.get_run("hierarchical", 2, "test-model")
        assert run is not None
        assert run.paradigm == "hierarchical"
        assert run.tier == 2

    def test_get_run_not_found(self, sample_suite: BenchmarkSuite) -> None:
        assert sample_suite.get_run("in_context", 1, "no-model") is None
        assert sample_suite.get_run("unknown", 1, "test-model") is None

    def test_dict_round_trip(self, sample_suite: BenchmarkSuite) -> None:
        d = sample_suite.to_dict()
        restored = BenchmarkSuite.from_dict(d)
        assert restored == sample_suite

    def test_json_file_round_trip(
        self,
        sample_suite: BenchmarkSuite,
        tmp_path,
    ) -> None:
        path = tmp_path / "suite.json"
        sample_suite.to_json(path)

        raw = json.loads(path.read_text())
        assert "runs" in raw
        assert "metadata" in raw
        assert len(raw["runs"]) == 9  # 3 paradigms x 3 tiers

        restored = BenchmarkSuite.from_json(path)
        assert restored == sample_suite

    def test_metadata_default(self) -> None:
        suite = BenchmarkSuite(runs=[])
        assert suite.metadata == {}
        d = suite.to_dict()
        assert d == {"runs": [], "metadata": {}}
