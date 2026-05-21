"""Data models for channel finder benchmarks.

Defines dataclasses for benchmark results at three levels of granularity:
  - QueryResult: single query evaluation
  - BenchmarkRun: one benchmark execution (one model, one query set)
  - BenchmarkSuite: bundle of runs with shared metadata, JSON round-trip

Cross-paradigm matrix aggregation (grouping runs by paradigm/tier/model/
backend, computing mean/std across repeats, cost inference from tokens)
lives in the companion paper repository — it is paper-experiment-specific
and not part of osprey's single-paradigm benchmark surface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass
class QueryResult:
    """Evaluation result for a single query in a single benchmark cell."""

    query_id: int
    user_query: str
    expected: list[str]
    predicted: list[str]
    precision: float
    recall: float
    f1: float
    cost_usd: float = 0.0
    num_turns: int = 1
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    eval_meta: dict[str, Any] = field(default_factory=dict)
    # SDK session traces for post-hoc analysis
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    response_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "query_id": self.query_id,
            "user_query": self.user_query,
            "expected": self.expected,
            "predicted": self.predicted,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "cost_usd": self.cost_usd,
            "num_turns": self.num_turns,
            "latency_s": self.latency_s,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "eval_meta": self.eval_meta,
            "tool_calls": self.tool_calls,
            "response_text": self.response_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueryResult:
        """Reconstruct a QueryResult from a dictionary."""
        return cls(
            query_id=data["query_id"],
            user_query=data["user_query"],
            expected=data["expected"],
            predicted=data["predicted"],
            precision=data["precision"],
            recall=data["recall"],
            f1=data["f1"],
            cost_usd=data.get("cost_usd", 0.0),
            num_turns=data.get("num_turns", 1),
            latency_s=data.get("latency_s", 0.0),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            eval_meta=data.get("eval_meta", {}),
            tool_calls=data.get("tool_calls", []),
            response_text=data.get("response_text", ""),
        )

    def to_json(self, path: Path) -> None:
        """Write to a JSON file with indent=2."""
        path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path) -> QueryResult:
        """Read from a JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass
class BenchmarkRun:
    """One benchmark execution: aggregated query results for a single model.

    ``paradigm`` and ``tier`` are optional labels consumed by the companion
    paper repo's matrix aggregation to group runs across pipeline modes and
    database tiers. OSPREY's single-project benchmark flow leaves them as
    ``None``; the configured pipeline and database are recorded elsewhere.
    """

    model: str
    timestamp: str
    query_results: list[QueryResult]
    aggregate_f1: float
    aggregate_precision: float
    aggregate_recall: float
    total_cost_usd: float = 0.0
    avg_latency_s: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    channel_count: int = 0
    num_failed: int = 0
    backend: str = "sdk"
    repeat_idx: int = 0
    # Paradigm / tier are optional labels. Kept after repeat_idx so the
    # invariant "no required field follows a defaulted one" holds.
    paradigm: str | None = None
    tier: int | None = None

    @classmethod
    def from_query_results(
        cls,
        model: str,
        results: list[QueryResult],
        *,
        paradigm: str | None = None,
        tier: int | None = None,
        channel_count: int = 0,
        num_failed: int = 0,
        backend: str = "sdk",
        repeat_idx: int = 0,
    ) -> BenchmarkRun:
        """Factory that computes aggregates from query_results.

        Generates the timestamp automatically and derives aggregate
        metrics (mean F1, precision, recall, total cost, mean latency)
        from the provided per-query results.
        """
        if results:
            agg_f1 = mean(r.f1 for r in results)
            agg_precision = mean(r.precision for r in results)
            agg_recall = mean(r.recall for r in results)
            total_cost = sum(r.cost_usd for r in results)
            avg_latency = mean(r.latency_s for r in results)
            total_in = sum(r.input_tokens for r in results)
            total_out = sum(r.output_tokens for r in results)
            avg_in = mean(r.input_tokens for r in results)
            avg_out = mean(r.output_tokens for r in results)
        else:
            agg_f1 = 0.0
            agg_precision = 0.0
            agg_recall = 0.0
            total_cost = 0.0
            avg_latency = 0.0
            total_in = 0
            total_out = 0
            avg_in = 0.0
            avg_out = 0.0

        return cls(
            paradigm=paradigm,
            tier=tier,
            model=model,
            timestamp=datetime.now(UTC).isoformat(),
            query_results=results,
            aggregate_f1=agg_f1,
            aggregate_precision=agg_precision,
            aggregate_recall=agg_recall,
            total_cost_usd=total_cost,
            avg_latency_s=avg_latency,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            avg_input_tokens=avg_in,
            avg_output_tokens=avg_out,
            channel_count=channel_count,
            num_failed=num_failed,
            backend=backend,
            repeat_idx=repeat_idx,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "paradigm": self.paradigm,
            "tier": self.tier,
            "model": self.model,
            "timestamp": self.timestamp,
            "query_results": [r.to_dict() for r in self.query_results],
            "aggregate_f1": self.aggregate_f1,
            "aggregate_precision": self.aggregate_precision,
            "aggregate_recall": self.aggregate_recall,
            "total_cost_usd": self.total_cost_usd,
            "avg_latency_s": self.avg_latency_s,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "avg_input_tokens": self.avg_input_tokens,
            "avg_output_tokens": self.avg_output_tokens,
            "channel_count": self.channel_count,
            "num_failed": self.num_failed,
            "backend": self.backend,
            "repeat_idx": self.repeat_idx,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkRun:
        """Reconstruct a BenchmarkRun from a dictionary."""
        return cls(
            paradigm=data.get("paradigm"),
            tier=data.get("tier"),
            model=data["model"],
            timestamp=data["timestamp"],
            query_results=[QueryResult.from_dict(r) for r in data["query_results"]],
            aggregate_f1=data["aggregate_f1"],
            aggregate_precision=data["aggregate_precision"],
            aggregate_recall=data["aggregate_recall"],
            total_cost_usd=data.get("total_cost_usd", 0.0),
            avg_latency_s=data.get("avg_latency_s", 0.0),
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            avg_input_tokens=data.get("avg_input_tokens", 0.0),
            avg_output_tokens=data.get("avg_output_tokens", 0.0),
            channel_count=data.get("channel_count", 0),
            num_failed=data.get("num_failed", 0),
            backend=data.get("backend", "sdk"),
            repeat_idx=data.get("repeat_idx", 0),
        )

    def to_json(self, path: Path) -> None:
        """Write to a JSON file with indent=2."""
        path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path) -> BenchmarkRun:
        """Read from a JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass
class BenchmarkSuite:
    """A bundle of ``BenchmarkRun`` instances with metadata and JSON round-trip.

    Cross-paradigm matrix aggregation (mean/std across repeats, cost
    inference from tokens) lives in the companion paper repository.
    """

    runs: list[BenchmarkRun]
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_run(
        self,
        paradigm: str,
        tier: int,
        model: str,
    ) -> BenchmarkRun | None:
        """Find a specific run by paradigm, tier, and model.

        Returns the first matching BenchmarkRun, or None if no match.
        """
        for run in self.runs:
            if run.paradigm == paradigm and run.tier == tier and run.model == model:
                return run
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "runs": [r.to_dict() for r in self.runs],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkSuite:
        """Reconstruct a BenchmarkSuite from a dictionary."""
        return cls(
            runs=[BenchmarkRun.from_dict(r) for r in data["runs"]],
            metadata=data.get("metadata", {}),
        )

    def to_json(self, path: Path) -> None:
        """Write to a JSON file with indent=2."""
        path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path) -> BenchmarkSuite:
        """Read from a JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)
