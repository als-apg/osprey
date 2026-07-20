"""E2E benchmark tests for the channel-finder MCP server.

Thin pytest wrapper over ``BenchmarkRunner`` that exercises each of the three
channel-finder paradigms (hierarchical, middle_layer, in_context) against the
preset's channel database. Each project ships the tier-resolved unified query
set (one source of truth, paradigm-resolved at build time), so this test reads
the full materialized queries file and slices the first ``SLICE_SIZE`` queries.

The runner, queries, and evaluator all live in
``osprey.services.channel_finder.benchmarks``; this file only wires them into
pytest, parameterizes per-query result inspection, and asserts aggregate
quality thresholds.

Skip-gating mirrors the safety/SDK suite in this directory: ``pytestmark`` is
module-local in pytest, so conftest's gates do not cascade — each test file
must declare its own.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from osprey.services.channel_finder.benchmarks.models import BenchmarkRun
from osprey.services.channel_finder.benchmarks.runner import BenchmarkRunner
from tests.e2e.sdk_helpers import (
    HAS_SDK,
    init_project,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.e2e_benchmark,
    pytest.mark.slow,
    pytest.mark.requires_als_apg,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
]

PARADIGMS = ("hierarchical", "middle_layer", "in_context")
SLICE_SIZE = 10
F1_THRESHOLD = 0.75
PERFECT_THRESHOLD = 0.80


def perfect_match_rate(run: BenchmarkRun) -> float:
    if not run.query_results:
        return 0.0
    return sum(1 for r in run.query_results if r.f1 == 1.0) / len(run.query_results)


def _resolve_dataset_path(project_dir: Path) -> Path:
    config = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))
    rel = config["channel_finder"]["benchmark"]["dataset_path"]
    path = Path(rel)
    return path if path.is_absolute() else project_dir / path


NEAR_MISS_CATEGORY = "near-miss"


def _slice_indices(dataset_path: Path) -> list[int]:
    """Deterministic, category-stratified slice with a reserved near-miss slot.

    Slot 1 is always a discrimination (``near-miss``) query when the dataset
    ships one, guaranteeing a distractor-family probe (QFA/SHF/SHD/SD) executes
    in the gate. The remaining slots round-robin over the *other* categories in
    sorted order — one query per category per round, in file order within a
    category — until ``SLICE_SIZE`` is reached. With more categories than slots
    (tier 3: 11 categories, 10 slots) exactly one non-near-miss category goes
    unsampled per run; with fewer (tier 1: 6 categories) every category is
    covered. A final file-order fallback fills any slots a degenerate dataset
    (e.g. near-miss only) leaves open, so the slice always reaches
    ``min(SLICE_SIZE, len(queries))`` indices with no duplicates.
    """
    queries = json.loads(dataset_path.read_text(encoding="utf-8"))
    limit = min(SLICE_SIZE, len(queries))

    by_category: dict[str, list[int]] = {}
    for idx, query in enumerate(queries):
        by_category.setdefault(query.get("category", ""), []).append(idx)

    selected: list[int] = []
    seen: set[int] = set()

    def take(idx: int) -> None:
        if idx not in seen:
            seen.add(idx)
            selected.append(idx)

    # Reserve slot 1 for a discrimination query when the dataset ships one.
    near_miss = by_category.get(NEAR_MISS_CATEGORY, [])
    if near_miss:
        take(near_miss[0])

    # Round-robin the remaining categories (near-miss excluded so it cannot
    # crowd out breadth) in sorted order to fill the slice.
    pools = [by_category[cat] for cat in sorted(by_category) if cat != NEAR_MISS_CATEGORY]
    depth = max((len(pool) for pool in pools), default=0)
    for round_idx in range(depth):
        if len(selected) >= limit:
            break
        for pool in pools:
            if round_idx < len(pool):
                take(pool[round_idx])
                if len(selected) >= limit:
                    break

    # Fallback: fill any remaining slots from unused indices in file order.
    if len(selected) < limit:
        for idx in range(len(queries)):
            take(idx)
            if len(selected) >= limit:
                break

    return selected


def _run_paradigm_benchmark(tmp_path_factory, paradigm: str) -> BenchmarkRun:
    tmp = tmp_path_factory.mktemp(f"cf-bench-{paradigm}")
    # Build with provider=als-apg so the gate (ALS_APG_API_KEY) lines up with
    # the routing the test will actually exercise. The runner picks up the
    # provider/wire-id mapping from the project's config.yml — no model knob
    # to plumb through this fixture.
    project_dir = init_project(
        tmp,
        f"cf-bench-{paradigm}",
        provider="als-apg",
        channel_finder_mode=paradigm,
    )

    indices = _slice_indices(_resolve_dataset_path(project_dir))
    runner = BenchmarkRunner(
        project_dir,
        model="als-apg/claude-haiku-4-5-20251001",
        max_concurrent=3,
        max_budget_per_query=0.20,
        use_llm_judge=True,
    )
    return asyncio.run(
        runner.run_queries(
            query_indices=indices,
            output_dir=tmp / "benchmark_results",
        )
    )


@pytest.fixture
def hierarchical_run(tmp_path_factory) -> BenchmarkRun:
    return _run_paradigm_benchmark(tmp_path_factory, "hierarchical")


@pytest.fixture
def middle_layer_run(tmp_path_factory) -> BenchmarkRun:
    return _run_paradigm_benchmark(tmp_path_factory, "middle_layer")


@pytest.fixture
def in_context_run(tmp_path_factory) -> BenchmarkRun:
    return _run_paradigm_benchmark(tmp_path_factory, "in_context")


# ---------------------------------------------------------------------------
# Aggregate tests — load-bearing assertions per project_channel_finder_
# benchmark_thresholds memory: target ~90%, assert at 80%; on a miss,
# investigate dataset/DB drift, do not lower the threshold.
#
# Per-query F1 is surfaced via review.html / BenchmarkRun output files; it is
# intentionally NOT asserted in pytest. Earlier `f1 >= 0.0` per-query asserts
# were tautological and created false-positive CI signal.
#
# reruns=2: the run is a real LLM agent, so the perfect-match rate has draw
# variance around the 0.80 bar. The `*_run` fixtures are function-scoped so each
# rerun re-executes the benchmark (a fresh draw); CI only turns red if all three
# attempts miss. The threshold is unchanged — a genuine (reproducible) regression
# still fails every attempt; only single-draw variance is absorbed.
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=2)
def test_hierarchical_aggregate(hierarchical_run: BenchmarkRun) -> None:
    assert hierarchical_run.aggregate_f1 >= F1_THRESHOLD, (
        f"aggregate_f1={hierarchical_run.aggregate_f1:.3f} below {F1_THRESHOLD}"
    )
    pmr = perfect_match_rate(hierarchical_run)
    assert pmr >= PERFECT_THRESHOLD, f"perfect_match_rate={pmr:.3f} below {PERFECT_THRESHOLD}"


@pytest.mark.flaky(reruns=2)
def test_middle_layer_aggregate(middle_layer_run: BenchmarkRun) -> None:
    assert middle_layer_run.aggregate_f1 >= F1_THRESHOLD, (
        f"aggregate_f1={middle_layer_run.aggregate_f1:.3f} below {F1_THRESHOLD}"
    )
    pmr = perfect_match_rate(middle_layer_run)
    assert pmr >= PERFECT_THRESHOLD, f"perfect_match_rate={pmr:.3f} below {PERFECT_THRESHOLD}"


@pytest.mark.flaky(reruns=2)
def test_in_context_aggregate(in_context_run: BenchmarkRun) -> None:
    assert in_context_run.aggregate_f1 >= F1_THRESHOLD, (
        f"aggregate_f1={in_context_run.aggregate_f1:.3f} below {F1_THRESHOLD}"
    )
    pmr = perfect_match_rate(in_context_run)
    assert pmr >= PERFECT_THRESHOLD, f"perfect_match_rate={pmr:.3f} below {PERFECT_THRESHOLD}"


# ---------------------------------------------------------------------------
# Hermetic unit tests for the slice selector. Pure function, no live run — they
# pin the reserved near-miss slot and the deterministic stratification so a
# discrimination query is guaranteed in the gate slice by construction. Live
# scoring happens in the aggregate lanes above.
# ---------------------------------------------------------------------------


def _write_dataset(tmp_path: Path, categories: list[tuple[str, int]]) -> tuple[Path, list[dict]]:
    """Write a synthetic queries file; return (path, queries) in file order."""
    queries: list[dict] = []
    for category, count in categories:
        for k in range(count):
            queries.append(
                {
                    "user_query": f"{category}-{k}",
                    "targeted_pv": [f"PV:{category}:{k:02d}"],
                    "category": category,
                }
            )
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(queries), encoding="utf-8")
    return path, queries


# An 11-category shape mirroring the shipped tier-3 query set (near-miss is the
# largest category since it holds the discrimination queries).
_TIER3_SHAPE = [
    ("aggregate", 4),
    ("ambiguous", 3),
    ("cross-ring", 5),
    ("device-specific", 7),
    ("multi-target", 7),
    ("near-miss", 11),
    ("range", 4),
    ("sector-based", 3),
    ("semantic", 3),
    ("single-target", 9),
    ("tier-boundary", 4),
]

# A 6-category shape mirroring the leaner tier-1 query set.
_TIER1_SHAPE = [
    ("aggregate", 3),
    ("device-specific", 4),
    ("multi-target", 3),
    ("near-miss", 3),
    ("range", 3),
    ("single-target", 4),
]


def test_slice_reserves_near_miss_slot(tmp_path: Path) -> None:
    path, queries = _write_dataset(tmp_path, _TIER3_SHAPE)
    indices = _slice_indices(path)

    assert queries[indices[0]]["category"] == NEAR_MISS_CATEGORY
    # Exactly one near-miss query is reserved — it must not crowd out breadth.
    near_miss_hits = [i for i in indices if queries[i]["category"] == NEAR_MISS_CATEGORY]
    assert len(near_miss_hits) == 1


def test_slice_is_full_and_unique(tmp_path: Path) -> None:
    path, _ = _write_dataset(tmp_path, _TIER3_SHAPE)
    indices = _slice_indices(path)

    assert len(indices) == SLICE_SIZE
    assert len(set(indices)) == len(indices)


def test_slice_tier3_leaves_one_category_unsampled(tmp_path: Path) -> None:
    path, queries = _write_dataset(tmp_path, _TIER3_SHAPE)
    indices = _slice_indices(path)

    sampled = {queries[i]["category"] for i in indices}
    all_categories = {cat for cat, _ in _TIER3_SHAPE}
    # 11 categories, 10 slots, near-miss reserved -> exactly one non-near-miss
    # category goes unsampled.
    assert NEAR_MISS_CATEGORY in sampled
    unsampled = all_categories - sampled
    assert unsampled == {"tier-boundary"}, unsampled


def test_slice_tier1_covers_every_category(tmp_path: Path) -> None:
    path, queries = _write_dataset(tmp_path, _TIER1_SHAPE)
    indices = _slice_indices(path)

    sampled = {queries[i]["category"] for i in indices}
    assert sampled == {cat for cat, _ in _TIER1_SHAPE}
    assert len(indices) == SLICE_SIZE


def test_slice_is_deterministic(tmp_path: Path) -> None:
    path, _ = _write_dataset(tmp_path, _TIER3_SHAPE)
    assert _slice_indices(path) == _slice_indices(path)


def test_slice_without_near_miss_category(tmp_path: Path) -> None:
    shape = [(cat, cnt) for cat, cnt in _TIER3_SHAPE if cat != NEAR_MISS_CATEGORY]
    path, queries = _write_dataset(tmp_path, shape)
    indices = _slice_indices(path)

    assert len(indices) == SLICE_SIZE
    assert len(set(indices)) == len(indices)
    assert all(queries[i]["category"] != NEAR_MISS_CATEGORY for i in indices)


def test_slice_smaller_than_slice_size(tmp_path: Path) -> None:
    path, queries = _write_dataset(tmp_path, [("near-miss", 2), ("range", 3)])
    indices = _slice_indices(path)

    assert len(indices) == len(queries)
    assert len(set(indices)) == len(indices)
    assert queries[indices[0]]["category"] == NEAR_MISS_CATEGORY
