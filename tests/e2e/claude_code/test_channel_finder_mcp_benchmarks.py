"""E2E benchmark tests for the channel-finder MCP server.

Thin pytest wrapper over ``BenchmarkRunner`` that exercises each of the three
channel-finder paradigms (hierarchical, middle_layer, in_context) against the
preset's full channel database, using a tier-1 query slice from the shipped
benchmark datasets.

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
    is_claude_code_available,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.e2e_benchmark,
    pytest.mark.slow,
    pytest.mark.requires_als_apg,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
    pytest.mark.skipif(not is_claude_code_available(), reason="Claude Code CLI not installed"),
]

PARADIGMS = ("hierarchical", "middle_layer", "in_context")
TIER = 1
SLICE_SIZE = 10
F1_THRESHOLD = 0.75
PERFECT_THRESHOLD = 0.80


def perfect_match_rate(run: BenchmarkRun) -> float:
    if not run.query_results:
        return 0.0
    return sum(1 for r in run.query_results if r.f1 == 1.0) / len(run.query_results)


def _resolve_dataset_path(project_dir: Path, paradigm: str) -> Path:
    config = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))
    rel = config["channel_finder"]["pipelines"][paradigm]["benchmark"]["dataset_path"]
    path = Path(rel)
    return path if path.is_absolute() else project_dir / path


def _tier_slice_indices(dataset_path: Path) -> list[int]:
    queries = json.loads(dataset_path.read_text(encoding="utf-8"))
    return [i for i, q in enumerate(queries) if q.get("tier") == TIER][:SLICE_SIZE]


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

    indices = _tier_slice_indices(_resolve_dataset_path(project_dir, paradigm))
    runner = BenchmarkRunner(
        project_dir,
        model="als-apg/claude-haiku-4-5-20251001",
        max_concurrent=3,
        max_budget_per_query=0.20,
    )
    return asyncio.run(
        runner.run_queries(
            query_indices=indices,
            output_dir=tmp / "benchmark_results",
        )
    )


@pytest.fixture(scope="module")
def hierarchical_run(tmp_path_factory) -> BenchmarkRun:
    return _run_paradigm_benchmark(tmp_path_factory, "hierarchical")


@pytest.fixture(scope="module")
def middle_layer_run(tmp_path_factory) -> BenchmarkRun:
    return _run_paradigm_benchmark(tmp_path_factory, "middle_layer")


@pytest.fixture(scope="module")
def in_context_run(tmp_path_factory) -> BenchmarkRun:
    return _run_paradigm_benchmark(tmp_path_factory, "in_context")


# ---------------------------------------------------------------------------
# Per-query informational tests — surface individual results in pytest output
# without gating CI on noisy single-query outcomes. Real bar is on aggregates.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("idx", range(SLICE_SIZE))
def test_hierarchical_per_query(hierarchical_run: BenchmarkRun, idx: int) -> None:
    assert hierarchical_run.query_results[idx].f1 >= 0.0


@pytest.mark.parametrize("idx", range(SLICE_SIZE))
def test_middle_layer_per_query(middle_layer_run: BenchmarkRun, idx: int) -> None:
    assert middle_layer_run.query_results[idx].f1 >= 0.0


@pytest.mark.parametrize("idx", range(SLICE_SIZE))
def test_in_context_per_query(in_context_run: BenchmarkRun, idx: int) -> None:
    assert in_context_run.query_results[idx].f1 >= 0.0


# ---------------------------------------------------------------------------
# Aggregate tests — load-bearing assertions per project_channel_finder_
# benchmark_thresholds memory: target ~90%, assert at 80%; on a miss,
# investigate dataset/DB drift, do not lower the threshold.
# ---------------------------------------------------------------------------


def test_hierarchical_aggregate(hierarchical_run: BenchmarkRun) -> None:
    assert hierarchical_run.aggregate_f1 >= F1_THRESHOLD, (
        f"aggregate_f1={hierarchical_run.aggregate_f1:.3f} below {F1_THRESHOLD}"
    )
    pmr = perfect_match_rate(hierarchical_run)
    assert pmr >= PERFECT_THRESHOLD, f"perfect_match_rate={pmr:.3f} below {PERFECT_THRESHOLD}"


def test_middle_layer_aggregate(middle_layer_run: BenchmarkRun) -> None:
    assert middle_layer_run.aggregate_f1 >= F1_THRESHOLD, (
        f"aggregate_f1={middle_layer_run.aggregate_f1:.3f} below {F1_THRESHOLD}"
    )
    pmr = perfect_match_rate(middle_layer_run)
    assert pmr >= PERFECT_THRESHOLD, f"perfect_match_rate={pmr:.3f} below {PERFECT_THRESHOLD}"


def test_in_context_aggregate(in_context_run: BenchmarkRun) -> None:
    assert in_context_run.aggregate_f1 >= F1_THRESHOLD, (
        f"aggregate_f1={in_context_run.aggregate_f1:.3f} below {F1_THRESHOLD}"
    )
    pmr = perfect_match_rate(in_context_run)
    assert pmr >= PERFECT_THRESHOLD, f"perfect_match_rate={pmr:.3f} below {PERFECT_THRESHOLD}"
