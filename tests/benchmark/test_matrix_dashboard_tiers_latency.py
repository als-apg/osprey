"""Tier labels, latency columns and model rows of the benchmark dashboard
(scripts/benchmark/matrix_dashboard.py)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parents[2] / "scripts" / "benchmark"


def _load_dashboard():
    spec = importlib.util.spec_from_file_location(
        "benchmark_dashboard_tiers", _BENCH / "matrix_dashboard.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_dashboard_tiers"] = mod
    spec.loader.exec_module(mod)
    return mod


dash = _load_dashboard()

_CAP = "tests/e2e/test_cap.py::test_task"
_CAP2 = "tests/e2e/test_cap.py::test_other_task"
_HARNESS = "tests/e2e/test_harness.py::test_hook_blocks"


def _summary(model: str, seed: int, tests: list[dict]) -> dict:
    counts = {"passed": 0, "failed": 0, "timeout": 0, "skipped": 0, "errors": 0}
    for t in tests:
        counts[{"error": "errors"}.get(t["outcome"], t["outcome"])] += 1
    return {
        "model": model,
        "seed": seed,
        "provider": "ds4",
        "route": "proxy",
        **counts,
        "total": len(tests),
        "total_duration_s": int(sum(t["duration_s"] for t in tests)),
        "tests": tests,
    }


def _write_results(results: Path) -> None:
    results.mkdir()
    (results / "m__seed1.lanes.json").write_text(
        json.dumps({_CAP: "agentic", _CAP2: "agentic", _HARNESS: "harness"})
    )
    tests = [
        {"name": _CAP, "outcome": "passed", "duration_s": 100.0},
        {"name": _CAP2, "outcome": "failed", "duration_s": 300.0},
        {"name": _HARNESS, "outcome": "passed", "duration_s": 20.0},
    ]
    (results / "m__seed1.json").write_text(json.dumps(_summary("m", 1, tests)))
    queries = [
        {"test": _CAP, "duration_ms": 100_000, "duration_api_ms": 90_000, "output_tokens": 900},
        {"test": _CAP2, "duration_ms": 300_000, "duration_api_ms": 210_000, "output_tokens": 2100},
    ]
    (results / "m__seed1.queries.jsonl").write_text("".join(json.dumps(q) + "\n" for q in queries))


def test_latency_stats_from_tests_and_query_log(tmp_path):
    results = tmp_path / "results"
    _write_results(results)
    runs = dash.load(str(results))
    lanes = dash.load_lanes(str(results))
    queries = dash.load_query_timing(str(results))

    stats = dash.latency_stats("m", runs, queries, lanes)

    # median over the model's tier-1 tests only (100 s and 300 s), harness excluded
    assert stats["median_task_s"] == 200.0
    # 300 s of 400 s spent waiting on the model
    assert stats["model_share"] == 0.75
    # 3000 output tokens over 300 s of model time
    assert stats["tokens_per_s"] == 10.0


def test_latency_stats_without_query_log_keeps_the_task_time(tmp_path):
    results = tmp_path / "results"
    _write_results(results)
    (results / "m__seed1.queries.jsonl").unlink()
    runs = dash.load(str(results))
    stats = dash.latency_stats(
        "m", runs, dash.load_query_timing(str(results)), dash.load_lanes(str(results))
    )
    assert stats["median_task_s"] == 200.0
    assert stats["model_share"] is None
    assert stats["tokens_per_s"] is None


def test_query_log_is_not_read_as_a_run_summary(tmp_path):
    results = tmp_path / "results"
    _write_results(results)
    assert set(dash.load(str(results))) == {("m", 1)}


def test_rendered_page_names_the_tiers_and_shows_only_models_with_data(tmp_path):
    results = tmp_path / "results"
    _write_results(results)
    out = tmp_path / "dash.html"
    subprocess.run(
        [
            sys.executable,
            str(_BENCH / "matrix_dashboard.py"),
            "--results-dir",
            str(results),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    page = out.read_text()
    assert "Tier 1" in page and "Tier 0" in page
    assert "median task" in page
    # models from earlier runs with no results here get no empty row
    assert "gpt-oss-20b" not in page
    assert "CBORG open models" not in page


def test_running_cell_takes_its_provider_from_the_worker_log(tmp_path):
    """A cell still filling has no summary yet; its provider label must come from
    the worker's banner, not a guess."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "k__seed2.live.jsonl").write_text(
        json.dumps({"name": _CAP, "outcome": "passed", "duration_s": 5.0}) + "\n"
    )
    (results / "k__seed2.run.log").write_text(">> model=k seed=2 provider=ds4 route=proxy\n")
    runs = dash.load(str(results))
    assert dash.provider_of("k", runs) == "ds4"
