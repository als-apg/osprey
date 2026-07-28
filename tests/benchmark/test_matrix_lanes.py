"""Benchmark-lane split tests (scripts/benchmark, tests/e2e lane markers).

The model matrix scores two lanes separately:

  * ``agentic_benchmark`` — model-capability tests (the headline score);
  * ``harness_benchmark`` — model-independent OSPREY safety/plumbing assertions.

Lane membership is declared by pytest markers on the e2e tests; each matrix
cell re-derives a ``*.lanes.json`` manifest from live collection (conftest's
``OSPREY_E2E_LANES`` hook), and the dashboard joins outcomes against it. Two
invariants keep the split trustworthy:

  1. every in-scope e2e test carries exactly one lane marker — gated by
     ``check_e2e_coverage.py --check-lanes`` (also run at matrix-cell startup);
  2. the dashboard's lane join handles both test-id formats and never mistakes
     a lane manifest for a run summary.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_BENCH = _REPO / "scripts" / "benchmark"
_CONFIG = _BENCH / "matrix_e2e_config.json"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _BENCH / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dash = _load("matrix_dashboard")


# ---------------------------------------------------------------------------
# 1. The lane gate: every in-scope e2e test declares its lane
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_every_in_scope_e2e_test_declares_a_benchmark_lane():
    """``check_e2e_coverage.py --check-lanes`` must pass against the real suite.

    This is the CI half of the drift guard (the matrix runner runs the same
    gate per cell): a new e2e file cannot silently join the matrix scope
    without declaring whether its pass/fail measures model capability or
    harness integrity. Collection runs in a subprocess — never in-process
    (nested pytest against tests/e2e/ leaks registry state).
    """
    proc = subprocess.run(
        [
            sys.executable,
            str(_BENCH / "check_e2e_coverage.py"),
            "--repo-root",
            str(_REPO),
            "--config",
            str(_CONFIG),
            "--check-lanes",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        "lane gate failed — an in-scope e2e test is missing (or double-carries) its "
        "agentic_benchmark/harness_benchmark marker:\n" + proc.stderr
    )


# ---------------------------------------------------------------------------
# 2. Dashboard lane join
# ---------------------------------------------------------------------------

_NODEID = "tests/e2e/claude_code/test_sdk_workflows.py::TestClaudeCodeSDKIntegration::test_a"
_JUNIT = "tests.e2e.claude_code.test_sdk_workflows.TestClaudeCodeSDKIntegration::test_a"


def test_load_lanes_joins_manifest_nodeids_to_junit_names(tmp_path):
    """A lane recorded under the pytest NODEID must be found for the same test
    named in JUnit dotted-classname form — the two formats the dashboard
    ingests (live stream vs completed summary)."""
    (tmp_path / "m__seed1.lanes.json").write_text(json.dumps({_NODEID: "agentic"}))
    lanes = dash.load_lanes(str(tmp_path))
    assert lanes[dash.canon_name(_NODEID)] == "agentic"
    assert lanes[dash.canon_name(_JUNIT)] == "agentic"


def test_lane_conclusive_scores_only_the_requested_lane(tmp_path):
    """Capability pass-rate must count agentic-lane outcomes only; harness and
    manifest-absent tests must not pad (or dilute) it. Timeouts count against
    the denominator, skips never do."""
    harness_id = "tests/e2e/claude_code/test_safety_writes.py::test_h"
    stale_id = "tests/e2e/test_gone.py::test_renamed_since"
    (tmp_path / "m__seed1.lanes.json").write_text(
        json.dumps({_NODEID: "agentic", harness_id: "harness"})
    )
    lanes = dash.load_lanes(str(tmp_path))
    cell = {
        "tests": [
            {"name": _JUNIT, "outcome": "passed"},
            {"name": _NODEID.replace("test_a", "test_b"), "outcome": "timeout"},
            {"name": harness_id, "outcome": "passed"},
            {"name": stale_id, "outcome": "failed"},
        ]
    }
    (tmp_path / "x.lanes.json").write_text(
        json.dumps({_NODEID.replace("test_a", "test_b"): "agentic"})
    )
    lanes = dash.load_lanes(str(tmp_path))
    assert dash.lane_conclusive(cell, lanes, "agentic") == (1, 2)  # pass + timeout
    assert dash.lane_conclusive(cell, lanes, "harness") == (1, 1)
    skipped_cell = {"tests": [{"name": _JUNIT, "outcome": "skipped"}]}
    assert dash.lane_conclusive(skipped_cell, lanes, "agentic") == (0, 0)


def test_load_skips_lane_manifests_in_results_dir(tmp_path):
    """``load()`` globs ``*__seed*.json`` — a ``*__seed1.lanes.json`` manifest
    matches that glob and must be skipped, not parsed as a run summary (it has
    no 'model' key and used to crash the dashboard)."""
    (tmp_path / "m__seed1.lanes.json").write_text(json.dumps({_NODEID: "agentic"}))
    (tmp_path / "m__seed1.json").write_text(
        json.dumps(
            {
                "model": "m",
                "seed": 1,
                "passed": 1,
                "failed": 0,
                "timeout": 0,
                "skipped": 0,
                "errors": 0,
                "total": 1,
                "tests": [{"name": _JUNIT, "outcome": "passed", "duration_s": 1.0}],
            }
        )
    )
    runs = dash.load(str(tmp_path))
    assert set(runs) == {("m", 1)}


def test_load_lanes_empty_dir_falls_back(tmp_path):
    """No manifest (results predate the lane split) → empty map, which the
    dashboard renders as the legacy blended single score."""
    assert dash.load_lanes(str(tmp_path)) == {}
