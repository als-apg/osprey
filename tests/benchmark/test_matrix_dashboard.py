"""Regression tests for the benchmark dashboard's exclusion handling
(scripts/benchmark/matrix_dashboard.py).

The matrix excludes ~13 e2e files from per-model scoring (matrix_e2e_config.json).
Two mechanisms must agree, or excluded tests leak into the rendered grid:

  1. the RUNNER ``--ignore``s them at collection time (fresh runs never have them);
  2. the DASHBOARD retroactively strips them from STALE result jsons that ran the
     test before it was excluded (``apply_exclusions``).

(2) silently broke for every file under a subdirectory (``claude_code/*``) because
``load_exclusions`` basenamed the path (``test_x.py``) while ``canon_name`` keys on
the path relative to ``tests/e2e`` (``claude_code/test_x.py``) — so the two never
matched and 4 excluded files rendered in the dashboard for months. These tests pin
the invariant against the REAL config so it can't regress.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "scripts" / "benchmark"
_CONFIG = _BENCH / "matrix_e2e_config.json"


def _load_dashboard():
    spec = importlib.util.spec_from_file_location("benchmark_dashboard", _BENCH / "matrix_dashboard.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_dashboard"] = mod
    spec.loader.exec_module(mod)
    return mod


dash = _load_dashboard()
EXCLUDED = json.load(open(_CONFIG))["excluded_files"]


def _name(path: str, fmt: str) -> str:
    """A test id named the way the dashboard ingests it.

    ``junit``  — completed run, dotted classname (``tests.e2e.sub.test_x::test_y``).
    ``nodeid`` — live stream, pytest nodeid (``tests/e2e/sub/test_x.py::test_y``).
    """
    if fmt == "junit":
        dotted = path.replace(".py", "").replace("/", ".")
        return f"{dotted}::test_something[case0]"
    return f"{path}::test_something[case0]"


@pytest.mark.parametrize("fmt", ["junit", "nodeid"])
def test_apply_exclusions_strips_every_configured_file(fmt):
    """Every file in matrix_e2e_config.json — top-level AND nested — must be
    stripped by apply_exclusions, in both id formats. Zero survivors."""
    excluded_tests = [
        {"name": _name(e["path"], fmt), "outcome": "failed", "duration_s": 1.0} for e in EXCLUDED
    ]
    # two genuinely model-driving tests that must SURVIVE the strip
    kept_tests = [
        {"name": _name("test_agent_delegation.py", fmt), "outcome": "passed", "duration_s": 2.0},
        {"name": _name("claude_code/test_agent_safety.py", fmt), "outcome": "passed", "duration_s": 3.0},
    ]
    runs = {
        ("m", 1): {
            "model": "m", "seed": 1,
            "tests": excluded_tests + kept_tests,
            "passed": 2, "failed": len(excluded_tests), "timeout": 0, "skipped": 0,
            "errors": 0, "total": len(excluded_tests) + 2,
        }
    }
    excl_set = {rel for rel, _ in dash.load_exclusions(str(_CONFIG))}
    dash.apply_exclusions(runs, excl_set)

    d = runs[("m", 1)]
    survivors = {dash.canon_name(t["name"])[0] for t in d["tests"]}
    leaked = survivors & {e["path"].replace("tests/e2e/", "", 1) for e in EXCLUDED}
    assert not leaked, f"excluded files still in the grid: {sorted(leaked)}"
    # the two model-driving tests survive, and counts are recomputed to match
    assert d["total"] == 2
    assert d["passed"] == 2
    assert d["failed"] == 0


def test_load_exclusions_keys_match_canon_name():
    """The exclusion key MUST equal what canon_name emits for the same file —
    the invariant whose violation caused the nested-file leak. Checked against
    realistic ids in both formats (dotted junit classname / pytest nodeid)."""
    for rel, _ in dash.load_exclusions(str(_CONFIG)):
        dotted = "tests.e2e." + rel.replace(".py", "").replace("/", ".")
        nodeid = "tests/e2e/" + rel
        assert dash.canon_name(f"{dotted}::test_x")[0] == rel, f"junit key != {rel!r}"
        assert dash.canon_name(f"{nodeid}::test_x")[0] == rel, f"nodeid key != {rel!r}"


def test_missing_config_warns_loudly_not_silently(tmp_path, capsys):
    """A missing/unreadable config must WARN (not silently disable every
    exclusion and let non-model tests flood the grid — the failure mode that
    can make a 'cleaned up' exclusion silently un-apply)."""
    out = dash.load_exclusions(str(tmp_path / "nope.json"))
    assert out == []
    assert "WARNING" in capsys.readouterr().err


def test_dispatch_tutorial_is_excluded_in_config():
    """The specific file that triggered this: it must be in the exclusion set
    (it hangs to the worker timeout under CBORG cells and measures no capability)."""
    rels = {rel for rel, _ in dash.load_exclusions(str(_CONFIG))}
    assert "test_dispatch_tutorial.py" in rels
