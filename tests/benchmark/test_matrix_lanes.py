"""Benchmark-lane and run-set tests (scripts/benchmark, tests/e2e lane markers).

Every in-scope e2e test declares one lane:

  * ``agentic`` — an agentic test, with its tier and plain-words description
    (tests/e2e/agentic_tiers.py); the board runs these and only these;
  * ``harness_benchmark`` — an agent may run, but the assertion is OSPREY's.

Lane membership is declared by pytest markers on the e2e tests; each matrix
cell re-derives a ``*.lanes.json`` manifest from live collection (conftest's
``OSPREY_E2E_LANES`` hook), and the dashboard joins outcomes against it. Three
invariants keep the board trustworthy:

  1. every in-scope e2e test carries exactly one lane marker — gated by
     ``check_e2e_coverage.py --check-lanes`` (also run at matrix-cell startup);
  2. a cell runs exactly the in-scope agentic tests, first attempt only;
  3. the dashboard's tier join handles both test-id formats and never mistakes
     a lane manifest for a run summary.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

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


# import-time required because scripts/ is not a package: the dashboard is
# loaded by path and registered in sys.modules so its dataclasses can resolve
# cls.__module__. The module object is what the tests below are written against.
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
        "agentic/harness_benchmark marker:\n" + proc.stderr
    )


class TestLaneGateReadsTheCollectorsVerdict:
    """A collection that was refused cannot be scored from what it left behind.

    The manifest is written by a collection hook that runs even when collection
    itself was refused, so its presence says only that pytest got far enough to
    call the hook. The collector's own exit status is the verdict, and the gate
    reads it.

    Collection runs in a subprocess here too — the fake stands in for the gate
    module's ``subprocess`` attribute, never for ``subprocess.run`` itself,
    which is process-global.
    """

    _NODEID = "tests/e2e/test_stub.py::test_stub"

    def _violations(self, monkeypatch, returncode: int, stderr: str) -> list[str]:
        gate = _load("check_e2e_coverage")

        def _run(cmd, cwd=None, env=None, capture_output=False, text=False):  # noqa: ARG001 - subprocess.run is called with these by keyword
            Path(env["OSPREY_E2E_LANES"]).write_text(
                json.dumps({self._NODEID: {"lane": "agentic", "tier": 1}}), encoding="utf-8"
            )
            return subprocess.CompletedProcess(cmd, returncode, "", stderr)

        monkeypatch.setattr(gate, "subprocess", SimpleNamespace(run=_run))
        return gate.check_lanes(_REPO, {})

    def test_a_collection_that_succeeded_is_scored(self, monkeypatch):
        assert self._violations(monkeypatch, 0, "") == []

    def test_a_collection_that_failed_is_a_violation(self, monkeypatch):
        """A refusal during collection must not read as a clean suite."""
        violations = self._violations(monkeypatch, 4, "UsageError: no provider named")

        assert len(violations) == 1
        assert "rc=4" in violations[0]
        assert "UsageError: no provider named" in violations[0]


# ---------------------------------------------------------------------------
# 2. The board's run set: the agentic tests, first attempt only
# ---------------------------------------------------------------------------


def test_a_cell_selects_the_agentic_tests_with_reruns_off():
    """The selection a cell hands pytest, beside the exclusions."""
    gate = _load("check_e2e_coverage")
    args = gate.run_args({"tests/e2e/test_x.py": "why"})
    assert args == [
        "--ignore=tests/e2e/test_x.py",
        "-m",
        "agentic",
        "-p",
        "no:rerunfailures",
        "-o",
        "timeout_func_only=true",
        "-o",
        "junit_duration_report=call",
    ]


def test_the_timeout_and_the_reported_time_leave_out_a_fixture_wait(tmp_path):
    """A cell can queue for a host-wide stack lock in a fixture for longer than the
    per-test cap. Under the board's options that wait is neither a timeout scored
    against the model nor time reported as the model's."""
    (tmp_path / "test_wait.py").write_text(
        "import time\nimport pytest\n\n\n"
        "@pytest.fixture\ndef queue():\n    time.sleep(2)\n\n\n"
        "def test_task(queue):\n    pass\n"
    )

    def run(*options: str) -> tuple[int, float]:
        report = tmp_path / "junit.xml"
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "test_wait.py", "-q", "-p", "no:cacheprovider"]
            + ["--timeout=1", "--timeout-method=signal", f"--junitxml={report}", *options],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=120,
        )
        case = ET.parse(report).getroot().find(".//testcase")
        return proc.returncode, float(case.get("time"))

    assert run()[0] != 0, "control: without the options the wait trips the timeout"
    code, reported = run("-o", "timeout_func_only=true", "-o", "junit_duration_report=call")
    assert code == 0
    assert reported < 1.0


def _fixture_parameters(module: Path, name: str) -> list[str]:
    tree = ast.parse(module.read_text(encoding="utf-8"))
    function = next(n for n in ast.walk(tree) if getattr(n, "name", None) == name)
    return [a.arg for a in function.args.args]


def test_the_plan_stack_holds_the_host_lock_until_it_is_torn_down():
    """Its project name and ports are fixed, so the deploy fixture must depend on
    the lock's fixture: pytest then tears the stack down before releasing it."""
    module = _REPO / "tests" / "e2e" / "test_plan_stack_agentic.py"
    assert "_host_stack_turn" in _fixture_parameters(module, "deployed_scan_stack")
    source = ast.get_source_segment(
        module.read_text(encoding="utf-8"),
        next(
            n
            for n in ast.walk(ast.parse(module.read_text(encoding="utf-8")))
            if getattr(n, "name", None) == "_host_stack_turn"
        ),
    )
    assert "host_stack_lock(STACK_LOCK)" in source


class TestBoardImagesGate:
    """A full cell refuses to start unless the images it needs were built at this
    revision. The runtime and git are stood in for; no container is touched."""

    ENTRIES: ClassVar = [
        {"image": "va:latest", "override_env": "BOARD_TEST_VA_IMAGE", "build": "b1"},
        {"image": "ioc:latest", "build": "b2"},
    ]

    @pytest.fixture
    def gate(self, monkeypatch):
        gate = _load("check_e2e_coverage")
        ids = {"va:latest": "sha:va", "ioc:latest": "sha:ioc"}
        monkeypatch.setattr(gate, "_image_id", ids.get)
        monkeypatch.setattr(gate, "_revision", lambda _root: "rev1")
        monkeypatch.delenv("BOARD_TEST_VA_IMAGE", raising=False)
        gate.test_ids = ids
        return gate

    def test_recorded_images_at_this_revision_pass(self, gate, tmp_path):
        stamp = tmp_path / "stamp.json"
        assert gate.record_board_images(_REPO, self.ENTRIES, stamp) == []
        assert gate.check_board_images(_REPO, self.ENTRIES, stamp) == []

    def test_a_missing_image_is_named_with_the_build_command(self, gate, tmp_path):
        stamp = tmp_path / "stamp.json"
        gate.record_board_images(_REPO, self.ENTRIES, stamp)
        del gate.test_ids["ioc:latest"]
        problems = gate.check_board_images(_REPO, self.ENTRIES, stamp)
        assert len(problems) == 1
        assert "ioc:latest is not on this host" in problems[0]
        assert gate.BUILD_IMAGES_COMMAND in problems[0]

    def test_an_image_nobody_recorded_is_refused(self, gate, tmp_path):
        problems = gate.check_board_images(_REPO, self.ENTRIES, tmp_path / "none.json")
        assert len(problems) == 2

    def test_an_image_rebuilt_since_the_record_is_refused(self, gate, tmp_path):
        stamp = tmp_path / "stamp.json"
        gate.record_board_images(_REPO, self.ENTRIES, stamp)
        gate.test_ids["va:latest"] = "sha:other"
        assert "va:latest is not the one" in gate.check_board_images(_REPO, self.ENTRIES, stamp)[0]

    def test_images_from_another_revision_are_refused(self, gate, tmp_path, monkeypatch):
        stamp = tmp_path / "stamp.json"
        gate.record_board_images(_REPO, self.ENTRIES, stamp)
        monkeypatch.setattr(gate, "_revision", lambda _root: "rev2")
        problems = gate.check_board_images(_REPO, self.ENTRIES, stamp)
        assert "built at rev1" in problems[0] and "this checkout is rev2" in problems[0]

    def test_the_override_variable_names_the_image_to_check(self, gate, tmp_path, monkeypatch):
        monkeypatch.setenv("BOARD_TEST_VA_IMAGE", "side:build")
        gate.test_ids["side:build"] = "sha:side"
        stamp = tmp_path / "stamp.json"
        gate.record_board_images(_REPO, self.ENTRIES, stamp)
        assert "side:build" in json.loads(stamp.read_text())["images"]

    def test_a_cell_runs_the_gate_unless_it_is_a_smoke_selection(self):
        runner = (_BENCH / "run_e2e_for_model.sh").read_text(encoding="utf-8")
        assert '[ -z "${OSPREY_BENCH_SELECT:-}" ]' in runner
        assert "--check-board-images" in runner


def test_the_config_names_the_images_the_fixtures_require():
    """The config cannot drift from what the switch module's fixtures fail on."""
    from tests.fixtures import bench_ioc
    from tests.va.e2e import _image as va_image

    entries = {e["image"]: e for e in json.loads(_CONFIG.read_text())["board_images"]}
    assert entries[va_image.DEFAULT_IMAGE]["build"] == va_image.BUILD_COMMAND
    assert entries[va_image.DEFAULT_IMAGE]["override_env"] == va_image.IMAGE_ENV
    assert entries[bench_ioc.IMAGE]["build"] == bench_ioc.BUILD_COMMAND


def test_the_cell_runner_takes_its_selection_from_the_gate_script():
    """One source for the selection: the runner splices the script's output in."""
    runner = (_BENCH / "run_e2e_for_model.sh").read_text(encoding="utf-8")
    assert "--print-run-args" in runner
    assert "$RUN_ARGS" in runner


@pytest.mark.slow
def test_the_board_collects_every_in_scope_agentic_test_and_nothing_else(tmp_path):
    """Collected with the board's own arguments, the run set is the in-scope
    tests whose manifest entry is agentic with a tier — no harness check, no
    model-free floor — and every one of them is there."""
    gate = _load("check_e2e_coverage")
    excluded = {e["path"]: e["reason"] for e in json.loads(_CONFIG.read_text())["excluded_files"]}

    def collect(manifest: Path, args: list[str]) -> dict:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/e2e/", "--collect-only", "-q"]
            + ["-p", "no:cacheprovider", "-o", "addopts=", "--assert=plain", *args],
            cwd=_REPO,
            env={**os.environ, "OSPREY_E2E_LANES": str(manifest)},
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stdout[-800:] + proc.stderr[-800:]
        return json.loads(manifest.read_text())

    in_scope = collect(tmp_path / "scope.json", [f"--ignore={p}" for p in excluded])
    selected = collect(tmp_path / "board.json", gate.run_args(excluded))
    agentic = {n for n, e in in_scope.items() if e["lane"] == "agentic"}
    assert set(selected) == agentic
    assert all(selected[n]["tier"] in (1, 2) for n in selected)
    switch = {n for n in selected if "test_target_switch_agentic.py" in n}
    assert len(switch) == 4, "the machine-switch tests are on the board"


# ---------------------------------------------------------------------------
# 3. Dashboard tier join
# ---------------------------------------------------------------------------

_NODEID = "tests/e2e/claude_code/test_sdk_workflows.py::TestClaudeCodeSDKIntegration::test_a"
_JUNIT = "tests.e2e.claude_code.test_sdk_workflows.TestClaudeCodeSDKIntegration::test_a"
_META = {"lane": "agentic", "tier": 1, "title": "t", "asks": "a", "passes_if": "p"}


def test_load_tiers_joins_manifest_nodeids_to_junit_names(tmp_path):
    """A tier recorded under the pytest NODEID must be found for the same test
    named in JUnit dotted-classname form — the two formats the dashboard
    ingests (live stream vs completed summary)."""
    (tmp_path / "m__seed1.lanes.json").write_text(json.dumps({_NODEID: _META}))
    tiers = dash.load_tiers(str(tmp_path))
    assert tiers[dash.canon_name(_NODEID)]["tier"] == 1
    assert tiers[dash.canon_name(_JUNIT)]["title"] == "t"


def test_a_result_takes_its_tier_from_the_manifest_or_its_own_record():
    tiers = {dash.canon_name(_NODEID): _META}
    assert dash.tier_of({"name": _JUNIT}, tiers) == 1
    assert dash.tier_of({"name": "tests.e2e.test_y::test_y", "tier": 2}, tiers) == 2
    assert dash.tier_of({"name": "tests.e2e.test_y::test_y"}, tiers) is None


def test_conclusive_where_scores_only_the_selected_tests():
    """A tier's pass count covers its own tests only. Timeouts count against
    the denominator, skips never do."""
    other = "tests/e2e/test_other.py::test_h"
    cell = {
        "tests": [
            {"name": _JUNIT, "outcome": "passed"},
            {"name": _NODEID.replace("test_a", "test_b"), "outcome": "timeout"},
            {"name": other, "outcome": "passed"},
        ]
    }
    keep = {dash.canon_name(_NODEID), dash.canon_name(_NODEID.replace("test_a", "test_b"))}
    assert dash.conclusive_where(cell, keep.__contains__) == (1, 2)
    skipped_cell = {"tests": [{"name": _JUNIT, "outcome": "skipped"}]}
    assert dash.conclusive_where(skipped_cell, keep.__contains__) == (0, 0)


def test_load_skips_lane_manifests_in_results_dir(tmp_path):
    """``load()`` globs ``*__seed*.json`` — a ``*__seed1.lanes.json`` manifest
    matches that glob and must be skipped, not parsed as a run summary (it has
    no 'model' key and used to crash the dashboard)."""
    (tmp_path / "m__seed1.lanes.json").write_text(json.dumps({_NODEID: _META}))
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


def test_load_tiers_ignores_manifests_from_before_the_tiers(tmp_path):
    """A manifest of bare lane strings (results that predate the tiers) yields
    no tiers, which the dashboard renders as one untiered section."""
    (tmp_path / "m__seed1.lanes.json").write_text(json.dumps({_NODEID: "agentic"}))
    assert dash.load_tiers(str(tmp_path)) == {}
    assert dash.load_tiers(str(tmp_path / "missing")) == {}
