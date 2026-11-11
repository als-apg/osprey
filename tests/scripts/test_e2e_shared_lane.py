"""Tests for ``scripts/ci/e2e_shared_lane.py``: the shared e2e lane's ``--ignore``
arguments and the compose projects its cleanup step removes.

Which module runs where is held by the exactly-one-place invariant in
``tests/scripts/test_check.py``; this file holds the script's own output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._by_path import load_module

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "e2e_shared_lane.py"
lane = load_module("e2e_shared_lane", _SCRIPT)
check = lane.check
fw = lane.FW
FILES = check.repository_files()


def test_ignores_is_one_argument_per_module_another_host_runs() -> None:
    hosts = fw.e2e_hosts(check.LANES, FILES)
    args = lane.ignores(FILES)
    assert args == sorted(args)
    assert {arg.removeprefix("--ignore=") for arg in args} == {
        path for path, host in hosts.items() if host != (fw.SHARED_E2E_LANE,)
    }
    assert "--ignore=tests/e2e/test_two_shape_boot.py" in args
    assert "--ignore=tests/e2e/test_watchdog_live.py" in args


def test_ignores_prints_the_arguments_and_names_each_host(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert lane.main(["ignores"]) == 0
    out, err = capsys.readouterr()
    assert out.splitlines() == lane.ignores(FILES)
    assert f"e2e-tests leaves {len(out.splitlines())} modules to their hosts:" in err
    assert "  tests/e2e/test_two_shape_boot.py  (two-shape-boot-e2e)" in err.splitlines()


@pytest.mark.parametrize(
    ("e2e_filter", "projects"),
    [
        ("tests/e2e/test_single_user_narrowing_e2e.py -k narrowing", ["suproj"]),
        ("tests/e2e/test_dispatch_deploy.py::test_x", ["osprey-e2e-dispatch"]),
        ("tests/e2e/tier2/", ["chart-value-reading", "pv-directory-agentic"]),
        ("tests/va/e2e/test_bluesky_lanes.py", ["lane-e2e"]),
        ("tests/e2e/test_rf_cavity_correlation_scenario.py", []),
        ("-k nothing_is_a_path", []),
    ],
)
def test_stacks_are_the_projects_of_the_filtered_modules(
    e2e_filter: str, projects: list[str]
) -> None:
    assert lane.stacks(FILES, e2e_filter) == projects


def test_the_shared_sweep_brings_up_no_stack_a_lane_owns() -> None:
    """With no filter the cleanup names only the shared sweep's own projects,
    never one a dedicated lane's module deploys."""
    owned = lane.stacks(FILES, " ".join(fw.e2e_shared_ignores(check.LANES, FILES)))
    assert owned, "no dedicated module names a PROJECT_NAME; the probe is stale"
    assert set(lane.stacks(FILES, "")).isdisjoint(owned)


def test_stacks_prints_nothing_when_no_module_names_a_project(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty line would reach the cleanup script as one empty project name."""
    monkeypatch.setenv("E2E_FILTER", "tests/e2e/test_rf_cavity_correlation_scenario.py")
    assert lane.main(["stacks"]) == 0
    assert capsys.readouterr().out == ""
