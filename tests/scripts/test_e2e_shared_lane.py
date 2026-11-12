"""Tests for ``scripts/ci/e2e_shared_lane.py``: the shared e2e lane's ``--ignore``
arguments, the compose projects its cleanup step removes, and the module list a
dedicated lane's run step takes from its ``Lane``.

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


def test_modules_are_the_lanes_own_paths_in_declaration_order() -> None:
    (no_model,) = [entry for entry in check.LANES if entry.id == "e2e-no-model"]
    assert no_model.modules
    assert lane.modules("e2e-no-model") == list(no_model.modules)


def test_modules_keeps_a_path_that_is_gone() -> None:
    """Resolved against the tree, a vanished module would drop out of the run;
    handed to pytest as declared, it exits pytest 4."""
    gone = "tests/e2e/test_no_such_module.py"
    lanes = (check.Lane("x-e2e", "containers", ("tests/e2e/**",), modules=(gone,)),)
    assert lane.modules("x-e2e", lanes) == [gone]


GLOBBED = check.Lane("glob-e2e", "containers", ("tests/e2e/**",), modules=("tests/e2e/x_*.py",))
EMPTY = check.Lane("empty-e2e", "containers", ("tests/e2e/**",), modules=())


@pytest.mark.parametrize(
    ("lane_id", "refusal"),
    [
        ("no-such-lane", "0 lanes have the id"),
        ("e2e-tests", "names no module list"),
        ("glob-e2e", "names no module list"),
        ("empty-e2e", "names no module list"),
    ],
    ids=["unknown", "no-modules", "globs", "empty"],
)
def test_modules_refuses_a_lane_with_no_list_of_plain_paths(lane_id: str, refusal: str) -> None:
    with pytest.raises(ValueError, match=refusal):
        lane.modules(lane_id, (*check.LANES, GLOBBED, EMPTY))


def test_modules_prints_one_path_per_line(capsys: pytest.CaptureFixture[str]) -> None:
    assert lane.main(["modules", "e2e-no-model"]) == 0
    assert capsys.readouterr().out.splitlines() == lane.modules("e2e-no-model")


@pytest.mark.parametrize(
    "argv", [["modules"], ["ignores", "e2e-no-model"], ["modules", "e2e-tests"]]
)
def test_modules_takes_exactly_one_lane_with_a_list(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        lane.main(argv)
    assert exited.value.code == 2


@pytest.mark.parametrize(
    ("argv", "refusal"),
    [
        (["ignores"], "derived no --ignore list"),
        (["modules", "empty-e2e"], "lane 'empty-e2e' names no module list"),
    ],
    ids=["ignores", "modules"],
)
def test_an_empty_list_exits_non_zero(
    argv: list[str],
    refusal: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handed no list, the shared lane would run every module and the model-free
    lane would collect the whole tree; the step's ``bash -e`` stops on the exit
    status instead. An empty tree empties ``ignores``, ``EMPTY`` ``modules``."""
    monkeypatch.setattr(check, "repository_files", list)
    monkeypatch.setattr(check, "LANES", (*check.LANES, EMPTY))
    with pytest.raises(SystemExit) as exited:
        lane.main(argv)
    assert exited.value.code == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert refusal in err


@pytest.mark.parametrize(
    ("e2e_filter", "projects"),
    [
        ("tests/e2e/test_single_user_narrowing_e2e.py -k narrowing", ["suproj"]),
        ("tests/e2e/test_dispatch_deploy.py::test_x", ["osprey-e2e-dispatch"]),
        ("tests/e2e/tier2/", ["chart-value-reading"]),
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


@pytest.mark.parametrize("files", [None, []], ids=["no-project", "no-module"])
def test_stacks_prints_nothing_when_no_module_names_a_project(
    files: list[str] | None, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty line would reach the cleanup script as one empty project name.
    Nothing to remove is no error, unlike an empty ``ignores`` or ``modules``:
    the cleanup step reads an empty result as none."""
    monkeypatch.setenv("E2E_FILTER", "tests/e2e/test_rf_cavity_correlation_scenario.py")
    if files is not None:
        monkeypatch.setattr(check, "repository_files", lambda: files)
    assert lane.main(["stacks"]) == 0
    assert capsys.readouterr().out == ""
