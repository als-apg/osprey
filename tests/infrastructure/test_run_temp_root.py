"""A run takes a temporary root of its own when it starts.

Each case starts a fresh pytest with only ``tests._scoped_temp`` loaded, from a
launcher that drops the ``PYTEST_DEBUG_TEMPROOT`` ``run_nested_pytest`` sets,
or sets one of its own, so the child starts the way a run from a shell does. Its one test writes its
base temp to a file the calling test names.
"""

from __future__ import annotations

import getpass
import os
import shutil
import sys
import tempfile
from pathlib import Path

from tests._nested_pytest import NestedRun, outer_run_free_env, run_nested_pytest
from tests._scoped_temp import TEMPROOT_VAR

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The line a run prints when it keeps its root.
_KEPT = "temporary trees kept under "

_TEST = """
def test_reports_its_base_temp(tmp_path_factory):
    with open({report!r}, "a") as handle:
        handle.write(str(tmp_path_factory.getbasetemp()) + "\\n")
    assert {passes}
"""


def _run(
    tmp_path: Path,
    *,
    passes: bool = True,
    temproot: Path | None = None,
    extra_args: tuple[str, ...] = (),
) -> tuple[NestedRun, list[Path]]:
    """Run a one-test suite and return the run with the base temps its test saw."""
    suite = tmp_path / "suite"
    suite.mkdir(parents=True)
    report = tmp_path / "basetemps"
    (suite / "test_one.py").write_text(_TEST.format(report=str(report), passes=passes))
    args = [
        "-p",
        "tests._scoped_temp",
        "-p",
        "no:cacheprovider",
        *extra_args,
        "-c",
        os.devnull,
        "--rootdir",
        str(suite),
        "-o",
        "tmp_path_retention_policy=failed",
        "-q",
        str(suite),
    ]
    if temproot is None:
        start = f"os.environ.pop({TEMPROOT_VAR!r}, None)"
    else:
        start = f"os.environ[{TEMPROOT_VAR!r}] = {str(temproot)!r}"
    launcher = f"import os, sys, pytest; {start}; sys.exit(pytest.main({args!r}))"
    env = outer_run_free_env()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    run = run_nested_pytest([sys.executable, "-c", launcher], cwd=_REPO_ROOT, env=env)
    assert "passed" in run.stdout or "failed" in run.stdout, run.stdout + run.stderr
    seen = [Path(line) for line in report.read_text().splitlines()]
    return run, seen


def _root_of(basetemp: Path) -> Path:
    """The run root a serial run's base temp ``<root>/pytest-of-<user>/pytest-N`` sits in."""
    return basetemp.parent.parent


def test_two_runs_from_one_entry_point_take_different_temp_parents(tmp_path):
    _, (green,) = _run(tmp_path / "green")
    _, (red,) = _run(tmp_path / "red", passes=False)
    try:
        assert green.parent.parent != red.parent.parent
        host_shared = Path(tempfile.gettempdir()) / f"pytest-of-{getpass.getuser()}"
        assert host_shared not in (green.parent, red.parent)
    finally:
        shutil.rmtree(_root_of(red), ignore_errors=True)


def test_a_green_run_leaves_no_root_and_a_red_run_prints_it(tmp_path):
    _, (green,) = _run(tmp_path / "green")
    red_run, (red,) = _run(tmp_path / "red", passes=False)
    try:
        assert not _root_of(green).exists()
        assert _root_of(red).is_dir()
        assert f"{_KEPT}{_root_of(red)}" in red_run.stdout, red_run.stdout
    finally:
        shutil.rmtree(_root_of(red), ignore_errors=True)


def test_a_temproot_the_caller_set_is_kept(tmp_path):
    temproot = tmp_path / "callers-root"
    temproot.mkdir()
    _, (seen,) = _run(tmp_path / "run", temproot=temproot)
    assert seen.is_relative_to(temproot)


def test_xdist_workers_share_the_run_root(tmp_path):
    # Every worker runs the one test, so each reports its own base temp.
    _, seen = _run(tmp_path, extra_args=("-p", "xdist", "-n", "2", "--dist", "each"))
    assert len({base.name for base in seen}) == 2, seen
    # A worker's base temp is <root>/pytest-of-<user>/pytest-N/popen-gwX.
    assert len({base.parents[2] for base in seen}) == 1, seen
