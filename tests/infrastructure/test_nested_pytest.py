"""The bound on a nested pytest run: its own progress, not a clock.

A child that keeps writing runs to its end however long it takes; a child that
writes nothing for the stall window is stopped and its output reported. A
nested run's temporary directories, and the pruning pytest does of them at
exit, stay inside a root of its own.
"""

from __future__ import annotations

import getpass
import os
import sys

import pytest

from tests._nested_pytest import run_nested_pytest

_TALKER = "import time\nfor _ in range(6):\n    print('.', flush=True)\n    time.sleep(0.5)\n"


def test_a_run_that_keeps_writing_is_not_stopped_by_its_length():
    result = run_nested_pytest(
        [sys.executable, "-c", _TALKER], cwd=None, env=None, stall_seconds=1.5
    )
    assert result.returncode == 0
    assert result.stdout.count(".") == 6


def test_a_run_that_writes_nothing_for_the_stall_window_is_stopped_with_its_stacks():
    with pytest.raises(pytest.fail.Exception, match="no new output for 1.0 s") as stopped:
        run_nested_pytest(
            [sys.executable, "-c", "print('started', flush=True)\nimport time\ntime.sleep(60)"],
            cwd=None,
            env=None,
            stall_seconds=1.0,
        )
    report = str(stopped.value)
    assert "started" in report
    assert "most recent call first" in report


def test_a_nested_run_leaves_the_shared_temporary_root_alone(tmp_path):
    shared = tmp_path / "shared"
    numbered = shared / f"pytest-of-{getpass.getuser()}"
    for name in ("pytest-1", "pytest-2", "pytest-3"):
        (numbered / name).mkdir(parents=True)
    (numbered / "pytest-1" / "left-by-a-finished-run").write_text("")
    work = tmp_path / "work"
    work.mkdir()
    module = work / "test_uses_tmp_path.py"
    module.write_text(
        "def test_writes_into_tmp_path(tmp_path):\n    (tmp_path / 'out').write_text('x')\n"
    )

    result = run_nested_pytest(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-c",
            os.devnull,
            "--rootdir",
            str(work),
            str(module),
            "-q",
        ],
        cwd=work,
        env={
            **os.environ,
            "PYTEST_DEBUG_TEMPROOT": str(shared),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert sorted(p.name for p in numbered.iterdir()) == ["pytest-1", "pytest-2", "pytest-3"]
    assert (numbered / "pytest-1" / "left-by-a-finished-run").exists()
