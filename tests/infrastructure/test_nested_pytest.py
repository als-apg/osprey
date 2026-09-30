"""The bound on a nested pytest run: its own progress, not a clock.

A child that keeps writing runs to its end however long it takes; a child that
writes nothing for the stall window is stopped and its output reported.
"""

from __future__ import annotations

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
