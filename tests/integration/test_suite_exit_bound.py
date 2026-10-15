"""A green run of the suite ends on its own, promptly, after its summary.

One facility test module, which imports the product, runs as a nested pytest
under xdist, so one run puts every kind of pytest process the suite starts
under the exit contract of ``tests/_live_threads.py``, and each assertion
proves a different one:

* ``quiet_before_exit`` is the time between the run's last word and its exit.
  That last word is the controller's summary, so this bounds the controller's
  exit, as it would a serial run's. It cannot see a worker: xdist ends every
  worker before the controller prints its summary, and kills one that has not
  exited in time without a word, after which the run still exits promptly.
* No ``still exiting`` line on stderr is the workers' proof. A worker that
  outlives its own bound after its exit began writes that line, with the stack
  holding it, to its stderr, which is the controller's terminal, before it can
  be killed.

The bound is ``EXIT_BOUND_SECONDS`` itself, read from the report that enforces
it. The run is timed alone, in the serial step that selects ``wallclock``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests import _live_threads
from tests._nested_pytest import outer_run_free_env, run_nested_pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.wallclock
def test_a_facility_module_exits_within_its_bound_after_its_summary() -> None:
    result = run_nested_pytest(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/facility/test_errors.py",
            "-q",
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            "--dist",
            "loadgroup",
        ],
        cwd=_REPO_ROOT,
        env=outer_run_free_env(),
    )

    tail = result.stdout[-4000:] + "\n--- stderr ---\n" + result.stderr[-8000:]
    assert result.returncode == 0, tail
    assert "still exiting" not in result.stderr, tail
    assert result.quiet_before_exit <= _live_threads.EXIT_BOUND_SECONDS, (
        f"the run sat silent for {result.quiet_before_exit:.1f} s before it exited\n{tail}"
    )
