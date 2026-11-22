"""What a test's child process inherits from the host and from the harness.

A nested run starts with the host settings in its environment and loads the
shared guards; its one test spawns a child that prints what it sees.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

from tests._nested_pytest import outer_run_free_env, run_nested_pytest, with_harness_path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_TEST = """
import os
import subprocess
import sys

def test_child_reports_malloc_settings():
    seen = subprocess.run(
        [sys.executable, "-c", "import os; print(sorted(k for k in os.environ if k.startswith('Malloc')))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert seen == "[]", seen
"""

_FAULTHANDLER_TEST = """
import subprocess
import sys

def test_child_reports_its_faulthandler_setting():
    seen = subprocess.run(
        [sys.executable, "-c", "import os; print(os.environ.get('PYTHONFAULTHANDLER'))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert seen == "1", seen
"""

#: Starts the interpreter named by its arguments with ``PYTHONFAULTHANDLER``
#: empty, the host setting that disables it, in place of the ``1`` that
#: ``run_nested_pytest`` gives every nested run, so only the guards the nested
#: run loads can give it back to the children of its tests.
_WITHOUT_FAULTHANDLER = (
    "import os, sys; "
    "os.environ['PYTHONFAULTHANDLER'] = ''; "
    "os.execv(sys.executable, [sys.executable, *sys.argv[1:]])"
)


def _guarded_pytest(suite: Path) -> list[str]:
    return [
        "-m",
        "pytest",
        "-p",
        "osprey_connectors_tests._harness._guards",
        "-p",
        "no:cacheprovider",
        "-c",
        os.devnull,
        "--rootdir",
        str(suite),
        "-q",
        str(suite),
    ]


def test_a_child_never_sees_the_hosts_malloc_settings(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_child.py").write_text(textwrap.dedent(_TEST))
    env = with_harness_path(outer_run_free_env())
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["MallocStackLogging"] = "1"
    env["MallocNanoZone"] = "0"
    run = run_nested_pytest([sys.executable, *_guarded_pytest(suite)], cwd=_REPO_ROOT, env=env)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "1 passed" in run.stdout, run.stdout + run.stderr


def test_every_child_writes_its_stack_on_a_fatal_signal(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_child.py").write_text(textwrap.dedent(_FAULTHANDLER_TEST))
    env = with_harness_path(outer_run_free_env())
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    run = run_nested_pytest(
        [sys.executable, "-c", _WITHOUT_FAULTHANDLER, *_guarded_pytest(suite)],
        cwd=_REPO_ROOT,
        env=env,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "1 passed" in run.stdout, run.stdout + run.stderr
