"""A shared fixture's temporary directories go when its scope ends.

Each case runs a small suite in a fresh pytest with only ``tests._scoped_temp``
loaded and its own ``--basetemp``, then reads what that base temp directory
still holds: pytest itself never removes a ``tmp_path_factory`` directory
before the session ends, so whatever is gone was removed by the plugin.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

from tests._nested_pytest import outer_run_free_env, run_nested_pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_CONFTEST = """
import pytest

@pytest.fixture(scope="session")
def long_lived(tmp_path_factory):
    return tmp_path_factory.mktemp("long-lived")

@pytest.fixture(scope="module")
def shared(tmp_path_factory, request):
    made = tmp_path_factory.mktemp("module-made")
    # A session fixture first set up inside this one keeps its own directory.
    request.getfixturevalue("long_lived")
    return made
"""


def _run(tmp_path: Path, test_body: str, policy: str = "failed") -> set[str]:
    """Run a one-module suite over ``shared`` and return what its base temp holds."""
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "conftest.py").write_text(textwrap.dedent(_CONFTEST))
    (suite / "test_shared.py").write_text(textwrap.dedent(test_body))
    (suite / "test_later.py").write_text("def test_later():\n    pass\n")
    basetemp = tmp_path / "basetemp"
    env = outer_run_free_env()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    run = run_nested_pytest(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests._scoped_temp",
            "-p",
            "no:cacheprovider",
            "-c",
            os.devnull,
            "--rootdir",
            str(suite),
            "--basetemp",
            str(basetemp),
            "-o",
            f"tmp_path_retention_policy={policy}",
            "-q",
            str(suite),
        ],
        cwd=_REPO_ROOT,
        env=env,
    )
    assert "passed" in run.stdout or "failed" in run.stdout, run.stdout + run.stderr
    return {path.name for path in basetemp.iterdir()}


def test_a_module_fixtures_directory_goes_with_its_module(tmp_path):
    left = _run(tmp_path, "def test_uses(shared):\n    assert shared.is_dir()\n")
    assert "module-made0" not in left
    assert "long-lived0" in left


def test_a_failed_modules_directory_is_kept(tmp_path):
    left = _run(tmp_path, "def test_uses(shared):\n    assert False\n")
    assert "module-made0" in left


def test_policy_all_keeps_every_directory(tmp_path):
    left = _run(tmp_path, "def test_uses(shared):\n    pass\n", policy="all")
    assert "module-made0" in left


def test_a_function_scoped_fixture_does_not_list_the_base_temp():
    from types import SimpleNamespace

    from tests import _scoped_temp

    class _Unlistable:
        def getbasetemp(self):
            raise AssertionError("the base temp was read for a function-scoped fixture")

    fixturedef = SimpleNamespace(scope="function", argnames=("tmp_path_factory",))
    request = SimpleNamespace(config=SimpleNamespace(_tmp_path_factory=_Unlistable()))
    setup = _scoped_temp.pytest_fixture_setup(fixturedef, request)
    next(setup)
    with pytest.raises(StopIteration) as done:
        setup.send("value")
    assert done.value.value == "value"
