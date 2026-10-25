"""A fixture's temporary directories go when its scope ends.

Each case runs a small suite in a fresh pytest with only ``tests._scoped_temp``
loaded and its own ``--basetemp``, then reads what that base temp directory
still holds: pytest itself never removes a ``tmp_path_factory`` directory, so
whatever is gone was removed by the plugin.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

from tests._nested_pytest import outer_run_free_env, run_nested_pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_CONFTEST = """
import pytest

from tests._scoped_temp import shared_dir

@pytest.fixture(scope="session")
def long_lived(tmp_path_factory):
    return tmp_path_factory.mktemp("long-lived")

@pytest.fixture(scope="session")
def shared_tree(tmp_path_factory):
    return shared_dir(tmp_path_factory, "shared-tree")

@pytest.fixture(scope="module")
def shared(tmp_path_factory, request):
    made = tmp_path_factory.mktemp("module-made")
    # A session fixture first set up inside this one keeps its own directory.
    request.getfixturevalue("long_lived")
    return made

@pytest.fixture(scope="module")
def hand_made(tmp_path_factory):
    made = tmp_path_factory.getbasetemp() / "hand-made"
    made.mkdir()
    return made

@pytest.fixture
def fn_made(tmp_path_factory):
    return tmp_path_factory.mktemp("fn-made")
"""

#: Runs the nested suite on two xdist workers.
_XDIST = ("-p", "xdist", "-n", "2")


def _run(
    tmp_path: Path,
    test_body: str,
    policy: str = "failed",
    extra_args: tuple[str, ...] = (),
) -> set[str]:
    """Run a one-module suite over ``shared`` and return what its base temp holds."""
    suite = tmp_path / "suite"
    suite.mkdir(parents=True)
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
            *extra_args,
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


def _shared_trees(left: set[str]) -> set[str]:
    return {name for name in left if name.startswith("shared-tree-")}


def test_a_module_fixtures_directory_goes_with_its_module(tmp_path):
    left = _run(tmp_path, "def test_uses(shared):\n    assert shared.is_dir()\n")
    assert "module-made0" not in left


def test_a_failed_modules_directory_is_kept(tmp_path):
    left = _run(tmp_path, "def test_uses(shared):\n    assert False\n")
    assert "module-made0" in left


def test_a_session_fixtures_directory_goes_when_the_session_ends(tmp_path):
    left = _run(tmp_path, "def test_uses(long_lived):\n    assert long_lived.is_dir()\n")
    assert "long-lived0" not in left


def test_a_session_directory_is_kept_for_a_failed_consumer(tmp_path):
    left = _run(tmp_path, "def test_uses(long_lived):\n    assert False\n")
    assert "long-lived0" in left


def test_a_failure_elsewhere_keeps_nothing_it_did_not_use(tmp_path):
    body = """
    def test_uses(shared, long_lived):
        pass

    def test_fails():
        assert False
    """
    left = _run(tmp_path, body)
    assert "module-made0" not in left
    assert "long-lived0" not in left


def test_a_dynamic_request_counts_as_consuming(tmp_path):
    body = """
    def test_uses(request):
        request.getfixturevalue("long_lived")
        assert False
    """
    left = _run(tmp_path, body)
    assert "long-lived0" in left


def test_a_function_fixtures_directory_follows_its_test(tmp_path):
    passed = _run(tmp_path / "pass", "def test_uses(fn_made):\n    assert fn_made.is_dir()\n")
    assert "fn-made0" not in passed
    failed = _run(tmp_path / "fail", "def test_uses(fn_made):\n    assert False\n")
    assert "fn-made0" in failed


def test_a_dynamic_request_of_a_function_fixture_keeps_it(tmp_path):
    body = """
    def test_uses(request):
        request.getfixturevalue("fn_made")
        assert False
    """
    left = _run(tmp_path, body)
    assert "fn-made0" in left


def test_a_setup_error_keeps_what_the_test_used(tmp_path):
    body = """
    import pytest

    @pytest.fixture
    def broken(long_lived):
        raise RuntimeError("setup fails")

    def test_uses(broken):
        pass
    """
    left = _run(tmp_path, body)
    assert "long-lived0" in left


def test_a_shared_tree_goes_after_a_green_xdist_run(tmp_path):
    body = "def test_uses(shared_tree):\n    assert shared_tree.is_dir()\n"
    left = _run(tmp_path, body, extra_args=_XDIST)
    assert any(name.startswith("popen-gw") for name in left), left
    assert not _shared_trees(left)


def test_a_shared_tree_is_kept_for_a_failed_consumer_under_xdist(tmp_path):
    left = _run(tmp_path, "def test_uses(shared_tree):\n    assert False\n", extra_args=_XDIST)
    assert _shared_trees(left)


def test_a_shared_tree_goes_when_only_a_non_consumer_failed_under_xdist(tmp_path):
    body = """
    def test_uses(shared_tree):
        assert shared_tree.is_dir()

    def test_fails():
        assert False
    """
    left = _run(tmp_path, body, extra_args=_XDIST)
    assert any(name.startswith("popen-gw") for name in left), left
    assert not _shared_trees(left)


def test_policy_all_keeps_every_directory(tmp_path):
    left = _run(tmp_path, "def test_uses(shared):\n    pass\n", policy="all")
    assert "module-made0" in left


def test_only_what_mktemp_made_is_removed(tmp_path):
    left = _run(tmp_path, "def test_uses(shared, hand_made):\n    pass\n")
    assert "module-made0" not in left
    assert "hand-made" in left
