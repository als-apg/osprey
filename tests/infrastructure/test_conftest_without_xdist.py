"""The shared conftest loads in a pytest that has no xdist.

``tests/conftest.py`` implements ``pytest_xdist_make_scheduler``, a hook whose
specification only xdist provides. pytest validates every hook implementation
of a conftest against the specifications of the loaded plugins and ends the
session on one it cannot match, unless the implementation declares itself
optional. The proof runs a nested pytest over this module under the
repository's own conftests, once with no plugin loaded at all and once with
xdist disabled by name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests._nested_pytest import outer_run_free_env, run_nested_pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Whether the inner run loads installed plugins, which each case below decides
#: for itself, so the outer run's choice must not reach it either.
_PLUGIN_AUTOLOAD_KEY = "PYTEST_DISABLE_PLUGIN_AUTOLOAD"


@pytest.mark.parametrize(
    ("plugin_env", "plugin_args"),
    [
        pytest.param({"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}, (), id="xdist-absent"),
        pytest.param({}, ("-p", "no:xdist"), id="xdist-disabled"),
    ],
)
def test_the_shared_conftest_collects_without_xdist(
    plugin_env: dict[str, str], plugin_args: tuple[str, ...]
) -> None:
    env = outer_run_free_env(also_drop=(_PLUGIN_AUTOLOAD_KEY,))
    env.update(plugin_env)
    result = run_nested_pytest(
        [
            sys.executable,
            "-m",
            "pytest",
            *plugin_args,
            "-p",
            "no:cacheprovider",
            "--collect-only",
            "-q",
            "--color=no",
            str(Path(__file__).relative_to(_REPO_ROOT)),
        ],
        cwd=_REPO_ROOT,
        env=env,
    )
    output = result.stdout + result.stderr
    assert "unknown hook" not in output, output
    assert result.returncode == 0, output
