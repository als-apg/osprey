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

import os
import sys
from pathlib import Path

import pytest

from tests._nested_pytest import run_nested_pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Keys of the outer run that must not reach an inner pytest: its worker id,
#: its diagnostics directory, any options it was started with, and whether it
#: loads installed plugins, which each case below decides for itself.
_OUTER_RUN_KEYS = (
    "PYTEST_XDIST_",
    "PYTEST_ADDOPTS",
    "OSPREY_CI_DIAG_DIR",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
)


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
    env = {k: v for k, v in os.environ.items() if not k.startswith(_OUTER_RUN_KEYS)}
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
