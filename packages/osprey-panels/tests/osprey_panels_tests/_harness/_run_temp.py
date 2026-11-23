"""A run owns its temporary root; nothing else on the host numbers or prunes it.

Left to itself, pytest numbers every run's base temporary directory under the
shared ``{temproot}/pytest-of-<user>/`` and, when a run exits, deletes all but
the newest three entries there. Two runs on one host would delete each other's
live trees. So the run gives itself a fresh directory as ``--basetemp`` before
pytest's temporary-path setup reads the option, outside ``pytest-of-<user>``,
in ``PYTEST_DEBUG_TEMPROOT`` or the system temp dir. An xdist run's workers
each get ``<root>/popen-gwN`` from their controller and claim nothing.

The run removes the root it claimed itself, as ``tmp_path_retention_policy``
says: ``none`` always, ``failed`` after a green run, ``all`` never. A kept root
is named once in the terminal summary. A ``--basetemp`` the caller gives, or
one another suite's harness already claimed in a run that loads both, is used
as is and never removed here: its owner removes it.

This is the framework-free twin of the facility suite's ``_run_temp``; the
panels suite cannot import that one, which ships in no distribution.
Nothing here imports ``osprey`` or the framework's ``tests`` package.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

#: Prefix of every root a run claims.
ROOT_PREFIX = "osprey-tests-"

#: Line naming a root the run kept, followed by its path.
KEPT_LINE = "temporary files kept at"

#: Name the release and summary hooks are registered under.
PLUGIN_NAME = "osprey-panels-run-temp"

_OWNED_ROOT = pytest.StashKey[Path]()
_KEPT_ROOT = pytest.StashKey[Path]()


def claim(config: pytest.Config) -> None:
    """Give the run a fresh temporary root unless it already has one."""
    if _OWNED_ROOT in config.stash:
        return
    if config.option.basetemp or hasattr(config, "workerinput"):
        return
    temproot = os.environ.get("PYTEST_DEBUG_TEMPROOT") or tempfile.gettempdir()
    root = Path(tempfile.mkdtemp(prefix=ROOT_PREFIX, dir=temproot)).resolve()
    config.option.basetemp = str(root)
    config.stash[_OWNED_ROOT] = root
    if not config.pluginmanager.is_registered(sys.modules[__name__]):
        config.pluginmanager.register(sys.modules[__name__], PLUGIN_NAME)


def release(config: pytest.Config, exitstatus: int) -> None:
    """Remove the root this run claimed, as the retention policy says."""
    root = config.stash.get(_OWNED_ROOT, None)
    if root is None:
        return
    policy = config.getini("tmp_path_retention_policy")
    if policy == "none" or (policy == "failed" and exitstatus == 0):
        shutil.rmtree(root, ignore_errors=True)
    if root.exists():
        config.stash[_KEPT_ROOT] = root


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_sessionfinish(session: pytest.Session):
    """Release the root after every other ``pytest_sessionfinish`` settled the exit status."""
    result = yield
    release(session.config, session.exitstatus)
    return result


def pytest_terminal_summary(terminalreporter, config: pytest.Config) -> None:
    """Name the temporary root the run kept."""
    root = config.stash.get(_KEPT_ROOT, None)
    if root is not None:
        terminalreporter.write_line(f"{KEPT_LINE} {root}")
