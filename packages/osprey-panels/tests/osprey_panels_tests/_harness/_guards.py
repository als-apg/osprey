"""The test guards of the panels suite, none of which needs the framework.

The suite runs where only ``osprey-panels`` is installed, so its guards live
here and ``tests/conftest.py`` imports them by name, not as a plugin: a
conftest's autouse fixtures are ordered by name within that conftest. In a
run that also loads the framework's conftest the session-level hook is called
once per conftest, so it is idempotent.

Nothing here imports ``osprey`` or the framework's ``tests`` package.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from osprey_panels_tests._harness import _run_temp
from osprey_panels_tests._harness._paths import REPO_ROOT

__all__ = [
    "pytest_configure",
    "restore_cwd",
    "restore_environ",
]


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Claim the run's own temporary root before pytest reads ``--basetemp``."""
    _run_temp.claim(config)


@pytest.fixture(autouse=True)
def restore_environ() -> Iterator[None]:
    """Snapshot and restore the whole of ``os.environ`` around every test.

    A test that writes ``os.environ`` directly rather than through
    ``monkeypatch`` would otherwise leak into its siblings, and under xdist the
    result would depend on how the files happened to be distributed.
    """
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


@pytest.fixture(autouse=True)
def restore_cwd() -> Iterator[None]:
    """Restore the process working directory after every test.

    A raw ``os.chdir`` whose restore an exception skipped leaves the process in
    a foreign, often deleted, ``tmp_path``; later tests that resolve a relative
    path would read from there. A backstop: each ``os.chdir`` site still owns
    its own restore.
    """
    try:
        original = os.getcwd()
    except OSError:
        original = str(REPO_ROOT)
    yield
    try:
        os.chdir(original)
    except OSError:
        os.chdir(REPO_ROOT)
