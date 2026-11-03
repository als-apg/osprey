"""pytest plugin: the suite runs against the installed packages, not a source tree.

The connectors compatibility lanes run a suite against packages installed in
a venv -- the floor connectors under the framework at HEAD (Lane B), or a
framework release with the connectors under test (the nightly) -- from a
checkout that also holds those packages' sources. A ``pythonpath`` entry, a
conftest's ``sys.path`` edit or an editable install reaching one of those
trees would run the suite green against the wrong code. Every module named in
``CONNECTORS_COMPAT_INSTALLED`` (space-separated) must therefore import, in
every worker and at test time, from the interpreter's site-packages; a test
errors in its setup otherwise.

Loaded with ``-p connectors_compat_guard`` and this directory on
``PYTHONPATH`` (scripts/ci/connectors_compat_lane.py does both). Standard
library and pytest only: a release's own test environment loads it.
"""

from __future__ import annotations

import importlib
import os
import sysconfig
from pathlib import Path

import pytest

ENV = "CONNECTORS_COMPAT_INSTALLED"


def _misplaced() -> list[str]:
    purelib = Path(sysconfig.get_path("purelib")).resolve()
    found = []
    for name in os.environ.get(ENV, "").split():
        origin = Path(importlib.import_module(name).__file__ or "").resolve()
        if purelib not in origin.parents:
            found.append(f"{name} imports from {origin}, not from {purelib}")
    return found


def pytest_report_header() -> list[str]:
    names = os.environ.get(ENV, "").split()
    if not names:
        return [f"connectors compat guard: {ENV} names no module, nothing is held"]
    return [f"connectors compat guard: {', '.join(names)} must import from site-packages"]


@pytest.fixture(autouse=True, scope="session")
def _connectors_compat_installed() -> None:
    misplaced = _misplaced()
    if misplaced:
        pytest.fail("; ".join(misplaced), pytrace=False)
