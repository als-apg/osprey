"""Shared collection-time setup for the Virtual Accelerator test suite.

docker/virtual-accelerator/{manifest,lattice,ioc} are not importable dotted
packages (their parent directory name contains a hyphen), so every test
module under this directory imports them by putting docker/virtual-accelerator
directly on sys.path -- "manifest", "lattice", and "ioc" are then valid
top-level module names.

pytest loads this conftest before collecting any test module in this
directory tree (including tests/va/e2e/), so the path is set up exactly once
for the whole suite instead of being repeated per test module.
"""

from __future__ import annotations

import atexit
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VA_PARENT = REPO_ROOT / "docker" / "virtual-accelerator"
if str(VA_PARENT) not in sys.path:
    sys.path.insert(0, str(VA_PARENT))


# --- Skip the libca teardown segfault on Linux --------------------------------
#
# Importing ``softioc.builder`` -- which test_record_factory does, to build
# records for its manifest assertions -- leaves EPICS libca with a
# half-initialised, name-server-only Channel Access client context in the host
# process.  On Linux that context's C++ static destructor asserts on a null
# ``pudpiiu`` during interpreter shutdown, so the process dies with SIGSEGV
# (exit 139) *after* every test has passed and every report (summary, coverage,
# JUnit) has already been written.  macOS is unaffected.
#
# The VA suite already sidesteps this for its own CA helpers by running them in
# throwaway subprocesses that ``os._exit`` instead of unwinding libca (see
# test_record_factory and tests/va/e2e/conftest.py).  The pytest host process
# imports ``softioc.builder`` too, so it needs the same treatment -- but it must
# exit with pytest's *real* status so genuine failures still fail CI.
#
# This runs from an ``atexit`` hook registered at import time.  atexit fires
# last-registered-first, so cleanup hooks other suites register during the run
# (e.g. ``atexit.register(container.stop)``) still run first; we then ``os._exit``
# before the crashing C++ static destructors, skipping only them.

_PYTEST_EXIT_CODE: int | None = None


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ANN001
    """Record pytest's real exit status for the teardown guard below."""
    global _PYTEST_EXIT_CODE
    _PYTEST_EXIT_CODE = int(exitstatus)


def _skip_libca_teardown_segfault() -> None:
    # Only intervene when softioc actually poisoned this process and pytest ran
    # to a normal finish; otherwise let the interpreter shut down as usual.
    if _PYTEST_EXIT_CODE is None or "softioc.builder" not in sys.modules:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_PYTEST_EXIT_CODE)


atexit.register(_skip_libca_teardown_segfault)
