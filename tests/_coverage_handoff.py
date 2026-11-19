"""A child process's environment without coverage's subprocess hand-off.

The coverage cell runs pytest with ``--cov`` under ``[tool.coverage.run]
patch = ["subprocess"]``, which puts ``COVERAGE_PROCESS_CONFIG`` into
``os.environ``. Every Python child that inherits it starts coverage from the
``.pth`` file coverage installs in site-packages, measures itself, and writes a
data file at exit; every child of that child does the same. That is the point
for a child whose own framework code is what the test drives — a hook script, a
connector-host process — so those keep the variables. A child that runs only
test code (which ``omit`` leaves out), a stub, the stdlib, or framework code the
in-process tests already execute pays coverage's start-up and data write for
nothing, and a test that spawns many of them pays it many times.

The leading underscore keeps the module out of pytest collection.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

#: The prefix of the variables coverage's ``.pth`` start-up reads:
#: ``COVERAGE_PROCESS_CONFIG`` (what ``patch = ["subprocess"]`` sets) and
#: ``COVERAGE_PROCESS_START`` (the older, manual switch).
COVERAGE_HANDOFF_PREFIX = "COVERAGE_PROCESS_"


def without_coverage_handoff(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """A copy of *env* (default ``os.environ``) a Python child starts without coverage in.

    Args:
        env: The environment the child would otherwise get. Not modified.

    Returns:
        *env* without any key starting with :data:`COVERAGE_HANDOFF_PREFIX`.
    """
    source = os.environ if env is None else env
    return {
        key: value for key, value in source.items() if not key.startswith(COVERAGE_HANDOFF_PREFIX)
    }
