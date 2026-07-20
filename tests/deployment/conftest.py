"""Shared fixtures for the deployment test suite."""

from __future__ import annotations

import pytest

from osprey.deployment.compose_generator import _reset_wheel_build_cache


@pytest.fixture(autouse=True)
def reset_wheel_build_cache():
    """Isolate every test from the process-wide dev-wheel build memo.

    ``compose_generator`` memoizes the ``python -m build`` invocation per
    process (one build, many staged copies). Left unreset, a wheel built (or a
    failure memoized) by one test would leak into the next — e.g. a test that
    mocks the build subprocess and expects it to be invoked would silently get
    the cached result instead.
    """
    _reset_wheel_build_cache()
    yield
    _reset_wheel_build_cache()
