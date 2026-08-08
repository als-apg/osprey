"""Shared fixtures for the deployment test suite."""

from __future__ import annotations

import pytest

from osprey.deployment.compose_generator import _reset_wheel_build_cache
from osprey.deployment.runtime_helper import reset_runtime_cache as _reset_runtime_cache


@pytest.fixture(autouse=True)
def reset_runtime_cache():
    """Isolate every test from the process-wide docker-vs-podman memo.

    ``runtime_helper`` detects the container runtime once per process and
    memoizes the result. Many tests here (and under ``web_terminals``) fake
    ``runtime_helper.subprocess.run``, so the first one to run would otherwise
    pin its fake answer — or the host's real runtime — for every test after it.
    """
    _reset_runtime_cache()
    yield
    _reset_runtime_cache()


@pytest.fixture(autouse=True)
def running_a_released_osprey(monkeypatch):
    """Build image argv as though a released osprey were running.

    Production image builds pin ``osprey-framework==<release>`` and refuse when
    the running framework is a development build, since no published
    distribution matches it. The test suite itself runs from a development
    checkout, so without this every argv-shape test here would trip that refusal
    instead of asserting on the argv it cares about.

    Tests that are *about* the refusal re-patch these inside the test body, which
    takes precedence — see ``TestResolvePipSpec`` in
    ``test_container_lifecycle.py``.
    """
    monkeypatch.setattr("osprey.version.is_release", lambda: True)
    monkeypatch.setattr("osprey.version.get_release_version", lambda: "2026.6.2")


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
