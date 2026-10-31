"""Remove a shared fixture's temporary directories when its scope ends, used by ``tests/conftest.py``.

pytest removes a passing test's ``tmp_path`` as the test ends
(``tmp_path_retention_policy = "failed"``), but a directory a fixture makes with
``tmp_path_factory.mktemp`` stays until the session ends. Module-scoped
fixtures that run a real ``osprey build`` make one of a few hundred megabytes
each, so a run's temporary directory grows with every module that has one, and
a CI runner whose ``/tmp`` has a quota runs out part-way through ``tests/cli``.

:func:`pytest_fixture_setup` records the directories ``tmp_path_factory.mktemp``
returns while a module-, class- or package-scoped fixture is being set up, and
:func:`pytest_fixture_post_finalizer` removes them once that fixture is torn
down -- unless a test of its scope failed and the retention policy keeps what
failed. The base temp directory is never listed: it holds an entry for every
test the worker has run, so reading it costs more the longer a run goes. A
directory made by a longer-lived fixture while being set up inside one is that
fixture's, not the enclosing one's, and is left alone. A fixture that hands
back a function making directories on demand makes their parent with
``mktemp`` during its setup, so the parent is the one recorded.

Registered as a plugin for the reason the harness's ``_env_scope_guard.py`` gives:
``pytest_fixture_setup`` is dispatched on the node that owns the fixture's
scope, and only a plugin is in scope for every node.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

PLUGIN_NAME = "osprey-scoped-temp"

#: Scopes whose fixtures are torn down before the session ends.
SCOPES_ENDING_EARLY = frozenset({"package", "module", "class"})

#: Directories each live fixture's setup made, by fixture definition.
_MADE: dict[pytest.FixtureDef, list[Path]] = {}

#: The directories of each fixture setup under way, innermost last.
_SETUPS: list[list[Path]] = []

#: Node ids of the tests that failed, for the retention policy.
_FAILED: set[str] = set()

#: ``TempPathFactory.mktemp`` as pytest defines it, while this plugin wraps it.
_UNWRAPPED_MKTEMP = None


def _recording_mktemp(self, basename, numbered=True):
    path = _UNWRAPPED_MKTEMP(self, basename, numbered)
    if _SETUPS:
        _SETUPS[-1].append(path)
    return path


def pytest_configure():
    global _UNWRAPPED_MKTEMP
    # A pytest run inside this one finds mktemp already wrapped and leaves it be.
    if pytest.TempPathFactory.mktemp is _recording_mktemp:
        return
    _UNWRAPPED_MKTEMP = pytest.TempPathFactory.mktemp
    pytest.TempPathFactory.mktemp = _recording_mktemp


def pytest_unconfigure():
    global _UNWRAPPED_MKTEMP
    if _UNWRAPPED_MKTEMP is not None:
        pytest.TempPathFactory.mktemp = _UNWRAPPED_MKTEMP
        _UNWRAPPED_MKTEMP = None


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(fixturedef):
    """Note the directories a fixture's setup makes."""
    # A function-scoped fixture can neither be set up inside a longer-lived one
    # nor outlive its test, so what it makes is pytest's to remove.
    if fixturedef.scope == "function":
        return (yield)
    made: list[Path] = []
    _SETUPS.append(made)
    try:
        return (yield)
    finally:
        _SETUPS.pop()
        if made and fixturedef.scope in SCOPES_ENDING_EARLY:
            _MADE.setdefault(fixturedef, []).extend(made)


def pytest_runtest_logreport(report):
    if report.failed:
        _FAILED.add(report.nodeid)


def pytest_fixture_post_finalizer(fixturedef, request):
    """Remove what the fixture's setup added, as the retention policy allows."""
    made = _MADE.pop(fixturedef, None)
    if not made:
        return
    policy = request.config.getini("tmp_path_retention_policy")
    scope = request.node.nodeid
    # A package's tests sit under its directory, a module's and a class's under "::".
    inside = (f"{scope}::", f"{scope}/")
    failed = any(nodeid == scope or nodeid.startswith(inside) for nodeid in _FAILED)
    if policy == "all" or (policy == "failed" and failed):
        return
    for path in made:
        shutil.rmtree(path, ignore_errors=True)
