"""Remove a shared fixture's temporary directories when its scope ends, used by ``tests/conftest.py``.

pytest removes a passing test's ``tmp_path`` as the test ends
(``tmp_path_retention_policy = "failed"``), but a directory a fixture makes with
``tmp_path_factory.mktemp`` stays until the session ends. Module-scoped
fixtures that run a real ``osprey build`` make one of a few hundred megabytes
each, so a run's temporary directory grows with every module that has one, and
a CI runner whose ``/tmp`` has a quota runs out part-way through ``tests/cli``.

:func:`pytest_fixture_setup` records the entries a module-, class- or
package-scoped fixture's setup adds to the worker's base temp directory, and
:func:`pytest_fixture_post_finalizer` removes them once that fixture is torn
down -- unless a test of its scope failed and the retention policy keeps what
failed. Entries a longer-lived fixture adds while being set up inside it are
that fixture's, not the enclosing one's, and are left alone.

Registered as a plugin for the reason ``tests/_env_scope_guard.py`` gives:
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

#: Base temp entries each live fixture's setup added, by fixture definition.
_MADE: dict[pytest.FixtureDef, list[Path]] = {}

#: Entries added by a fixture that outlives the setup measuring them.
_LONG_LIVED: set[Path] = set()

#: Node ids of the tests that failed, for the retention policy.
_FAILED: set[str] = set()


def _basetemp(request: pytest.FixtureRequest) -> Path | None:
    factory = getattr(request.config, "_tmp_path_factory", None)
    return None if factory is None else factory.getbasetemp()


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(fixturedef, request):
    """Note the base temp entries a fixture's setup adds."""
    base = _basetemp(request) if "tmp_path_factory" in fixturedef.argnames else None
    if base is None:
        return (yield)
    before = set(base.iterdir())
    try:
        return (yield)
    finally:
        added = set(base.iterdir()) - before
        if fixturedef.scope in SCOPES_ENDING_EARLY:
            _MADE.setdefault(fixturedef, []).extend(added - _LONG_LIVED)
        else:
            _LONG_LIVED.update(added)


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
