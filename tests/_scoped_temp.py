"""Remove a fixture's temporary directories when its scope ends, used by ``tests/conftest.py``.

Every directory a fixture's setup makes with ``tmp_path_factory.mktemp``, at
any scope, goes when that fixture is torn down, unless a failed test used the
fixture and the retention policy (``tmp_path_retention_policy``) keeps what
failed. Policy ``all`` keeps every directory; ``none`` keeps none.

:func:`pytest_fixture_setup` records the directories ``mktemp`` returns while a
fixture is being set up, and :func:`pytest_fixture_post_finalizer` removes them
once that fixture is torn down. A directory made by a longer-lived fixture
while being set up inside another is that fixture's, not the enclosing one's.
A fixture that hands back a function making directories on demand makes their
parent with ``mktemp`` during its setup, so the parent is the one recorded. The
base temp directory is never listed: it holds an entry for every test the
worker has run, so reading it costs more the longer a run goes.

A test used a fixture when the fixture is in its fixture closure or the test
requested it with ``request.getfixturevalue``. A function-scoped fixture serves
exactly one test, so it is kept iff that test failed; a longer-lived one is
kept iff a failed test inside its scope used it by name.

A tree every xdist worker reads is made with :func:`shared_dir` in the run's
root, beside the workers' own base temps. A worker cannot tell when the others
are done with it, so each worker reports the shared trees it made and the
fixtures its failed tests used, and the controller removes the shared trees no
failed test used once every worker is down. A worker the controller does not
hear down from cleanly leaves what it used unknown, and every shared tree is
kept.

Registered as a plugin for the reason
``packages/osprey-connectors/tests/osprey_connectors_tests/_harness/_env_scope_guard.py``
gives:
``pytest_fixture_setup`` is dispatched on the node that owns the fixture's
scope, and only a plugin is in scope for every node.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

PLUGIN_NAME = "osprey-scoped-temp"

#: Directories each live fixture's setup made, by fixture definition.
_MADE: dict[pytest.FixtureDef, list[Path]] = {}

#: Each fixture setup under way with the directories it made, innermost last.
_SETUPS: list[tuple[pytest.FixtureDef, list[Path]]] = []

#: Node id of each failed test -> the names of the fixtures it used.
_CONSUMED_BY_FAILED: dict[str, set[str]] = {}

#: Shared trees this worker made, with the name of the fixture that made each.
_SHARED_MADE: list[tuple[str, Path]] = []

#: Controller side: gateway id of every worker that came up.
_NODES_READY: set[str] = set()

#: Controller side: gateway id -> the report a worker went down with, ``None``
#: when it went down with an error or without one.
_NODES_DOWN: dict[str, dict | None] = {}

#: ``TempPathFactory.mktemp`` as pytest defines it, while this plugin wraps it.
_UNWRAPPED_MKTEMP = None


def _recording_mktemp(self, basename, numbered=True):
    path = _UNWRAPPED_MKTEMP(self, basename, numbered)
    if _SETUPS:
        _SETUPS[-1][1].append(path)
    return path


def shared_dir(tmp_path_factory: pytest.TempPathFactory, name: str) -> Path:
    """A new directory every xdist worker of this run can read, for a fixture's setup.

    Outside xdist it is an ordinary ``mktemp`` directory of the calling
    fixture. Under xdist it sits in the run's root, outlives this worker's
    session, and the controller removes it after the run.
    """
    if not _SETUPS:
        raise RuntimeError("shared_dir is called only while a fixture is being set up")
    if os.environ.get("PYTEST_XDIST_WORKER") is None:
        return tmp_path_factory.mktemp(name)
    root = tmp_path_factory.getbasetemp().parent
    path = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=root))
    _SHARED_MADE.append((_SETUPS[-1][0].argname, path))
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
    made: list[Path] = []
    _SETUPS.append((fixturedef, made))
    try:
        return (yield)
    finally:
        _SETUPS.pop()
        if made:
            _MADE.setdefault(fixturedef, []).extend(made)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item):
    """Note the fixtures a failed test used."""
    report = yield
    if report.failed:
        names = _CONSUMED_BY_FAILED.setdefault(item.nodeid, set())
        names.update(getattr(item, "fixturenames", ()))
    return report


@pytest.fixture(autouse=True)
def _scoped_temp_dynamic_requests(request):
    """Add the fixtures a failed test requested by name to what it used."""
    yield
    names = _CONSUMED_BY_FAILED.get(request.node.nodeid)
    if names is not None:
        names.update(request.fixturenames)


def _used_by_a_failure(fixturedef: pytest.FixtureDef, request: pytest.FixtureRequest) -> bool:
    scope = request.node.nodeid
    if fixturedef.scope == "function":
        return scope in _CONSUMED_BY_FAILED
    # A package's tests sit under its directory, a module's and a class's under "::".
    inside = (f"{scope}::", f"{scope}/")
    return any(
        fixturedef.argname in names
        and (scope == "" or nodeid == scope or nodeid.startswith(inside))
        for nodeid, names in _CONSUMED_BY_FAILED.items()
    )


def pytest_fixture_post_finalizer(fixturedef, request):
    """Remove what the fixture's setup made, as the retention policy allows."""
    made = _MADE.pop(fixturedef, None)
    if not made:
        return
    policy = request.config.getini("tmp_path_retention_policy")
    if policy == "all" or (policy == "failed" and _used_by_a_failure(fixturedef, request)):
        return
    for path in made:
        shutil.rmtree(path, ignore_errors=True)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodeready(node):
    _NODES_READY.add(node.gateway.id)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    report = None if error else getattr(node, "workeroutput", {}).get(PLUGIN_NAME)
    _NODES_DOWN[node.gateway.id] = report


def pytest_sessionfinish(session):
    """Report the shared trees to the controller, or remove them on it."""
    config = session.config
    if hasattr(config, "workeroutput"):
        config.workeroutput[PLUGIN_NAME] = {
            "made": [[argname, str(path)] for argname, path in _SHARED_MADE],
            "kept": sorted(set().union(*_CONSUMED_BY_FAILED.values())),
        }
        return
    if hasattr(config, "workerinput") or not _NODES_READY:
        return
    reports = [_NODES_DOWN.get(node) for node in sorted(_NODES_READY)]
    policy = config.getini("tmp_path_retention_policy")
    if policy == "all" or any(report is None for report in reports):
        return
    kept = {name for report in reports for name in report["kept"]} if policy == "failed" else set()
    for report in reports:
        for argname, path in report["made"]:
            if argname not in kept:
                shutil.rmtree(path, ignore_errors=True)
