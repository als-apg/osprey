"""Scope-aware ``os.environ`` restoration, used by ``tests/conftest.py``.

Two guards there snapshot and restore the environment: one per test, one per
test *module*. The module-scoped one exists because a module-scoped fixture --
a real ``osprey build`` over a seeded repo, which loads that repo's ``.env``
with override semantics -- is set up *before* the per-test snapshot is taken,
so its writes land inside the copy every later test is restored to.

Restoring a module's snapshot wholesale would be wrong, though: pytest sets a
fixture up on first request, so a *session*-scoped fixture can be created
part-way through a module. Its writes are made after that module's snapshot,
and the fixture instance outlives the module -- rolling them back at module
teardown leaves a live fixture whose environment is gone, and the next module
runs against it without the variables it published.

The environment alone cannot say which fixture wrote a value, so this module
asks pytest instead: :func:`pytest_fixture_setup` measures the setup of every
fixture that outlives a module, and :func:`restore_module_environment` replays
what it recorded on top of the snapshot it puts back.

``tests/conftest.py`` registers this as a *plugin* rather than keeping the hook
in its own body. ``pytest_fixture_setup`` is dispatched on the node that owns
the fixture's scope, and a session-scoped fixture's node is the Session --
whose hook proxy carries only the conftests above it, which ``tests/conftest.py``
is not. A plainly registered plugin is in scope for every node, which is exactly
the set this has to see.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import pytest

#: Env writes made while a fixture that outlives a test module was being set
#: up: key -> the value that setup left behind, or ``None`` when it removed the
#: key. Reset on the way into every guarded region, so it only ever holds
#: writes made after the snapshot that region is restored to.
LONG_LIVED_ENV_WRITES: dict[str, str | None] = {}

#: Fixture scopes whose instances outlive the module that first requests them.
SCOPES_OUTLIVING_A_MODULE = frozenset({"session", "package"})

#: The guarded regions currently open, outermost first. Each entry is the record
#: its region set aside on the way in. Only :func:`restore_module_environment`
#: touches it; it is what tells an exiting region whether anything encloses it.
_OPEN_REGIONS: list[dict[str, str | None]] = []


def record_env_delta(before: Mapping[str, str]) -> None:
    """Record every environment key that changed since *before*.

    Args:
        before: The environment as it stood before the setup being measured.
    """
    for key in set(before) | set(os.environ):
        if before.get(key) != os.environ.get(key):
            LONG_LIVED_ENV_WRITES[key] = os.environ.get(key)


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(fixturedef, request):
    """Measure the environment across the setup of a longer-lived fixture.

    A pluggy *wrapper*: it does not produce the fixture value, it brackets
    whoever does. Only scopes that outlive a module are measured -- recording a
    module-scoped fixture's writes here would mark them long-lived and defeat
    :func:`restore_module_environment` entirely.
    """
    if fixturedef.scope not in SCOPES_OUTLIVING_A_MODULE:
        return (yield)
    before = dict(os.environ)
    try:
        return (yield)
    finally:
        record_env_delta(before)


@contextmanager
def restore_module_environment() -> Iterator[None]:
    """Restore ``os.environ`` at module teardown, minus the long-lived writes.

    Snapshots the environment on the way in and puts that snapshot back on the
    way out, then re-applies everything :func:`record_env_delta` attributed to a
    fixture that outlives this module. A module-scoped fixture's writes are in
    neither set, so they are the ones that go away.

    Re-entrant, because the autouse fixture wraps every test in this repo and a
    test of this guard therefore opens a second region inside it. The record is
    a single module-level dict, so an inner region borrows it: it sets the
    record aside on the way in and puts it back on the way out, with its own
    long-lived writes folded in -- those keys belong to a fixture that outlives
    the *enclosing* region too, and clearing the record outright would leave
    that region rolling a live fixture's environment back under it.

    A region that encloses nothing hands its record to nobody: one module's
    long-lived writes are not the next module's to replay, and carrying them
    would grow the record for the length of the run.
    """
    saved = dict(os.environ)
    _OPEN_REGIONS.append(dict(LONG_LIVED_ENV_WRITES))
    LONG_LIVED_ENV_WRITES.clear()
    try:
        yield
    finally:
        own = dict(LONG_LIVED_ENV_WRITES)
        enclosing = _OPEN_REGIONS.pop()
        os.environ.clear()
        os.environ.update(saved)
        for key, value in own.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        LONG_LIVED_ENV_WRITES.clear()
        if _OPEN_REGIONS:
            LONG_LIVED_ENV_WRITES.update(enclosing)
            LONG_LIVED_ENV_WRITES.update(own)
