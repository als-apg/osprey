"""Restore the process-wide state a ``raw_put_block.install`` call changes.

``install`` patches attributes in place on whatever its rows resolve to, and
follows each patched attribute back to the module that defined it, so a fake
row can reach well past the fake module a test registered — a fake function
defined in a test module has that test module as its home. It also leaves an
import-hook finder on ``sys.meta_path``. A test that installs the block must
put all of that back, or the next test runs against a patched process.

:func:`patch_restore` snapshots the namespaces a test names (modules and
classes) together with ``sys.meta_path``, and on exit restores every attribute
that changed, removes every attribute that was added, and drops every guard
finder. The :func:`restore_patches` fixture wraps it for pytest: call the
yielded ``track(*namespaces)`` before ``install`` for each object the rows can
reach.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

#: The attribute every finder and loader the block installs carries.
GUARD_ATTR = "_osprey_readonly_guard"

_MISSING = object()


class _Snapshot:
    """The attributes of a set of namespaces at one moment."""

    def __init__(self) -> None:
        self._saved: list[tuple[Any, dict[str, Any]]] = []
        self._seen: set[int] = set()

    def track(self, *namespaces: Any) -> None:
        for namespace in namespaces:
            if id(namespace) in self._seen:
                continue
            self._seen.add(id(namespace))
            self._saved.append((namespace, dict(vars(namespace))))

    def restore(self) -> None:
        for namespace, saved in reversed(self._saved):
            current = vars(namespace)
            for name in [n for n in current if n not in saved]:
                try:
                    delattr(namespace, name)
                except (AttributeError, TypeError):
                    pass
            for name, value in saved.items():
                if current.get(name, _MISSING) is not value:
                    setattr(namespace, name, value)


def drop_guard_finders() -> None:
    """Remove every finder the block registered from ``sys.meta_path``."""
    sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, GUARD_ATTR, False)]


@contextmanager
def patch_restore(*namespaces: Any) -> Iterator[_Snapshot]:
    """Snapshot *namespaces* and ``sys.meta_path``; restore both on exit.

    More namespaces can be added through the yielded snapshot's ``track``.
    """
    snapshot = _Snapshot()
    snapshot.track(*namespaces)
    meta_path = list(sys.meta_path)
    try:
        yield snapshot
    finally:
        snapshot.restore()
        sys.meta_path[:] = meta_path
        drop_guard_finders()


@pytest.fixture
def restore_patches() -> Iterator[Any]:
    """Yield ``track(*namespaces)``; everything tracked is restored afterwards."""
    with patch_restore() as snapshot:
        yield snapshot.track
