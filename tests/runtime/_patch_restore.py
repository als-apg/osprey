"""Restore the process-wide state a ``raw_put_block.install`` call changes.

``install`` patches attributes in place on whatever its rows resolve to, and
follows each patched attribute back to the module that defined it, so a fake
row can reach well past the fake module a test registered — a fake function
defined in a test module has that test module as its home. It also leaves an
import-hook finder on ``sys.meta_path``. A test that installs the block must
put all of that back, or the next test runs against a patched process.

:func:`patch_restore` snapshots the namespaces a test names (modules and
classes), and on exit restores every attribute that changed, removes every
attribute that was added, and drops only the guard finders from
``sys.meta_path``, leaving every other finder where it is. It also restores the shared-library loaders the ``no_client`` mode
replaces whatever the test names: ``ctypes._dlopen``, ``_ctypes.dlopen`` and,
where ``cffi`` is importable, ``cffi.api.FFI.dlopen``. The
:func:`restore_patches` fixture wraps it for pytest: call the yielded
``track(*namespaces)`` before ``install`` for each object the rows can reach.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

import pytest

from osprey.runtime import raw_put_block

#: The attribute every finder and loader the block installs carries.
GUARD_ATTR = raw_put_block._GUARD_ATTR

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
                with suppress(AttributeError, TypeError):
                    delattr(namespace, name)
            for name, value in saved.items():
                if current.get(name, _MISSING) is not value:
                    setattr(namespace, name, value)


def _loader_slots() -> list[tuple[Any, str, Any]]:
    """Each shared-library loader the ``no_client`` mode replaces, with its value now."""
    import _ctypes
    import ctypes

    owners: list[tuple[Any, str]] = [(ctypes, "_dlopen")]
    owners += [(_ctypes, attr) for attr in ("dlopen", "LoadLibrary") if hasattr(_ctypes, attr)]
    try:
        from cffi.api import FFI
    except ImportError:
        pass
    else:
        owners.append((FFI, "dlopen"))
    return [(owner, attr, getattr(owner, attr)) for owner, attr in owners]


def _is_guard_finder(finder: Any) -> bool:
    return bool(getattr(finder, GUARD_ATTR, False))


def guard_finders() -> list[Any]:
    """The finders on ``sys.meta_path`` the block registered."""
    return [f for f in sys.meta_path if _is_guard_finder(f)]


def drop_guard_finders() -> None:
    """Remove every finder the block registered from ``sys.meta_path``."""
    sys.meta_path[:] = [f for f in sys.meta_path if not _is_guard_finder(f)]


@contextmanager
def patch_restore(*namespaces: Any) -> Iterator[_Snapshot]:
    """Snapshot *namespaces* and the loaders; on exit restore them and drop only the guard finders.

    Every other finder on ``sys.meta_path`` stays where it is, including one
    another library inserted while the block was open.

    More namespaces can be added through the yielded snapshot's ``track``.
    """
    snapshot = _Snapshot()
    snapshot.track(*namespaces)
    loaders = _loader_slots()
    try:
        yield snapshot
    finally:
        snapshot.restore()
        for owner, attr, value in loaders:
            setattr(owner, attr, value)
        drop_guard_finders()


@pytest.fixture
def restore_patches() -> Iterator[Any]:
    """Yield ``track(*namespaces)``; everything tracked is restored afterwards."""
    with patch_restore() as snapshot:
        yield snapshot.track
