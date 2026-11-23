"""Reading the installed distributions' entry points afresh.

A test that installs a distribution (a fixture kind package, say) must not see,
nor leave behind, a reading of the entry points made without it. The package
suite does not memoize ``importlib.metadata.entry_points`` itself, but a run
from the repository root may (the framework suite wraps it in
``functools.cache`` once per session), and a run that loads both suites can
stack one memo on another. So the reset clears every cached layer it finds, not
only the outermost.

Nothing here imports ``osprey`` or the framework's ``tests`` package.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from importlib import metadata

import pytest


def read_entry_points_afresh() -> None:
    """Drop every cached reading of the entry points, and the import system's path caches."""
    reader = metadata.entry_points
    seen: set[int] = set()
    while reader is not None and id(reader) not in seen:
        seen.add(id(reader))
        clear = getattr(reader, "cache_clear", None)
        if callable(clear):
            clear()
        reader = getattr(reader, "__wrapped__", None)
    importlib.invalidate_caches()


@pytest.fixture
def entry_points_read_afresh() -> Iterator[None]:
    """The test reads the entry points of its own path, and leaves none read behind.

    For a module that installs a fixture distribution: any cached reading is
    dropped before the test and again after it, so a reading made while the
    distribution was on the path does not outlive it.
    """
    read_entry_points_afresh()
    yield
    read_entry_points_afresh()
