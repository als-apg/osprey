"""Throwaway installed packages that register measurement kinds, for the registry's tests.

Each directory beside this file is a ``sys.path`` entry holding one package and
its ``<name>.dist-info`` (``METADATA`` and ``entry_points.txt``), the shape an
installed wheel leaves: ``kind`` registers ``registry_kind``, ``collision``
registers a second ``orm``, ``unknown_hook`` a kind that names a hook OSPREY
does not offer. :func:`install` puts one on the path for one test.
"""

from __future__ import annotations

import functools
import importlib
import sys
from importlib import metadata
from pathlib import Path

import pytest

#: The kind the ``kind`` package registers.
REGISTRY_KIND = "registry_kind"

_HERE = Path(__file__).resolve().parent

#: Each site's top-level package, dropped from ``sys.modules`` when the site goes.
_PACKAGES = {
    "kind": "registry_kind_fixture",
    "collision": "registry_kind_collision",
    "unknown_hook": "registry_kind_hooks",
}


def install(monkeypatch: pytest.MonkeyPatch, site: str = "kind") -> Path:
    """Put the ``site`` package and its dist-info on ``sys.path`` for the test's duration.

    The suite reads the entry points once per session; a test that installs a
    distribution hands the code a registry read after the install, cached for
    that test alone.

    Args:
        monkeypatch: The test's monkeypatch, which takes the path entry back off.
        site: ``kind``, ``collision`` or ``unknown_hook``.

    Returns:
        The site directory.
    """
    directory = _HERE / site
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.delitem(sys.modules, _PACKAGES[site], raising=False)
    importlib.invalidate_caches()
    read = getattr(metadata.entry_points, "__wrapped__", metadata.entry_points)
    monkeypatch.setattr(metadata, "entry_points", functools.cache(read))
    return directory
