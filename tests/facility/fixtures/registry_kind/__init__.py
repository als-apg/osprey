"""Throwaway installed packages that register measurement kinds, for the registry's tests.

Each directory beside this file is a ``sys.path`` entry holding one package and
its ``<name>.dist-info`` (``METADATA`` and ``entry_points.txt``), the shape an
installed wheel leaves: ``kind`` registers ``registry_kind``, ``collision``
registers a second ``orm``, ``unknown_hook`` a kind that names a hook OSPREY
does not offer. :func:`install` puts one on the path for one test.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from tests.facility._fixture_kind import read_entry_points_afresh

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

    The suite reads the entry points once per session; the install drops that
    reading, so the code reads a registry that holds the package. A module that
    calls this requests the ``entry_points_read_afresh`` fixture
    (``tests/facility/conftest.py``), which drops it again once the test is over.

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
    read_entry_points_afresh()
    return directory
