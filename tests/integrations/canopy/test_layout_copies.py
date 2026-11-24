"""The canopy package's own spellings of its paths under ``data/`` equal the declaration.

The package imports nothing outside the standard library, so it cannot read
:mod:`osprey.repo_layout`; it keeps private copies of the screen library's and
the facility collections' positions, and this test holds each equal to the
child the deployment-root declaration names.
"""

from __future__ import annotations

from pathlib import Path

from osprey.integrations.canopy import FACILITY_COLLECTIONS_SUBPATH, LIBRARY_SUBPATH
from osprey.repo_layout import child_display_path


def test_the_screen_library_copy_equals_the_declared_child() -> None:
    assert Path(child_display_path("data", "canopy-library")) == LIBRARY_SUBPATH


def test_the_facility_collections_copy_equals_the_declared_child() -> None:
    assert Path(child_display_path("data", "canopy-collections")) == FACILITY_COLLECTIONS_SUBPATH
