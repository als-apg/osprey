"""The panel tests' loader hands out the checked-in fixture document.

The document is rebuilt from its tree and held to the build byte for byte by
the package suite
(``packages/osprey-facility/tests/osprey_facility_tests/test_panel_fixture_rebuild.py``),
which needs neither the panels nor the framework. What stays here is the root
panel tests' own loader, ``tests/facility/panels/_fixture.py``.
"""

from __future__ import annotations

import json

from tests.facility.panels._fixture import FACILITY_TREE, FIXTURE_DOCUMENT, load_fixture_document


def test_the_loader_reads_the_checked_in_document_fresh_each_time() -> None:
    first = load_fixture_document()
    assert first == json.loads(FIXTURE_DOCUMENT.read_text(encoding="utf-8"))
    first["devices"].clear()
    assert load_fixture_document()["devices"], "each call returns its own copy"


def test_the_fixture_tree_is_read_from_the_package_suite() -> None:
    assert (FACILITY_TREE / "documents.yaml").is_file()
    assert FACILITY_TREE.parts[-4:] == (
        "osprey-facility",
        "tests",
        "osprey_facility_tests",
        "panel_fixture",
    )
