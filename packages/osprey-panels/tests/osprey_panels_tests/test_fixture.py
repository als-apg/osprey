"""The panel tests' loader hands out the checked-in fixture document.

The document is rebuilt from its tree and held to the build byte for byte by
the facility package suite
(``packages/osprey-facility/tests/osprey_facility_tests/test_panel_fixture_rebuild.py``).
What is tested here is this suite's own loader, ``_fixture.py``.
"""

from __future__ import annotations

import json

from osprey_panels_tests._fixture import FIXTURE_DOCUMENT, load_fixture_document


def test_the_loader_reads_the_checked_in_document_fresh_each_time() -> None:
    first = load_fixture_document()
    assert first == json.loads(FIXTURE_DOCUMENT.read_text(encoding="utf-8"))
    first["devices"].clear()
    assert load_fixture_document()["devices"], "each call returns its own copy"
