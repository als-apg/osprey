"""The ``canopy-screens`` skill: a build-time guide to the project's CANOPY screen library.

The guide is read from the library the deployment ships. A project without the CANOPY
panel renders none.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.integrations.canopy.screen_guide import compress, screen_guide_context


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["BPM 39", "BPM 37", "BPM 38"], ["BPM 37–39"]),
        (["EPU7", "EPU8"], ["EPU7", "EPU8"]),
        ([f"BPM {n:02d}" for n in range(1, 73)], ["BPM 01–72"]),
        (["QF 13", "QF 14", "SEK", "IG ST07A"], ["IG ST07A", "QF 13", "QF 14", "SEK"]),
        (["U1", "U14", "U9", "U10", "U11"], ["U1", "U9", "U10", "U11", "U14"]),
    ],
)
def test_compress_writes_numbered_runs_first_to_last(labels, expected) -> None:
    assert compress(labels) == expected


def test_a_project_without_the_canopy_panel_gets_no_guide(tmp_path: Path) -> None:
    assert screen_guide_context({"web": {"panels": {}}}, tmp_path) is None
    assert (
        screen_guide_context({"web": {"panels": {"canopy": {"enabled": False}}}}, tmp_path) is None
    )
