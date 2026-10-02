"""The built graph answers a sector question by place path and position in metres.

The beam position monitors under ``SR/`` whose section code is ``SECT1``,
ordered by ``sPositionM``, are BPM01 to BPM06 at the deck's positions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

# xdist_group("built_control_assistant"): every module reading the session's one
# control-assistant build shares a worker, so the build runs once per run.
pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("built_control_assistant")]

SECTOR_QUERY = """
PREFIX narad_p: <https://narad.example.org/property/>
PREFIX narad_sem: <https://narad.example.org/schema/shared_semantics/>
SELECT ?id ?s WHERE {
    ?device a narad_sem:BeamPositionMonitor ;
        narad_p:deviceId ?id ;
        narad_p:placePath ?path ;
        narad_p:sectionCode "SECT1" ;
        narad_p:sPositionM ?s .
    FILTER(STRSTARTS(?path, "SR/"))
}
ORDER BY ?s
"""

SECTOR_BPMS = [
    ("SR/BPM01", 4.800),
    ("SR/BPM02", 6.287),
    ("SR/BPM03", 8.800),
    ("SR/BPM04", 11.177),
    ("SR/BPM05", 13.690),
    ("SR/BPM06", 14.499),
]


def test_the_first_sector_bpms_by_position(built_control_assistant: BuiltProject) -> None:
    from rdflib import Graph

    from osprey.facility.views.graph import GRAPH_FILE

    graph = Graph().parse(
        built_control_assistant.build_dir / "data" / "graph" / GRAPH_FILE, format="turtle"
    )

    rows = [(str(row.id), float(row.s)) for row in graph.query(SECTOR_QUERY)]

    assert [device for device, _ in rows] == [device for device, _ in SECTOR_BPMS]
    for (_, position), (_, expected) in zip(rows, SECTOR_BPMS, strict=True):
        assert position == pytest.approx(expected, abs=1e-3)
