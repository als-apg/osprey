"""The built search index carries the graph view's places and positions.

A control-assistant render seeds its store from the graph view the build
writes, and its search index is derived from that same file: the index's
``corpus_sha256`` is the view's digest, and the beam position monitors under
``SR/`` in section ``SECT1``, ordered by ``s_position_m``, are BPM01 to BPM06
at the deck's positions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

# xdist_group("built_control_assistant"): every module reading the session's one
# control-assistant build shares a worker, so the build runs once per run.
pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("built_control_assistant")]

#: The corpus every render with a graph store is seeded from.
CORPUS = "./data/graph/facility.ttl"

#: Where the build writes the index, relative to the render.
INDEX = "data/channel_databases/graph.duckdb"

BPM_CLASS = "https://narad.example.org/schema/shared_semantics/BeamPositionMonitor"

SECTOR_QUERY = """
SELECT DISTINCT device_uri, s_position_m
FROM bindings
WHERE place_path LIKE 'SR/%'
  AND section = 'SECT1'
  AND list_contains(class_uris, ?)
ORDER BY s_position_m
"""

SECTOR_BPMS = [
    ("SR/BPM01", 4.800),
    ("SR/BPM02", 6.287),
    ("SR/BPM03", 8.800),
    ("SR/BPM04", 11.177),
    ("SR/BPM05", 13.690),
    ("SR/BPM06", 14.499),
]


def _read(built: BuiltProject, sql: str, params: list[object] | None = None) -> list[tuple]:
    import duckdb

    con = duckdb.connect(str(built.build_dir / INDEX), read_only=True)
    try:
        return con.execute(sql, params or []).fetchall()
    finally:
        con.close()


def test_the_render_seeds_its_store_from_the_graph_view(
    built_control_assistant: BuiltProject,
) -> None:
    import yaml

    config = yaml.safe_load((built_control_assistant.build_dir / "config.yml").read_text())

    assert config["services"]["graphdb"]["ttl_path"] == CORPUS


def test_the_index_digest_is_the_graph_views(built_control_assistant: BuiltProject) -> None:
    from osprey.services.facility_knowledge.seeder.graph_seeder import ttl_sha256

    corpus = built_control_assistant.build_dir / CORPUS
    expected = ttl_sha256(corpus.read_text(encoding="utf-8"))

    assert _read(built_control_assistant, "SELECT corpus_sha256 FROM meta") == [(expected,)]


def test_the_first_sector_bpms_by_position(built_control_assistant: BuiltProject) -> None:
    from osprey.facility.views.graph_iri import decode

    code = str(built_control_assistant.facility["identity"]["code"])

    rows = _read(built_control_assistant, SECTOR_QUERY, [BPM_CLASS])

    assert [decode(uri, code) for uri, _ in rows] == [bpm for bpm, _ in SECTOR_BPMS]
    assert [s for _, s in rows] == pytest.approx([s for _, s in SECTOR_BPMS], abs=1e-3)
