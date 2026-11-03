"""The storage ring's beam lifetime and operation mode are findable in every channel-finder view.

``SR:DIAG:DCCT:01:LIFETIME:RB`` and ``SR:OPS:MODE:RB`` are records of the demo
facility. An operator asking for the beam lifetime or the operation mode finds
each one by the words they would use: in the graph view's search, as a leaf of
the hierarchical view, and as a row of the in_context view with its
description.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from tests._builds import BuiltProject

LIFETIME = "SR:DIAG:DCCT:01:LIFETIME:RB"
MODE = "SR:OPS:MODE:RB"

#: The words an operator asks with, each finding its record.
ASKED_FOR = {
    LIFETIME: (["beam", "lifetime"], ["lifetime"]),
    MODE: (["operation", "mode"], ["operation mode"]),
}


def _inputs(doc: dict[str, Any]) -> Any:
    from osprey.facility.records import FacilityRecords
    from osprey.facility.views import ViewInputs

    return ViewInputs(
        records=FacilityRecords(doc), rendered_config={}, facility_dir=Path("."), served=[]
    )


@pytest.fixture(scope="module")
def graph_index_path(
    built_control_assistant: BuiltProject, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    from osprey.services.channel_finder.graph_index.builder import parse_corpus
    from tests.services.channel_finder.graph_index.test_search import _write

    view = built_control_assistant.build_dir / "data" / "graph" / "facility.ttl"
    parsed = parse_corpus(view.read_text(encoding="utf-8"))
    return _write(parsed, tmp_path_factory.mktemp("findability") / "graph.duckdb")


@pytest.mark.slow
@pytest.mark.parametrize("address", sorted(ASKED_FOR))
def test_the_graph_search_finds_the_record_by_its_words_and_its_address(
    graph_index_path: Path, address: str
) -> None:
    from tests.services.channel_finder.graph_index.test_search import _open

    with _open(graph_index_path) as index:
        for tokens in ASKED_FOR[address]:
            payload = index.search(tokens=tokens, page_size=500)
            found = [row["fullPv"] for row in payload["rows"]]
            assert address in found, (tokens, found)
        exact = index.search(tokens=[address])
    assert [row["fullPv"] for row in exact["rows"]] == [address]
    (row,) = exact["rows"]
    assert row["description"]
    assert row["section"] == "SR"


@pytest.mark.slow
def test_the_hierarchical_view_holds_each_record_as_a_leaf(
    built_control_assistant: BuiltProject, tmp_path: Path
) -> None:
    from osprey.facility.views.channel_finder import write_hierarchical
    from osprey.services.channel_finder.databases.hierarchical import (
        HierarchicalChannelDatabase,
    )

    (target,) = write_hierarchical(
        tmp_path / "channel_finder", _inputs(built_control_assistant.facility)
    )
    database = HierarchicalChannelDatabase(str(target))

    for address in ASKED_FOR:
        entry = database.channel_map[address]
        assert entry["channel"] == address
        assert entry["path"]["leaf"] == address


@pytest.mark.slow
def test_the_in_context_view_holds_each_record_with_its_description(
    built_control_assistant: BuiltProject, tmp_path: Path
) -> None:
    from osprey.facility.views.channel_finder import write_in_context

    (target,) = write_in_context(tmp_path, _inputs(built_control_assistant.facility))
    rows = {row["address"]: row for row in json.loads(target.read_bytes())["channels"]}
    described = {
        str(channel["id"]): channel["description"]
        for channel in built_control_assistant.facility["channels"]
    }

    for address in ASKED_FOR:
        assert rows[address]["channel"] == address
        assert rows[address]["description"] == described[address]
    assert "lifetime" in rows[LIFETIME]["description"].lower()
    assert "operation mode" in rows[MODE]["description"].lower()
