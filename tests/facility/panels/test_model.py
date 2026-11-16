"""The panel graph over the fixture facility and its rule tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osprey.facility.panels import PanelGraph, PanelTableError, load_graph
from tests.facility.panels._fixture import RING_FRAME, copy_tables, write_fixture_facility

EXCLUDED_THING = "XR02:SAF1"
EXCLUDED_PLACE = "XR/XR02/SAFE"
HIDDEN_THINGS = (EXCLUDED_PLACE, "XR/XR02/SAFE/CAB1", "XR02:SAF2")
HIDDEN_CHANNELS = ("XR02:SAF2:Door-Sts", "XR02:Cab1Temp-RB")
EXCLUDED_CHANNELS = ("XR02:SAF1:Trip-Sts", "XR02:IP2:Interlock-Sts")
VA_CHANNEL = "VA:XR:InjectionKnob"


def _every_channel(graph: PanelGraph) -> set[str]:
    return {channel.name for thing in graph.things() for channel in graph.channels(thing.id)}


def _every_edge_end(graph: PanelGraph) -> set[str]:
    return {
        edge.other
        for thing in graph.things()
        for edges in graph.neighbours(thing.id).values()
        for edge in edges
    }


def test_things_are_the_devices_places_and_groups(panel_graph: PanelGraph) -> None:
    kinds: dict[str, set[str]] = {}
    for thing in panel_graph.things():
        kinds.setdefault(thing.kind, set()).add(thing.id)
    assert kinds["place"] == {"XR", "XR/XR01", "XR/XR02", "XR/XR01/RK1", "XR/XR02/RK2"}
    assert kinds["group"] == {"XR01-VAC", "XR02-VAC", "XR-QUAD"}
    assert len(kinds["device"]) == 6 + 4 + 3 + 1 + 1 + 1 + 1  # XR02:BLOCK1, XR02:CH5


def test_a_thing_carries_kind_class_place_and_citations(panel_graph: PanelGraph) -> None:
    magnet = panel_graph.thing("XR01:QF1")
    assert (magnet.kind, magnet.cls, magnet.place) == ("device", "Quadrupole", "XR/XR01")
    (citation,) = magnet.citations
    assert (citation.document, citation.version, citation.state, citation.locator) == (
        "XR-LAYOUT",
        "A",
        "released",
        "sheet 1",
    )
    rack = panel_graph.thing("XR/XR01/RK1")
    assert (rack.kind, rack.cls, rack.place) == ("place", None, "XR/XR01")
    assert panel_graph.thing("XR").place is None
    assert panel_graph.thing("XR-QUAD").kind == "group"


def test_the_excluded_thing_is_absent_from_every_accessor(panel_graph: PanelGraph) -> None:
    assert EXCLUDED_THING not in {thing.id for thing in panel_graph.things()}
    for accessor in (panel_graph.thing, panel_graph.neighbours, panel_graph.channels):
        with pytest.raises(KeyError):
            accessor(EXCLUDED_THING)
    assert panel_graph.pose(EXCLUDED_THING) is None
    assert EXCLUDED_THING not in {thing.id for thing in panel_graph.members("XR02-VAC")}
    assert EXCLUDED_THING not in {thing.id for thing in panel_graph.within("XR", deep=True)}
    assert EXCLUDED_THING not in _every_edge_end(panel_graph)
    assert "interlocked_by" not in panel_graph.neighbours("XR02:IP1")


@pytest.mark.parametrize("name", [*EXCLUDED_CHANNELS, VA_CHANNEL])
def test_an_excluded_or_simulation_only_channel_is_absent(
    panel_graph: PanelGraph, name: str
) -> None:
    assert name not in _every_channel(panel_graph)
    with pytest.raises(KeyError):
        panel_graph.channel(name)
    assert panel_graph.rule(name) == []
    assert panel_graph.pair(name) is None


def test_a_pair_whose_readback_is_excluded_is_absent(tmp_path: Path) -> None:
    tables = copy_tables(tmp_path / "panels")
    with (tables / "excluded.csv").open("a", encoding="utf-8") as handle:
        handle.write("channel,XR01:QF1:Current-RB,stand-in for the test,XR-LAYOUT,A,sheet 1\n")
    graph = load_graph(write_fixture_facility(tmp_path / "facility.json"), tables)
    assert graph.pair("XR01:QF1:Current-SP") is None


def test_a_channel_carries_name_drawn_as_writable_and_alias(panel_graph: PanelGraph) -> None:
    channels = {channel.name: channel for channel in panel_graph.channels("XR01:QF1")}
    assert list(channels) == ["XR01:QF1:Current-RB", "XR01:QF1:Current-SP"]
    setpoint, readback = channels["XR01:QF1:Current-SP"], channels["XR01:QF1:Current-RB"]
    assert (setpoint.drawn_as, setpoint.writable, setpoint.alias) == ("setpoint", True, None)
    assert (readback.drawn_as, readback.writable, readback.alias) == (
        "value",
        False,
        "XR-ML:QF1:AM",
    )
    assert readback.record_type == "ai"
    assert readback.thing == "XR01:QF1"
    assert panel_graph.channel("XR01:QD1:Current-RB").drawn_as is None


def test_a_place_holds_the_channels_on_it(panel_graph: PanelGraph) -> None:
    assert [channel.name for channel in panel_graph.channels("XR/XR01/RK1")] == ["XR01:RackTemp-RB"]
    assert panel_graph.channels("XR") == []


def test_neighbours_are_keyed_by_relation_kind_in_both_directions(
    panel_graph: PanelGraph,
) -> None:
    supply = panel_graph.neighbours("XR01:PS1")
    assert set(supply) == {"powered_by"}
    assert {(edge.other, edge.outgoing) for edge in supply["powered_by"]} == {
        ("XR01:QF1", False),
        ("XR01:QD1", False),
    }
    (fed,) = panel_graph.neighbours("XR01:QF1")["powered_by"]
    assert (fed.other, fed.other_kind, fed.outgoing, fed.standing, fed.basis) == (
        "XR01:PS1",
        "device",
        True,
        "accepted",
        "stated",
    )
    (racked,) = panel_graph.neighbours("XR01:CH1")["located_in"]
    assert (racked.other, racked.other_kind, racked.unit) == ("XR/XR01/RK1", "place", 10)
    assert racked.source is not None and racked.source.document == "XR-RACKS"
    assert panel_graph.neighbours("XR01:CH3") == {}


def _add_unknown_relation(document: dict[str, Any]) -> None:
    document["relations"].append(
        {
            "id": "xr01-ip2-custom",
            "type": "xr_custom_link",
            "subject": {"device": "XR01:IP2"},
            "object": {"device": "XR01:CH3"},
            "standing": "accepted",
        }
    )


def test_an_unknown_relation_kind_is_kept_under_its_own_key(tmp_path: Path) -> None:
    facility = write_fixture_facility(tmp_path / "facility.json", _add_unknown_relation)
    graph = load_graph(facility, copy_tables(tmp_path / "panels"))
    (edge,) = graph.neighbours("XR01:IP2")["xr_custom_link"]
    assert (edge.kind, edge.other, edge.outgoing) == ("xr_custom_link", "XR01:CH3", True)
    (back,) = graph.neighbours("XR01:CH3")["xr_custom_link"]
    assert (back.other, back.outgoing) == ("XR01:IP2", False)


def test_members_and_within(panel_graph: PanelGraph) -> None:
    assert [thing.id for thing in panel_graph.members("XR01-VAC")] == ["XR01:IP1", "XR01:IP2"]
    assert [thing.id for thing in panel_graph.members("XR02-VAC")] == ["XR02:IP1", "XR02:IP2"]
    direct = {thing.id for thing in panel_graph.within("XR/XR01")}
    assert "XR/XR01/RK1" in direct
    assert "XR01:QF1" in direct
    assert "XR01:CH1" not in direct
    deep = {thing.id for thing in panel_graph.within("XR/XR01", deep=True)}
    assert {"XR01:CH1", "XR01:CH2", "XR01:CH3"} <= deep
    assert {thing.id for thing in panel_graph.within("XR")} == {"XR/XR01", "XR/XR02"}
    with pytest.raises(KeyError):
        panel_graph.members("XR/XR01")
    with pytest.raises(KeyError):
        panel_graph.within("XR01:QF1")


def test_the_ring_frame_is_the_frame_with_most_true_positions(panel_graph: PanelGraph) -> None:
    assert panel_graph.ring_frame == RING_FRAME


def test_pose_prefers_the_ring_frame_row(panel_graph: PanelGraph) -> None:
    pose = panel_graph.pose("XR01:QF1")
    assert pose is not None
    assert (pose.frame, pose.s_m, pose.is_ring_position) == (RING_FRAME, 1.25, True)
    assert [row.frame for row in panel_graph.poses("XR01:QF1")] == [RING_FRAME, "XR-GIRDER"]
    rack = panel_graph.pose("XR/XR01/RK1")
    assert rack is not None
    assert (rack.frame, rack.is_ring_position) == ("XR-HALL", False)


def test_a_derived_ring_row_is_not_a_ring_position(panel_graph: PanelGraph) -> None:
    pose = panel_graph.pose("XR01:BEND1")
    assert pose is not None
    assert pose.frame == RING_FRAME
    assert pose.derived_assumes
    assert not pose.is_ring_position


def test_two_things_at_one_s_are_both_placed(panel_graph: PanelGraph) -> None:
    magnet, pump = panel_graph.pose("XR01:QF1"), panel_graph.pose("XR01:IP1")
    assert magnet is not None and pump is not None
    assert magnet.s_m == pump.s_m


def test_a_thing_with_a_pose_and_no_channel(panel_graph: PanelGraph) -> None:
    assert panel_graph.channels("XR02:SHIELD1") == []
    assert panel_graph.pose("XR02:SHIELD1") is not None
    assert panel_graph.pose("XR02:QD1") is None


def test_the_ring_frame_can_be_passed(panel_facility: Path, panel_tables: Path) -> None:
    graph = load_graph(panel_facility, panel_tables, ring_frame="XR-GIRDER")
    pose = graph.pose("XR01:QF1")
    assert pose is not None
    assert (pose.frame, pose.is_ring_position) == ("XR-GIRDER", True)
    other = graph.pose("XR02:QF1")
    assert other is not None
    assert (other.frame, other.is_ring_position) == (RING_FRAME, False)


def test_rule_returns_every_row_of_every_source(panel_graph: PanelGraph) -> None:
    rules = panel_graph.rule("XR01:IP1:State-Sts")
    assert [(rule.value, rule.state, rule.source) for rule in rules] == [
        ("0", "bad", "xr-screens"),
        ("0", "normal", "xr-ioc"),
        ("1", "normal", "xr-screens"),
    ]
    assert panel_graph.rule("XR01:QF1:Current-RB") == []


def test_pair_comes_from_the_table_only(panel_graph: PanelGraph) -> None:
    stated = panel_graph.pair("XR01:QF1:Current-SP")
    assert stated is not None
    assert (stated.readback, stated.basis, stated.source) == (
        "XR01:QF1:Current-RB",
        "stated",
        "xr-ioc",
    )
    inferred = panel_graph.pair("XR01:PS1:On-Cmd")
    assert inferred is not None
    assert (inferred.readback, inferred.basis) == ("XR01:PS1:On-Sts", "inferred")
    # The facility file pairs these, but pairs.csv does not: no pair.
    assert panel_graph.pair("XR02:QF1:Current-SP") is None
    assert panel_graph.pair("XR01:QD1:Current-SP") is None
    assert panel_graph.pair("XR01:QF1:Current-RB") is None


def test_an_excluded_document_drops_what_it_sources(tmp_path: Path, panel_facility: Path) -> None:
    tables = copy_tables(tmp_path / "panels")
    with (tables / "excluded.csv").open("a", encoding="utf-8") as handle:
        handle.write("document,XR-RACKS,stand-in for the test,XR-LAYOUT,A,sheet 1\n")
    graph = load_graph(panel_facility, tables)
    assert "located_in" not in graph.neighbours("XR01:CH1")
    assert graph.pose("XR/XR01/RK1") is None
    assert graph.pose("XR01:QF1") is not None


def test_an_unknown_id_is_a_key_error(panel_graph: PanelGraph) -> None:
    with pytest.raises(KeyError, match="XR09:NOPE"):
        panel_graph.thing("XR09:NOPE")


def _every_name(graph: PanelGraph) -> set[str]:
    """Every id and channel name any accessor of the graph returns."""
    names: set[str] = set()
    for thing in graph.things():
        names.add(thing.id)
        if thing.place is not None:
            names.add(thing.place)
        names |= {channel.name for channel in graph.channels(thing.id, thing.kind)}
        names |= {channel.thing for channel in graph.channels(thing.id, thing.kind)}
        for edges in graph.neighbours(thing.id, thing.kind).values():
            names |= {edge.other for edge in edges}
        if thing.kind == "group":
            names |= {member.id for member in graph.members(thing.id)}
        if thing.kind == "place":
            names |= {inner.id for inner in graph.within(thing.id, deep=True)}
        names |= {pose.thing for pose in graph.poses(thing.id)}
    return names


def test_an_excluded_place_hides_everything_within_it(panel_graph: PanelGraph) -> None:
    names = _every_name(panel_graph)
    for hidden in (*HIDDEN_THINGS, *HIDDEN_CHANNELS):
        assert hidden not in names
    for thing_id in HIDDEN_THINGS:
        with pytest.raises(KeyError):
            panel_graph.thing(thing_id)
        assert panel_graph.pose(thing_id) is None
        assert panel_graph.poses(thing_id) == []
    for name in HIDDEN_CHANNELS:
        with pytest.raises(KeyError):
            panel_graph.channel(name)
        assert panel_graph.rule(name) == []
    assert {edge.other for edge in panel_graph.neighbours("XR01:PS1")["powered_by"]} == {
        "XR01:QF1",
        "XR01:QD1",
    }
    assert {thing.id for thing in panel_graph.within("XR/XR02")} == {
        "XR02:QF1",
        "XR02:QD1",
        "XR02:BEND1",
        "XR02:IP1",
        "XR02:IP2",
        "XR02:SHIELD1",
        "XR02:BLOCK1",
        "XR/XR02/RK2",
    }


def test_the_fixture_has_the_hidden_records(panel_facility: Path) -> None:
    import json

    document = json.loads(panel_facility.read_text(encoding="utf-8"))
    ids = {record["id"] for key in ("places", "devices", "channels") for record in document[key]}
    assert set(HIDDEN_THINGS) | set(HIDDEN_CHANNELS) <= ids


def test_a_ring_row_without_s_is_not_a_ring_position(panel_graph: PanelGraph) -> None:
    pose = panel_graph.pose("XR02:IP1")
    assert pose is not None
    assert (pose.frame, pose.s_m, pose.derived_assumes) == (RING_FRAME, None, None)
    assert pose.in_ring_frame
    assert not pose.is_ring_position


def test_poses_put_the_ring_row_first_then_file_order(panel_graph: PanelGraph) -> None:
    assert [pose.frame for pose in panel_graph.poses("XR01:QD1")] == [RING_FRAME, "XR-GIRDER"]
    assert [pose.in_ring_frame for pose in panel_graph.poses("XR01:QD1")] == [True, False]
    assert len(panel_graph.poses("XR02:SHIELD1")) == 1
    assert panel_graph.poses("XR02:QD1") == []
    assert panel_graph.poses(EXCLUDED_THING) == []
    assert panel_graph.poses("XR09:NOPE") == []


def test_a_tie_between_ring_frames_is_refused(tmp_path: Path, panel_facility: Path) -> None:
    tables = copy_tables(tmp_path / "panels")
    header = (tables / "positions.csv").read_text(encoding="utf-8").splitlines()[0]
    rows = [
        "XR01:QF1,XR-A,0,0,0,0,0,0,1.0,0,0,,,stated,XR-LAYOUT,A,released,sheet 1",
        "XR01:QD1,XR-B,0,0,0,0,0,0,2.0,0,0,,,stated,XR-LAYOUT,A,released,sheet 1",
    ]
    (tables / "positions.csv").write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    with pytest.raises(PanelTableError, match=r"positions\.csv.*XR-A, XR-B.*ring frame") as raised:
        load_graph(panel_facility, tables)
    assert raised.value.line is None
    graph = load_graph(panel_facility, tables, ring_frame="XR-B")
    assert graph.ring_frame == "XR-B"
    assert graph.pose("XR01:QD1").is_ring_position  # type: ignore[union-attr]


def test_rules_lists_every_row_of_every_held_channel(panel_graph: PanelGraph) -> None:
    rows = panel_graph.rules()
    # 16 table rows, less the one on an excluded thing, the one on a hidden thing
    # and the one on a simulation-only channel
    assert len(rows) == 13
    assert {row.channel for row in rows}.isdisjoint({VA_CHANNEL, *HIDDEN_CHANNELS})


def test_a_place_carries_its_level_and_a_device_none(panel_graph):
    assert panel_graph.thing("XR/XR01/RK1").level == "rack"
    assert panel_graph.thing("XR/XR01").level == "sector"
    assert panel_graph.thing("XR01:QF1").level is None


def test_a_channel_carries_its_stated_tolerance(panel_graph):
    assert panel_graph.channel("XR01:QF1:Current-SP").tolerance == ("absolute", 0.05)
    assert panel_graph.channel("XR01:QD1:Current-SP").tolerance is None


def test_record_type_gives_the_table_row_of_a_held_channel(panel_graph):
    row = panel_graph.record_type("XR01:QF1:Current-SP")
    assert row is not None and (row.record_type, row.source) == ("ao", "xr-ioc")
    assert panel_graph.record_type("XR01:QD1:Current-SP") is None
    assert panel_graph.record_type(VA_CHANNEL) is None
    assert panel_graph.record_type("XR02:SAF1:Trip-Sts") is None  # on an excluded thing


def test_pictures_give_a_class_rows_in_file_order(panel_graph):
    parts = panel_graph.pictures("Quadrupole")
    assert [(part.view, part.part) for part in parts] == [
        ("end", "pole 1"),
        ("end", "thermal switch 1"),
    ]
    assert panel_graph.pictures("Dipole") == []
