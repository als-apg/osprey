"""The panel graph over the fixture facility and its rule tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osprey.facility.panels import PanelGraph, load_graph
from tests.facility.panels._fixture import (
    PANELS_DIR,
    RING_FRAME,
    add_part,
    copy_tables,
    graph_with_part,
    write_fixture_facility,
)

EXCLUDED_THING = "XR02:SAF1"
EXCLUDED_PLACE = "XR/XR02/SAFE"
HIDDEN_THINGS = (EXCLUDED_PLACE, "XR/XR02/SAFE/CAB1", "XR02:SAF2")
HIDDEN_CHANNELS = ("XR02-SAF2:Door-Sts", "XR02:Cab1Temp-RB")
EXCLUDED_CHANNELS = ("XR02-SAF1:Trip-Sts", "XR02-IP2:Interlock-Sts")
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
    # XR02:BLOCK1, XR02:CH5, XR09:LOOSE1
    assert len(kinds["device"]) == 6 + 4 + 3 + 1 + 1 + 1 + 1 + 1


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
        handle.write("channel,XR01-QF1:Current-RB,stand-in for the test,XR-LAYOUT,A,sheet 1\n")
    graph = load_graph(write_fixture_facility(tmp_path / "facility.json"), tables)
    assert graph.pair("XR01-QF1:Current-SP") is None


def test_a_channel_carries_name_drawn_as_writable_and_alias(panel_graph: PanelGraph) -> None:
    channels = {channel.name: channel for channel in panel_graph.channels("XR01:QF1")}
    assert list(channels) == ["XR01-QF1:Current-RB", "XR01-QF1:Current-SP"]
    setpoint, readback = channels["XR01-QF1:Current-SP"], channels["XR01-QF1:Current-RB"]
    assert (setpoint.drawn_as, setpoint.writable, setpoint.alias) == ("setpoint", True, None)
    assert (readback.drawn_as, readback.writable, readback.alias) == (
        "value",
        False,
        "XR-ML:QF1:AM",
    )
    assert readback.record_type == "ai"
    assert readback.thing == "XR01:QF1"
    # no record_types.csv row: drawn by its role in the facility file
    assert panel_graph.channel("XR01-QD1:Current-RB").drawn_as == "value"


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


def test_the_beam_frame_is_the_frame_with_most_true_positions(panel_graph: PanelGraph) -> None:
    assert panel_graph.beam_frames == frozenset({RING_FRAME})  # XR-GIRDER: a local frame


def test_pose_prefers_the_beam_frame_row(panel_graph: PanelGraph) -> None:
    pose = panel_graph.pose("XR01:QF1")
    assert pose is not None
    assert (pose.frame, pose.s_m, pose.is_beam_position) == (RING_FRAME, 1.25, True)
    assert [row.frame for row in panel_graph.poses("XR01:QF1")] == [RING_FRAME, "XR-GIRDER"]
    rack = panel_graph.pose("XR/XR01/RK1")
    assert rack is not None
    assert (rack.frame, rack.is_beam_position) == ("XR-HALL", False)


def test_a_derived_ring_row_is_not_a_beam_position(panel_graph: PanelGraph) -> None:
    pose = panel_graph.pose("XR01:BEND1")
    assert pose is not None
    assert pose.frame == RING_FRAME
    assert pose.derived_assumes
    assert not pose.is_beam_position


def test_two_things_at_one_s_are_both_placed(panel_graph: PanelGraph) -> None:
    magnet, pump = panel_graph.pose("XR01:QF1"), panel_graph.pose("XR01:IP1")
    assert magnet is not None and pump is not None
    assert magnet.s_m == pump.s_m


def test_a_thing_with_a_pose_and_no_channel(panel_graph: PanelGraph) -> None:
    assert panel_graph.channels("XR02:SHIELD1") == []
    assert panel_graph.pose("XR02:SHIELD1") is not None
    assert panel_graph.pose("XR02:QD1") is None


def test_each_model_has_its_own_beam_frame(tmp_path: Path, panel_facility: Path) -> None:
    import json

    document = json.loads(panel_facility.read_text(encoding="utf-8"))
    for device in document["devices"]:
        if device["id"].startswith("XR02:"):
            device["model"] = "XB"  # a second ring, posed in a frame of its own
    facility = tmp_path / "facility.json"
    facility.write_text(json.dumps(document), encoding="utf-8")
    tables = copy_tables(tmp_path / "panels")
    rows = (tables / "positions.csv").read_text(encoding="utf-8").splitlines()
    rows = [r.replace("XR02:QF1,XR-RING", "XR02:QF1,XB-RING") for r in rows]
    rows.append("XR02:SHIELD1,XB-RING,0,0,0,0,0,0,4.0,0,0,,,stated,XR-LAYOUT,A,released,sheet 9")
    (tables / "positions.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    graph = load_graph(facility, tables)
    assert graph.beam_frames == frozenset({RING_FRAME, "XB-RING"})
    other = graph.thing("XR02:QF1", "device")
    assert graph.beam_position(other) == ("XB-RING", 11.0)  # its own ring
    shield = graph.pose("XR02:SHIELD1")  # also posed in XR-RING: the other model's frame
    assert shield is not None and (shield.frame, shield.s_m) == ("XB-RING", 4.0)
    assert graph.beam_position(graph.thing("XR01:QF1", "device")) == (RING_FRAME, 1.25)


def test_rule_returns_every_row_of_every_source(panel_graph: PanelGraph) -> None:
    rules = panel_graph.rule("XR01-IP1:State-Sts")
    assert [(rule.value, rule.state, rule.source) for rule in rules] == [
        ("0", "bad", "xr-screens"),
        ("0", "normal", "xr-ioc"),
        ("1", "normal", "xr-screens"),
    ]
    assert panel_graph.rule("XR01-QF1:Current-RB") == []


def test_pair_comes_from_the_table_only(panel_graph: PanelGraph) -> None:
    stated = panel_graph.pair("XR01-QF1:Current-SP")
    assert stated is not None
    assert (stated.readback, stated.basis, stated.source) == (
        "XR01-QF1:Current-RB",
        "stated",
        "xr-ioc",
    )
    inferred = panel_graph.pair("XR01-PS1:On-Cmd")
    assert inferred is not None
    assert (inferred.readback, inferred.basis) == ("XR01-PS1:On-Sts", "inferred")
    # The facility file pairs these and pairs.csv does not: the facility file's pair.
    from osprey.facility.panels.model import FACILITY_PAIR

    pair = panel_graph.pair("XR02-QF1:Current-SP")
    assert pair is not None and (pair.evidence, pair.basis) == (FACILITY_PAIR, "stated")
    assert pair.readback == "XR02-QF1:Current-RB"
    by_signal = panel_graph.pair("XR01-QD1:Current-SP")  # the build pairs it by signal
    assert by_signal is not None and by_signal.evidence == FACILITY_PAIR
    assert panel_graph.pair("XR01-QF1:Current-RB") is None
    # each row states a source or an inferred basis the screens read: none restates it
    assert panel_graph.redundant.get("pairs.csv") is None


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
    assert pose.on_beam
    assert not pose.is_beam_position


def test_poses_put_the_ring_row_first_then_file_order(panel_graph: PanelGraph) -> None:
    assert [pose.frame for pose in panel_graph.poses("XR01:QD1")] == [RING_FRAME, "XR-GIRDER"]
    assert [pose.on_beam for pose in panel_graph.poses("XR01:QD1")] == [True, False]
    assert len(panel_graph.poses("XR02:SHIELD1")) == 1
    assert panel_graph.poses("XR02:QD1") == []
    assert panel_graph.poses(EXCLUDED_THING) == []
    assert panel_graph.poses("XR09:NOPE") == []


def test_a_tie_between_beam_frames_goes_to_the_frame_with_most_rows(
    tmp_path: Path, panel_facility: Path
) -> None:
    tables = copy_tables(tmp_path / "panels")
    header = (tables / "positions.csv").read_text(encoding="utf-8").splitlines()[0]
    rows = [
        "XR01:QF1,XR-A,0,0,0,0,0,0,1.0,0,0,,,stated,XR-LAYOUT,A,released,sheet 1",
        "XR01:QD1,XR-B,0,0,0,0,0,0,2.0,0,0,,,stated,XR-LAYOUT,A,released,sheet 1",
    ]
    (tables / "positions.csv").write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    graph = load_graph(panel_facility, tables)
    assert graph.beam_frames == frozenset({"XR-A"})  # equal rows: by name
    assert graph.pose("XR01:QF1").is_beam_position  # type: ignore[union-attr]
    assert not graph.pose("XR01:QD1").is_beam_position  # type: ignore[union-attr]


def test_rules_lists_every_row_of_every_held_channel(panel_graph: PanelGraph) -> None:
    rows = panel_graph.rules()
    # 17 table rows, less the one on an excluded thing, the one on a hidden thing
    # and the one on a simulation-only channel
    assert len(rows) == 14
    assert {row.channel for row in rows}.isdisjoint({VA_CHANNEL, *HIDDEN_CHANNELS})


def test_a_place_carries_its_level_and_a_device_none(panel_graph):
    assert panel_graph.thing("XR/XR01/RK1").level == "rack"
    assert panel_graph.thing("XR/XR01").level == "sector"
    assert panel_graph.thing("XR01:QF1").level is None


def test_a_channel_carries_its_stated_tolerance(panel_graph):
    assert panel_graph.channel("XR01-QF1:Current-SP").tolerance == ("absolute", 0.05)
    assert panel_graph.channel("XR01-QD1:Current-SP").tolerance is None


def test_record_type_gives_the_table_row_of_a_held_channel(panel_graph):
    row = panel_graph.record_type("XR01-QF1:Current-SP")
    assert row is not None and (row.record_type, row.source) == ("ao", "xr-ioc")
    assert panel_graph.record_type("XR01-QD1:Current-SP") is None
    assert panel_graph.record_type(VA_CHANNEL) is None
    assert panel_graph.record_type("XR02-SAF1:Trip-Sts") is None  # on an excluded thing


def test_pictures_give_a_class_rows_in_file_order(panel_graph):
    parts = panel_graph.pictures("Quadrupole")
    assert [(part.view, part.part) for part in parts] == [
        ("end", "pole 1"),
        ("end", "thermal switch 1"),
    ]
    assert panel_graph.pictures("Dipole") == []


def test_contents_of_a_place_are_built_there_or_claimed_there(panel_graph: PanelGraph) -> None:
    sector = [thing.id for thing in panel_graph.contents("XR/XR02")]
    # built within, any depth, then claimed into it or a place within it (XR/XR02/RK2)
    assert sector[: len(panel_graph.within("XR/XR02", deep=True))] == [
        thing.id for thing in panel_graph.within("XR/XR02", deep=True)
    ]
    assert "XR02:CH5" in sector and len(sector) == len(set(sector))
    rack = [thing.id for thing in panel_graph.contents("XR/XR01/RK1")]
    assert rack == ["XR01:CH1", "XR01:CH2", "XR01:CH3", "XR02:CH5"]  # each once
    assert panel_graph.claimed("XR/XR02/RK2") == [panel_graph.thing("XR02:CH5")]


def test_a_device_in_no_place_is_unplaced(panel_graph: PanelGraph) -> None:
    assert [thing.id for thing in panel_graph.unplaced()] == ["XR09:LOOSE1"]
    assert panel_graph.placed("XR02:CH5")  # by its claims
    assert not panel_graph.placed("XR09:LOOSE1")
    every = {thing.id for place in ("XR",) for thing in panel_graph.contents(place)}
    assert "XR09:LOOSE1" not in every


def test_an_excluded_document_drops_the_rule_rows_it_sources(tmp_path: Path, panel_facility: Path):
    tables = copy_tables(tmp_path / "panels")
    with (tables / "excluded.csv").open("a") as rows:
        rows.write("document,xr-ioc,an invented source left out,,,\n")
    graph = load_graph(panel_facility, tables)
    assert {row.source for row in graph.rules()} == {"xr-screens"}
    pair = graph.pair("XR01-QF1:Current-SP")  # its pair row cites xr-ioc: the file's pair
    assert pair is not None and pair.source is None
    assert graph.record_type("XR01-QF1:Current-SP") is None
    assert graph.channel("XR01-QF1:Current-SP").drawn_as == "setpoint"  # by its role


def test_a_channel_on_no_thing_is_kept_and_one_on_a_left_out_thing_is_not(tmp_path: Path):
    def orphan(document) -> None:
        document["channels"].append({"id": "XR-NOWHERE:Beam-Sts"})
        document["channels"].append({"id": "XR-GONE:Beam-Sts", "on": {"device": "XR-NO-SUCH"}})

    graph = load_graph(write_fixture_facility(tmp_path / "f.json", orphan), PANELS_DIR)
    assert {c.name for c in graph.unheld()} == {"XR-GONE:Beam-Sts", "XR-NOWHERE:Beam-Sts"}
    assert graph.channel("XR-NOWHERE:Beam-Sts").thing is None
    with pytest.raises(KeyError):
        graph.channel("XR02-SAF1:Trip-Sts")  # on an excluded thing: left out with it


def test_a_unit_that_is_no_integer_keeps_its_claim_as_text_and_is_counted(tmp_path: Path):
    def odd_unit(document) -> None:
        for claim in document["relations"]:
            if claim["id"] == "xr01-ch2-rack":
                claim["object"]["unit"] = "A5"

    graph = load_graph(write_fixture_facility(tmp_path / "f.json", odd_unit), PANELS_DIR)
    assert graph.unread == 1
    (edge,) = [e for e in graph.neighbours("XR01:CH2")["located_in"] if e.claim == "xr01-ch2-rack"]
    assert (edge.unit, edge.unit_text) == (None, "A5")


# -- parts follow their whole ---------------------------------------------------------


def test_a_part_with_no_place_is_in_the_place_of_its_whole(tmp_path: Path) -> None:
    graph = graph_with_part(tmp_path)
    assert graph.whole_of("XR01:CARD1") == ("XR01:CH1", ("XR/XR01/RK1",))
    assert graph.placed("XR01:CARD1")
    assert "XR01:CARD1" not in [thing.id for thing in graph.unplaced()]
    for place in ("XR/XR01/RK1", "XR/XR01", "XR"):
        found = [thing.id for thing in graph.contents(place)]
        assert found.count("XR01:CARD1") == 1, place
    assert graph.derived("XR/XR01/RK1") == [graph.thing("XR01:CARD1")]
    assert "XR01:CARD1" not in [thing.id for thing in graph.contents("XR/XR02")]


def test_a_part_follows_its_whole_up_to_the_nearest_whole_with_a_place(tmp_path: Path):
    def deeper(document: dict[str, Any]) -> None:
        add_part(document, "XR01:CHIP1", "XR01:CARD1")

    graph = graph_with_part(tmp_path, deeper)
    assert graph.whole_of("XR01:CHIP1") == ("XR01:CH1", ("XR/XR01/RK1",))
    assert "XR01:CHIP1" in [thing.id for thing in graph.contents("XR/XR01/RK1")]


def test_a_whole_placed_by_its_claims_lends_the_claimed_places(tmp_path: Path) -> None:
    def on_contested(document: dict[str, Any]) -> None:
        add_part(document, "XR02:CARD5", "XR02:CH5")

    graph = graph_with_part(tmp_path, on_contested)
    assert graph.whole_of("XR02:CARD5") == ("XR02:CH5", ("XR/XR01/RK1", "XR/XR02/RK2"))
    assert "XR02:CARD5" in [thing.id for thing in graph.contents("XR/XR02/RK2")]


def test_a_part_never_overrides_a_stated_place_or_claim(tmp_path: Path) -> None:
    def placed_parts(document: dict[str, Any]) -> None:
        for part, whole in (("XR02:CH5", "XR01:CH1"), ("XR01:CH2", "XR02:BLOCK1")):
            document["relations"].append(
                {
                    "id": f"{part.lower()}-part",
                    "type": "part_of",
                    "subject": {"device": part},
                    "object": {"device": whole},
                    "standing": "accepted",
                }
            )

    graph = graph_with_part(tmp_path, placed_parts)
    assert graph.whole_of("XR02:CH5") is None  # its claims stand
    assert graph.whole_of("XR01:CH2") is None  # its built place stands
    assert "XR01:CH2" not in [thing.id for thing in graph.contents("XR/XR02")]


def test_a_part_of_cycle_with_no_place_stays_unplaced(tmp_path: Path) -> None:
    def loop(document: dict[str, Any]) -> None:
        add_part(document, "XR09:LOOPA", "XR09:LOOPB")
        add_part(document, "XR09:LOOPB", "XR09:LOOPA")
        add_part(document, "XR09:LOOPC", "XR09:LOOPA")

    graph = graph_with_part(tmp_path, loop)
    for part in ("XR09:LOOPA", "XR09:LOOPB", "XR09:LOOPC"):
        assert graph.whole_of(part) is None
        assert not graph.placed(part)
    assert {"XR09:LOOPA", "XR09:LOOPB", "XR09:LOOPC"} <= {t.id for t in graph.unplaced()}


# -- parts of an excluded thing -------------------------------------------------------


def _withheld(tmp_path: Path) -> PanelGraph:
    """The fixture with ``XR02:SAF1:CARD``, a ruled part of the excluded ``XR02:SAF1``,
    and ``XR02:SAF1:CHIP``, a part of that part."""

    def parts(document: dict[str, Any]) -> None:
        add_part(document, "XR02:SAF1:CARD", "XR02:SAF1", "XR02-SAF1-CARD:Fault-Sts")
        add_part(document, "XR02:SAF1:CHIP", "XR02:SAF1:CARD")

    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as rows:
        rows.write(
            "XR02-SAF1-CARD:Fault-Sts,1,bad,database-severity,stated,xr-ioc,0a1b2c3,c.db:1\n"
        )
    return load_graph(write_fixture_facility(tmp_path / "facility.json", parts), tables)


def test_a_part_of_an_excluded_thing_is_withheld_with_its_channels(tmp_path: Path) -> None:
    graph = _withheld(tmp_path)
    ids = {thing.id for thing in graph.things()}
    assert not {"XR02:SAF1:CARD", "XR02:SAF1:CHIP"} & ids
    with pytest.raises(KeyError):
        graph.channel("XR02-SAF1-CARD:Fault-Sts")
    assert "XR02-SAF1-CARD:Fault-Sts" not in {c.name for c in graph.unheld()}
    assert not graph.rule("XR02-SAF1-CARD:Fault-Sts")
    assert graph.withheld == 2
    assert "XR02:SAF1:CARD" not in {thing.id for thing in graph.unplaced()}


def test_a_thing_carries_the_short_name_the_names_table_states(panel_graph) -> None:
    assert panel_graph.thing("XR01:QF1", "device").short_name == "F1"
    assert panel_graph.thing("XR01:BEND1", "device").short_name is None


# -- calibrations ------------------------------------------------------------------


def test_the_claim_in_force_is_the_newest_accepted_one_not_superseded(panel_graph) -> None:
    found = panel_graph.calibration("XR01:QF1", "integrated_gradient")
    assert found.in_force is not None and found.in_force.id == "ig/QF1/bench"
    assert found.state == "bench" and not found.below_wanted
    assert found.design is not None and found.design.id == "ig/QF/design"
    assert [c.id for c in found.earlier] == ["ig/QF/design"]


def test_two_accepted_claims_of_one_basis_beyond_the_tolerance_are_contested(panel_graph):
    found = panel_graph.calibration("XR02:QF1", "integrated_gradient")
    assert found.state == "contested"
    assert found.in_force is not None and found.in_force.id == "ig/XR02QF1/bench-b"
    assert [c.id for c in found.rivals] == ["ig/XR02QF1/bench-a"]


def test_a_kind_with_no_claim_is_none_and_below_its_wanted_basis(panel_graph) -> None:
    found = panel_graph.calibration("XR02:QD1", "integrated_gradient")
    assert (found.in_force, found.state, found.below_wanted) == (None, "none", True)
    design = panel_graph.calibration("XR01:QD1", "integrated_gradient")
    assert design.state == "design" and design.below_wanted  # bench wanted


def test_a_curve_claim_carries_its_points_and_kinds_go_by_order(panel_graph) -> None:
    found = panel_graph.calibration("XR01:QF1", "excitation")
    assert [(p.x, p.y) for p in found.points] == [(0.0, -0.02376), (73.33, -4.9198)]
    kinds = [k.kind for k in panel_graph.calibration_kinds("Quadrupole")]
    assert kinds == [
        "integrated_gradient", "excitation", "centre", "strength_scale", "current_offset"
    ]  # fmt: skip
    assert panel_graph.calibration_kinds("Pump") == []


def test_a_claim_from_an_excluded_document_is_dropped(tmp_path) -> None:
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    tables = copy_tables(tmp_path / "panels")
    with (tables / "excluded.csv").open("a") as rows:
        rows.write("document,XR-MAG-TEST,withdrawn,XR-LAYOUT,A,sheet 3\n")
    graph = load_graph(write_fixture_facility(tmp_path / "f.json"), tables)
    found = graph.calibration("XR01:QF1", "integrated_gradient")
    assert found.in_force is not None and found.in_force.id == "ig/QF/design"
    assert found.state == "design"


def _graph_with(tmp_path: Path, kinds: list[str] = (), claims: list[str] = (), change=None):
    """The fixture graph with *kinds* and *claims* rows added to the calibration tables."""
    tables = copy_tables(tmp_path / "panels")
    for name, rows in (("calibration_kinds.csv", kinds), ("calibrations.csv", claims)):
        with (tables / name).open("a") as handle:
            handle.writelines(f"{row}\n" for row in rows)
    return load_graph(write_fixture_facility(tmp_path / "f.json", change), tables)


def test_a_kind_applies_by_class_and_under_a_place(panel_graph) -> None:
    assert [k.kind for k in panel_graph.calibration_kinds_of("XR02:BEND1")] == ["centre"]
    assert panel_graph.calibration_kinds_of("XR01:BEND1") == []
    assert panel_graph.calibration_kind("centre", "Dipole").title == "Field centre"
    assert panel_graph.calibration_kind("centre", "Quadrupole").title == "Magnetic centre"
    assert panel_graph.calibration("XR02:BEND1", "centre").kind.cls == "Dipole"
    with pytest.raises(KeyError):
        panel_graph.calibration("XR01:BEND1", "centre")
    with pytest.raises(KeyError):
        panel_graph.calibration_kind("centre", "Pump")


def test_a_kind_applies_under_a_model(tmp_path) -> None:
    def models(document: dict[str, Any]) -> None:
        for device in document["devices"]:
            if device["id"].startswith("XR0"):
                device["model"] = "XA" if device["id"].startswith("XR01") else "XB"

    graph = _graph_with(tmp_path, ["roll,Roll,Pump,scalar,urad,survey,,,,1,XB"], change=models)
    assert [k.kind for k in graph.calibration_kinds_of("XR02:IP1")] == ["roll"]
    assert graph.calibration_kinds_of("XR01:IP1") == []
    assert graph.thing("XR02:IP1", "device").model == "XB"


def test_an_absolute_tolerance_holds_for_each_number_of_a_tuple(tmp_path) -> None:
    near = "centre/QF1/near,centre,XR01:QF1,-2.5;-10.0,um,,bench,,2023-02-24,D,A,Released,,accepted"
    graph = _graph_with(tmp_path, claims=[near])
    assert graph.calibration("XR01:QF1", "centre").state == "bench"  # 4.5 um apart, 5 allowed
    far = near.replace("-10.0", "-20.0").replace("near", "far")
    graph = _graph_with(tmp_path / "far", claims=[far])
    found = graph.calibration("XR01:QF1", "centre")
    assert found.state == "contested" and [c.id for c in found.rivals] == ["centre/QF1/bench"]


def test_a_relative_tolerance_is_a_fraction_of_the_value_in_force(tmp_path) -> None:
    near = "ig/QF1/near,integrated_gradient,XR01:QF1,-4.9100,T,,bench,,2023-03-01,D,A,Released,,accepted"
    graph = _graph_with(tmp_path, claims=[near])
    assert graph.calibration("XR01:QF1", "integrated_gradient").state == "bench"  # 0.2 % of 4.91
    far = near.replace("-4.9100", "-4.9000").replace("near", "far")
    graph = _graph_with(tmp_path / "far", claims=[far])
    assert graph.calibration("XR01:QF1", "integrated_gradient").state == "contested"


def test_a_kind_with_no_tolerance_is_contested_only_by_standing(tmp_path) -> None:
    rows = [
        "ss/QF1/a,strength_scale,XR01:QF1,1.0,,,beam,,2024-01-01,D,A,Released,,accepted",
        "ss/QF1/b,strength_scale,XR01:QF1,1.5,,,beam,,2024-02-01,D,A,Released,,accepted",
    ]
    graph = _graph_with(tmp_path, claims=rows)
    assert graph.calibration("XR01:QF1", "strength_scale").state == "beam"
    graph = _graph_with(
        tmp_path / "c",
        claims=[*rows, rows[0].replace("/a,", "/c,").replace("accepted", "contested")],
    )
    assert graph.calibration("XR01:QF1", "strength_scale").state == "contested"
