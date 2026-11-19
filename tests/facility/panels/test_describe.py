"""View descriptions over the fixture: what each screen holds, without layout."""

from __future__ import annotations

import dataclasses
import gc
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.panels import load_graph
from osprey.facility.panels.describe import (
    HOME,
    HOME_LINK,
    HOME_TARGET,
    NOT_PLACED,
    RACK_ORDER,
    RELATION_TITLES,
    SETS_SECTION,
    checkout,
    describe,
    device_card,
    health_rollup,
    home,
    place_view,
    rack_elevation,
    set_overview,
    targets,
)
from osprey.facility.panels.model import PanelGraph
from osprey.facility.panels.tables import HealthRule, Pair, PanelTables, Pose, RecordType
from osprey.facility.panels.view import (
    CardHead,
    CheckRow,
    Elevation,
    Fact,
    Grid,
    Light,
    Link,
    Node,
    Picture,
    Placement,
    Section,
    Strip,
    Target,
    Totals,
    ValueRow,
    View,
)
from osprey.facility.records import FacilityRecords
from tests.facility.panels._fixture import (
    PANELS_DIR,
    RING_FRAME,
    copy_tables,
    graph_with_part,
    write_fixture_facility,
)

SECTORS = frozenset({"sector"})  # a level marked as a rack by parameter
EXCLUDED = ("XR02:SAF1", "XR/XR02/SAFE", "XR/XR02/SAFE/CAB1", "XR02:SAF2")


def card(thing_id: str) -> Target:
    return Target("card", "device", thing_id)


def _section(view: View, title: str):
    """A section by title; ``where`` and ``sources`` are the card foot's claims and
    statements, as a section of them."""
    if title in ("where", "sources"):
        (foot,) = [s.items[0] for s in view.sections if s.title == "foot"]
        if title == "where":
            return Section(title, foot.claims)
        return Section(
            title, tuple(Fact(c.document, c.locator or "", citation=c) for c in foot.citations)
        )
    found = [section for section in view.sections if section.title == title]
    assert len(found) == 1, f"{view.id}: sections {[s.title for s in view.sections]}"
    return found[0]


def _blocks(view: View) -> list:
    """The blocks of a drawn card's relations chain."""
    (chain,) = _section(view, "relations").items
    return list(chain.blocks)


def _titles(view: View) -> list[str]:
    return [section.title for section in view.sections]


def _walk(value: Any):
    """Every dataclass instance reachable from *value*, itself included."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        yield value
        for field in dataclasses.fields(value):
            yield from _walk(getattr(value, field.name))
    elif isinstance(value, tuple | list):
        for entry in value:
            yield from _walk(entry)


def _links(view: View) -> list[Link]:
    return [item for item in _walk(view) if isinstance(item, Link)]


def _ids(view: View) -> set[str]:
    """Every thing id and channel name a view names anywhere."""
    found: set[str] = set()
    for item in _walk(view):
        for field in dataclasses.fields(item):
            value = getattr(item, field.name)
            if isinstance(value, str):
                found.add(value)
    return found


def _graph(tmp_path: Path, change) -> PanelGraph:
    facility = write_fixture_facility(tmp_path / "facility.json", change)
    return load_graph(facility, PANELS_DIR)


# -- device card -------------------------------------------------------------


def test_a_magnet_fed_by_the_shared_supply_links_to_it(panel_graph):
    view = device_card(panel_graph, "XR01:QF1")
    (feeder,) = [b for b in _blocks(view) if b.link is not None and b.title.startswith("Power")]
    assert feeder.link.target == card("XR01:PS1")


def test_the_supply_links_back_to_both_magnets(panel_graph):
    blocks = _blocks(device_card(panel_graph, "XR01:PS1"))
    feeds = [b.link.target for b in blocks if b.title.startswith("Feeds")]
    # graph order: the built facility sorts claims by id
    assert feeds == [card("XR01:QD1"), card("XR01:QF1")]


def test_a_setpoint_is_an_entry_and_its_readback_stands_in_the_channels(panel_graph):
    view = device_card(panel_graph, "XR01:QF1")
    (operate,) = _section(view, "operate").items
    assert [(g.kind, [c.channel for c in g.controls]) for g in operate.groups] == [
        ("entry", ["XR01-QF1:Current-SP"])
    ]
    (grid,) = _section(view, "all channels").items
    assert [channel for _short, channel in grid.rows] == [
        "XR01-QF1:Current-RB",
        "XR01-QF1:Current-SP",  # every channel, the one operate sets too
    ]
    assert grid.rows[0][0] == "Current-RB"  # no device prefix: the shared segment is cut


def test_the_channels_grid_keeps_the_channel_order(panel_graph):
    (grid,) = _section(device_card(panel_graph, "XR01:IP1"), "all channels").items
    assert [channel for _short, channel in grid.rows] == [
        channel.name for channel in panel_graph.channels("XR01:IP1", "device")
    ]


def test_a_light_per_rule_bearing_channel_marks_inferred_and_disagree(panel_graph):
    from osprey.facility.panels.describe import light

    assert light(panel_graph, "XR01-PS1:On-Sts") == Light("XR01-PS1:On-Sts", inferred=True)
    # one stated row and one inferred row: not all inferred; they disagree on 0
    pump = light(panel_graph, "XR01-IP1:State-Sts")
    assert pump == Light("XR01-IP1:State-Sts", disagree=True)


def test_a_light_whose_rules_the_engine_cannot_evaluate_says_so(panel_graph):
    from osprey.facility.panels.describe import light

    lit = {
        c.name: light(panel_graph, c.name)
        for thing in ("XR02:BEND1", "XR02:QF1", "XR02:IP1")
        for c in panel_graph.channels(thing, "device")
        if panel_graph.rule(c.name)
    }
    assert [x.unsupported for x in lit.values() if x.channel.startswith("XR02-BEND1")] == [True]
    assert lit["XR02-QF1:Current-RB"].unsupported is True
    assert all(not x.unsupported for x in lit.values() if x.channel.startswith("XR02-IP1"))


def test_a_channel_less_thing_has_place_pose_and_no_signal_or_health(panel_graph):
    view = device_card(panel_graph, "XR02:SHIELD1")
    assert _titles(view) == ["head", "relations", "foot"]
    where = _section(view, "where").items
    assert where[0] == Link("place", Target("card", "place", "XR/XR02"))
    pose, beam = [item for item in where if isinstance(item, Fact)]
    assert beam.text == "s 12.500 m along XR-RING, from its pose in positions.csv"
    assert pose.label == "XR-RING"
    assert "s 12.500 m" in pose.text
    assert pose.caveat is None


def test_a_derived_pose_carries_its_assumption_as_a_caveat(panel_graph):
    where = _section(device_card(panel_graph, "XR01:BEND1"), "where").items
    (pose,) = [item for item in where if isinstance(item, Fact)]
    assert pose.caveat == "fixture: frame treated as the ring frame"
    assert pose.citation is not None and pose.citation.document == "XR-LAYOUT"


def test_card_sections_come_in_their_fixed_order(panel_graph):
    assert _titles(device_card(panel_graph, "XR01:QF1")) == [
        "head",
        "tiles",  # its channels' signals are its roles: key tiles by default
        "relations",
        "picture",
        "calibration",
        "operate",
        "all channels",
        "foot",
    ]
    assert _titles(device_card(panel_graph, "XR01:IP1")) == [
        "head",
        "tiles",
        "relations",
        "operate",  # it says no operator action is stated
        "all channels",
        "foot",
    ]


def test_a_card_with_every_section_has_them_in_order(tmp_path):
    def everything(document: dict[str, Any]) -> None:
        for device in document["devices"]:
            if device["id"] == "XR01:QD1":
                device["citations"] = [{"document": "XR-LAYOUT", "locator": "sheet 2"}]
        document["relations"].append(
            {
                "id": "xr01-qd1-feeds",
                "type": "powered_by",
                "subject": {"device": "XR01:IP2"},
                "object": {"device": "XR01:QD1"},
            }
        )

    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as health:
        health.write("XR01-QD1:Current-RB,0,warn,database-severity,stated,xr-ioc,0a1b2c3,m.db:1\n")
    facility = write_fixture_facility(tmp_path / "facility.json", everything)
    view = device_card(load_graph(facility, tables), "XR01:QD1")
    assert _titles(view) == [
        "head",
        "tiles",
        "relations",
        "picture",
        "calibration",
        "operate",
        "all channels",
        "foot",
    ]
    words = [b.title.split()[0] for b in _blocks(view)]
    assert "Power" in words and "Feeds" in words  # fed by the supply, feeding the pump


def test_the_foot_counts_the_statements_and_names_their_documents(panel_graph):
    view = device_card(panel_graph, "XR01:QF1")
    fact = _section(view, "sources").items[0]  # the device's own citation first
    assert (fact.citation.document, fact.citation.version) == ("XR-LAYOUT", "A")
    (foot,) = [s.items[0] for s in view.sections if s.title == "foot"]
    n = len(foot.citations)
    documents = sorted({c.document for c in foot.citations})
    assert foot.sources.startswith(f"{n} statement")
    assert all(d in foot.sources for d in documents)
    assert len(foot.provenance) == n and "sheet 1" in foot.provenance[0]
    text, link = foot.where[0]
    assert text.startswith(link.target.id)


def test_the_place_relation_is_shown_under_where_with_its_unit(panel_graph):
    view = device_card(panel_graph, "XR01:CH1")
    assert "located_in" not in _titles(view)
    where = _section(view, "where").items
    assert Link("place", Target("card", "place", "XR/XR01/RK1")) in where
    (placed,) = [item for item in where if isinstance(item, Placement)]
    assert placed.link.target == Target("rack", "place", "XR/XR01/RK1")
    assert (placed.unit, placed.size_u, placed.citation.document) == (10, 2, "XR-RACKS")
    assert placed.link.standing == "accepted"


def test_a_card_shows_each_claim_even_when_two_sources_agree(panel_graph):
    where = _section(device_card(panel_graph, "XR01:CH2"), "where").items
    placed = [item for item in where if isinstance(item, Placement)]
    assert [(p.unit, p.citation.document if p.citation else None) for p in placed] == [
        (20, None),
        (20, "XR-LAYOUT"),
    ]


def test_unit_claims_show_under_where_without_a_built_place(panel_graph):
    view = device_card(panel_graph, "XR02:CH5")
    assert view.crumbs == (HOME_LINK,)  # no place: only the way home
    assert _titles(view) == ["head", "relations", "foot"]  # no raw located_in section
    where = _section(view, "where").items
    assert [(p.link.target, p.unit, p.link.standing) for p in where] == [
        (Target("rack", "place", "XR/XR01/RK1"), 40, "contested"),
        (Target("rack", "place", "XR/XR01/RK1"), 41, "contested"),
        (Target("rack", "place", "XR/XR02/RK2"), 5, "contested"),
    ]


def test_a_place_card_lists_a_device_claimed_into_it(panel_graph):
    view = device_card(panel_graph, "XR/XR01/RK1")
    assert "located_in" not in _titles(view)
    within = _section(view, "within").items
    (claimed,) = [item for item in within if item.target == card("XR02:CH5")]
    assert claimed.standing == "contested"


def test_an_excluded_neighbour_never_appears(panel_graph):
    pump = device_card(panel_graph, "XR02:IP1")
    assert "protected by" not in _titles(pump)
    supply = device_card(panel_graph, "XR01:PS1")
    for view in (pump, supply):
        assert not _ids(view) & set(EXCLUDED)


def test_relation_titles_cover_both_directions():
    assert RELATION_TITLES["powered_by"] == ("fed by", "feeds")
    assert RELATION_TITLES["interlocked_by"] == ("protected by", "protects")
    assert RELATION_TITLES["same_as"] == ("also known as", "also known as")
    assert RELATION_TITLES["cabled_to"] == ("cabled to", "cabled to")
    assert RELATION_TITLES["networked_to"] == ("network", "network")


def _add_claims(document: dict[str, Any]) -> None:
    for claim in document["relations"]:
        if claim["id"] == "xr01-ip1-readout":
            claim["standing"] = "contested"
            claim["basis"] = "inferred"
    document["relations"].append(
        {
            "id": "xr01-qf1-aligned",
            "type": "aligned_with",
            "subject": {"device": "XR01:QF1"},
            "object": {"device": "XR01:QD1"},
        }
    )


def test_a_contested_or_inferred_relation_is_marked_on_its_link(tmp_path):
    graph = _graph(tmp_path, _add_claims)
    (block,) = [b for b in _blocks(device_card(graph, "XR01:IP1")) if b.title.startswith("Read")]
    link = block.link
    assert (link.target, link.standing, link.basis) == (card("XR01:CH2"), "contested", "inferred")
    assert block.standing is None  # one thing: the block is not contested, its claim is
    (back,) = [b for b in _blocks(device_card(graph, "XR01:CH2")) if b.title.startswith("Reads")]
    assert (back.link.standing, back.link.basis) == ("contested", "inferred")


def test_an_unknown_relation_kind_is_titled_by_its_own_name(tmp_path):
    graph = _graph(tmp_path, _add_claims)
    blocks = _blocks(device_card(graph, "XR01:QF1"))
    (aligned,) = [b for b in blocks if b.title.startswith("Aligned with")]
    assert aligned.link.target == card("XR01:QD1")
    back = _blocks(device_card(graph, "XR01:QD1"))
    assert any(b.title.startswith("Aligned with") for b in back)


def test_a_class_with_picture_rows_yields_a_picture_and_one_without_none(panel_graph):
    (picture,) = _section(device_card(panel_graph, "XR01:QF1"), "picture").items
    assert isinstance(picture, Picture)
    assert (picture.device_class, picture.view) == ("Quadrupole", "end")
    assert [(part.kind, part.x_mm, part.y_mm, part.z_mm) for part in picture.parts] == [
        ("pole", 60.0, 60.0, 0.0),
        ("thermal_switch", 120.0, -40.0, 0.0),
    ]
    assert "picture" not in _titles(device_card(panel_graph, "XR01:BEND1"))


def test_a_device_card_has_its_place_chain_as_crumbs(panel_graph):
    view = device_card(panel_graph, "XR01:CH1")
    assert view.id == card("XR01:CH1")
    assert [crumb.target.id for crumb in view.crumbs] == [HOME, "XR", "XR/XR01", "XR/XR01/RK1"]
    # up to each place's view where it has one, else its card
    assert [crumb.target.view for crumb in view.crumbs] == ["home", "place", "place", "place"]


def test_a_place_card_lists_its_screens_and_what_is_within(panel_graph):
    view = device_card(panel_graph, "XR/XR01/RK1")
    screens = [item.target for item in _section(view, "screens").items]
    assert Target("place", "place", "XR/XR01/RK1") in screens
    assert Target("rack", "place", "XR/XR01/RK1") in screens
    assert [item.target for item in _section(view, "within").items] == [
        card("XR01:CH1"),
        card("XR01:CH2"),
        card("XR01:CH3"),
        card("XR02:CH5"),  # claimed into the rack, no built place
    ]
    assert [row.channel for row in _section(view, "signals").items] == ["XR01:RackTemp-RB"]
    sector = Target("rack", "place", "XR/XR01")
    assert sector not in [
        item.target for item in _section(device_card(panel_graph, "XR/XR01"), "screens").items
    ]
    marked = device_card(panel_graph, "XR/XR01", rack_classes=SECTORS)
    assert sector in [item.target for item in _section(marked, "screens").items]


def test_a_thing_with_no_screen_is_a_fact_not_a_link(panel_graph):
    within = _section(device_card(panel_graph, "XR/XR02", kind="place"), "within").items
    block = [item for item in within if getattr(item, "text", None) == "XR02:BLOCK1"]
    assert len(block) == 1 and isinstance(block[0], Fact)


def test_a_group_card_links_its_members_and_its_set(panel_graph):
    view = device_card(panel_graph, "XR02-VAC", kind="group")
    assert [item.target for item in _section(view, "members").items] == [
        card("XR02:IP1"),
        card("XR02:IP2"),
    ]
    assert [item.target for item in _section(view, "screens").items] == [
        Target("set", "group", "XR02-VAC")
    ]


# -- targets ------------------------------------------------------------------


def test_targets_name_each_screen_with_view_kind_thing_kind_and_id(panel_graph):
    found = set(targets(panel_graph))
    assert card("XR01:QF1") in found
    assert card("XR01:CH1") in found  # a channel and the located_in to its own place
    assert card("XR02:SHIELD1") not in found  # a pose and no channel: no card by default
    assert card("XR01:CH2") in found  # no channel, kept by the fixture's cards.csv
    assert card("XR02:BLOCK1") not in found  # a citation alone is no screen
    for place in ("XR", "XR/XR01", "XR/XR02", "XR/XR01/RK1"):
        assert Target("card", "place", place) in found
        assert Target("place", "place", place) in found
    assert Target("rack", "place", "XR/XR01/RK1") in found  # unit claims, no parameter
    assert Target("rack", "place", "XR/XR01") not in found
    marked = set(targets(panel_graph, rack_classes=SECTORS))
    assert Target("rack", "place", "XR/XR01") in marked
    assert Target("rack", "place", "XR") not in marked  # level machine
    for group in ("XR01-VAC", "XR02-VAC", "XR-QUAD"):
        assert Target("card", "group", group) in found
        assert Target("set", "group", group) in found
    for place in ("XR", "XR/XR01", "XR/XR02"):
        assert Target("rollup", "place", place, by="place") in found
        assert Target("rollup", "place", place, by="system") in found
        assert Target("checkout", "place", place) in found
    assert Target("rollup", "place", "XR/XR01/RK1", by="place") not in found
    assert Target("checkout", "place", "XR/XR01/RK1") not in found
    assert not {target.id for target in found} & set(EXCLUDED)
    assert len(found) == len(targets(panel_graph))


def test_a_set_needs_members_that_share_one_class(tmp_path):
    def mixed(document: dict[str, Any]) -> None:
        document["groups"].append({"id": "XR01-MIX", "members": ["XR01:QF1", "XR01:IP1"]})

    found = targets(_graph(tmp_path, mixed))
    assert Target("card", "group", "XR01-MIX") in found
    assert Target("set", "group", "XR01-MIX") not in found


def _every_view(
    graph: PanelGraph, rack_classes: frozenset[str] = frozenset(), **params: Any
) -> list[View]:
    found = targets(graph, rack_classes=rack_classes)
    return [describe(graph, target, rack_classes=rack_classes, **params) for target in found]


def test_every_link_target_is_a_screen(panel_graph):
    params = {"symbols": {"Pump": "pump"}, "above": frozenset({"pump"})}
    found = set(targets(panel_graph, rack_classes=SECTORS))
    views = _every_view(panel_graph, SECTORS, **params)
    assert [view.id for view in views] == targets(panel_graph, rack_classes=SECTORS)
    links = [link for view in views for link in _links(view)]
    assert links
    assert {link.target for link in links} <= found


def test_every_link_target_is_a_screen_with_contested_and_unknown_relations(tmp_path):
    graph = _graph(tmp_path, _add_claims)
    found = set(targets(graph))
    assert {link.target for view in _every_view(graph) for link in _links(view)} <= found


def test_views_carry_channel_names_never_values(panel_graph):
    for view in _every_view(panel_graph, SECTORS):
        for item in _walk(view):
            assert not {"value", "values"} & {field.name for field in dataclasses.fields(item)}


# -- set overview ---------------------------------------------------------------


def test_a_set_overview_is_a_grid_of_members_by_place(panel_graph):
    view = set_overview(panel_graph, "XR-QUAD")
    assert view.id == Target("set", "group", "XR-QUAD")
    (grid,) = _section(view, "members").items
    assert isinstance(grid, Grid)
    # rows: the sector, the nearest place with a sibling among the members' places
    assert [row.label.target for row in grid.rows] == [
        Target("card", "place", "XR/XR01"),
        Target("card", "place", "XR/XR02"),
    ]
    # within a row beam order: s first, a member with no ring position after, by id
    assert [[c.member.target for c in row.cells] for row in grid.rows] == [
        [card("XR01:QF1"), card("XR01:QD1")],
        [card("XR02:QF1"), card("XR02:QD1")],
    ]
    # one light per member: its rule channels
    cells = {c.member.target.id: c for row in grid.rows for c in row.cells}
    assert cells["XR02:QD1"].channels == ("XR02-QD1:Current-RB",)
    assert cells["XR01:QF1"].channels == ()


def test_the_primary_pair_is_keyed_on_the_stated_signal_never_the_name(panel_graph):
    assert not any(
        channel.name.startswith(thing.id)
        for thing in panel_graph.things()
        for channel in panel_graph.channels(thing.id, thing.kind)
    )
    (grid,) = _section(set_overview(panel_graph, "XR-QUAD"), "members").items
    assert grid.signal == "current_setpoint"  # the pair two members have
    cells = {c.member.target.id: c for row in grid.rows for c in row.cells}
    assert cells["XR01:QF1"].pair == ValueRow(
        label="current setpoint",
        channel="XR01-QF1:Current-SP",
        readback="XR01-QF1:Current-RB",
        tolerance=("absolute", 0.05),
        writable=True,
        drawn_as="setpoint",
        unit="A",
        readback_unit="A",
    )
    # no pairs.csv row: the pair the facility file states (the build pairs by signal)
    assert (cells["XR01:QD1"].pair.channel, cells["XR01:QD1"].pair.readback) == (
        "XR01-QD1:Current-SP",
        "XR01-QD1:Current-RB",
    )


def test_a_set_member_with_no_place_is_in_a_last_row(tmp_path):
    def loose(document: dict[str, Any]) -> None:
        document["groups"].append({"id": "XR-MISC", "members": ["XR09:LOOSE1", "XR02:SHIELD1"]})
        document["groups"].append({"id": "XR01-MAG", "members": ["XR01:QD1", "XR01:QF1"]})

    graph = _graph(tmp_path, loose)
    (grid,) = _section(set_overview(graph, "XR-MISC"), "members").items
    assert [row.label for row in grid.rows][-1] == Fact("row", "not placed")
    assert [c.member for c in grid.rows[-1].cells] == [Link("XR09:LOOSE1", card("XR09:LOOSE1"))]
    assert grid.signal is None
    # all members in one sector: no sibling among their places, so the row is that sector
    (grid,) = _section(set_overview(graph, "XR01-MAG"), "members").items
    assert [row.label.target.id for row in grid.rows] == ["XR/XR01"]
    assert [c.member.target.id for c in grid.rows[0].cells] == ["XR01:QF1", "XR01:QD1"]


# -- place view -----------------------------------------------------------------


BEAMLINE = {"Quadrupole": "quadrupole", "Dipole": "bend", "Pump": "ion_pump"}


def test_place_view_strips_ring_positions_and_trays_the_rest(panel_graph):
    view = place_view(panel_graph, "XR/XR02", symbols=BEAMLINE)
    (strip,) = _section(view, "strip").items
    assert isinstance(strip, Strip)
    # only a class with a symbol is drawn: XR02:SHIELD1 (Infrastructure) has a
    # ring position but no symbol, so it is counted, not drawn
    assert [(element.thing, element.s_m) for element in strip.elements] == [("XR02:QF1", 11.0)]
    assert strip.extent_m == (11.0, 11.0)
    # only classes with a symbol belong on the beamline: the tray, with each reason
    tray = {entry.thing: entry.reason for entry in strip.tray}
    assert tray == {
        "XR02:QD1": "no position",
        "XR02:BEND1": "no position",
        "XR02:IP1": "no position",  # a ring-frame row with no s
        "XR02:IP2": "no position",
    }
    # the rest is one line to the place's card: XR02:SHIELD1, XR02:BLOCK1 and
    # XR02:CH5 (by its claim)
    assert strip.others == Link("3 other things in this place", Target("card", "place", "XR/XR02"))
    assert not _ids(view) & set(EXCLUDED)


def test_place_view_two_at_one_s_are_both_drawn_on_their_lanes(panel_graph):
    # a class meant to show without a picture of its own is mapped to generic on purpose
    symbols = {"Pump": "pump", "Quadrupole": "generic"}
    view = place_view(panel_graph, "XR/XR01", symbols=symbols, above=frozenset({"pump"}))
    (strip,) = _section(view, "strip").items
    assert [(e.thing, e.cls, e.s_m, e.symbol, e.lane) for e in strip.elements] == [
        ("XR01:IP1", "Pump", 1.25, "pump", "above"),
        ("XR01:QF1", "Quadrupole", 1.25, "generic", "beam"),
        ("XR01:QD1", "Quadrupole", 2.0, "generic", "beam"),
    ]
    assert strip.elements[1].link == Link("XR01:QF1", card("XR01:QF1"))
    assert strip.extent_m == (1.25, 2.0)
    view = place_view(panel_graph, "XR/XR01", symbols={**BEAMLINE, "ElectronicsChassis": "generic"})
    (strip,) = _section(view, "strip").items
    tray = {entry.thing: entry.reason for entry in strip.tray}
    assert tray["XR01:BEND1"] == "fixture: frame treated as the ring frame"
    assert tray["XR01:CH1"] == "no position"  # within the rack, any depth


def test_a_ring_position_without_a_symbol_is_counted_not_drawn(panel_graph):
    drawn = place_view(panel_graph, "XR/XR02", symbols={**BEAMLINE, "Infrastructure": "generic"})
    (strip,) = _section(drawn, "strip").items
    assert [(e.thing, e.symbol) for e in strip.elements] == [
        ("XR02:QF1", "quadrupole"),
        ("XR02:SHIELD1", "generic"),
    ]
    # mapped, XR02:BLOCK1 (no position) joins the tray; only XR02:CH5 is counted
    assert "XR02:BLOCK1" in {entry.thing for entry in strip.tray}
    assert strip.others.label == "1 other things in this place"
    (strip,) = _section(place_view(panel_graph, "XR/XR02", symbols=BEAMLINE), "strip").items
    assert "XR02:SHIELD1" not in {e.thing for e in strip.elements}
    assert strip.others.label == "3 other things in this place"
    # no symbols at all: nothing is drawn and every device is counted
    (strip,) = _section(place_view(panel_graph, "XR/XR02"), "strip").items
    assert strip.elements == () and strip.extent_m is None and strip.tray == ()
    assert strip.others.target == Target("card", "place", "XR/XR02")


def test_place_view_links_child_places_and_the_parent(panel_graph):
    view = place_view(panel_graph, "XR/XR01")
    (racks,) = _section(view, "places within").items
    assert (racks.title, racks.form, racks.more) == ("racks", "chips", None)
    assert [(n.label, n.link.target) for n in racks.nodes] == [
        ("RK1", Target("rack", "place", "XR/XR01/RK1"))  # a rack opens its elevation
    ]
    assert [crumb.target for crumb in view.crumbs] == [HOME_TARGET, Target("place", "place", "XR")]


def test_places_within_are_grouped_by_what_the_data_states_and_counted_past_the_limit(
    panel_graph, monkeypatch
):
    described = sys.modules[place_view.__module__]
    view = place_view(panel_graph, "XR")
    groups = _section(view, "places within").items
    assert [g.title for g in groups] == ["sector"]  # the level the data states
    assert {n.label for n in groups[0].nodes} == {"XR01", "XR02"}  # short: the parent cut
    monkeypatch.setattr(described, "MAX_CHIPS", 1)
    (group,) = _section(place_view(panel_graph, "XR"), "places within").items
    assert len(group.nodes) == 1
    assert group.more == Link("+1 more", Target("card", "place", "XR"))


def test_a_place_view_ends_with_its_roll_up_by_system(panel_graph):
    view = place_view(panel_graph, "XR/XR01")
    (group,) = _section(view, "by system").items
    assert group.form == "rows"
    *_band, section = health_rollup(panel_graph, "XR/XR01", "system").sections
    assert group.nodes == section.items[0].children


def test_a_place_with_no_ring_position_has_no_extent(panel_graph):
    (strip,) = _section(place_view(panel_graph, "XR/XR01/RK1"), "strip").items
    assert strip.elements == () and strip.extent_m is None
    assert strip.tray == ()  # chassis have no symbol: they are not beamline devices
    # XR02:CH5 by its claims into the rack
    assert strip.others.label == "4 other things in this place"


# -- rack elevation ---------------------------------------------------------------


def test_rack_elevation_orders_units_top_down_and_lists_the_rest(panel_graph):
    assert RACK_ORDER == "top-down"
    view = rack_elevation(panel_graph, "XR/XR01/RK1")
    assert view.id == Target("rack", "place", "XR/XR01/RK1")
    (elevation,) = _section(view, "units").items
    assert isinstance(elevation, Elevation)
    assert elevation.order == RACK_ORDER
    assert [
        (u.position, u.size_u, u.standing, u.conflicting, u.link.target) for u in elevation.units
    ] == [
        # no built place: from its claims, which name two units, so both are drawn
        (41, None, "contested", True, card("XR02:CH5")),
        (40, None, "contested", True, card("XR02:CH5")),
        (20, None, "accepted", False, card("XR01:CH2")),
        (10, 2, "accepted", False, card("XR01:CH1")),  # the claim states size_u
    ]
    assert [(u.position, u.link.target) for u in elevation.unpositioned] == [
        (None, card("XR01:CH3"))
    ]
    every = (*elevation.units, *elevation.unpositioned)
    assert all(unit.lights == () for unit in every)  # a rack with no rule


def test_rack_elevation_lights_are_the_units_roll_up_channels(tmp_path):
    def racked_pump(document: dict[str, Any]) -> None:
        for device in document["devices"]:
            if device["id"] == "XR01:IP1":
                device["place"] = "XR/XR01/RK1"

    (elevation,) = _section(
        rack_elevation(_graph(tmp_path, racked_pump), "XR/XR01/RK1"), "units"
    ).items
    pump = [unit for unit in elevation.unpositioned if unit.link.target.id == "XR01:IP1"]
    assert pump[0].lights == (Light("XR01-IP1:State-Sts", disagree=True),)


def test_a_device_claimed_into_two_racks_is_in_both(panel_graph):
    assert Target("rack", "place", "XR/XR02/RK2") in targets(panel_graph)
    (elevation,) = _section(rack_elevation(panel_graph, "XR/XR02/RK2"), "units").items
    assert [(u.position, u.standing, u.link.target) for u in elevation.units] == [
        (5, "contested", card("XR02:CH5"))
    ]
    assert elevation.unpositioned == ()


def test_an_elevation_draws_every_claimed_slot_once(panel_graph):
    (elevation,) = _section(rack_elevation(panel_graph, "XR/XR01/RK1"), "units").items
    slots = [(u.link.target.id, u.position) for u in elevation.units]
    assert sorted(slots) == sorted(set(slots))
    assert "XR01:CH1" not in [u.link.target.id for u in elevation.unpositioned]


def test_sources_agreeing_on_one_unit_give_one_entry_with_both_citations(panel_graph):
    (elevation,) = _section(rack_elevation(panel_graph, "XR/XR01/RK1"), "units").items
    (ch2,) = [u for u in elevation.units if u.link.target.id == "XR01:CH2"]
    assert (ch2.position, ch2.conflicting) == (20, False)
    # one claim cites nothing, the other cites the layout
    assert [c.document for c in ch2.citations] == ["XR-LAYOUT"]


def test_a_device_with_one_unit_per_rack_does_not_conflict(panel_graph):
    (elevation,) = _section(rack_elevation(panel_graph, "XR/XR02/RK2"), "units").items
    assert [(u.position, u.conflicting) for u in elevation.units] == [(5, False)]


def test_a_size_that_does_not_parse_is_left_out(tmp_path):
    def odd_size(document: dict[str, Any]) -> None:
        for claim in document["relations"]:
            if claim["id"] == "xr01-ch1-rack":
                claim["attributes"]["size_u"] = "half"

    (elevation,) = _section(
        rack_elevation(_graph(tmp_path, odd_size), "XR/XR01/RK1"), "units"
    ).items
    sizes = {u.link.target.id: u.size_u for u in elevation.units}
    assert sizes["XR01:CH1"] is None


# -- health roll-up ---------------------------------------------------------------


def test_rollup_by_place_is_a_tree_of_children_with_their_channels(panel_graph):
    view = health_rollup(panel_graph, "XR/XR01", "place")
    assert view.id == Target("rollup", "place", "XR/XR01", by="place")
    (root,) = _section(view, "roll-up").items
    assert isinstance(root, Node)
    assert root.channels == ("XR01-IP1:State-Sts", "XR01-PS1:On-Sts")
    nodes = {node.label: node for node in root.children}
    assert nodes["XR01:IP1"].channels == ("XR01-IP1:State-Sts",)
    assert nodes["XR01:IP1"].link.target == card("XR01:IP1")
    # with a rule first, in beam order (XR01:IP1 at s 1.25, the supply has none),
    # then one row for the five without a rule, the rack among them, to the card
    assert [node.label for node in root.children] == [
        "XR01:IP1",
        "XR01:PS1",
        "5 without a rule",
    ]
    rest = root.children[-1]
    assert rest.no_rule is True and rest.channels == ()
    assert rest.link.target == Target("card", "place", "XR/XR01")


def test_rollup_by_place_links_down_to_child_rollups(panel_graph):
    (root,) = _section(health_rollup(panel_graph, "XR", "place"), "roll-up").items
    assert [(node.label, node.link.target) for node in root.children] == [
        ("XR/XR01", Target("rollup", "place", "XR/XR01", by="place")),
        ("XR/XR02", Target("rollup", "place", "XR/XR02", by="place")),
        ("not placed", Target("rollup", "place", NOT_PLACED, by="place")),
    ]


def test_rollup_by_system_groups_devices_by_class_by_default(panel_graph):
    view = health_rollup(panel_graph, "XR", "system")
    assert view.id == Target("rollup", "place", "XR", by="system")
    (root,) = _section(view, "roll-up").items
    systems = {node.label: node for node in root.children}
    assert systems["Pump"].channels == ("XR01-IP1:State-Sts", "XR02-IP1:Pressure-RB")
    assert systems["PowerSupply"].channels == ("XR01-PS1:On-Sts",)
    assert "ElectronicsChassis" not in systems  # among the systems without a rule
    assert root.children[-1].label == "2 without a rule"
    # one level: a system carries its devices' channels, not the devices, and
    # links to that system's roll-up within the place when it has a rule channel
    assert all(node.children == () for node in root.children)
    assert systems["Pump"].link.target == Target(
        "rollup", "place", "XR", by="system", system="Pump"
    )


def test_a_rollup_shows_one_level_below_its_root(panel_graph):
    for by in ("place", "system"):
        (root,) = _section(health_rollup(panel_graph, "XR", by), "roll-up").items
        assert root.children
        assert all(child.children == () for child in root.children)


def test_rollup_by_system_takes_the_system_key_from_a_function(panel_graph):
    view = health_rollup(
        panel_graph,
        "XR",
        "system",
        system_of=lambda thing: "vacuum" if thing.cls == "Pump" else None,
    )
    (root,) = _section(view, "roll-up").items
    assert [node.label for node in root.children] == ["vacuum", "no system", NOT_PLACED]


def test_rollup_refuses_an_unknown_grouping(panel_graph):
    with pytest.raises(ValueError, match="by"):
        health_rollup(panel_graph, "XR", "colour")


# -- checkout ---------------------------------------------------------------------


def test_checkout_links_each_child_place_to_its_checkout_or_card(panel_graph):
    view = checkout(panel_graph, "XR")
    links = _section(view, "places within").items
    assert [(link.label, link.target) for link in links] == [
        ("XR/XR01", Target("checkout", "place", "XR/XR01")),
        ("XR/XR02", Target("checkout", "place", "XR/XR02")),
        ("not placed", Target("checkout", "place", NOT_PLACED)),
    ]
    rack = _section(checkout(panel_graph, "XR/XR01"), "places within").items
    assert [link.target for link in rack] == [Target("card", "place", "XR/XR01/RK1")]


def test_checkout_has_one_row_per_device_channel_and_totals(panel_graph):
    view = checkout(panel_graph, "XR/XR01")
    assert view.id == Target("checkout", "place", "XR/XR01")
    rows = {row.channel: row for row in _section(view, "channels").items}
    assert "XR01:RackTemp-RB" not in rows  # on a place, not a device
    assert rows["XR01-IP1:State-Sts"] == CheckRow(
        channel="XR01-IP1:State-Sts",
        thing="XR01:IP1",
        on_device=True,
        typed=True,
        has_rule=True,
        has_source=True,
    )
    assert rows["XR01-CH1:Temp-RB"] == CheckRow(  # typed by its role: a readback is a value
        "XR01-CH1:Temp-RB", "XR01:CH1", True, True, False, False
    )
    # sourced by its health rows alone
    assert (rows["XR01-PS1:On-Sts"].typed, rows["XR01-PS1:On-Sts"].has_source) == (True, True)
    (totals,) = _section(view, "totals").items
    assert totals == Totals(
        channels=len(rows),
        on_device=len(rows),
        typed=sum(row.typed for row in rows.values()),
        has_rule=sum(row.has_rule for row in rows.values()),
        has_source=sum(row.has_source for row in rows.values()),
    )
    assert (totals.channels, totals.typed, totals.has_rule) == (13, 13, 2)  # all by role at least


def test_describe_refuses_a_rollup_with_no_grouping(panel_graph):
    with pytest.raises(ValueError, match="by"):
        describe(panel_graph, Target("rollup", "place", "XR"))


def test_describe_refuses_a_view_kind_it_does_not_know(panel_graph):
    with pytest.raises(ValueError, match="view"):
        describe(panel_graph, Target("map", "place", "XR"))  # type: ignore[arg-type]


# -- scale --------------------------------------------------------------------------


def _big_graph(devices: int) -> PanelGraph:
    sectors = 500
    places = [{"id": "R"}] + [{"id": f"R/S{n}"} for n in range(sectors)]
    document: dict[str, Any] = {
        "places": places,
        "devices": [],
        "channels": [],
        "groups": [{"id": f"G{n}", "members": []} for n in range(90)],
        "relations": [],
    }
    poses, types, health, pairs = [], [], [], []
    for n in range(devices):
        device = f"D{n}"
        document["devices"].append(
            {"id": device, "class": f"C{n % 40}", "place": f"R/S{n % sectors}"}
        )
        document["groups"][n % 90]["members"].append(device)
        for suffix, role in (("SP", "setpoint"), ("RB", None), ("St", None)):
            channel = {"id": f"{device}:{suffix}", "on": {"device": device}}
            if role:
                channel["role"] = role
            document["channels"].append(channel)
            types.append(RecordType(f"{device}:{suffix}", "ai", "value", "stated", "s", "r", "l"))
        pairs.append(Pair(f"{device}:SP", f"{device}:RB", "e", "stated", "s", "r", "l"))
        health.append(HealthRule(f"{device}:St", "1", "bad", "e", "stated", "s", "r", "l"))
        if n:
            document["relations"].append(
                {
                    "id": f"r{n}",
                    "type": "cabled_to",
                    "subject": {"device": device},
                    "object": {"device": f"D{n - 1}"},
                }
            )
        poses.append(
            Pose(
                device,
                "RING",
                0,
                0,
                0,
                0,
                0,
                0,
                n * 0.01,
                0,
                0,
                None,
                None,
                "stated",
                None,
                None,
                None,
                None,
            )
        )
    tables = PanelTables(tuple(poses), tuple(types), tuple(pairs), tuple(health), (), ())
    return PanelGraph(FacilityRecords(document), tables)


def test_targets_and_views_scale_near_linearly():
    small, large = _big_graph(2_000), _big_graph(8_000)

    def once(graph: PanelGraph) -> float:
        started = time.perf_counter()
        found = targets(graph)
        place_view(graph, "R")
        health_rollup(graph, "R", "place")
        health_rollup(graph, "R", "system")
        checkout(graph, "R")
        for target in found:
            if target.view == "card" and target.kind == "device":
                device_card(graph, target.id)
        return time.perf_counter() - started

    def cost(graph: PanelGraph) -> float:
        # The best of three, with the cyclic collector off: its passes grow
        # with the heap, not with this code, and made the ratio noisy.
        gc.collect()
        gc.disable()
        try:
            return min(once(graph) for _ in range(3))
        finally:
            gc.enable()

    ratio = cost(large) / cost(small)
    assert ratio < 8, f"4x the size took {ratio:.1f}x the time"


# -- what is in no place ------------------------------------------------------------


def test_every_root_roll_up_has_a_not_placed_child_that_reads_the_unplaced(panel_graph):
    loose = "XR09-LOOSE1:Fault-Sts"
    for by in ("place", "system"):
        (root,) = _section(health_rollup(panel_graph, "XR", by), "roll-up").items
        unplaced = next(node for node in root.children if node.label == NOT_PLACED)
        assert unplaced.channels == (loose,)
        assert unplaced.link.target == Target("rollup", "place", NOT_PLACED, by="place")
        assert loose in root.channels
    (sector,) = _section(health_rollup(panel_graph, "XR/XR01"), "roll-up").items
    assert NOT_PLACED not in [node.label for node in sector.children]  # a root's child only


def test_the_not_placed_screens_are_targets_with_their_devices(panel_graph):
    found = targets(panel_graph)
    assert Target("rollup", "place", NOT_PLACED, by="place") in found
    assert Target("checkout", "place", NOT_PLACED) in found
    view = describe(panel_graph, Target("rollup", "place", NOT_PLACED, by="place"))
    (root,) = _section(view, "roll-up").items
    assert [(node.label, node.channels) for node in root.children] == [
        ("XR09:LOOSE1", ("XR09-LOOSE1:Fault-Sts",))
    ]
    rows = _section(describe(panel_graph, Target("checkout", "place", NOT_PLACED)), "channels")
    assert [row.channel for row in rows.items] == ["XR09-LOOSE1:Fault-Sts"]


def test_a_roll_up_reads_a_device_its_claims_place_and_no_built_place_does(tmp_path):
    def ruled_chassis(document: dict[str, Any]) -> None:
        document["channels"].append({"id": "XR02-CH5:Fault-Sts", "on": {"device": "XR02:CH5"}})

    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as rows:
        rows.write("XR02-CH5:Fault-Sts,1,bad,database-severity,stated,xr-ioc,0a1b2c3,ch.db:1\n")
    graph = load_graph(write_fixture_facility(tmp_path / "facility.json", ruled_chassis), tables)
    for place in ("XR", "XR/XR02", "XR/XR02/RK2", "XR/XR01/RK1"):
        (root,) = _section(health_rollup(graph, place), "roll-up").items
        assert "XR02-CH5:Fault-Sts" in root.channels, place
    assert Target("rollup", "place", "XR/XR02/RK2", by="place") in targets(graph)


def test_a_channel_on_no_thing_is_in_the_root_checkout_and_under_not_placed(tmp_path):
    def orphan(document: dict[str, Any]) -> None:
        document["channels"].append({"id": "XR-NOWHERE:Beam-Sts"})

    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as rows:
        rows.write("XR-NOWHERE:Beam-Sts,1,bad,database-severity,stated,xr-ioc,0a1b2c3,b.db:1\n")
    graph = load_graph(write_fixture_facility(tmp_path / "facility.json", orphan), tables)
    rows = {row.channel: row for row in _section(checkout(graph, "XR"), "channels").items}
    assert rows["XR-NOWHERE:Beam-Sts"] == CheckRow(
        "XR-NOWHERE:Beam-Sts", None, False, False, True, True
    )
    assert "XR-NOWHERE:Beam-Sts" not in {
        row.channel for row in _section(checkout(graph, "XR/XR01"), "channels").items
    }
    (root,) = _section(health_rollup(graph, "XR"), "roll-up").items
    assert "XR-NOWHERE:Beam-Sts" in root.children[-1].channels  # the "not placed" child
    view = describe(graph, Target("rollup", "place", NOT_PLACED, by="place"))
    (unplaced,) = _section(view, "roll-up").items
    assert unplaced.children[-1].channels == ("XR-NOWHERE:Beam-Sts",)


def test_a_unit_that_is_no_integer_lists_the_device_unpositioned_with_its_text(tmp_path):
    def odd_unit(document: dict[str, Any]) -> None:
        for claim in document["relations"]:
            if claim["id"] == "xr02-ch5-rack2":
                claim["object"]["unit"] = "A5"

    graph = _graph(tmp_path, odd_unit)
    (elevation,) = _section(rack_elevation(graph, "XR/XR02/RK2"), "units").items
    assert elevation.units == ()
    assert [(u.link.target.id, u.unit_text) for u in elevation.unpositioned] == [("XR02:CH5", "A5")]


def test_a_system_within_a_place_has_its_own_roll_up_of_its_devices(panel_graph):
    found = targets(panel_graph)
    pumps = Target("rollup", "place", "XR/XR02", by="system", system="Pump")
    assert pumps in found
    assert Target("rollup", "place", "XR", by="system", system="ElectronicsChassis") not in found
    view = describe(panel_graph, pumps)
    assert view.id == pumps and view.crumbs[-1].target == Target(
        "rollup", "place", "XR/XR02", by="system"
    )
    (root,) = _section(view, "roll-up").items
    assert root.label == "Pump" and root.channels == ("XR02-IP1:Pressure-RB",)
    assert [node.label for node in root.children] == ["XR02:IP1", "1 without a rule"]


def test_a_places_own_rule_channels_are_a_child_of_its_roll_ups(tmp_path):
    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as rows:
        rows.write("XR01:RackTemp-RB,>=40,bad,database-severity,stated,xr-ioc,0a1b2c3,r.db:1\n")
    graph = load_graph(write_fixture_facility(tmp_path / "facility.json"), tables)
    for by in ("place", "system"):
        (root,) = _section(health_rollup(graph, "XR/XR01/RK1", by), "roll-up").items
        (own,) = [node for node in root.children if node.label == "XR/XR01/RK1 itself"]
        assert own.channels == ("XR01:RackTemp-RB",)


# -- parts follow their whole ---------------------------------------------------------


def test_a_part_card_says_where_it_is_as_part_of_its_whole_inferred(tmp_path) -> None:
    graph = graph_with_part(tmp_path)
    where = _section(describe(graph, card("XR01:CARD1")), "where").items
    assert where[0] == Link(
        "in XR/XR01/RK1, as part of XR01:CH1", card("XR01:CH1"), basis="inferred"
    )


def test_a_part_is_a_child_of_its_wholes_place_roll_up(tmp_path) -> None:
    graph = graph_with_part(tmp_path)
    (rack,) = _section(health_rollup(graph, "XR/XR01/RK1"), "roll-up").items
    child = next(node for node in rack.children if node.label == "XR01:CARD1")
    assert child.channels == ("XR01-CARD1:Fault-Sts",)
    (root,) = _section(health_rollup(graph, "XR"), "roll-up").items
    assert "XR01-CARD1:Fault-Sts" in root.channels
    unplaced = next(node for node in root.children if node.label == NOT_PLACED)
    assert "XR01-CARD1:Fault-Sts" not in unplaced.channels


def test_a_place_card_lists_a_part_placed_by_its_whole_as_inferred(tmp_path) -> None:
    graph = graph_with_part(tmp_path)
    within = _section(describe(graph, Target("card", "place", "XR/XR01/RK1")), "within").items
    assert within[-1] == Link("XR01:CARD1", card("XR01:CARD1"), basis="inferred")


# -- parts in the index and the rack ----------------------------------------------------


def test_a_rack_whose_rules_sit_on_parts_has_its_roll_ups_and_system_rows(tmp_path) -> None:
    graph = graph_with_part(tmp_path)
    found = targets(graph)
    system = Target("rollup", "place", "XR/XR01/RK1", by="system", system="ElectronicsModule")
    assert system in found
    assert Target("rollup", "place", "XR", by="system", system="ElectronicsModule") in found
    (root,) = _section(health_rollup(graph, "XR/XR01/RK1", "system"), "roll-up").items
    row = next(node for node in root.children if node.label == "ElectronicsModule")
    assert row.link is not None and row.link.target == system


def test_a_rack_unit_light_reads_the_parts_placed_through_its_device(tmp_path) -> None:
    graph = graph_with_part(tmp_path)
    (elevation,) = _section(rack_elevation(graph, "XR/XR01/RK1"), "units").items
    unit = next(u for u in elevation.units if u.link.label == "XR01:CH1")
    assert "XR01-CARD1:Fault-Sts" in [item.channel for item in unit.lights]
    assert unit.link.target == card("XR01:CH1")
    assert "XR01:CARD1" not in [u.link.label for u in (*elevation.units, *elevation.unpositioned)]


def test_a_device_claimed_into_a_rack_with_no_unit_is_unpositioned_there(tmp_path) -> None:
    def claimed(document: dict[str, Any]) -> None:
        document["relations"].append(
            {
                "id": "xr02-block1-rack",
                "type": "located_in",
                "subject": {"device": "XR02:BLOCK1"},
                "object": {"place": "XR/XR01/RK1"},
                "standing": "accepted",
            }
        )

    graph = load_graph(write_fixture_facility(tmp_path / "facility.json", claimed), PANELS_DIR)
    (elevation,) = _section(rack_elevation(graph, "XR/XR01/RK1"), "units").items
    # no card: named as a fact, its id the text
    assert "XR02:BLOCK1" in [
        u.link.text for u in elevation.unpositioned if isinstance(u.link, Fact)
    ]


def test_every_roll_up_counts_a_channel_once_at_every_level(tmp_path) -> None:
    def ruled_chassis(document: dict[str, Any]) -> None:
        document["channels"].append({"id": "XR02-CH5:Fault-Sts", "on": {"device": "XR02:CH5"}})

    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as rows:
        rows.write("XR02-CH5:Fault-Sts,1,bad,database-severity,stated,xr-ioc,0a1b2c3,ch.db:1\n")
    graph = load_graph(write_fixture_facility(tmp_path / "facility.json", ruled_chassis), tables)

    def walk(node):
        yield node
        for child in node.children:
            yield from walk(child)

    for place in ("XR", "XR/XR01", "XR/XR02"):
        for by in ("place", "system"):
            (root,) = _section(health_rollup(graph, place, by), "roll-up").items
            for node in walk(root):
                assert len(node.channels) == len(set(node.channels)), (place, by, node.label)
    (root,) = _section(health_rollup(graph, "XR"), "roll-up").items
    assert root.channels.count("XR02-CH5:Fault-Sts") == 1


def test_a_class_the_draw_settings_lift_is_drawn_on_the_upper_lane(panel_graph, tmp_path):
    import json

    from osprey.facility.panels.emit import load_draw

    path = tmp_path / "draw.json"
    path.write_text(
        json.dumps({"symbols": {"Pump": "pump", "Quadrupole": "generic"}, "above": ["Pump"]})
    )
    draw = load_draw(path)
    view = place_view(panel_graph, "XR/XR01", symbols=draw.symbols, above=draw.above)
    (strip,) = _section(view, "strip").items
    lanes = {e.thing: (e.symbol, e.lane) for e in strip.elements}
    assert lanes["XR01:IP1"] == ("pump", "above")
    assert lanes["XR01:QF1"][1] == "beam"


def test_a_strip_element_carries_the_values_its_symbols_annotation_names(panel_graph):
    annotate = {"quadrupole": ("current_readback",), "ion_pump": ("vacuum_readback",)}
    view = place_view(
        panel_graph, "XR/XR01", symbols={**BEAMLINE, "Pump": "ion_pump"}, annotate=annotate
    )
    (strip,) = _section(view, "strip").items
    values = {e.thing: e.annotations for e in strip.elements}
    assert values["XR01:QF1"] == (("XR01-QF1:Current-RB", "A"),)
    assert values["XR01:IP1"] == (("XR01-IP1:Pressure-RB", None),)
    plain = place_view(panel_graph, "XR/XR01", symbols={**BEAMLINE, "Pump": "ion_pump"})
    (strip,) = _section(plain, "strip").items
    assert all(e.annotations == () for e in strip.elements)  # nothing named, nothing shown


# -- calibration ----------------------------------------------------------------


def _calibration_rows(graph, device):
    (table,) = _section(device_card(graph, device), "calibration").items
    return {row.title: row for row in table.rows}


def test_a_card_has_a_calibration_row_per_kind_of_its_class(panel_graph):
    rows = _calibration_rows(panel_graph, "XR01:QF1")
    assert list(rows) == [
        "Integrated gradient", "Excitation curve", "Magnetic centre",
        "Strength scale from the beam", "Current offset",
    ]  # fmt: skip
    gradient = rows["Integrated gradient"]
    assert (gradient.shown, gradient.note) == ("-4.9198 T", "+0.21 % from design")
    assert gradient.conditions == "at 73.33 A" and gradient.state == "bench"
    assert gradient.chip == "bench · stretched wire" and gradient.source == "XR-MAG-TEST A.2"
    assert gradient.history == "replaces design -4.9094 T"
    assert rows["Excitation curve"].shown == "2 points (0 A, 73.33 A)"
    assert rows["Magnetic centre"].shown == "-2.5 · -14.5 um"
    assert rows["Magnetic centre"].history == "first value"
    scale = rows["Strength scale from the beam"]
    assert (scale.shown, scale.chip, scale.below_wanted) == (None, "beam wanted", True)
    live = rows["Current offset"]
    assert (live.channels, live.chip) == (("XR01-QF1:Current-SP",), "EPICS")
    assert "calibration" not in _titles(device_card(panel_graph, "XR01:IP1"))


def test_a_contested_row_names_the_other_claim(panel_graph):
    row = _calibration_rows(panel_graph, "XR02:QF1")["Integrated gradient"]
    assert row.state == "contested" and row.chip == "two sources disagree"
    assert row.note == "also claimed: -4.9701 T"


def _networks(*others: tuple[str, str, str]):
    def add(document: dict[str, Any]) -> None:
        for n, (kind, other, standing) in enumerate(others):
            document["relations"].append(
                {
                    "id": f"xr01-qf1-net-{n}",
                    "type": "networked_to",
                    "subject": {"device": "XR01:QF1"},
                    "object": {kind: other},
                    "standing": standing,
                }
            )

    return add


def test_the_default_network_hop_takes_a_device_and_one_thing_is_not_contested(tmp_path):
    place = ("place", "XR01", "accepted")
    one = _graph(tmp_path / "a", _networks(place, ("device", "XR01:CH2", "contested")))
    (block,) = [b for b in _blocks(device_card(one, "XR01:QF1")) if b.title.startswith("Network")]
    assert block.title == "Network  XR01:CH2" and block.standing is None and not block.dashed
    two = _graph(
        tmp_path / "b",
        _networks(place, ("device", "XR01:CH2", "accepted"), ("device", "XR01:PS1", "accepted")),
    )
    blocks = [b for b in _blocks(device_card(two, "XR01:QF1")) if b.title.startswith("Network")]
    assert [b.standing for b in blocks] == ["contested", "contested"]  # both named, none picked


def test_grid_names_lose_the_segments_they_share_when_no_device_id_cuts_them():
    from osprey.facility.panels.describe import _shared_cut

    rows = [
        ("PLC:IN2:Relay_Sts", "PLC:IN2:Relay_Sts"),
        ("PLC:IN2:Relay_FF", "PLC:IN2:Relay_FF"),
        ("P", "DEV:P"),
    ]
    assert _shared_cut(rows) == (
        ("Relay_Sts", "PLC:IN2:Relay_Sts"),
        ("Relay_FF", "PLC:IN2:Relay_FF"),
        ("P", "DEV:P"),
    )
    assert _shared_cut(rows[:1]) == tuple(rows[:1])  # one alone keeps its name


# -- the ways down: home, by place and by system, and back -------------------------


def _reached(graph, start):
    """Every target a walk over the links of the described views reaches from *start*."""
    from tests.facility.panels._views import links

    seen, todo = {start}, [start]
    while todo:
        for link in links(describe(graph, todo.pop())):
            if link.target not in seen:
                seen.add(link.target)
                todo.append(link.target)
    return seen


def test_every_screen_is_reached_from_home(panel_graph):
    found = targets(panel_graph)
    assert found[0] == HOME_TARGET  # the one way in, written first
    assert set(found) <= _reached(panel_graph, HOME_TARGET)


def test_home_holds_the_top_places_the_sets_and_what_is_in_no_place(panel_graph):
    view = home(panel_graph)
    assert (view.id, view.crumbs) == (HOME_TARGET, ())
    (places,) = _section(view, "places").items
    assert [(n.label, n.link.target) for n in places.nodes] == [
        ("XR", Target("place", "place", "XR"))
    ]
    sets = _section(view, SETS_SECTION).items
    assert [(g.title, [n.label for n in g.nodes]) for g in sets] == [
        ("Pump", ["XR01-VAC", "XR02-VAC"]),  # under their class, in name order
        ("Quadrupole", ["XR-QUAD"]),
    ]
    rollup, by_class = _section(view, NOT_PLACED).items
    assert rollup.target == Target("rollup", "place", NOT_PLACED, by="place")
    page = Target("home", "place", HOME, by="unplaced", system="Infrastructure")
    assert [(n.label, n.link.target) for n in by_class.nodes] == [("Infrastructure (1)", page)]
    (chips,) = _section(describe(panel_graph, page), NOT_PLACED).items  # the class's index
    assert [n.link.target for n in chips.nodes] == [card("XR09:LOOSE1")]


def test_home_puts_the_machines_first_and_the_other_places_below(tmp_path):
    def change(document: dict[str, Any]) -> None:
        document["places"].append({"id": "YARD"})
        for device in document["devices"]:
            if device["id"] == "XR01:QF1":
                device["model"] = "XR"

    graph = load_graph(write_fixture_facility(tmp_path / "f.json", change), PANELS_DIR)
    machines, others = _section(home(graph), "places").items
    assert (machines.title, [n.label for n in machines.nodes]) == ("machines", ["XR"])
    assert (others.title, [n.label for n in others.nodes]) == ("other places", ["YARD"])


def test_a_device_card_links_back_to_its_sets(panel_graph):
    (foot,) = _section(device_card(panel_graph, "XR01:QF1"), "foot").items
    assert foot.sets == (Link("XR-QUAD", Target("set", "group", "XR-QUAD")),)
    (none,) = _section(device_card(panel_graph, "XR01:CH1"), "foot").items
    assert none.sets == ()


def test_a_place_view_and_a_system_roll_up_open_the_sets_in_them(panel_graph):
    quad = Link("XR-QUAD", Target("set", "group", "XR-QUAD"))
    pumps = Link("XR01-VAC", Target("set", "group", "XR01-VAC"))
    # the sector's sets: the one across both sectors and its own pumps, not the other's
    assert _section(place_view(panel_graph, "XR/XR01"), SETS_SECTION).items == (quad, pumps)
    one = health_rollup(panel_graph, "XR/XR01", "system", system="Quadrupole")
    assert _section(one, SETS_SECTION).items == (quad,)  # a set is one class
    every = health_rollup(panel_graph, "XR/XR01", "system")  # every system: every set here
    assert _section(every, SETS_SECTION).items == (quad, pumps)
    assert SETS_SECTION not in _titles(health_rollup(panel_graph, "XR/XR01", "place"))


def test_a_place_card_draws_its_own_lights_up_to_the_cap_and_counts_the_rest(tmp_path, monkeypatch):
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    def faults(document) -> None:
        for n in range(3):
            document["channels"].append({"id": f"XR:Fault{n}-Sts", "on": {"place": "XR"}})

    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as handle:
        for n in range(3):
            handle.write(f"XR:Fault{n}-Sts,1,bad,screen,stated,xr-screens,4d5e6f7,f.screen:{n}\n")
    graph = load_graph(write_fixture_facility(tmp_path / "f.json", faults), tables)
    described = sys.modules[device_card.__module__]
    monkeypatch.setattr(described, "HEALTH_CAP", 2)
    health = _section(device_card(graph, "XR", "place"), "health").items
    assert [type(i).__name__ for i in health] == ["Light", "Light", "Link"]
    assert health[-1] == Link(
        "1 more with a health rule, in the roll-up", Target("rollup", "place", "XR", by="place")
    )


def test_a_part_stands_where_its_whole_is(tmp_path):
    from osprey.facility.panels.describe import _where_on_strip
    from osprey.facility.panels.layout import _BY_WHOLE
    from tests.facility.panels._fixture import add_part
    from tests.facility.panels._views import nodes, stub_render

    def coil(document: dict[str, Any]) -> None:
        add_part(document, "XR01:QF1:PUMP", "XR01:QF1")
        document["devices"][-1]["class"] = "Pump"

    graph = graph_with_part(tmp_path, coil)
    # crumbs: home, then the places round the whole (a part has no place of its own)
    view = device_card(graph, "XR01:CARD1")
    assert [c.label for c in view.crumbs] == ["home", "XR", "XR/XR01", "XR/XR01/RK1"]
    head = next(i for s in view.sections for i in s.items if isinstance(i, CardHead))
    assert head.by_whole == ("XR/XR01/RK1", "XR01:CH1")
    texts = [n["props"].get("text") for n in nodes(stub_render(view, graph))]
    assert _BY_WHOLE.format("XR/XR01/RK1", "XR01:CH1") in texts
    # the strip: at its whole's position
    part = graph.thing("XR01:QF1:PUMP", "device")
    assert _where_on_strip(graph, part) == (1.25, None, RING_FRAME)
    strip = place_view(graph, "XR/XR01", symbols={"Pump": "pump"}).sections
    elements = [e for s in strip for i in s.items for e in getattr(i, "elements", ())]
    assert ("XR01:QF1:PUMP", 1.25) in [(e.thing, e.s_m) for e in elements]


def test_a_contested_part_of_does_not_place_a_part(tmp_path):
    def contest(document: dict[str, Any]) -> None:
        document["relations"][-1]["standing"] = "contested"

    graph = graph_with_part(tmp_path, contest)
    assert graph.whole_of("XR01:CARD1") is None
    assert graph.place_of("XR01:CARD1") is None
    assert [c.label for c in device_card(graph, "XR01:CARD1").crumbs] == ["home"]


def test_a_rack_row_with_no_unit_takes_the_unit_another_row_of_the_device_states(tmp_path):
    def rows(document: dict[str, Any]) -> None:
        for n, unit in enumerate((None, 7)):
            obj: dict[str, Any] = {"place": "XR/XR01/RK1"}
            if unit is not None:
                obj["unit"] = unit
            document["relations"].append(
                {
                    "id": f"xr02-block1-rack-{n}",
                    "type": "located_in",
                    "subject": {"device": "XR02:BLOCK1"},
                    "object": obj,
                    "standing": "accepted",
                }
            )

    graph = load_graph(write_fixture_facility(tmp_path / "facility.json", rows), PANELS_DIR)
    (elevation,) = _section(rack_elevation(graph, "XR/XR01/RK1"), "units").items
    named = {id(u): u.link.text if isinstance(u.link, Fact) else u.link.label for u in [
        *elevation.units, *elevation.unpositioned
    ]}  # fmt: skip
    placed = [(u.position, u.conflicting) for u in elevation.units if named[id(u)] == "XR02:BLOCK1"]
    assert placed == [(7, False)]
    assert "XR02:BLOCK1" not in [named[id(u)] for u in elevation.unpositioned]


def test_a_stretch_across_a_rings_start_is_read_on_in_one_piece():
    from osprey.facility.panels.describe import _unwrapped

    last_sector = {"a": 171.6, "b": 175.0, "c": 181.9, "valve": 0.07}
    assert _unwrapped(last_sector, 181.9) == {**last_sector, "valve": 0.07 + 181.9}
    whole_ring = {str(i): i * 10.0 for i in range(19)}  # no gap over half the ring
    assert _unwrapped(whole_ring, 180.0) == whole_ring
    assert _unwrapped(last_sector, None) == last_sector  # a line: as stated


def test_a_line_model_is_drawn_from_its_devices_s(tmp_path):
    def line(document: dict[str, Any]) -> None:
        document["places"].append({"id": "XL"})
        document["models"] = [
            *document.get("models", []),
            {"name": "XL", "engine": "none", "length": 8.0, "periodic": False},
        ]
        for n, (cls, s) in enumerate((("Quadrupole", 2.0), ("Pump", 0.5), ("Quadrupole", 7.5))):
            document["devices"].append(
                {"id": f"XL:D{n}", "class": cls, "place": "XL", "model": "XL", "s": s}
            )

    graph = load_graph(write_fixture_facility(tmp_path / "f.json", line), PANELS_DIR)
    assert graph.beam_position(graph.thing("XL:D0", "device")) == ("model XL", 2.0)
    assert graph.frame_length("model XL") is None  # one pass: never wrapped
    assert graph.frame_length(RING_FRAME) is None  # no model states the ring's length
    view = place_view(graph, "XL", symbols={"Quadrupole": "quad", "Pump": "pump"})
    (strip,) = [i for s in view.sections for i in s.items if isinstance(i, Strip)]
    assert [(e.thing, e.s_m) for e in strip.elements] == [
        ("XL:D1", 0.5),
        ("XL:D0", 2.0),
        ("XL:D2", 7.5),
    ]
    assert strip.extent_m == (0.5, 7.5) and not strip.tray


def ring_model(length: float):
    """A change giving every device with a ring position the periodic model ``XRM``."""

    def change(document: dict[str, Any]) -> None:
        document["models"] = [
            *document.get("models", []),
            {"name": "XRM", "engine": "none", "length": length, "periodic": True},
        ]
        for device in document["devices"]:
            if device["id"].startswith(("XR01:", "XR02:")):
                device["model"] = "XRM"

    return change


def test_the_ring_frame_takes_the_length_of_the_periodic_model_of_its_devices(tmp_path):
    graph = load_graph(write_fixture_facility(tmp_path / "f.json", ring_model(12.0)), PANELS_DIR)
    assert graph.frame_length(RING_FRAME) == 12.0
    assert graph.frame_length("model XRM") == 12.0


def test_a_card_says_which_beam_position_it_uses_and_from_where(tmp_path):
    def stated(document: dict[str, Any]) -> None:
        for device in document["devices"]:
            if device["id"] == "XR01:QF1":
                device.update(model="XRM", s=1.5)  # the deck's s, another origin
            if device["id"] == "XR01:QD1":
                device.update(model="XRM", s=2.0)

    graph = load_graph(write_fixture_facility(tmp_path / "f.json", stated), PANELS_DIR)

    def beam(device: str) -> Fact:
        (fact,) = [
            i for i in _section(device_card(graph, device), "where").items
            if isinstance(i, Fact) and i.label == "beam position"
        ]  # fmt: skip
        return fact

    qf1 = beam("XR01:QF1")
    assert qf1.text == "s 1.250 m along XR-RING, from its pose in positions.csv"
    assert qf1.caveat == "the facility file states s 1.500 m along model XRM"
    assert beam("XR01:QD1").caveat is None  # the two agree
