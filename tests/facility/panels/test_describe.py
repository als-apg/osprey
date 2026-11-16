"""View descriptions over the fixture: what each screen holds, without layout."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.panels import load_graph
from osprey.facility.panels.describe import (
    RELATION_TITLES,
    checkout,
    describe,
    device_card,
    health_rollup,
    place_view,
    rack_elevation,
    set_overview,
    targets,
)
from osprey.facility.panels.model import PanelGraph
from osprey.facility.panels.tables import HealthRule, Pair, PanelTables, Pose, RecordType
from osprey.facility.panels.view import (
    CheckRow,
    Elevation,
    Fact,
    Grid,
    Light,
    Link,
    Node,
    Picture,
    Strip,
    Target,
    Totals,
    ValueRow,
    View,
)
from osprey.facility.records import FacilityRecords
from tests.facility.panels._fixture import PANELS_DIR, write_fixture_facility

RACKS = frozenset({"rack"})
EXCLUDED = ("XR02:SAF1", "XR/XR02/SAFE", "XR/XR02/SAFE/CAB1", "XR02:SAF2")


def card(thing_id: str) -> Target:
    return Target("card", "device", thing_id)


def _section(view: View, title: str):
    found = [section for section in view.sections if section.title == title]
    assert len(found) == 1, f"{view.id}: sections {[s.title for s in view.sections]}"
    return found[0]


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
    fed = _section(view, "fed by")
    assert [(item.label, item.target) for item in fed.items] == [("XR01:PS1", card("XR01:PS1"))]


def test_the_supply_links_back_to_both_magnets(panel_graph):
    feeds = _section(device_card(panel_graph, "XR01:PS1"), "feeds")
    # graph order: the built facility sorts claims by id
    assert [item.target for item in feeds.items] == [card("XR01:QD1"), card("XR01:QF1")]


def test_a_paired_setpoint_sits_beside_its_readback_in_signals(panel_graph):
    view = device_card(panel_graph, "XR01:QF1")
    signals = _section(view, "signals")
    assert signals.items == (
        ValueRow(
            label="Current-SP",
            channel="XR01:QF1:Current-SP",
            readback="XR01:QF1:Current-RB",
            tolerance=("absolute", 0.05),
            writable=True,
            drawn_as="setpoint",
            inferred=False,
        ),
    )
    assert "settings" not in _titles(view)


def test_a_writable_channel_with_no_pair_lands_in_settings(panel_graph):
    view = device_card(panel_graph, "XR01:QD1")
    settings = _section(view, "settings")
    assert [(row.channel, row.writable, row.readback) for row in settings.items] == [
        ("XR01:QD1:Current-SP", True, None)
    ]
    signals = _section(view, "signals")
    assert [(row.channel, row.writable) for row in signals.items] == [
        ("XR01:QD1:Current-RB", False)
    ]


def test_readable_signals_are_ordered_by_how_they_are_drawn(panel_graph):
    signals = _section(device_card(panel_graph, "XR01:IP1"), "signals")
    assert [row.channel for row in signals.items] == [
        "XR01:IP1:State-Sts",  # drawn as state
        "XR01:IP1:Pressure-RB",  # untyped, last
    ]


def test_an_inferred_pair_marks_its_value_row(panel_graph):
    signals = _section(device_card(panel_graph, "XR01:PS1"), "signals")
    row = signals.items[0]
    assert (row.channel, row.readback, row.inferred) == (
        "XR01:PS1:On-Cmd",
        "XR01:PS1:On-Sts",
        True,
    )


def test_a_light_per_rule_bearing_channel_marks_inferred_and_disagree(panel_graph):
    supply = _section(device_card(panel_graph, "XR01:PS1"), "health")
    assert supply.items == (Light("XR01:PS1:On-Sts", inferred=True),)
    pump = _section(device_card(panel_graph, "XR01:IP1"), "health")
    # one stated row and one inferred row: not all inferred; they disagree on 0
    assert pump.items == (Light("XR01:IP1:State-Sts", disagree=True),)


def test_a_light_whose_rules_the_engine_cannot_evaluate_says_so(panel_graph):
    (bend,) = _section(device_card(panel_graph, "XR02:BEND1"), "health").items
    assert bend.unsupported is True
    (mixed,) = _section(device_card(panel_graph, "XR02:QF1"), "health").items
    assert (mixed.channel, mixed.unsupported) == ("XR02:QF1:Current-RB", True)
    (plain,) = _section(device_card(panel_graph, "XR02:IP1"), "health").items
    assert plain.unsupported is False


def test_a_channel_less_thing_has_place_pose_and_no_signal_or_health(panel_graph):
    view = device_card(panel_graph, "XR02:SHIELD1")
    assert _titles(view) == ["where"]
    where = _section(view, "where").items
    assert where[0] == Link("place", Target("card", "place", "XR/XR02"))
    (pose,) = [item for item in where if isinstance(item, Fact)]
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
        "signals",
        "fed by",
        "where",
        "picture",
        "sources",
    ]
    assert _titles(device_card(panel_graph, "XR01:IP1")) == [
        "signals",
        "health",
        "read out by",
        "where",
    ]


def test_sources_hold_a_fact_per_citation(panel_graph):
    (fact,) = _section(device_card(panel_graph, "XR01:QF1"), "sources").items
    assert (fact.label, fact.citation.document, fact.citation.version) == (
        "XR-LAYOUT",
        "XR-LAYOUT",
        "A",
    )
    assert "sheet 1" in fact.text


def test_the_place_relation_is_shown_under_where_with_its_unit(panel_graph):
    view = device_card(panel_graph, "XR01:CH1")
    assert "located_in" not in _titles(view)
    where = _section(view, "where").items
    assert Link("place", Target("card", "place", "XR/XR01/RK1")) in where
    (unit,) = [item for item in where if isinstance(item, Fact) and item.label == "unit"]
    assert (unit.text, unit.citation.document) == ("10", "XR-RACKS")


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
    (link,) = _section(device_card(graph, "XR01:IP1"), "read out by").items
    assert (link.target, link.standing, link.basis) == (card("XR01:CH2"), "contested", "inferred")
    (back,) = _section(device_card(graph, "XR01:CH2"), "reads out").items
    assert (back.standing, back.basis) == ("contested", "inferred")


def test_an_unknown_relation_kind_is_titled_by_its_own_name(tmp_path):
    graph = _graph(tmp_path, _add_claims)
    assert _section(device_card(graph, "XR01:QF1"), "aligned_with").items[0].target == card(
        "XR01:QD1"
    )
    assert "aligned_with" in _titles(device_card(graph, "XR01:QD1"))


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
    assert [crumb.target.id for crumb in view.crumbs] == ["XR", "XR/XR01", "XR/XR01/RK1"]


def test_a_place_card_lists_its_screens_and_what_is_within(panel_graph):
    view = device_card(panel_graph, "XR/XR01/RK1", rack_classes=RACKS)
    screens = [item.target for item in _section(view, "screens").items]
    assert Target("place", "place", "XR/XR01/RK1") in screens
    assert Target("rack", "place", "XR/XR01/RK1") in screens
    assert [item.target for item in _section(view, "within").items] == [
        card("XR01:CH1"),
        card("XR01:CH2"),
        card("XR01:CH3"),
    ]
    assert [row.channel for row in _section(view, "signals").items] == ["XR01:RackTemp-RB"]
    no_rack = device_card(panel_graph, "XR/XR01/RK1")
    assert Target("rack", "place", "XR/XR01/RK1") not in [
        item.target for item in _section(no_rack, "screens").items
    ]


def test_a_thing_with_no_screen_is_a_fact_not_a_link(panel_graph):
    within = _section(device_card(panel_graph, "XR/XR02", kind="place"), "within").items
    shield = [item for item in within if getattr(item, "text", None) == "XR02:SHIELD1"]
    assert len(shield) == 1 and isinstance(shield[0], Fact)


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
    found = set(targets(panel_graph, rack_classes=RACKS))
    assert card("XR01:QF1") in found
    assert card("XR01:CH1") in found  # a channel and the located_in to its own place
    assert card("XR02:SHIELD1") not in found  # no channel, no relation but its place
    for place in ("XR", "XR/XR01", "XR/XR02", "XR/XR01/RK1"):
        assert Target("card", "place", place) in found
        assert Target("place", "place", place) in found
    assert Target("rack", "place", "XR/XR01/RK1") in found
    assert Target("rack", "place", "XR/XR01") not in found
    assert Target("rack", "place", "XR/XR01/RK1") not in set(targets(panel_graph))
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
    assert len(found) == len(targets(panel_graph, rack_classes=RACKS))


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
    found = set(targets(panel_graph, rack_classes=RACKS))
    views = _every_view(panel_graph, RACKS, **params)
    assert [view.id for view in views] == targets(panel_graph, rack_classes=RACKS)
    links = [link for view in views for link in _links(view)]
    assert links
    assert {link.target for link in links} <= found


def test_every_link_target_is_a_screen_with_contested_and_unknown_relations(tmp_path):
    graph = _graph(tmp_path, _add_claims)
    found = set(targets(graph))
    assert {link.target for view in _every_view(graph) for link in _links(view)} <= found


def test_views_carry_channel_names_never_values(panel_graph):
    for view in _every_view(panel_graph, RACKS):
        for item in _walk(view):
            assert not {"value", "values"} & {field.name for field in dataclasses.fields(item)}


# -- set overview ---------------------------------------------------------------


def test_a_set_overview_is_a_grid_of_members_by_common_suffix(panel_graph):
    view = set_overview(panel_graph, "XR-QUAD")
    assert view.id == Target("set", "group", "XR-QUAD")
    (grid,) = _section(view, "members").items
    assert isinstance(grid, Grid)
    # XR02:QD1 has no pose, so the rows go by id
    assert [row.target for row in grid.rows] == [
        card("XR01:QD1"),
        card("XR01:QF1"),
        card("XR02:QD1"),
        card("XR02:QF1"),
    ]
    assert grid.columns == ("Current-RB",)
    assert [row[0].channel for row in grid.cells] == [
        "XR01:QD1:Current-RB",
        "XR01:QF1:Current-RB",
        "XR02:QD1:Current-RB",
        "XR02:QF1:Current-RB",
    ]
    # a rule-bearing readback is a light, the others value rows
    assert isinstance(grid.cells[2][0], Light) and isinstance(grid.cells[0][0], ValueRow)


def test_a_set_whose_members_all_have_ring_positions_goes_by_s(tmp_path):
    def magnets(document: dict[str, Any]) -> None:
        document["groups"].append({"id": "XR01-MAG", "members": ["XR01:QD1", "XR01:QF1"]})

    (grid,) = _section(set_overview(_graph(tmp_path, magnets), "XR01-MAG"), "members").items
    assert [row.target.id for row in grid.rows] == ["XR01:QF1", "XR01:QD1"]  # s 1.25, 2.0
    # the readback beside a paired setpoint is not a column of its own only
    # when every member pairs it
    assert grid.columns == ("Current-RB", "Current-SP")  # the first row's channel order
    assert grid.cells[0][1] == ValueRow(
        label="Current-SP",
        channel="XR01:QF1:Current-SP",
        readback="XR01:QF1:Current-RB",
        tolerance=("absolute", 0.05),
        writable=True,
        drawn_as="setpoint",
    )


# -- place view -----------------------------------------------------------------


def test_place_view_strips_ring_positions_and_trays_the_rest(panel_graph):
    view = place_view(panel_graph, "XR/XR02")
    (strip,) = _section(view, "strip").items
    assert isinstance(strip, Strip)
    assert [(element.thing, element.s_m) for element in strip.elements] == [
        ("XR02:QF1", 11.0),
        ("XR02:SHIELD1", 12.5),
    ]
    assert strip.extent_m == (11.0, 12.5)
    tray = {entry.thing: entry.reason for entry in strip.tray}
    assert tray == {
        "XR02:QD1": "no position",
        "XR02:BEND1": "no position",
        "XR02:IP1": "no position",  # a ring-frame row with no s
        "XR02:IP2": "no position",
    }
    assert not _ids(view) & set(EXCLUDED)


def test_place_view_two_at_one_s_are_both_drawn_on_their_lanes(panel_graph):
    view = place_view(panel_graph, "XR/XR01", symbols={"Pump": "pump"}, above=frozenset({"pump"}))
    (strip,) = _section(view, "strip").items
    assert [(e.thing, e.cls, e.s_m, e.symbol, e.lane) for e in strip.elements] == [
        ("XR01:IP1", "Pump", 1.25, "pump", "above"),
        ("XR01:QF1", "Quadrupole", 1.25, "generic", "beam"),
        ("XR01:QD1", "Quadrupole", 2.0, "generic", "beam"),
    ]
    assert strip.elements[1].link == Link("XR01:QF1", card("XR01:QF1"))
    tray = {entry.thing: entry.reason for entry in strip.tray}
    assert tray["XR01:BEND1"] == "fixture: frame treated as the ring frame"
    assert tray["XR01:CH1"] == "no position"  # within the rack, any depth
    assert strip.extent_m == (1.25, 2.0)


def test_place_view_links_child_places_and_the_parent(panel_graph):
    view = place_view(panel_graph, "XR/XR01")
    assert [item.target for item in _section(view, "places within").items] == [
        Target("place", "place", "XR/XR01/RK1")
    ]
    assert [crumb.target for crumb in view.crumbs] == [Target("place", "place", "XR")]


def test_a_place_with_no_ring_position_has_no_extent(panel_graph):
    (strip,) = _section(place_view(panel_graph, "XR/XR01/RK1"), "strip").items
    assert strip.elements == () and strip.extent_m is None
    assert [entry.thing for entry in strip.tray] == ["XR01:CH1", "XR01:CH2", "XR01:CH3"]


# -- rack elevation ---------------------------------------------------------------


def test_rack_elevation_orders_units_by_rack_unit_then_id(panel_graph):
    view = rack_elevation(panel_graph, "XR/XR01/RK1")
    assert view.id == Target("rack", "place", "XR/XR01/RK1")
    (elevation,) = _section(view, "units").items
    assert isinstance(elevation, Elevation)
    assert [(unit.position, unit.link.target) for unit in elevation.units] == [
        (10, card("XR01:CH1")),
        (None, card("XR01:CH2")),
        (None, card("XR01:CH3")),
    ]
    assert all(unit.lights == () for unit in elevation.units)  # a rack with no rule


def test_rack_elevation_lights_are_the_units_roll_up_channels(tmp_path):
    def racked_pump(document: dict[str, Any]) -> None:
        for device in document["devices"]:
            if device["id"] == "XR01:IP1":
                device["place"] = "XR/XR01/RK1"

    (elevation,) = _section(
        rack_elevation(_graph(tmp_path, racked_pump), "XR/XR01/RK1"), "units"
    ).items
    pump = [unit for unit in elevation.units if unit.link.target.id == "XR01:IP1"]
    assert pump[0].lights == (Light("XR01:IP1:State-Sts", disagree=True),)


# -- health roll-up ---------------------------------------------------------------


def test_rollup_by_place_is_a_tree_of_children_with_their_channels(panel_graph):
    view = health_rollup(panel_graph, "XR/XR01", "place")
    assert view.id == Target("rollup", "place", "XR/XR01", by="place")
    (root,) = _section(view, "roll-up").items
    assert isinstance(root, Node)
    assert root.channels == ("XR01:IP1:State-Sts", "XR01:PS1:On-Sts")
    nodes = {node.label: node for node in root.children}
    assert nodes["XR01:IP1"].channels == ("XR01:IP1:State-Sts",)
    assert nodes["XR01:IP1"].link.target == card("XR01:IP1")
    assert nodes["XR01:QF1"].no_rule is True and nodes["XR01:QF1"].channels == ()
    # the rack has no rule: present, marked, and its link goes to its card
    rack = nodes["XR/XR01/RK1"]
    assert rack.no_rule is True
    assert rack.link.target == Target("card", "place", "XR/XR01/RK1")


def test_rollup_by_place_links_down_to_child_rollups(panel_graph):
    (root,) = _section(health_rollup(panel_graph, "XR", "place"), "roll-up").items
    assert [(node.label, node.link.target) for node in root.children] == [
        ("XR/XR01", Target("rollup", "place", "XR/XR01", by="place")),
        ("XR/XR02", Target("rollup", "place", "XR/XR02", by="place")),
    ]


def test_rollup_by_system_groups_devices_by_class_by_default(panel_graph):
    view = health_rollup(panel_graph, "XR", "system")
    assert view.id == Target("rollup", "place", "XR", by="system")
    (root,) = _section(view, "roll-up").items
    systems = {node.label: node for node in root.children}
    assert systems["Pump"].channels == ("XR01:IP1:State-Sts", "XR02:IP1:Pressure-RB")
    assert systems["PowerSupply"].channels == ("XR01:PS1:On-Sts",)
    assert systems["ElectronicsChassis"].no_rule is True
    assert [node.label for node in systems["Pump"].children] == [
        "XR01:IP1",
        "XR01:IP2",
        "XR02:IP1",
        "XR02:IP2",
    ]


def test_rollup_by_system_takes_the_system_key_from_a_function(panel_graph):
    view = health_rollup(
        panel_graph,
        "XR",
        "system",
        system_of=lambda thing: "vacuum" if thing.cls == "Pump" else None,
    )
    (root,) = _section(view, "roll-up").items
    assert [node.label for node in root.children] == ["vacuum", "no system"]


def test_rollup_refuses_an_unknown_grouping(panel_graph):
    with pytest.raises(ValueError, match="by"):
        health_rollup(panel_graph, "XR", "colour")


# -- checkout ---------------------------------------------------------------------


def test_checkout_has_one_row_per_device_channel_and_totals(panel_graph):
    view = checkout(panel_graph, "XR/XR01")
    assert view.id == Target("checkout", "place", "XR/XR01")
    rows = {row.channel: row for row in _section(view, "channels").items}
    assert "XR01:RackTemp-RB" not in rows  # on a place, not a device
    assert rows["XR01:IP1:State-Sts"] == CheckRow(
        channel="XR01:IP1:State-Sts",
        thing="XR01:IP1",
        on_device=True,
        typed=True,
        has_rule=True,
        has_source=True,
    )
    assert rows["XR01:CH1:Temp-RB"] == CheckRow(
        "XR01:CH1:Temp-RB", "XR01:CH1", True, False, False, False
    )
    # sourced by its health rows alone
    assert (rows["XR01:PS1:On-Sts"].typed, rows["XR01:PS1:On-Sts"].has_source) == (False, True)
    (totals,) = _section(view, "totals").items
    assert totals == Totals(
        channels=len(rows),
        on_device=len(rows),
        typed=sum(row.typed for row in rows.values()),
        has_rule=sum(row.has_rule for row in rows.values()),
        has_source=sum(row.has_source for row in rows.values()),
    )
    assert (totals.channels, totals.typed, totals.has_rule) == (13, 4, 2)


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

    def cost(graph: PanelGraph) -> float:
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

    cost(small)  # warm the caches of the code paths
    ratio = cost(large) / cost(small)
    assert ratio < 8, f"4x the size took {ratio:.1f}x the time"
