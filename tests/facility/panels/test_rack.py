"""The rack's looks: a drawn rack, chassis at their true unit, a column of facts."""

from __future__ import annotations

from osprey.facility.panels import layout, symbols
from osprey.facility.panels.describe import rack_elevation
from tests.facility.panels._views import nodes, stub_render


def _doc(panel_graph):
    return stub_render(rack_elevation(panel_graph, "XR/XR01/RK1"), panel_graph)


def _bars(doc: dict) -> dict[str, dict]:
    """Each chassis bar by the device it opens."""
    left, right = layout.MARGIN + layout.RACK_LABEL_WIDTH, layout.MARGIN + layout.RACK_COLUMN
    return {
        n["props"]["tooltip"]: n
        for n in nodes(doc)
        if n["type"] == "rectangle"
        and n["props"].get("actions")
        and n["geometry"]["width"] == layout.RACK_WIDTH
        and left < n["geometry"]["x"] < right
    }


def _texts(doc: dict) -> dict[str, dict]:
    return {n["props"]["text"]: n for n in nodes(doc) if n["type"] == "text-label"}


def test_chassis_stand_at_their_true_unit_spanning_the_rack(panel_graph) -> None:
    assert layout.RACK_U >= 12
    bars = _bars(_doc(panel_graph))
    ch1, ch2 = bars["XR01:CH1"]["geometry"], bars["XR01:CH2"]["geometry"]
    # CH2 at U20 (1 U), CH1 at U10 (2 U, so its top is U11): 9 units between the tops
    assert ch1["y"] - ch2["y"] == (20 - 11) * layout.RACK_U
    assert ch1["height"] == 2 * layout.RACK_U and ch2["height"] == layout.RACK_U
    assert ch1["width"] == ch2["width"] == layout.RACK_WIDTH
    assert ch1["x"] == ch2["x"]


def test_the_rack_is_drawn_round_its_chassis_with_unit_numbers_left(panel_graph) -> None:
    doc = _doc(panel_graph)
    bars = _bars(doc)
    frame = [
        n["geometry"]
        for n in nodes(doc)
        if n["type"] == "rectangle"
        and n["props"].get("lineColor") == symbols.PAGE_TEXT
        and n["geometry"]["width"] > layout.RACK_WIDTH
    ]
    assert len(frame) == 1
    (f,) = frame
    for bar in bars.values():
        g = bar["geometry"]
        assert f["x"] < g["x"] and g["x"] + g["width"] < f["x"] + f["width"]
        assert f["y"] < g["y"] and g["y"] + g["height"] < f["y"] + f["height"]
    u20 = _texts(doc)["U20"]["geometry"]
    assert u20["x"] + u20["width"] <= f["x"]
    assert u20["y"] == bars["XR01:CH2"]["geometry"]["y"]


def test_a_chassis_name_is_inside_its_bar_with_its_light_at_the_right_end(panel_graph) -> None:
    doc = _doc(panel_graph)
    bar = _bars(doc)["XR01:CH2"]["geometry"]
    name = _texts(doc)["XR01:CH2"]
    g = name["geometry"]
    assert name["props"]["fontSize"] == layout.SMALL_SIZE
    assert bar["x"] <= g["x"] and g["x"] + g["width"] <= bar["x"] + bar["width"]
    lights = [
        n["geometry"]
        for n in nodes(doc)
        if n["type"] == "polygon"
        and bar["y"] <= n["geometry"]["y"]
        and n["geometry"]["y"] + n["geometry"]["height"] <= bar["y"] + bar["height"]
        and n["geometry"]["x"] > g["x"] + g["width"]
    ]
    assert len(lights) == 1  # one light, at the right end
    assert lights[0]["x"] + lights[0]["width"] <= bar["x"] + bar["width"]


def test_devices_not_at_a_unit_are_a_framed_list_below_the_rack(panel_graph) -> None:
    doc = _doc(panel_graph)
    texts = _texts(doc)
    heading, ch3 = texts["NOT AT A UNIT"]["geometry"], texts["XR01:CH3"]["geometry"]
    lowest = max(b["geometry"]["y"] + b["geometry"]["height"] for b in _bars(doc).values())
    assert heading["y"] > lowest and ch3["y"] > heading["y"]
    assert ch3["x"] < layout.MARGIN + layout.RACK_COLUMN  # below the rack, not beside


def test_the_right_column_counts_the_classes_and_names_the_sources(panel_graph) -> None:
    doc = _doc(panel_graph)
    texts = _texts(doc)
    right = layout.MARGIN + layout.RACK_COLUMN
    for word in ("BY CLASS", "SOURCES", "4 at a stated unit · 1 not at a unit"):
        assert texts[word]["geometry"]["x"] >= right, word
    assert any(t.startswith("XR-LAYOUT") for t in texts)
    assert any(t.startswith("XR-RACKS") for t in texts)


def test_half_width_units_stand_on_their_stated_side() -> None:
    from osprey.facility.panels.layout import rack_lanes
    from osprey.facility.panels.view import Fact, RackUnit

    def unit(position, side=None):
        return RackUnit(position, Fact("x", "x"), (), None, side=side)

    # U7 holds a right half then a left half; U6 a right half alone; U5 full width
    lanes = rack_lanes([unit(7, "R"), unit(7, "L"), unit(6, "R"), unit(5)])
    assert lanes == [(1, 2), (0, 2), (1, 2), (0, 1)]


def test_a_located_in_side_attribute_makes_a_unit_half_width(tmp_path) -> None:
    from osprey.facility.panels import load_graph
    from osprey.facility.panels.describe import rack_elevation
    from osprey.facility.panels.view import Elevation
    from tests.facility.panels._fixture import PANELS_DIR, write_fixture_facility

    def sided(document) -> None:
        for claim in document["relations"]:
            if claim["type"] == "located_in" and claim["object"].get("unit") is not None:
                claim.setdefault("attributes", {})["side"] = "L"
                break

    graph = load_graph(write_fixture_facility(tmp_path / "f.json", sided), PANELS_DIR)
    racks = {
        c.other
        for t in graph.things()
        if t.kind == "device"
        for c in graph.neighbours(t.id, "device").get("located_in", ())
        if c.unit is not None
    }
    found = [
        u
        for r in racks
        for s in rack_elevation(graph, r).sections
        for i in s.items
        if isinstance(i, Elevation)
        for u in i.units
    ]
    assert [u.side for u in found].count("L") == 1
    assert {u.side for u in found} <= {"L", None}


def test_chassis_whose_units_overlap_stand_side_by_side() -> None:
    from osprey.facility.panels.layout import rack_lanes
    from osprey.facility.panels.view import Fact, RackUnit

    def unit(position, size=None):
        return RackUnit(position, Fact("x", "x"), (), size)

    # U13 to U10 (4 U) against U12 and U11: two lanes; U5 alone: one
    lanes = rack_lanes([unit(10, 4), unit(12), unit(11), unit(5)])
    assert lanes == [(0, 2), (1, 2), (1, 2), (0, 1)]


def test_rack_units_are_numbered_by_where_they_stand(panel_graph) -> None:
    """A rack unit's order tag follows its place in the rack: top down, then left
    to right, whatever order the view lists the units in."""
    import dataclasses
    import re

    view = rack_elevation(panel_graph, "XR/XR01/RK1")
    shuffled = []
    for section in view.sections:
        items = tuple(
            dataclasses.replace(i, units=tuple(reversed(i.units))) if hasattr(i, "units") else i
            for i in section.items
        )
        shuffled.append(dataclasses.replace(section, items=items))
    for v in (view, dataclasses.replace(view, sections=tuple(shuffled))):
        doc = stub_render(v, panel_graph)
        first: dict[int, tuple[float, float]] = {}
        for n in nodes(doc):
            comment = n.get("meta", {}).get("comment", "")
            if "item: rack-unit" not in comment:
                continue
            order = int(re.search(r"order: (\d+)", comment).group(1))
            g = n["geometry"]
            first[order] = min(first.get(order, (g["y"], g["x"])), (g["y"], g["x"]))
        assert len(first) > 1
        places = [first[k] for k in sorted(first)]
        assert places == sorted(places)  # by place: down the rack, then across
