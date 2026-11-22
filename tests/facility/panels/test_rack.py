"""The rack's looks: a drawn rack, chassis at their true unit, a column of facts."""

from __future__ import annotations

import re

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


def test_everything_of_the_rack_stands_beside_it_live_first(panel_graph) -> None:
    """The devices at no unit are listed beside the rack, not under it, above the
    counts and sources, which are reference."""
    doc = _doc(panel_graph)
    texts = _texts(doc)
    heading, ch3 = texts["NOT AT A UNIT"]["geometry"], texts["XR01:CH3"]["geometry"]
    right = layout.MARGIN + layout.RACK_COLUMN
    assert heading["x"] >= right and ch3["x"] >= right  # beside the rack
    assert ch3["y"] > heading["y"]
    assert texts["BY CLASS"]["geometry"]["y"] > ch3["y"]  # state before inventory
    lowest = max(b["geometry"]["y"] + b["geometry"]["height"] for b in _bars(doc).values())
    assert ch3["y"] < lowest  # nothing of the rack hangs under it


def test_every_unit_is_numbered_and_a_free_one_is_a_slot(panel_graph) -> None:
    """The units from the rack's floor to its highest stated one are numbered, those
    a device is stated at in the page's text and the rest muted; a unit no device
    takes is a faint slot; the rack's height, which nothing states, is said once."""
    from osprey.facility.panels.view import Elevation

    view = rack_elevation(panel_graph, "XR/XR01/RK1")
    (elevation,) = [i for s in view.sections for i in s.items if isinstance(i, Elevation)]
    stated = {u.position for u in elevation.units}
    doc = _doc(panel_graph)
    labels = [n for n in nodes(doc) if n["type"] == "text-label"]
    numbers = {n["props"]["text"]: n for n in labels if re.fullmatch(r"U\d+", n["props"]["text"])}
    top = max(int(k[1:]) for k in numbers)
    assert set(numbers) == {f"U{u}" for u in range(1, top + 1)}
    for name, node in numbers.items():
        bright = node["props"]["foregroundColor"] == layout.PAGE_TEXT
        assert bright == (int(name[1:]) in stated), name
    slots = [
        n for n in nodes(doc)
        if n["type"] == "rectangle" and n["props"].get("fill") is False
        and n["geometry"]["height"] == layout.RACK_U - 2
    ]  # fmt: skip
    assert 0 < len(slots) < top
    assert any(t.startswith("the rack's height is not stated") for t in _texts(doc))


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


def test_a_list_longer_than_the_rack_continues_under_it_at_the_full_width(panel_graph) -> None:
    """Three times as many devices at no unit as the column beside the rack holds:
    the column is filled down to the rack's foot, the rest stands under the rack
    across the full width under the heading continued; every device is listed
    once, nothing is folded away, the rack does not grow and the classes and
    sources stay beside it."""
    import dataclasses

    from osprey.facility.panels.view import Elevation, Fact

    view = rack_elevation(panel_graph, "XR/XR01/RK1")
    (elevation,) = [i for s in view.sections for i in s.items if isinstance(i, Elevation)]
    template = elevation.unpositioned[0]

    def render(n: int) -> dict:
        names = [f"XR01:SPARE{k:03d}" for k in range(n)]
        units = tuple(dataclasses.replace(template, link=Fact(name, name)) for name in names)
        sections = tuple(
            dataclasses.replace(
                s,
                items=tuple(
                    dataclasses.replace(i, unpositioned=units) if isinstance(i, Elevation) else i
                    for i in s.items
                ),
            )
            for s in view.sections
        )
        return stub_render(dataclasses.replace(view, sections=sections), panel_graph)

    def frame(doc: dict) -> dict:
        (rack,) = [
            n["geometry"] for n in nodes(doc)
            if n["type"] == "rectangle" and n["props"].get("lineColor") == layout.PAGE_TEXT
            and n["geometry"]["width"] == layout.RACK_WIDTH + 2 * layout.RACK_PAD
        ]  # fmt: skip
        return rack

    def chips(doc: dict) -> list[dict]:
        return [
            n["geometry"] for n in nodes(doc)
            if n["type"] == "text-label" and n["props"]["text"].startswith("XR01:SPARE")
        ]  # fmt: skip

    short = render(1)
    rack = frame(short)
    foot = rack["y"] + rack["height"]
    column = layout.MARGIN + layout.RACK_COLUMN
    # how many the column holds: render one too many for it and count those beside
    beside = len([g for g in chips(render(200)) if g["x"] >= column and g["y"] < foot])
    assert beside > 0
    doc = render(3 * beside)
    placed = chips(doc)
    assert frame(doc) == rack  # the rack never grows
    texts = _texts(doc)
    assert len(texts) >= 3 * beside and len(placed) == 3 * beside  # each once, none folded
    over = [g for g in placed if g["x"] >= column and g["y"] < foot]
    under = [g for g in placed if g["y"] > foot]
    assert len(over) == beside and len(under) == 2 * beside
    assert all(g["y"] + layout.CHIP_HEIGHT <= foot for g in over)  # down to the foot
    assert min(g["x"] for g in under) < column  # under it, from the left margin
    heading = texts["NOT AT A UNIT (CONTINUED)"]["geometry"]
    assert foot < heading["y"] < min(g["y"] for g in under)
    assert texts["BY CLASS"]["geometry"]["y"] < foot  # the reference stays beside it
    rows_over = len({g["y"] for g in over})
    rows_under = len({g["y"] for g in under})
    assert rows_under < 2 * rows_over  # width before height: more to a row under it


# -- a long list: one way of reading it, everywhere


def test_a_long_list_fills_balanced_columns_downward_in_bands_with_a_gap_every_fourth_row():
    from osprey.facility.panels.layout import list_cells, list_height, list_offset

    # ten in three columns: four, three, three, read down then across; the columns
    # of a band differ by one row at most
    assert list_cells(10, 3) == [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2),
                                 (2, 0), (2, 1), (2, 2)]  # fmt: skip
    for count, columns in ((54, 4), (26, 5), (52, 7)):
        rows = [0] * columns
        for column, _row in list_cells(count, columns):
            rows[column] += 1
        assert max(rows) - min(rows) <= 1, (count, columns, rows)
    # one or two rows fill across, not downward
    assert list_cells(5, 3) == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)]
    assert list_cells(2, 3) == [(0, 0), (1, 0)]
    # a band of twenty rows is read whole before the next band starts under it
    cells = list_cells(2 * layout.LIST_BAND + 3, 2)
    assert cells[layout.LIST_BAND - 1] == (0, layout.LIST_BAND - 1)
    assert cells[layout.LIST_BAND] == (1, 0)
    assert cells[2 * layout.LIST_BAND :] == [(0, 20), (0, 21), (1, 20)]
    pitch = 30
    assert list_offset(3, pitch) == 3 * pitch  # no gap inside a run of four
    assert list_offset(4, pitch) == 4 * pitch + layout.LIST_RUN_GAP  # the larger gap
    assert list_offset(20, pitch) == 20 * pitch + 4 * layout.LIST_RUN_GAP + layout.LIST_BAND_GAP
    assert list_height(0, 3, 26, pitch) == 0
    assert list_height(10, 3, 26, pitch) == list_offset(3, pitch) + 26


def _rack(panel_graph, unpositioned: int, crowded: int, prefix: str = "XR01:SPARE"):
    """The test rack with *unpositioned* devices at no unit and *crowded* side by side
    at one unit, drawn: the document, the rack's frame and the column's left edge."""
    import dataclasses

    from osprey.facility.panels.view import Elevation, Fact

    view = rack_elevation(panel_graph, "XR/XR01/RK1")
    (elevation,) = [i for s in view.sections for i in s.items if isinstance(i, Elevation)]
    unit = elevation.units[0]
    loose = tuple(
        dataclasses.replace(unit, position=None, link=Fact(n, n), conflicting=False)
        for n in (f"{prefix}{k:03d}" for k in range(unpositioned))
    )
    crowd = tuple(
        dataclasses.replace(unit, link=Fact(f"XR01:SLOT{k:02d}", f"XR01:SLOT{k:02d}"),
                            conflicting=False)
        for k in range(crowded)
    )  # fmt: skip
    elevation = dataclasses.replace(elevation, units=crowd or elevation.units, unpositioned=loose)
    sections = tuple(
        dataclasses.replace(s, items=tuple(elevation if isinstance(i, Elevation) else i
                                           for i in s.items))
        for s in view.sections
    )  # fmt: skip
    doc = stub_render(dataclasses.replace(view, sections=sections), panel_graph)
    (rack,) = [
        n["geometry"] for n in nodes(doc)
        if n["type"] == "rectangle" and n["props"].get("lineColor") == layout.PAGE_TEXT
        and n["geometry"]["width"] == layout.RACK_WIDTH + 2 * layout.RACK_PAD
    ]  # fmt: skip
    return doc, rack, layout.MARGIN + layout.RACK_COLUMN


def test_nothing_beside_the_rack_runs_past_its_foot_and_a_group_that_does_not_fit_goes_under(
    panel_graph,
) -> None:
    doc, rack, column = _rack(panel_graph, 30, 200)  # more side by side than the column holds
    foot = rack["y"] + rack["height"]
    beside = [n["geometry"] for n in nodes(doc) if n["geometry"]["x"] >= column - layout.GAP
              and n["geometry"]["y"] < foot]  # fmt: skip
    assert all(g["y"] + g["height"] <= foot + 1 for g in beside)  # down to the foot, no further
    heads = {n["props"]["text"]: n["geometry"] for n in nodes(doc) if n["type"] == "text-label"}
    (group,) = [t for t in heads if t.startswith("SIDE BY SIDE AT ONE UNIT")]
    assert re.search(r" · U\d+", group)  # its unit on its heading, the line's place
    assert heads[group]["y"] > foot and heads[group]["x"] < column  # under, at the full width
    lines = [n["geometry"] for n in nodes(doc) if n["type"] == "line"]
    assert all(g["y"] + g["height"] <= foot for g in lines)  # no joining line under the foot
    counts = next(t for t in heads if t.endswith("30 not at a unit"))
    assert heads[counts]["y"] < foot  # the reference stays beside
    bars = [t for t in heads if "devices side by side" in t]
    assert bars and all(t.endswith("listed under the rack") for t in bars)  # says where


def test_numbered_names_in_a_rack_list_stand_in_number_order(panel_graph) -> None:
    """A list of devices at no unit takes the numbers in their names by value:
    SPARE97 before SPARE100."""
    import dataclasses

    from osprey.facility.panels.view import Elevation, Fact

    view = rack_elevation(panel_graph, "XR/XR01/RK1")
    (elevation,) = [i for s in view.sections for i in s.items if isinstance(i, Elevation)]
    unit = elevation.units[0]
    names = [f"XR01:SPARE{k}" for k in (100, 98, 97, 99, 101)]
    loose = tuple(
        dataclasses.replace(unit, position=None, link=Fact(n, n), conflicting=False) for n in names
    )
    sections = tuple(
        dataclasses.replace(s, items=tuple(
            dataclasses.replace(elevation, unpositioned=loose) if isinstance(i, Elevation) else i
            for i in s.items))
        for s in view.sections
    )  # fmt: skip
    doc = stub_render(dataclasses.replace(view, sections=sections), panel_graph)
    shown = [
        (n["geometry"]["x"], n["geometry"]["y"], n["props"]["text"])
        for n in nodes(doc) if n["type"] == "text-label" and "SPARE" in n["props"]["text"]
    ]  # fmt: skip
    across = [t for _x, _y, t in sorted(shown, key=lambda g: (g[1], g[0]))]
    assert across == [f"XR01:SPARE{k}" for k in (97, 98, 99, 100, 101)]


def test_the_continued_list_takes_more_columns_than_beside_and_reads_downward(panel_graph):
    doc, rack, column = _rack(panel_graph, 300, 0)
    foot = rack["y"] + rack["height"]
    chips = [n["geometry"] for n in nodes(doc) if n["type"] == "text-label"
             and n["props"]["text"].startswith("XR01:SPARE")]  # fmt: skip
    over = [g for g in chips if g["y"] < foot]
    under = [g for g in chips if g["y"] > foot]
    assert len({g["x"] for g in under}) > len({g["x"] for g in over})
    names = [n["props"]["text"] for n in nodes(doc) if n["type"] == "text-label"
             and n["props"]["text"].startswith("XR01:SPARE")]  # fmt: skip
    first = min(g["x"] for g in under)
    down = [n for n, g in zip(names, chips, strict=True) if g["y"] > foot and g["x"] == first]
    assert down == sorted(down) and len(down) > 1  # the first column holds a run, in order


def test_a_rack_name_cut_keeps_the_part_that_tells_it_apart(panel_graph) -> None:
    long = "XR01:PLC1:A-VERY-LONG-SHARED-POINT-NAME-THAT-NO-TILE-HOLDS-WHOLE-"
    doc, _rack_frame, _column = _rack(panel_graph, 4, 0, prefix=long * 3)
    hover = [n for n in nodes(doc) if str(n["props"].get("tooltip", "")).startswith(long)]
    assert len(hover) == 4  # the whole name on hover
    shown = [n["props"]["text"] for n in nodes(doc) if n["type"] == "text-label"
             and "\u2026" in str(n["props"].get("text", ""))]  # fmt: skip
    assert len(shown) == 4 and len({text[-3:] for text in shown}) == 4  # each its own number


def test_a_cut_rack_label_under_its_link_shows_its_whole_text_on_the_link(panel_graph) -> None:
    """A rack unit's link box lies over its label; where the label is cut, the
    box's tooltip, the one the pointer reaches, holds the label's whole text,
    the contested standing the label adds to the link's name included."""
    import dataclasses

    from osprey.facility.panels.view import Elevation, Link

    view = rack_elevation(panel_graph, "XR/XR01/RK1")
    long = "XR01:CH1 " + "chassis with a stated name longer than its bar " * 2
    sections = []
    for section in view.sections:
        items = []
        for item in section.items:
            if isinstance(item, Elevation):
                units = tuple(
                    dataclasses.replace(
                        u, link=dataclasses.replace(u.link, label=long), standing="contested"
                    )
                    if isinstance(u.link, Link) and u.link.target.id == "XR01:CH1"
                    else u
                    for u in item.units
                )
                item = dataclasses.replace(item, units=units)
            items.append(item)
        sections.append(dataclasses.replace(section, items=tuple(items)))
    doc = stub_render(dataclasses.replace(view, sections=tuple(sections)), panel_graph)
    cut = [
        n for n in nodes(doc)
        if n["type"] == "text-label" and n["props"]["text"].endswith("…")
        and n["props"]["text"].startswith("XR01:CH1")
    ]  # fmt: skip
    assert cut  # the label is cut in its room
    for label in cut:
        g = label["geometry"]
        middle = (g["x"] + g["width"] / 2, g["y"] + g["height"] / 2)
        over = [
            n for n in nodes(doc)
            if n["type"] == "rectangle" and n["props"].get("actions")
            and n["geometry"]["x"] <= middle[0] <= n["geometry"]["x"] + n["geometry"]["width"]
            and n["geometry"]["y"] <= middle[1] <= n["geometry"]["y"] + n["geometry"]["height"]
        ]  # fmt: skip
        assert over  # the link lies over it
        assert f"{long} · contested" in over[-1]["props"]["tooltip"]
        # its one-line hover label is the link's own label, held short, never the added lines
        hover = over[-1]["props"]["hoverLabel"]
        assert len(hover) <= layout.HOVER_LABEL_MOST and hover.endswith("…")
        assert long.startswith(hover[:-1])


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
