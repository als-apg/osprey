"""Layout: view descriptions drawn as Screen-IR documents.

The schema check of every rendered view needs CANOPY and is in
``test_layout_canopy.py``; everything here runs without it.
"""

from __future__ import annotations

import ast
import itertools
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from osprey.facility.panels import layout, symbols
from osprey.facility.panels.describe import (
    health_rollup,
    light,
    place_view,
    rack_elevation,
)
from osprey.facility.panels.health import channel_state, worst
from osprey.facility.panels.layout import (
    FORMAT_REQUESTS,
    MIN_STRIP_SCALE,
    SYMBOL_UNIT,
    aggregates,
    engine_length,
    ladder,
    needs,
    render,
    text_width,
    unexpressed,
)
from osprey.facility.panels.model import load_graph
from osprey.facility.panels.view import (
    Fact,
    Light,
    Link,
    Node,
    Section,
    StripElement,
    Target,
    ValueRow,
    View,
)
from tests.facility.panels._fixture import (
    copy_tables,
    graph_with_part,
    graph_with_wide,
    write_fixture_facility,
)
from tests.facility.panels._views import (
    ABOVE,
    RULED,
    SYMBOLS,
    THREE,
    Screens,
    bound,
    card,
    fixture_views,
    invented_views,
    links,
    nodes,
    printed,
    resolved,
    rollup_view,
    strip_view,
    stub_render,
    with_symbols,
)

#: Widget types the pinned viewer's catalog (its ir-map.json) maps.
PINNED_WIDGETS = {
    "absolute-canvas",
    "text-label",
    "pv-label",
    "pv-text-input",
    "related-display-button",
    "pv-button",
    "waveform-plot",
    "rectangle",
    "ellipse",
    "line",
    "polygon",
}
VALUES = (-3, 0, 0.5, 1, 2, 2.5, 3, 4.5, 5, 6, 7)
TOKEN = re.compile(r"\{(\d+)\}")


def draw(view: View, graph) -> tuple[dict, Screens]:
    screens = Screens()
    return stub_render(view, graph, screens), screens


def evaluate(rule, values) -> str:
    """The state a ladder gives *values*, read as the viewer reads it: first true arm wins.

    The arms are the layout's own output, numbers and operators over ``{i}``;
    ``eval`` with no builtins reads them as the viewer's Python-grammar engine does.
    """
    for expression, state in rule.arms:
        try:
            if eval(TOKEN.sub(r"v[\1]", expression), {"__builtins__": {}}, {"v": values}):
                return state
        except TypeError:
            return "unknown"
    return "unknown"


def flagged(state: str, item: Light) -> str:
    """The engine's state as a light draws it: a light with unsupported rows is never
    normal; disagreement elsewhere does not colour a value every row calls normal."""
    if state != "normal":
        return state
    return "unknown" if item.unsupported else state


def _by_type(doc: dict, kind: str) -> list[dict]:
    return [n for n in nodes(doc) if n["type"] == kind]


def _texts(doc: dict) -> list[str]:
    return [n["props"]["text"] for n in _by_type(doc, "text-label")]


def _links_drawn(doc: dict) -> list[dict]:
    """The boxes that open a screen."""
    return [n for n in nodes(doc) if n["props"].get("actions")]


def _axes(doc: dict) -> list[dict]:
    """The beam axes: the page's lines at the axis width that span the strip."""
    return [
        n
        for n in _by_type(doc, "line")
        if n["props"]["lineWidth"] == layout.AXIS_WIDTH
        and n["geometry"]["width"] >= layout.STRIP_WIDTH
    ]


def _is_mark(n: dict) -> bool:
    """A locator's device mark: a short upright line that names its device."""
    return (
        n["type"] == "line"
        and "tooltip" in n["props"]
        and n["geometry"]["width"] <= layout.AXIS_WIDTH
        and n["geometry"]["height"] in (layout.LOCATOR_MARK, layout.LOCATOR_BIG)
    )


def _marks(doc: dict) -> list[dict]:
    """The locator's device marks."""
    return [n for n in nodes(doc) if _is_mark(n)]


# --------------------------------------------------------------------------- rules


@pytest.mark.parametrize("channel", RULED)
def test_a_lights_rule_gives_the_engines_state_for_every_value(panel_graph, channel) -> None:
    item = light(panel_graph, channel)
    rule = ladder(panel_graph, [item])
    assert not isinstance(rule, str)
    for value in VALUES:
        expected = flagged(channel_state(panel_graph, channel, value), item)
        assert evaluate(rule, [value]) == expected, (channel, value)


def test_a_light_with_no_value_is_unknown(panel_graph) -> None:
    rule = ladder(panel_graph, [light(panel_graph, "XR01-PS1:On-Sts")])
    assert evaluate(rule, [None]) == "unknown"


def test_a_roll_up_rule_gives_the_worst_channel(panel_graph) -> None:
    items = [light(panel_graph, c) for c in RULED]
    rule = ladder(panel_graph, items)
    for a in VALUES:
        for b in VALUES:
            values = [a, b, a, b, a, b]
            expected = worst(
                flagged(channel_state(panel_graph, item.channel, v), item)
                for item, v in zip(items, values, strict=True)
            )
            assert evaluate(rule, values) == expected, values


def test_a_light_is_disagree_only_where_its_rows_disagree(panel_graph) -> None:
    item = light(panel_graph, "XR01-IP1:State-Sts")  # 0: bad and normal; 1: normal
    assert item.disagree
    rule = ladder(panel_graph, [item])
    assert evaluate(rule, {0: 0}) == "disagree"
    assert evaluate(rule, {0: 1}) == "normal"  # every row calls 1 normal: grey, not lilac


def test_a_light_with_rows_it_cannot_evaluate_is_never_normal(panel_graph) -> None:
    item = light(panel_graph, "XR02-QF1:Current-RB")
    assert item.unsupported
    rule = ladder(panel_graph, [item])
    assert all(state != "normal" for _, state in rule.arms)


def test_a_text_label_row_cannot_be_written_and_the_light_is_unknown(tmp_path) -> None:
    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as rows:
        rows.write("XR01-PS1:On-Sts,ON,normal,screen,stated,xr-screens,4d5e6f7,ps.screen:1\n")
    graph = load_graph(write_fixture_facility(tmp_path / "facility.json"), tables)
    assert ladder(graph, [light(graph, "XR01-PS1:On-Sts")]) == "label condition"
    doc, _ = draw(card("label", light(graph, "XR01-PS1:On-Sts")), graph)
    assert unexpressed(doc) == 1
    assert "label condition" in needs(doc)
    (shape,) = _by_type(doc, "polygon")
    assert "rules" not in shape
    assert shape["props"]["lineStyle"] == "dashed" and shape["props"]["fill"] is False
    assert "1 light(s) drawn unknown" in doc["metadata"]["description"]


def _limit_between_one_channel_and_three(graph) -> int:
    singles = [engine_length(ladder(graph, [light(graph, c)])) for c in RULED]
    three = ladder(graph, [light(graph, c) for c in THREE])
    assert engine_length(three) > max(singles)
    return max(singles)


def test_the_rule_limit_is_the_formula_engines_expression_limit() -> None:
    assert layout.RULE_EXPRESSION_LIMIT == 2048
    assert "max_expression_len" in Path(layout.__file__).read_text()


def test_a_root_over_the_limit_says_its_summary_needs_a_server_side_aggregate(
    panel_graph, monkeypatch
):
    monkeypatch.setattr(
        layout, "RULE_EXPRESSION_LIMIT", _limit_between_one_channel_and_three(panel_graph)
    )
    assert ladder(panel_graph, [light(panel_graph, c) for c in RULED]) == "aggregate state"
    doc, _ = draw(rollup_view(), panel_graph)
    texts = _texts(doc)
    summary = f"{len(RULED)} channels: a live summary of this level needs a server-side aggregate"
    assert texts.count(summary) == 1
    assert "server-side aggregate" in needs(doc)
    assert aggregates(doc) == 1  # the three-channel child's tile: open, a link down
    assert unexpressed(doc) == 0  # an aggregate is not an unknown light
    shapes = _by_type(doc, "polygon")
    assert not [n for n in shapes if n["geometry"]["width"] == layout.MOSAIC_LIGHT]
    live = [n for n in shapes if n.get("rules")]
    armed = [c for c in RULED if ladder(panel_graph, [light(panel_graph, c)]).arms]
    assert len(live) == len(armed)  # one light per child that fits one rule (and has an arm)


def _subscriptions(doc: dict) -> int:
    """Rules that read a channel: a conditional rule's, and a formula rule's."""
    rules = [r for n in nodes(doc) for r in n.get("rules", ())]
    return sum(bool(r.get("conditions")) or "formula" in r for r in rules)


def _bound(doc: dict) -> set[str]:
    formulas = {f["name"]: f for f in doc.get("formulas", ())}
    found = set()
    for n in nodes(doc):
        for r in n.get("rules", ()):
            found |= {pv["name"] for pv in r.get("pvs", ())}
            if "formula" in r:
                found |= set(formulas[r["formula"].removeprefix("formula://")]["bindings"].values())
    return found


def test_a_roll_up_spends_its_budget_on_single_lights_then_opens(tmp_path, monkeypatch):
    from osprey.facility.panels.describe import health_rollup

    graph = graph_with_wide(tmp_path, 40)
    view = health_rollup(graph, "XR/XR01")
    doc, _ = draw(view, graph)
    assert not [n for n in nodes(doc) if str(n["props"].get("tooltip", "")).startswith("part ")]
    assert any(t.startswith("40 channels") for t in _texts(doc))  # its tile: it is open
    opens = [n for n in _links_drawn(doc) if n["props"]["tooltip"] == "XR01:WIDE1"]
    assert len(opens) == 1
    budget = _subscriptions(doc) - 1
    monkeypatch.setattr(layout, "SCREEN_RULE_BUDGET", budget)
    doc, _ = draw(view, graph)
    assert _subscriptions(doc) <= budget
    monkeypatch.setattr(layout, "SCREEN_CHANNEL_BUDGET", 2)
    doc, _ = draw(view, graph)
    assert len(_bound(doc)) <= 2


def test_tile_positions_wrap_at_the_margin_and_give_the_rows_height() -> None:
    places, height = layout.tile_positions(500, 600, 40, layout.ROW)
    step = layout.MOSAIC_LIGHT + layout.MOSAIC_GAP
    assert places[0] == (600, 0)
    assert all(x + layout.MOSAIC_LIGHT <= layout.MARGIN + layout.STRIP_WIDTH for x, _ in places)
    lines = sorted({dy for _, dy in places})
    assert lines[1] == layout.ROW and all(
        b - a == step for a, b in zip(lines[1:], lines[2:], strict=False)
    )
    assert height == lines[-1] + layout.MOSAIC_LIGHT


def test_a_roll_up_screen_draws_one_level(panel_graph) -> None:
    doc, _ = draw(rollup_view(), panel_graph)
    assert "deep" not in _texts(doc)
    assert aggregates(doc) == 0


def test_one_channel_over_the_limit_is_unknown_and_counted(panel_graph, monkeypatch) -> None:
    monkeypatch.setattr(layout, "RULE_EXPRESSION_LIMIT", 10)
    assert ladder(panel_graph, [light(panel_graph, "XR02-IP1:Pressure-RB")]) == "long rule"
    doc, _ = draw(card("long", light(panel_graph, "XR02-IP1:Pressure-RB")), panel_graph)
    assert unexpressed(doc) == 1 and "long rule" in needs(doc)


def test_every_fixture_rule_fits_the_limit(panel_graph) -> None:
    for view in fixture_views(panel_graph):
        assert unexpressed(draw(view, panel_graph)[0]) == 0, view.id


def test_a_lights_rules_carry_the_looks_of_their_states(panel_graph) -> None:
    item = light(panel_graph, "XR02-IP1:Pressure-RB")
    doc, _ = draw(card("one", item), panel_graph)
    (shape,) = _by_type(doc, "polygon")
    rule = ladder(panel_graph, [item])
    assert [r["targetProperty"] for r in shape["rules"]] == list(symbols.LIGHT_PROPS)
    assert "formulas" not in doc  # the arms are written into each rule
    for r in shape["rules"]:
        assert r["pvs"] == [{"name": "XR02-IP1:Pressure-RB", "trigger": True}]
        assert r["default"] == symbols.look("unknown")[r["targetProperty"]]
        assert len(r["conditions"]) <= len(rule.arms)
    for value in VALUES:
        assert drawn(doc, shape, {item.channel: value}) == shown(evaluate(rule, [value])), value
    (mark,) = [n for n in _by_type(doc, "text-label") if n.get("rules")]
    assert mark["rules"][0]["targetProperty"] == "text"


# --------------------------------------------------------------------------- items


def test_a_value_row_puts_the_readback_beside_the_setpoint(panel_graph) -> None:
    doc, _ = draw(invented_views()[3], panel_graph)
    inputs = {n["props"]["pv"]: n for n in _by_type(doc, "pv-text-input")}
    labels = {n["props"]["pv"]: n for n in _by_type(doc, "pv-label")}
    assert set(inputs) == {"A:SP", "B:SP"}  # writable rows only
    assert set(labels) == {"A:RB", "B:RB", "C:RB"}
    for sp, rb in (("A:SP", "A:RB"), ("B:SP", "B:RB")):
        assert labels[rb]["geometry"]["y"] == inputs[sp]["geometry"]["y"]
        assert labels[rb]["geometry"]["x"] > inputs[sp]["geometry"]["x"]


def test_a_deviation_light_only_where_the_row_has_a_tolerance(panel_graph) -> None:
    doc, _ = draw(invented_views()[3], panel_graph)
    read = [bound(doc, n) for n in _by_type(doc, "polygon") if n.get("rules")]
    assert ("A:SP", "A:RB") in read and ("B:SP", "B:RB") in read
    assert not any("C:RB" in names for names in read)
    absolute = layout._tolerance_ladder(invented_views()[3].sections[0].items[0])
    assert evaluate(absolute, [10.0, 10.04]) == "normal"
    assert evaluate(absolute, [10.0, 10.06]) == "warn"
    relative = layout._tolerance_ladder(invented_views()[3].sections[0].items[1])
    assert evaluate(relative, [100.0, 100.9]) == "normal"
    assert evaluate(relative, [100.0, 101.1]) == "warn"


def test_an_inferred_item_carries_the_mark_and_a_stated_one_does_not(panel_graph) -> None:
    marked, _ = draw(card("x", Light("XR01-PS1:On-Sts", inferred=True)), panel_graph)
    plain, _ = draw(card("x", Light("XR01-PS1:On-Sts")), panel_graph)
    assert _texts(marked).count(symbols.INFERRED_MARK) == 1
    assert symbols.INFERRED_MARK not in _texts(plain)
    doc, _ = draw(invented_views()[3], panel_graph)
    # the inferred value row, light, fact and link
    assert _texts(doc).count(symbols.INFERRED_MARK) == 4


def test_a_strip_centres_its_magnets_on_the_axis_and_lifts_the_upper_lane(panel_graph) -> None:
    doc, _ = draw(invented_views()[2], panel_graph)
    (axis,) = _axes(doc)
    y = axis["geometry"]["y"] + axis["geometry"]["height"] / 2
    shape = {
        thing: _symbol(doc, thing)[0][0]["geometry"] for thing in ("QF", "QD", "QX", "IP", "ODD")
    }
    for quad in ("QF", "QD", "QX"):
        assert shape[quad]["y"] + shape[quad]["height"] / 2 == pytest.approx(y, abs=0.1)
    assert shape["IP"]["y"] + shape["IP"]["height"] < y  # the lane above
    assert shape["ODD"]["width"] == pytest.approx(symbols.WIDTH_UNITS["generic"] * SYMBOL_UNIT)


def test_a_locator_puts_ticks_and_devices_in_proportion_to_s(panel_graph) -> None:
    doc, _ = draw(invented_views()[1], panel_graph)
    ticks = [
        (float(n["props"]["text"]), n["geometry"]["x"] + n["geometry"]["width"] / 2)
        for n in _by_type(doc, "text-label")
        if re.fullmatch(r"\d+(\.\d+)?", n["props"]["text"])
    ]
    (caption,) = [t for t in _texts(doc) if t.startswith("to scale · position along the beam")]
    assert caption.endswith("this row 0 to 30 of 0 to 30")  # one row: the whole place
    assert [m for m, _ in ticks] == list(range(0, 31, 2))
    (m0, x0), (m1, x1) = ticks[0], ticks[-1]
    per_metre = (x1 - x0) / (m1 - m0)
    for m, x in ticks:
        assert x == pytest.approx(x0 + (m - m0) * per_metre, abs=0.2)
    marks = sorted(n["geometry"]["x"] + n["geometry"]["width"] / 2 for n in _marks(doc))
    for s, x in zip((0, 7, 12.5, 30), marks, strict=True):
        assert x == pytest.approx(x0 + s * per_metre, abs=0.2)


def test_a_dense_strip_breaks_into_rows_instead_of_shrinking(panel_graph) -> None:
    doc, _ = draw(invented_views()[0], panel_graph)
    axes = _axes(doc)
    assert len(axes) > 1
    for n in nodes(doc):
        thing = n["props"].get("tooltip", "")
        if thing.startswith("D") and n["type"] == "polygon":
            kind = ("quadrupole_f", "quadrupole_d", "bend")[int(thing[1:]) % 3]
            natural = symbols.WIDTH_UNITS[kind] * SYMBOL_UNIT
            assert n["geometry"]["width"] >= natural * MIN_STRIP_SCALE - 0.1


def test_each_row_of_a_strip_has_a_locator_of_its_own_stretch(panel_graph) -> None:
    doc, _ = draw(invented_views()[0], panel_graph)
    stretches = [
        tuple(float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", t.rsplit("this row", 1)[1]))
        for t in _texts(doc)
        if t.startswith("to scale · position along the beam")
    ]
    assert len(stretches) == len(_axes(doc)) > 1
    for lo, hi, whole_lo, whole_hi in stretches:
        assert whole_lo <= lo < hi <= whole_hi and hi - lo < whole_hi - whole_lo
    # one row's stretch follows the last
    assert [s[0] for s in stretches] == sorted(s[0] for s in stretches)


def test_the_tray_is_a_dashed_box_of_its_entries_with_their_reason(panel_graph) -> None:
    view = place_view(with_symbols(panel_graph, SYMBOLS, ABOVE), "XR/XR02")
    (section,) = [s for s in view.sections if s.title == "strip"]
    (item,) = section.items
    assert item.tray
    doc, _ = draw(view, panel_graph)
    dashed = [n for n in _by_type(doc, "polygon") if n["props"].get("lineStyle") == "dashed"]
    assert dashed
    texts = _texts(doc)
    assert "IN OUR LISTS, NOT PLACED" in texts
    for entry in item.tray:
        assert entry.thing in texts  # the heading says it is not placed


def test_a_conflicting_rack_unit_is_marked(panel_graph) -> None:
    view = rack_elevation(panel_graph, "XR/XR01/RK1")
    (elevation,) = view.sections[0].items
    conflicting = [u for u in elevation.units if u.conflicting]
    assert conflicting
    doc, _ = draw(view, panel_graph)
    assert _texts(doc).count(symbols.CONFLICT_MARK) == len(conflicting)
    assert "NOT AT A UNIT" in _texts(doc)


def test_a_conflict_mark_leaves_room_before_the_name(panel_graph) -> None:
    view = rack_elevation(panel_graph, "XR/XR01/RK1")
    doc, _ = draw(view, panel_graph)
    texts = _by_type(doc, "text-label")
    for i, n in enumerate(texts):
        if n["props"]["text"] == symbols.CONFLICT_MARK:
            name = texts[i + 1]["geometry"]
            mark = n["geometry"]
            assert name["x"] >= mark["x"] + mark["width"] + layout.GAP - 0.01


def test_units_too_many_side_by_side_are_one_bar_and_listed_under_the_rack(panel_graph) -> None:
    view = rack_elevation(panel_graph, "XR/XR01/RK1")
    (elevation,) = view.sections[0].items
    unit = elevation.units[0]
    crowd = tuple(
        replace(unit, link=Fact("chassis", f"XR01:SLOT{i}"), conflicting=False) for i in range(8)
    )
    elevation = replace(elevation, units=crowd)
    view = replace(view, sections=(replace(view.sections[0], items=(elevation,)),))
    doc, _ = draw(view, panel_graph)
    texts = _texts(doc)
    assert "8 devices side by side · listed under the rack" in texts
    assert "SIDE BY SIDE AT ONE UNIT" in texts
    assert [t for t in texts if t.startswith("XR01:SLOT")] == [
        f"XR01:SLOT{i} · U{unit.position}" for i in range(8)
    ]


def test_a_unit_numbered_zero_stands_inside_the_rack_frame(panel_graph) -> None:
    view = rack_elevation(panel_graph, "XR/XR01/RK1")
    (elevation,) = view.sections[0].items
    floor = replace(
        elevation.units[0], position=0, size_u=1, link=Fact("x", "XR01:FLOOR"),
        standing="accepted", conflicting=False,
    )  # fmt: skip
    elevation = replace(elevation, units=(*elevation.units, floor))
    view = replace(view, sections=(replace(view.sections[0], items=(elevation,)),))
    doc, _ = draw(view, panel_graph)
    frame = max(
        (n for n in _by_type(doc, "rectangle") if n["props"].get("lineWidth") == layout.AXIS_WIDTH),
        key=lambda n: n["geometry"]["height"],
    )["geometry"]
    name = next(n for n in _by_type(doc, "text-label") if n["props"]["text"] == "XR01:FLOOR")
    g = name["geometry"]
    assert frame["y"] <= g["y"] and g["y"] + g["height"] <= frame["y"] + frame["height"]


def test_a_rack_unit_is_as_tall_as_its_size(panel_graph) -> None:
    view = rack_elevation(panel_graph, "XR/XR01/RK1")
    doc, _ = draw(view, panel_graph)
    heights = sorted({n["geometry"]["height"] for n in _by_type(doc, "rectangle")})
    assert layout.RACK_U in heights and 2 * layout.RACK_U in heights


def test_a_node_without_a_rule_reads_no_rule_not_unknown(panel_graph) -> None:
    view = health_rollup(panel_graph, "XR/XR01")
    (root,) = view.sections[-1].items  # after the key numbers

    def all_nodes(node):
        yield node
        for child in node.children:
            yield from all_nodes(child)

    found = list(all_nodes(root))
    without = [n for n in found if n.no_rule]
    assert without and len(without) < len(found)
    doc, _ = draw(view, panel_graph)
    polygons = [  # the tree's: left of the rows by system
        n
        for n in _by_type(doc, "polygon")
        if n["geometry"]["x"] < layout.MARGIN + layout.ROLLUP_LEFT
    ]
    assert len(polygons) == len(found)  # one light per node, those without a rule in "none"
    bare = [n for n in polygons if n["props"]["lineColor"] == symbols.look("none")["lineColor"]]
    assert len(bare) == len(without)
    assert all(not n.get("rules") and n["props"]["tooltip"] == "no rule" for n in bare)


def test_a_picture_part_is_a_marker_unless_its_row_names_a_channel(tmp_path) -> None:
    def describe_switch(document) -> None:
        for channel in document["channels"]:
            if channel["id"] == "XR01-QF1:Current-RB":
                channel["description"] = "Thermal switch 1"  # no matching by text

    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as rows:
        rows.write("XR01-QF1:Current-RB,1,bad,screen,stated,xr-screens,4d5e6f7,m.screen:1\n")
    graph = load_graph(write_fixture_facility(tmp_path / "f.json", describe_switch), tables)
    from osprey.facility.panels.describe import device_card

    view = device_card(graph, "XR01:QF1")
    (section,) = [s for s in view.sections if s.title == "picture"]
    (picture,) = section.items
    assert all(part.channel is None for part in picture.parts)
    corners = tuple(
        replace(picture.parts[0], part=f"o{i}", kind="outline", x_mm=x, y_mm=y)
        for i, (x, y) in enumerate(((-9.0, -9.0), (9.0, -9.0), (0.0, 9.0)))
    )
    outlined = replace(picture, parts=picture.parts + corners)
    doc, _ = draw(View(view.id, view.title, (), (Section("picture", (outlined,)),)), graph)
    assert not [n for n in _by_type(doc, "polygon") if n.get("rules")]
    plain = [n for n in nodes(doc) if n["type"] in ("ellipse", "rectangle")]
    tips = [n["props"].get("tooltip") for n in plain]
    assert tips.count("pole 1") == tips.count("thermal switch 1") == 1  # off the body: blocks

    named = replace(
        outlined,
        parts=(replace(picture.parts[1], channel="XR01-QF1:Current-RB"), *corners),
    )
    doc, _ = draw(View(view.id, view.title, (), (Section("picture", (named,)),)), graph)
    (shape,) = [n for n in _by_type(doc, "polygon") if n.get("rules")]
    assert bound(doc, shape) == ("XR01-QF1:Current-RB",)  # its state, where it sits
    assert not [
        n
        for n in nodes(doc)
        if n["props"].get("tooltip") == "thermal switch 1" and n["type"] in ("ellipse", "rectangle")
    ]


def _symbol(doc: dict, thing: str) -> tuple[list[dict], list[dict]]:
    """A strip symbol's shape nodes (they carry the device as tooltip) and its mark, if any."""
    found = [
        n
        for n in nodes(doc)
        if n["props"].get("tooltip") == thing and not n["props"].get("actions") and not _is_mark(n)
    ]
    shapes = [n for n in found if n["type"] != "text-label"]
    marks = [n for n in found if n["type"] == "text-label"]
    return shapes, marks


def _live(doc: dict, node: dict, values: dict[str, object]) -> dict:
    """The props a node's own rules give it for *values*, for the props they target."""
    values = resolved(doc, values)
    out = {}
    for rule in node.get("rules", ()):
        v = [values[p["name"]] for p in rule["pvs"]]
        out[rule["targetProperty"]] = rule["default"]
        for condition in rule["conditions"]:
            if eval(TOKEN.sub(r"v[\1]", condition["expression"]), {"__builtins__": {}}, {"v": v}):
                out[rule["targetProperty"]] = condition["value"]
                break
    return out


def _ring(doc: dict, thing: str) -> tuple[dict | None, dict | None]:
    """A strip symbol's state ring and the word above it (the node after the ring)."""
    found = nodes(doc)
    for i, n in enumerate(found):
        if (
            n["type"] == "polygon"
            and n["props"].get("tooltip") == thing
            and n["props"]["lineWidth"] == 2.5
        ):
            return n, found[i + 1]
    return None, None


def _shapes(doc: dict, thing: str) -> list[dict]:
    shapes, _marks_ = _symbol(doc, thing)
    return [n for n in shapes if n["props"].get("lineWidth") != 2.5]


def test_a_strip_symbols_state_is_a_live_ring_and_word_never_its_fill(tmp_path) -> None:
    channel = "XR01-QF1:Current-RB"
    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as rows:
        for value, state in ((">=7", "bad"), ("<1", "warn"), ("2", "normal")):
            rows.write(
                f"{channel},{value},{state},database-severity,stated,xr-ioc,0a1b2c3,m.db:1\n"
            )
    graph = load_graph(write_fixture_facility(tmp_path / "facility.json"), tables)
    view = place_view(with_symbols(graph, SYMBOLS, ABOVE), "XR/XR01")
    doc = stub_render(view, graph, colours={"quadrupole": "#2f9e52"})
    ring, word = _ring(doc, "XR01:QF1")
    for node in _shapes(doc, "XR01:QF1"):
        assert not node.get("rules")  # the fill is static: the type, not the state
        assert node["props"]["fillColor"] == "#2f9e52"
        assert node["props"]["lineStyle"] == "solid"
    assert bound(doc, ring) == (channel,) and bound(doc, word) == (channel,)
    for value, state in ((9, "bad"), (0.5, "warn"), (2, "normal")):
        look = symbols.ring_look(state)
        assert _live(doc, ring, {channel: value}) == {p: look[p] for p in symbols.RING_PROPS}
        assert _live(doc, word, {channel: value}) == {"text": look["text"]}
    unknown = symbols.ring_look("unknown")  # drawn unknown until the channel answers
    assert ring["props"]["lineStyle"] == unknown["lineStyle"] == "dashed"
    assert word["props"]["text"] == unknown["text"]


@pytest.mark.parametrize(("lane", "symbol"), [("beam", "corrector"), ("above", "ion_pump")])
def test_ring_words_never_stand_over_one_another(panel_graph, lane, symbol) -> None:
    """Ringed and unringed symbols closer than a word's width, alternating, on the
    axis and as badges over it: each ring's word, at its widest, touches no other
    word and no other ring. An unringed symbol on the axis takes no lane, and
    where two lanes would not do a further one is opened."""
    row = [
        StripElement(f"E{i:02d}", None, float(i), symbol, lane, None, RULED if i % 2 else ())
        for i in range(36)
    ] + [  # and a run of rings too close for two lanes
        StripElement(f"R{i}", None, 36 + 0.2 * i, symbol, lane, None, RULED) for i in range(4)
    ]
    doc, _ = draw(strip_view(row), panel_graph)
    widest = max(
        layout.text_width(str(symbols.ring_look(state)["text"]), layout.SMALL_SIZE)
        for state in symbols.STATES
    )
    assert widest + 2 * layout.LABEL_INSET <= layout.TAG_WIDTH  # the room holds every word

    def box(node: dict) -> tuple[float, float, float, float]:
        g = node["geometry"]
        return g["x"], g["y"], g["width"], g["height"]

    def touch(a: tuple, b: tuple) -> bool:
        return (
            a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]
        )

    words, rings = {}, {}
    for element in row:
        ring, word = _ring(doc, element.thing)
        assert (ring is None) == (not element.channels)
        if word is not None:
            words[element.thing], rings[element.thing] = box(word), box(ring)
    over = [(a, b) for a, b in itertools.combinations(words, 2) if touch(words[a], words[b])]
    over += [(a, b) for a in words for b in rings if a != b and touch(words[a], rings[b])]
    assert over == []
    assert len({w[1] for w in words.values()}) >= 3  # the close run needed a further lane


def test_a_stem_through_another_badge_s_word_is_named(panel_graph) -> None:
    """Rings too close for one stem length stand on longer stems, whose lines run
    down past the words below them; each such crossing is named, for ``check``."""
    close = [
        StripElement(f"R{i:02d}", None, 0.5 * i, "ion_pump", "above", None, RULED)
        for i in range(60)
    ]
    crossed: list[str] = []
    view = strip_view(close)
    layout.render(
        view, graph=panel_graph, screens=Screens(), doc_id=Screens.doc_id(view.id), crossed=crossed
    )
    assert crossed and all(" stem runs through " in c for c in crossed)
    apart = [
        StripElement(f"R{i}", None, 5.0 * i, "ion_pump", "above", None, RULED) for i in range(4)
    ]
    crossed = []
    view = strip_view(apart)
    layout.render(
        view, graph=panel_graph, screens=Screens(), doc_id=Screens.doc_id(view.id), crossed=crossed
    )
    assert crossed == []


def test_word_lanes_take_the_lowest_free_lane_and_open_one_when_none_is():
    assert layout.word_lanes([0, 50, 100, 150], 70) == [0, 1, 0, 1]
    assert layout.word_lanes([0, 10, 20, 100], 70) == [0, 1, 2, 0]
    assert layout.word_lanes([], 70) == []


def test_a_strip_symbol_with_no_rule_has_a_dashed_outline_and_no_ring(panel_graph) -> None:
    shown = {**SYMBOLS, "Infrastructure": "generic"}  # mapped on purpose: drawn, no rule
    view = place_view(with_symbols(panel_graph, shown, ABOVE), "XR/XR02")
    doc, _ = draw(view, panel_graph)
    shapes = _shapes(doc, "XR02:SHIELD1")
    assert shapes and _ring(doc, "XR02:SHIELD1") == (None, None)
    for node in shapes:
        assert not node.get("rules")
        assert node["props"]["lineStyle"] == "dashed"
        assert node["props"]["lineColor"] == symbols.PAGE_TEXT
        assert node["props"]["fillColor"] == symbols.PANEL_BACKGROUND  # no colour given


def test_a_strip_symbol_over_the_rule_limit_takes_the_aggregate_ring(
    panel_graph, monkeypatch
) -> None:
    big = StripElement("BIG", None, 1.0, "bend", "beam", None, RULED)
    monkeypatch.setattr(layout, "RULE_EXPRESSION_LIMIT", 0)
    doc, _ = draw(strip_view([big]), panel_graph)
    ring, word = _ring(doc, "BIG")
    assert not ring.get("rules") and not word.get("rules")
    look = symbols.ring_look("aggregate")
    assert {p: ring["props"][p] for p in symbols.RING_PROPS} == {
        p: look[p] for p in symbols.RING_PROPS
    }
    assert word["props"]["text"] == symbols.AGGREGATE_MARK


def test_every_fixture_view_renders_with_unique_ids_and_pinned_widgets(panel_graph) -> None:
    views = fixture_views(panel_graph)
    assert {v.id.view for v in views} == {
        "home", "card", "set", "calibration", "place", "rack", "rollup", "checkout"
    }  # fmt: skip
    for view in [*views, *invented_views()]:
        doc, _ = draw(view, panel_graph)
        ids = [n["id"] for n in [doc["root"], *nodes(doc)]]
        assert len(ids) == len(set(ids)), view.id
        assert {n["type"] for n in nodes(doc)} <= PINNED_WIDGETS - {"absolute-canvas"}
        assert doc["metadata"]["size"]["height"] >= max(
            n["geometry"]["y"] + n["geometry"]["height"] for n in nodes(doc)
        )


def test_rendering_is_deterministic(panel_graph) -> None:
    for view in [*fixture_views(panel_graph), *invented_views()]:
        first, _ = draw(view, panel_graph)
        second, _ = draw(view, panel_graph)
        assert json.dumps(first, sort_keys=False) == json.dumps(second, sort_keys=False)


def test_every_navigation_target_comes_from_screens(panel_graph) -> None:
    for view in [*fixture_views(panel_graph), *invented_views()]:
        doc, screens = draw(view, panel_graph)
        given = {Screens()(t) for t in screens.asked}  # a fresh stub names them alike
        assert not _by_type(doc, "related-display-button")  # every link is an action box
        files = [a["file"] for n in nodes(doc) for a in n["props"].get("actions", ())]
        assert set(files) <= given, view.id
        assert set(screens.asked) == {link.target for link in links(view)}, view.id


def test_the_format_requests_are_exactly_the_needs_the_views_hit(panel_graph, monkeypatch) -> None:
    hit: set[str] = set()
    for view in [*fixture_views(panel_graph), *invented_views(), rollup_view()]:
        hit |= set(needs(draw(view, panel_graph)[0]))
    # over the limit: an aggregate roll-up, and one channel's long rule
    monkeypatch.setattr(
        layout, "RULE_EXPRESSION_LIMIT", _limit_between_one_channel_and_three(panel_graph)
    )
    hit |= set(needs(draw(rollup_view(), panel_graph)[0]))
    monkeypatch.setattr(layout, "RULE_EXPRESSION_LIMIT", 10)
    hit |= set(needs(draw(card("long", Light("XR02-IP1:Pressure-RB")), panel_graph)[0]))
    assert FORMAT_REQUESTS
    # the label condition is hit in test_a_text_label_row_cannot_be_written...
    assert {name for name, _ in FORMAT_REQUESTS} - {"label condition"} == hit
    assert all(why for _, why in FORMAT_REQUESTS)


def test_the_document_is_a_screen_of_the_targets_id(panel_graph) -> None:
    doc, _ = draw(card("XR01:QF1"), panel_graph)
    assert doc["schemaVersion"] == "1.1" and doc["kind"] == "screen"
    assert doc["id"] == Screens.doc_id(Target("card", "device", "XR01:QF1"))  # the caller's
    assert not hasattr(layout, "screen_id")  # one screen id function: emit's
    named = render(card("x"), graph=panel_graph, screens=Screens(), doc_id="a/b")
    assert named["id"] == "a/b"


def test_an_empty_strip_with_only_a_tray_renders(panel_graph) -> None:
    from osprey.facility.panels.view import TrayEntry

    doc, _ = draw(strip_view([], (TrayEntry("T", None, "no position", None),)), panel_graph)
    assert "T" in _texts(doc) and "T · no position" not in _texts(doc)
    assert not _axes(doc) and not _marks(doc)


def test_text_width_grows_with_the_text_and_the_size() -> None:
    assert text_width("ab", 12) < text_width("abc", 12) < text_width("abc", 16)


def test_no_size_is_written_outside_the_constants_block() -> None:
    source = Path(layout.__file__).read_text()
    end = source.index("# --- end constants")
    end_line = source[:end].count("\n") + 1
    stray = [
        (node.lineno, node.value)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant)
        and type(node.value) in (int, float)
        and node.lineno > end_line
        and node.value not in (0, 1, 2)
    ]
    assert stray == []


def test_a_view_kind_not_drawn_is_refused(panel_graph) -> None:
    view = View(Target("card", "device", "x"), "x", (), (Section("s", ("not an item",)),))
    with pytest.raises(KeyError):
        stub_render(view, panel_graph)


def test_a_node_light_reads_its_channels(panel_graph) -> None:
    node = Node("n", None, ("XR01-PS1:On-Sts", "XR02-IP1:Pressure-RB"), False)
    doc, _ = draw(card("n", node), panel_graph)
    (shape,) = _by_type(doc, "polygon")
    assert list(bound(doc, shape)) == list(node.channels)


def test_a_long_table_is_cut_and_its_remaining_rows_counted(panel_graph) -> None:
    from osprey.facility.panels.view import CheckRow

    extra = 7
    rows = tuple(
        CheckRow(f"X:{i}", "X", True, True, False, False)
        for i in range(layout.MAX_TABLE_ROWS + extra)
    )
    doc, _ = draw(
        View(Target("checkout", "place", "X"), "x", (), (Section("c", rows),)), panel_graph
    )
    texts = _texts(doc)
    assert texts.count("X") == layout.MAX_TABLE_ROWS + 1  # and the header's
    (note,) = [t for t in texts if t.startswith(f"{extra} more ")]
    assert "rows not drawn" in note


def test_a_light_over_many_channels_names_their_count_not_each(panel_graph) -> None:
    doc, _ = draw(invented_views()[-1], panel_graph)
    (shape,) = _by_type(doc, "polygon")
    assert shape["props"]["tooltip"] == "40 channels"
    assert needs(doc) == ["state light"]  # forty channels with no rows make a short rule


def test_agreeing_sources_write_a_condition_once_and_literals_never_disagree(tmp_path) -> None:
    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as rows:
        rows.write("XR01-PS1:On-Sts,1,normal,screen,stated,xr-screens,4d5e6f7,ps.screen:1\n")
        rows.write("XR01-PS1:On-Sts,0,warn,screen,stated,xr-screens,4d5e6f7,ps.screen:2\n")
    graph = load_graph(write_fixture_facility(tmp_path / "facility.json"), tables)
    item = light(graph, "XR01-PS1:On-Sts")
    rule = ladder(graph, [item])
    assert rule.arms == (("{0} == 0", "warn"), ("{0} == 1", "normal"))
    for value in VALUES:
        assert evaluate(rule, [value]) == channel_state(graph, item.channel, value)


def test_a_one_channel_roll_up_node_over_the_limit_is_an_aggregate(panel_graph, monkeypatch):
    monkeypatch.setattr(layout, "RULE_EXPRESSION_LIMIT", 10)
    node = Node("one", None, ("XR02-IP1:Pressure-RB",), False)
    doc, _ = draw(card("one", node), panel_graph)
    assert aggregates(doc) == 1
    assert "1 channels, open for detail" in _texts(doc)
    # its one part is the channel alone, too long for any rule: unknown, and counted
    assert unexpressed(doc) == 1 and "long rule" in needs(doc)


def test_an_unwritable_channel_leaves_its_siblings_arms_and_never_normal(tmp_path) -> None:
    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as rows:
        rows.write("XR01-PS1:On-Sts,ON,normal,screen,stated,xr-screens,4d5e6f7,ps.screen:1\n")
    graph = load_graph(write_fixture_facility(tmp_path / "facility.json"), tables)
    items = [light(graph, "XR01-PS1:On-Sts"), light(graph, "XR02-IP1:Pressure-RB")]
    assert ladder(graph, items[:1]) == "label condition"  # alone, its light is unknown
    rule = ladder(graph, items)
    assert not isinstance(rule, str)
    assert rule.unwritten == ("XR01-PS1:On-Sts",)
    assert all(state != "normal" for _, state in rule.arms)
    assert evaluate(rule, [1, 7]) == "bad"  # the sibling's bad still shows
    assert evaluate(rule, [1, 0.5]) == "warn"
    assert evaluate(rule, [1, 3]) == "unknown"  # the sibling is normal: unknown, not normal
    doc, _ = draw(card("node", Node("n", None, tuple(i.channel for i in items), False)), graph)
    assert unexpressed(doc) == 0
    assert "label condition" in needs(doc)


def drawn(doc: dict, shape: dict, values: dict[str, object]) -> dict:
    """The props a drawn light's own rules give it for *values* by channel, as the viewer reads
    them: per rule, the first condition that holds sets the prop, else the rule's default;
    a prop no rule sets keeps its static value."""
    values = resolved(doc, values)
    out = {prop: shape["props"][prop] for prop in symbols.LIGHT_PROPS}
    for rule in shape.get("rules", ()):
        v = [values[p["name"]] for p in rule["pvs"]]
        out[rule["targetProperty"]] = rule["default"]
        for condition in rule["conditions"]:
            # the layout's own expressions: numbers and operators over {i}
            if eval(TOKEN.sub(r"v[\1]", condition["expression"]), {"__builtins__": {}}, {"v": v}):
                out[rule["targetProperty"]] = condition["value"]
                break
    return out


def shown(state: str) -> dict:
    return {prop: symbols.look(state)[prop] for prop in symbols.LIGHT_PROPS}


def test_a_drawn_node_with_an_unwritable_channel_shows_its_siblings_states(tmp_path) -> None:
    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as rows:
        rows.write("XR01-PS1:On-Sts,ON,normal,screen,stated,xr-screens,4d5e6f7,ps.screen:1\n")
        rows.write("XR02-IP1:Pressure-RB,3,normal,screen,stated,xr-screens,4d5e6f7,vac.screen:1\n")
    graph = load_graph(write_fixture_facility(tmp_path / "facility.json"), tables)
    a, b = "XR02-IP1:Pressure-RB", "XR01-PS1:On-Sts"  # A: >=5 bad, <1 warn, 3 normal; B: a label
    alone, _ = draw(card("a", Light(a)), graph)
    assert drawn(alone, _by_type(alone, "polygon")[0], {a: 3}) == shown(
        "normal"
    )  # A alone reads normal
    doc, _ = draw(card("node", Node("n", None, (a, b), False)), graph)
    (shape,) = _by_type(doc, "polygon")
    assert shape.get("rules"), "the writable channel's rules are drawn"
    static = {prop: shape["props"][prop] for prop in symbols.LIGHT_PROPS}
    assert static == shown("unknown")
    assert drawn(doc, shape, {a: 7, b: 1}) == shown("bad")
    assert drawn(doc, shape, {a: 0.5, b: 1}) == shown("warn")
    assert drawn(doc, shape, {a: 3, b: 1}) == shown("unknown")  # A normal: never normal
    assert "label condition" in needs(doc) and unexpressed(doc) == 0


def test_a_device_in_no_place_turns_the_root_roll_up_bad(panel_graph) -> None:
    loose = "XR09-LOOSE1:Fault-Sts"
    for by in ("place", "system"):
        doc, _ = draw(health_rollup(panel_graph, "XR", by), panel_graph)
        root = _by_type(doc, "polygon")[0]  # the root's row is drawn first
        names = set(bound(doc, root))
        assert loose in names
        quiet = dict.fromkeys(names, 1) | {loose: 0}
        assert drawn(doc, root, quiet)["fillColor"] != symbols.look("bad")["fillColor"]
        assert drawn(doc, root, quiet | {loose: 1}) == shown("bad"), by


def test_a_set_grid_draws_each_member_once_with_its_light_and_no_input(panel_graph) -> None:
    from osprey.facility.panels.describe import set_overview

    doc, _ = draw(set_overview(panel_graph, "XR-QUAD"), panel_graph)
    assert not _by_type(doc, "pv-text-input")  # nothing writable in a grid
    labels = [n["props"]["tooltip"] for n in _links_drawn(doc)]
    for member in ("XR01:QF1", "XR01:QD1", "XR02:QF1", "XR02:QD1"):
        assert labels.count(member) == 1
    values = [printed(doc, n) for n in _by_type(doc, "text-label") if printed(doc, n)]
    # each paired member's readback, a number held to its cell: a formula's text
    assert values[0] == "XR01-QF1:Current-RB" and "XR01-QD1:Current-RB" in values
    assert not _by_type(doc, "pv-label")
    read = [bound(doc, n) for n in _by_type(doc, "polygon") if n.get("rules")]
    assert ("XR01-QF1:Current-SP", "XR01-QF1:Current-RB") in read  # the deviation light
    assert ("XR02-QD1:Current-RB",) in read  # the member's own light


def test_a_rack_unit_and_a_grid_cell_with_no_rule_are_in_the_none_look(panel_graph) -> None:
    from osprey.facility.panels.describe import set_overview

    none = symbols.look("none")
    for view, count in (
        (rack_elevation(panel_graph, "XR/XR01/RK1"), 5),  # four units and one unpositioned
        (set_overview(panel_graph, "XR-QUAD"), 1),  # XR01:QD1: no rule, no pair (QF1 deviates)
    ):
        doc, _ = draw(view, panel_graph)
        bare = [n for n in _by_type(doc, "polygon") if n["props"].get("tooltip") == "no rule"]
        assert len(bare) == count, view.id
        assert all(not n.get("rules") for n in bare)
        for node in bare:
            assert {p: node["props"][p] for p in symbols.LIGHT_PROPS} == {
                p: none[p] for p in symbols.LIGHT_PROPS
            }


WRITABLE = {"pv-text-input", "pv-button"}


@pytest.mark.parametrize(
    ("drawn_as", "writable", "widget"),
    [
        ("value", False, "pv-label"),
        ("state", False, "pv-label"),
        ("text", False, "pv-label"),
        ("setpoint", True, "pv-text-input"),
        ("command", True, "pv-button"),
        ("waveform", False, "waveform-plot"),
        (None, True, "pv-text-input"),  # no row: a writable channel is a setpoint
        ("setpoint", False, "pv-label"),  # never an input on a channel that is no setpoint
    ],
)
def test_a_value_is_drawn_as_the_widget_its_drawn_as_names(panel_graph, drawn_as, writable, widget):
    row = ValueRow("x", "X:A", None, None, writable, drawn_as=drawn_as)
    doc, _ = draw(card("one", row), panel_graph)
    assert [n["type"] for n in nodes(doc) if n["type"].startswith(("pv-", "waveform"))] == [widget]
    assert layout.WIDGETS[drawn_as or ("setpoint" if writable else "untyped")]


def test_nothing_writable_is_drawn_outside_a_device_cards_own_rows(panel_graph) -> None:
    rows = (
        ValueRow("sp", "X:SP", "X:RB", None, True, drawn_as="setpoint"),
        ValueRow("cmd", "X:CMD", None, None, True, drawn_as="command"),
    )
    elsewhere = View(Target("place", "place", "X"), "x", (), (Section("s", rows),))
    doc, _ = draw(elsewhere, panel_graph)
    assert not {n["type"] for n in nodes(doc)} & WRITABLE
    for view in fixture_views(panel_graph):
        found = {n["type"] for n in nodes(draw(view, panel_graph)[0])} & WRITABLE
        if not (view.id.view == "card" and view.id.kind == "device"):
            assert not found, view.id


def test_a_command_shown_as_a_value_prints_whole_and_its_input_keeps_its_precision(
    panel_graph,
) -> None:
    row = ValueRow("cmd", "X:CMD", None, None, True, drawn_as="command")
    elsewhere = View(Target("place", "place", "X"), "x", (), (Section("s", (row,)),))
    doc, _ = draw(elsewhere, panel_graph)
    (label,) = [n for n in nodes(doc) if n["type"] == "pv-label"]
    assert label["props"]["precision"] == 0
    assert not [r for r in label.get("rules", ()) if r["targetProperty"] == "precision"]
    doc, _ = draw(card("one", row), panel_graph)
    (button,) = [n for n in nodes(doc) if n["type"] == "pv-button"]
    assert "precision" not in button["props"]


@pytest.mark.parametrize(
    ("channel", "word"),
    [
        ("XR01-PS1:Fault_Reset_Cmd", "Fault reset"),
        ("XR01-PS1:RampDown-CMD", "Ramp down"),
        ("XR01-PS1:resetRelay1", "Reset relay 1"),
        ("XR01-PS1:HV_On-Cmd", "HV on"),
        ("XR01-PS1:reset", "Reset"),
        ("XR01-PS1:Cmd", "Cmd"),  # all ending: the name whole, never an empty button
        ("NOCOLON", "NOCOLON"),
    ],
)
def test_a_command_names_its_action_from_its_channel(channel: str, word: str) -> None:
    assert layout.action_word(channel) == word


def test_a_command_button_reads_its_action_never_a_bare_send(tmp_path) -> None:
    command = "XR01-PS1:On-Cmd"
    row = ValueRow("on", command, None, None, True, drawn_as="command")
    tables = copy_tables(tmp_path / "plain")
    graph = load_graph(write_fixture_facility(tmp_path / "plain.json"), tables)
    (button,) = [n for n in nodes(draw(card("one", row), graph)[0]) if n["type"] == "pv-button"]
    assert button["props"]["label"] == "On"  # from the name
    stated = copy_tables(tmp_path / "stated")
    (stated / "state_labels.csv").write_text(
        "channel,value,label,source,revision,locator\n"
        f"{command},0,Rest,FIXTURE,X.1,l\n{command},1,Energise,FIXTURE,X.1,l\n"
    )
    graph = load_graph(write_fixture_facility(tmp_path / "stated.json"), stated)
    (button,) = [n for n in nodes(draw(card("one", row), graph)[0]) if n["type"] == "pv-button"]
    assert button["props"]["label"] == "Energise"  # the word for the value it writes
    assert button["props"]["pressValue"] == layout.COMMAND_VALUE


def test_a_paired_state_readback_prints_whole_and_a_two_valued_setpoints_input_too(
    panel_graph,
) -> None:
    # a bo setpoint paired with a bi state readback, as record_types.csv types them
    row = ValueRow("on", "XR01-PS1:On-Cmd", "XR01-IP1:State-Sts", None, True, drawn_as="setpoint")
    doc, _ = draw(card("one", row), panel_graph)
    (field,) = [n for n in nodes(doc) if n["type"] == "pv-text-input"]
    assert field["props"]["precision"] == 0  # an operator types 0 or 1
    (label,) = [n for n in nodes(doc) if n["type"] == "pv-label"]
    assert label["props"]["pv"] == "XR01-IP1:State-Sts"
    assert label["props"]["precision"] == 0
    assert not [r for r in label.get("rules", ()) if r["targetProperty"] == "precision"]


def test_an_analogue_setpoints_input_keeps_the_channels_own_precision(panel_graph) -> None:
    row = ValueRow("i", "XR01-QF1:Current-SP", "XR01-QF1:Current-RB", None, True)
    doc, _ = draw(card("one", row), panel_graph)
    (field,) = [n for n in nodes(doc) if n["type"] == "pv-text-input"]
    assert "precision" not in field["props"]


@pytest.mark.parametrize("channel", RULED)
def test_a_lights_screen_formula_gives_the_ladders_state_for_every_value(panel_graph, channel):
    item = light(panel_graph, channel)
    rule = ladder(panel_graph, [item])
    doc, _ = draw(card("one", item), panel_graph)
    if not rule.arms:  # rows the engine skips write no arm: no rule
        assert not [n for n in nodes(doc) if n.get("rules")]
        return
    (shape,) = _by_type(doc, "polygon")
    for value in VALUES:
        assert drawn(doc, shape, {channel: value}) == shown(evaluate(rule, [value])), value


def test_each_prop_of_a_light_reads_its_channels_and_writes_only_its_arms(panel_graph):
    items = [light(panel_graph, c) for c in RULED]
    doc, _ = draw(card("node", Node("n", None, RULED, False)), panel_graph)
    rule = ladder(panel_graph, items, node=True)
    arms = [condition for condition, _ in rule.arms]
    ruled = [n for n in nodes(doc) if n.get("rules")]
    assert len(ruled) == 2  # the shape and its mark
    for node in ruled:
        for found in node["rules"]:
            assert [pv["name"] for pv in found["pvs"]] == list(RULED)
            for condition in found["conditions"]:  # one arm, or arms of one look joined
                assert condition["expression"] in arms or all(
                    f"({arm})" in condition["expression"] or arm not in condition["expression"]
                    for arm in arms
                )
    assert max(_engine_lengths(doc)) == layout.engine_length(rule)


def test_a_child_with_no_live_light_opens_its_own_screen(panel_graph, monkeypatch) -> None:
    monkeypatch.setattr(layout, "SCREEN_RULE_BUDGET", 0)
    down = Target("rollup", "place", "XR/XR02", by="place")
    child = Node("XR/XR02", Link("XR/XR02", down), THREE, False)
    root = Node("XR", None, RULED, False, (child,))
    view = View(Target("rollup", "place", "XR", by="place"), "x", (), (Section("r", (root,)),))
    doc, screens = draw(view, panel_graph)
    opens = [n for n in _links_drawn(doc) if n["props"]["tooltip"] == "XR/XR02"]
    assert [n["props"]["actions"][0]["file"] for n in opens] == [screens(down)]
    assert any(t.startswith("3 channels") for t in _texts(doc))  # its tile says it is open
    assert not [n for n in nodes(doc) if n.get("rules")]


def test_a_link_is_neutral_text_under_an_outlined_box_that_opens_its_screen(panel_graph) -> None:
    target = Target("card", "device", "XR01:PS1")
    doc, screens = draw(card("x", Link("the supply", target)), panel_graph)
    (box,) = _links_drawn(doc)
    (label,) = [n for n in _by_type(doc, "text-label") if n["props"]["text"] == "the supply"]
    assert box["type"] == "rectangle" and box["geometry"] == label["geometry"]
    assert box["props"]["actions"] == [
        {"type": "open_display", "file": screens(target), "target": "tab"}
    ]
    assert box["props"]["lineColor"] == symbols.BOX_BORDER and box["props"]["fill"] is False
    assert label["props"]["foregroundColor"] == symbols.PAGE_TEXT
    assert label["props"]["backgroundColor"] == symbols.FIELD_BACKGROUND
    assert "clickable text" in needs(doc)


def test_every_locator_mark_and_line_has_extent_and_a_visible_stroke(panel_graph) -> None:
    for view in invented_views()[:3]:
        doc, _ = draw(view, panel_graph)
        assert _marks(doc)
        for n in _by_type(doc, "line"):
            assert n["geometry"]["width"] > 0 and n["geometry"]["height"] > 0
            assert n["props"]["lineWidth"] > 0
        for n in _marks(doc):
            assert n["geometry"]["height"] > layout.LOCATOR_BAR  # real length in y


def test_each_strip_symbol_has_its_short_name_under_it_and_no_two_names_collide(panel_graph):
    from osprey.facility.panels.layout import strip_labels

    doc, _ = draw(invented_views()[0], panel_graph)  # 300 symbols in rows
    (axis, *_rest) = _axes(doc)
    below = axis["geometry"]["y"] + layout.STRIP_LABEL_TOP - 1
    names = [
        n
        for n in _by_type(doc, "text-label")
        if n["props"]["fontSize"] == layout.STRIP_LABEL_SIZE and n["geometry"]["y"] >= below
    ]
    assert names
    by_level: dict[float, list[dict]] = {}
    for n in names:
        by_level.setdefault(n["geometry"]["y"], []).append(n)
    for row in by_level.values():
        row.sort(key=lambda n: n["geometry"]["x"])
        for a, b in zip(row, row[1:], strict=False):
            assert a["geometry"]["x"] + a["geometry"]["width"] <= b["geometry"]["x"]
    view = place_view(with_symbols(panel_graph, SYMBOLS, ABOVE), "XR/XR01")
    doc, _ = draw(view, panel_graph)
    assert {"F1", "D1"} <= set(_texts(doc))  # the stated short names
    assert "IP1" in _texts(doc)  # a badge's name stands right of it, not under the axis
    from osprey.facility.panels.strip import Placed

    wide = [
        Placed(f"X:LONGNAME{i}", "bpm", "beam", float(i), 10.0 * i, 8.0, 0, None) for i in range(4)
    ]
    placed = strip_labels(wide)
    assert {level for *_x, level in placed} == {0, 1}  # neighbours stagger


def test_the_devices_off_the_beamline_are_one_line_that_opens_the_place_card(panel_graph):
    view = place_view(with_symbols(panel_graph, SYMBOLS, ABOVE), "XR/XR02")
    doc, screens = draw(view, panel_graph)
    assert "3 other things in this place" in _texts(doc)  # XR02:SHIELD1 has no symbol
    files = [a["file"] for n in _links_drawn(doc) for a in n["props"]["actions"]]
    assert screens(Target("card", "place", "XR/XR02")) in files
    assert "XR02:BLOCK1" not in _texts(doc)  # listed on the place card, not in the tray


def test_a_cards_grid_prints_a_value_with_its_unit_beside_its_short_name(panel_graph) -> None:
    from osprey.facility.panels.describe import device_card

    doc, _ = draw(device_card(panel_graph, "XR01:QF1"), panel_graph)
    labels = {n["props"]["pv"]: n for n in _by_type(doc, "pv-label")}
    value = labels["XR01-QF1:Current-RB"]
    assert value["props"]["showUnits"] is True  # the unit with the value, whole
    assert value["geometry"]["width"] >= layout.value_width("A", layout.TEXT_SIZE)
    assert "Current-RB" in _texts(doc)  # its name, the shared segment cut, left of it


def _picture_view(parts, *, signal: str | None = None) -> View:
    """A card of one picture whose rows are parts at points, as a file of the first
    eleven columns states them; with *signal*, each part names it (and is named)."""
    from osprey.facility.panels.tables import PicturePart
    from osprey.facility.panels.view import Picture

    rows = tuple(
        PicturePart(
            "Magnet",
            "end",
            name,
            kind,
            x,
            y,
            0.0,
            None,
            None,
            None,
            None,
            signal=None if kind == "outline" else signal,
            as_point=kind != "outline",
        )
        for name, kind, x, y in parts
    )
    return card("pic", Picture("Magnet", "end", rows))


_CORNERS = [
    ("o1", "outline", -12.0, -9.0),
    ("o2", "outline", 12.0, -9.0),
    ("o3", "outline", 0, 9.0),
]


@pytest.mark.parametrize("tall", [False, True])
def test_no_two_picture_labels_overlap_and_no_two_leaders_cross(tall: bool) -> None:
    """The probe picture's label columns (:func:`layout.picture_labels`), with slots of
    one row, and of one or two rows."""
    from osprey.facility.panels.layout import ROW, ROW_GAP, _crossing, picture_labels

    centre, top, bottom, anchors = 300.0, 0.0, 300.0, (100.0, 500.0)
    crowded = [(f"part {i}", centre + (-1) ** i * (100 - 10 * i), 150 - 9 * i) for i in range(12)]
    scattered = [
        (f"bit {i}", centre + 20 * ((i * 7 % 11) - 5), 150 + 25 * ((i * 5 % 9) - 4))
        for i in range(14)
    ]
    for parts in (crowded, scattered):
        heights = [2 * ROW if tall and n % 3 == 0 else ROW for n in range(len(parts))]
        slots = picture_labels(parts, centre, top, bottom, anchors, heights if tall else None)
        for side in ("left", "right"):
            mine = sorted((y, h) for (s, y), h in zip(slots, heights, strict=True) if s == side)
            for (y1, h1), (y2, _h2) in itertools.pairwise(mine):
                assert y1 + h1 + ROW_GAP <= y2 + 1e-6
        if tall:
            continue  # a respacing after the swaps may cross two leaders again
        leaders = [
            ((x, y), (anchors[0] if s == "left" else anchors[1], slot + ROW / 2))
            for (_n, x, y), (s, slot) in zip(parts, slots, strict=True)
        ]
        for i, a in enumerate(leaders):
            for b in leaders[i + 1 :]:
                assert not _crossing(*a, *b), (a, b)


def test_a_picture_without_an_outline_draws_its_parts_in_a_frame(panel_graph) -> None:
    parts = [
        ("pole 1", "pole", 5.0, 5.0),
        ("probe", "a-kind-of-the-facility", -5.0, 0.0),
        ("pole 2", "pole", -5.0, 5.0),
    ]
    doc, _ = draw(_picture_view(parts), panel_graph)
    texts = _texts(doc)
    assert "END VIEW" in texts and "outline not stated" in texts
    tips = [n["props"]["tooltip"] for n in _by_type(doc, "ellipse")]
    assert tips == ["pole 1", "probe", "pole 2"]  # each part at its position, plain
    kinds = [t for t in texts if t in ("pole", "a-kind-of-the-facility")]
    # each kind's word above its parts on the drawing, then the legend in table order
    assert kinds.count("pole") >= 2 and kinds.count("a-kind-of-the-facility") == 2
    assert kinds[-2:] == ["pole", "a-kind-of-the-facility"]
    bare = [n for n in _by_type(doc, "polygon") if n["props"].get("tooltip") == "no rule"]
    assert not bare  # no chips with unbound lights


def test_a_picture_draws_the_outline_the_table_states(panel_graph) -> None:
    parts = [("pole 1", "pole", 5.0, 5.0), ("probe", "a-kind-of-the-facility", -5.0, 0.0)]
    corners = [
        ("o1", "outline", -9.0, -9.0),
        ("o2", "outline", 9.0, -9.0),
        ("o3", "outline", 0.0, 9.0),
    ]
    doc, _ = draw(_picture_view([*parts, *corners]), panel_graph)
    assert "outline not stated" not in _texts(doc)
    tips = [n["props"]["tooltip"] for n in _by_type(doc, "ellipse")]
    assert tips == ["yoke", "pole 1", "probe"]  # the cavity, then each part where it sits
    (yoke,) = [n for n in _by_type(doc, "polygon") if n["props"].get("tooltip") == "yoke"]
    (window,) = [n for n in _by_type(doc, "ellipse") if n["props"].get("tooltip") == "yoke"]
    assert len(yoke["props"]["points"]) == len(corners)  # the stated points: the yoke
    assert yoke["props"]["fillColor"] == symbols.STEEL_DARK
    assert window["props"]["fillColor"] == symbols.PANEL_BACKGROUND  # convex: its cavity
    assert "o1" not in _texts(doc)  # an outline point is no part


def test_a_bad_part_turns_the_place_of_its_whole_bad(tmp_path) -> None:
    graph = graph_with_part(tmp_path)
    card = "XR01-CARD1:Fault-Sts"
    for place in ("XR/XR01/RK1", "XR/XR01"):
        doc, _ = draw(health_rollup(graph, place), graph)
        root = _by_type(doc, "polygon")[0]  # the root's row is drawn first
        names = set(bound(doc, root))
        assert card in names, place
        quiet = dict.fromkeys(names, 1) | {card: 0}
        assert drawn(doc, root, quiet)["fillColor"] != symbols.look("bad")["fillColor"]
        assert drawn(doc, root, quiet | {card: 1}) == shown("bad"), place


# ------------------------------------------------------------- the engine's expression


def _engine_lengths(doc: dict) -> list[int]:
    """The length of every rule's expression as the viewer compiles it for the engine."""
    found = {f["name"]: f for f in doc.get("formulas", ())}
    return [len(layout.engine_expression(r, found)) for n in nodes(doc) for r in n.get("rules", ())]


def test_the_engine_expression_is_the_viewers_compile_of_a_rule() -> None:
    formula = {
        "name": "s00001",
        "kind": "scalar",
        "expression": "4 if (v0 == 1) else (1)",
        "bindings": {"v0": "A"},
    }
    rule = {
        "id": "r-0001",
        "name": "n",
        "targetProperty": "fill",
        "default": False,
        "pvs": [{"name": "formula://s00001", "trigger": True}],
        "conditions": [{"expression": "{0} == 4", "value": True}],
    }
    assert layout.engine_expression(rule, {"s00001": formula}) == (
        "True if ((4 if ({s00001.v0} == 1) else (1)) == 4) else (False)"
    )
    plain = dict(rule, pvs=[{"name": "A", "trigger": True}], default={"$role": "X"})
    assert layout.engine_expression(plain, {}) == 'True if ({0} == 4) else ({"$role":"X"})'


def test_the_bound_is_the_longest_expression_the_light_writes(panel_graph) -> None:
    for items in ([light(panel_graph, c)] for c in RULED):
        rule = ladder(panel_graph, items)
        if not rule.arms:
            continue  # no arm, no rule drawn
        doc, _ = draw(card("one", *items), panel_graph)
        assert layout.engine_length(rule) == max(_engine_lengths(doc)), items


def test_no_drawn_rule_is_longer_than_the_engine_takes_even_for_forty_channels(tmp_path):
    from osprey.facility.panels.describe import describe, health_rollup

    graph = graph_with_wide(tmp_path, 40)
    for view in [
        *fixture_views(graph),
        health_rollup(graph, "XR/XR01"),
        health_rollup(graph, "XR"),
    ]:
        doc, _ = draw(view, graph)
        assert max(_engine_lengths(doc), default=0) <= layout.RULE_EXPRESSION_LIMIT, view.id
    doc, _ = draw(describe(graph, layout_target("XR01:WIDE1")), graph)
    assert max(_engine_lengths(doc)) <= layout.RULE_EXPRESSION_LIMIT


def layout_target(thing: str):
    from osprey.facility.panels.view import Target

    return Target("card", "device", thing)


def test_a_channel_writes_the_shortest_condition_per_state(tmp_path) -> None:
    graph = graph_with_wide(tmp_path, 1)
    rule = ladder(graph, [light(graph, "XR01-WIDE1:Sig0-Sts")])
    assert rule.arms == (
        ("{0} == 1 or {0} >= 5", "bad"),
        ("{0} == 2", "warn"),
        ("{0} == 0", "normal"),
    )


def test_a_run_of_integer_values_is_one_test(tmp_path) -> None:
    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as rows:
        for value in (1, 2, 3, 4):
            rows.write(
                f"XR09-LOOSE1:Fault-Sts,{value},bad,database-severity,stated,xr-ioc,0a1b2c3,l.db:{value}\n"
            )
        rows.write(
            "XR09-LOOSE1:Fault-Sts,0,normal,database-severity,stated,xr-ioc,0a1b2c3,l.db:9\n"
        )
    graph = load_graph(write_fixture_facility(tmp_path / "facility.json"), tables)
    item = light(graph, "XR09-LOOSE1:Fault-Sts")
    rule = ladder(graph, [item])
    assert rule.arms[0] == ("{0} == {0} // 1 and 1 <= {0} <= 4", "bad")
    for value in (-1, 0, 0.5, 1, 1.5, 2, 3, 4, 4.5, 5):
        assert evaluate(rule, [value]) == flagged(channel_state(graph, item.channel, value), item)


def test_the_simplified_ladder_is_the_worst_state_for_every_combination(tmp_path) -> None:
    import itertools

    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as rows:
        rows.write(
            "XR09-LOOSE1:Fault-Sts,0,normal,database-severity,stated,xr-ioc,0a1b2c3,l.db:3\n"
        )
        rows.write("XR09-LOOSE1:Fault-Sts,3,warn,database-severity,stated,xr-ioc,0a1b2c3,l.db:4\n")
        rows.write("XR09-LOOSE1:Fault-Sts,>=4,bad,database-severity,stated,xr-ioc,0a1b2c3,l.db:5\n")
        rows.write("XR02-QD1:Current-RB,>=6,bad,database-severity,stated,xr-ioc,0a1b2c3,m.db:7\n")
    graph = load_graph(write_fixture_facility(tmp_path / "facility.json"), tables)
    # literals with agreeing and disagreeing sources, comparisons, bit tests beside a
    # comparison, a flagged light
    channels = ("XR09-LOOSE1:Fault-Sts", "XR01-IP1:State-Sts", "XR02-QD1:Current-RB")
    items = [light(graph, c) for c in channels]
    rule = ladder(graph, items)
    assert not isinstance(rule, str)
    values = (-1, 0, 0.5, 1, 2, 3, 4, 5, 6, 7)
    for combo in itertools.product(values, repeat=3):
        expected = worst(
            flagged(channel_state(graph, item.channel, v), item)
            for item, v in zip(items, combo, strict=True)
        )
        assert evaluate(rule, list(combo)) == expected, combo
    formula = " ".join(condition for condition, _ in rule.arms)
    assert "not (" not in formula  # no negation of what an earlier arm already ruled out


def test_a_channel_with_bit_tests_and_other_rows_is_exact_too(tmp_path) -> None:
    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as rows:
        rows.write("XR02-QD1:Current-RB,>=8,bad,database-severity,stated,xr-ioc,0a1b2c3,m.db:9\n")
    graph = load_graph(write_fixture_facility(tmp_path / "facility.json"), tables)
    item = light(graph, "XR02-QD1:Current-RB")
    rule = ladder(graph, [item])
    for value in (-3, -2, -1, 0, 0.5, 1, 2, 3, 4, 5, 6, 7, 7.5, 8, 9, 10, 11):
        assert evaluate(rule, [value]) == flagged(channel_state(graph, item.channel, value), item)


def test_a_channel_bad_above_its_last_value_writes_an_open_bound(tmp_path) -> None:
    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as rows:
        rows.write("XR09-LOOSE1:Fault-Sts,>1,bad,database-severity,stated,xr-ioc,0a1b2c3,l.db:3\n")
        rows.write("XR09-LOOSE1:Fault-Sts,<0,warn,database-severity,stated,xr-ioc,0a1b2c3,l.db:4\n")
    graph = load_graph(write_fixture_facility(tmp_path / "facility.json"), tables)
    item = light(graph, "XR09-LOOSE1:Fault-Sts")
    rule = ladder(graph, [item])
    assert rule.arms[:2] == (("{0} >= 1", "bad"), ("{0} < 0", "warn"))
    for value in (-1, -0.5, 0, 0.5, 1, 1.5, 2):
        assert evaluate(rule, [value]) == flagged(channel_state(graph, item.channel, value), item)


def test_a_rack_unit_draws_one_light_over_its_channels_and_its_parts(tmp_path):
    from tests.facility.panels._fixture import add_part

    second = "XR01-CARD2:Fault-Sts,1,bad,database-severity,stated,xr-ioc,0a1b2c3,card.db:2\n"
    graph_with_part(tmp_path, lambda d: add_part(d, "XR01:CARD2", "XR01:CH1", second[:20]))
    with (tmp_path / "panels" / "health.csv").open("a") as rows:
        rows.write(second)
    graph = load_graph(tmp_path / "facility.json", tmp_path / "panels")
    doc, _ = draw(rack_elevation(graph, "XR/XR01/RK1"), graph)
    lit = [n for n in _by_type(doc, "polygon") if n.get("rules")]
    tips = [n["props"]["tooltip"] for n in lit]
    assert "XR01-CARD1:Fault-Sts" not in tips  # no light of its own: the unit's light
    assert any("XR01-CARD1:Fault-Sts" in t and "XR01-CARD2:Fault-Sts" in t for t in tips)
    assert not [t for t in tips if t.startswith("part ")]


def test_a_tray_chip_names_a_reason_only_when_it_says_more_than_the_heading(panel_graph):
    from osprey.facility.panels.view import TrayEntry

    tray = (TrayEntry("T1", None, "no position", None), TrayEntry("T2", None, "frame odd", None))
    doc, _ = draw(strip_view([], tray), panel_graph)
    assert {"T1", "T2 · frame odd"} <= set(_texts(doc))


def test_the_upper_lane_is_reserved_only_when_something_stands_in_it(panel_graph) -> None:
    from tests.facility.panels._views import element

    flat = [element(f"E{i}", float(i), "bpm") for i in range(3)]
    lifted = [*flat, element("UP", 1.5, "ion_pump", lane="above")]
    tops = []
    for row in (flat, lifted):
        doc, _ = draw(strip_view(row), panel_graph)
        (axis,) = _axes(doc)
        tops.append(axis["geometry"]["y"])
    assert tops[0] + layout.STEM_LOW - layout.STRIP_AXIS <= tops[1]


def test_the_scale_caption_stands_above_the_bar_at_its_left_end(panel_graph) -> None:
    doc, _ = draw(invented_views()[1], panel_graph)
    (caption,) = [
        n for n in _by_type(doc, "text-label") if n["props"]["text"].startswith("to scale")
    ]
    bar = [n for n in _by_type(doc, "line") if n["geometry"]["width"] >= layout.STRIP_WIDTH][-1]
    assert caption["geometry"]["y"] + caption["geometry"]["height"] <= bar["geometry"]["y"]
    assert caption["geometry"]["x"] == layout.MARGIN


def test_a_strip_label_is_the_stated_short_name_and_the_link_keeps_the_id(panel_graph) -> None:
    view = place_view(with_symbols(panel_graph, SYMBOLS, ABOVE), "XR/XR01")
    doc, screens = draw(view, panel_graph)
    texts = _texts(doc)
    assert {"F1", "D1"} <= set(texts) and "QF1" not in texts
    tips = {n["props"]["tooltip"] for n in _links_drawn(doc)}
    assert "XR01:QF1" in tips  # the symbol's link names the full id


def test_a_card_header_shows_the_short_name_as_an_alias(panel_graph) -> None:
    from osprey.facility.panels.describe import device_card

    view = device_card(panel_graph, "XR01:QF1")
    assert view.alias == "F1"
    doc, _ = draw(view, panel_graph)
    header = [
        n
        for n in _by_type(doc, "text-label")
        if n["geometry"]["y"] + n["geometry"]["height"] <= layout.HEADER_HEIGHT
    ]
    texts = [n["props"]["text"] for n in header]
    assert "XR01:QF1" in texts and "F1" in texts


ANNOTATE = {"quadrupole": ("current_readback",), "ion_pump": ("vacuum_readback",)}


def _values(doc: dict) -> dict[str, dict]:
    """The strip's values by channel: static labels whose text a formula rule sets."""
    formulas = {f["name"]: f for f in doc.get("formulas", ())}
    found = {}
    for n in _by_type(doc, "text-label"):
        for rule in n.get("rules", ()):
            if rule["targetProperty"] == "text" and "formula" in rule:
                formula = formulas[rule["formula"].removeprefix("formula://")]
                found[formula["bindings"]["v"]] = n
    return found


def test_a_strip_shows_the_annotated_values_under_its_symbols_and_beside_its_badges(
    panel_graph,
) -> None:
    view = place_view(with_symbols(panel_graph, SYMBOLS, ABOVE), "XR/XR01", annotate=ANNOTATE)
    doc, _ = draw(view, panel_graph)
    values = _values(doc)
    quad, pump = values["XR01-QF1:Current-RB"], values["XR01-IP1:Pressure-RB"]
    assert quad["props"]["fontSize"] == layout.SMALL_SIZE
    assert quad["props"]["foregroundColor"] == symbols.MUTED_TEXT
    (label,) = [n for n in _by_type(doc, "text-label") if n["props"]["text"] == "F1"]
    centre = label["geometry"]["x"] + label["geometry"]["width"] / 2
    g = quad["geometry"]
    assert abs(g["x"] + g["width"] / 2 - centre) < 1 and g["y"] >= label["geometry"]["y"]
    (circle,) = [
        n
        for n in _by_type(doc, "polygon")
        if n["props"].get("tooltip") == "XR01:IP1" and len(n["props"]["points"]) > 8
    ]
    assert pump["geometry"]["x"] >= circle["geometry"]["x"] + circle["geometry"]["width"]
    assert any(t.startswith("values: ") and "current_readback (A)" in t for t in _texts(doc))


def _names_bottom(doc: dict, names: set[str]) -> float:
    return max(
        n["geometry"]["y"] + n["geometry"]["height"]
        for n in _by_type(doc, "text-label")
        if n["props"]["text"] in names
    )


def test_strip_values_are_bare_text_stacked_under_their_own_name(panel_graph) -> None:
    two = {"quadrupole": ("current_readback", "current_setpoint")}
    view = place_view(with_symbols(panel_graph, SYMBOLS, ABOVE), "XR/XR01", annotate=two)
    doc, _ = draw(view, panel_graph)
    values = _values(doc)
    first = [values["XR01-QF1:Current-RB"], values["XR01-QD1:Current-RB"]]
    second = [values["XR01-QF1:Current-SP"], values["XR01-QD1:Current-SP"]]
    for name, a, b in (("F1", first[0], second[0]), ("D1", first[1], second[1])):
        (label,) = [n for n in _by_type(doc, "text-label") if n["props"]["text"] == name]
        assert a["geometry"]["y"] == label["geometry"]["y"] + layout.VALUE_STEP
        assert b["geometry"]["y"] == a["geometry"]["y"] + layout.VALUE_STEP
        assert a["geometry"]["x"] == b["geometry"]["x"] == label["geometry"]["x"]
    for thing, value in (("XR01:QF1", first[0]), ("XR01:QD1", first[1])):
        shapes = [n for n in _by_type(doc, "polygon") if n["props"].get("tooltip") == thing]
        left = min(n["geometry"]["x"] for n in shapes)
        right = max(n["geometry"]["x"] + n["geometry"]["width"] for n in shapes)
        g = value["geometry"]
        edge = g["x"] == layout.MARGIN or g["x"] + g["width"] == layout.MARGIN + layout.STRIP_WIDTH
        assert edge or abs(g["x"] + g["width"] / 2 - (left + right) / 2) < 1  # centred, kept in
    for node in first + second:  # bare text: no ground, centred
        props = node["props"]
        assert props["backgroundColor"] == symbols.TRANSPARENT
        assert props["align"] == "center"


def test_a_lifted_badge_carries_its_name_right_of_it_and_its_value_under_the_name(
    panel_graph,
) -> None:
    view = place_view(with_symbols(panel_graph, SYMBOLS, ABOVE), "XR/XR01", annotate=ANNOTATE)
    doc, _ = draw(view, panel_graph)
    (circle,) = [
        n
        for n in _by_type(doc, "polygon")
        if n["props"].get("tooltip") == "XR01:IP1" and len(n["props"]["points"]) > 8
    ]
    right = circle["geometry"]["x"] + circle["geometry"]["width"]
    (name,) = [n for n in _by_type(doc, "text-label") if n["props"]["text"] == "IP1"]
    assert name["props"]["fontSize"] == layout.SMALL_SIZE
    assert name["props"]["foregroundColor"] == symbols.PAGE_TEXT
    assert name["geometry"]["x"] >= right
    value = _values(doc)["XR01-IP1:Pressure-RB"]
    assert value["geometry"]["x"] == name["geometry"]["x"]
    assert value["geometry"]["y"] >= name["geometry"]["y"] + name["geometry"]["height"]
    assert value["props"]["align"] == "left"


def test_a_badge_name_that_would_collide_is_left_out(panel_graph) -> None:
    near = [
        StripElement("XR:P1", None, 1.0, "ion_pump", "above", None, ()),
        StripElement("XR:P2", None, 1.0001, "ion_pump", "above", None, ()),
        StripElement("XR:P3", None, 1.0002, "ion_pump", "above", None, ()),
    ]
    doc, _ = draw(strip_view(near), panel_graph)
    texts = _texts(doc)
    drawn = [t for t in ("P1", "P2", "P3") if t in texts]
    assert len(drawn) < 3  # three badges at one place: their names cannot all fit
    boxes = [n["geometry"] for n in _by_type(doc, "text-label") if n["props"]["text"] in drawn]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            assert (
                a["x"] + a["width"] <= b["x"]
                or b["x"] + b["width"] <= a["x"]
                or a["y"] + a["height"] <= b["y"]
                or b["y"] + b["height"] <= a["y"]
            )


def test_over_the_budget_values_are_dropped_not_symbols_and_the_strip_says_so(
    panel_graph, monkeypatch
) -> None:
    view = place_view(with_symbols(panel_graph, SYMBOLS, ABOVE), "XR/XR01", annotate=ANNOTATE)
    full, _ = draw(view, panel_graph)
    monkeypatch.setattr(layout, "SCREEN_CHANNEL_BUDGET", 0)
    doc, _ = draw(view, panel_graph)
    assert not _values(doc)
    assert any("values not shown" in t for t in _texts(doc))
    shapes = [n for n in _by_type(doc, "polygon") if n["props"].get("tooltip") == "XR01:QF1"]
    assert shapes  # the symbol stays
    assert len(_values(full)) == 3  # two quadrupoles and the pump


def test_five_chain_blocks_fit_one_row_and_a_sixth_wraps() -> None:
    rows = layout._Page.chain_rows
    three = rows([layout.BLOCK_LEAST] * 3)
    assert three == [([layout.BLOCK_LEAST] * 3, [layout.BLOCK_GAP] * 2)]  # room: as drawn
    (five,) = rows([layout.BLOCK_MOST] * 5)
    widths, gaps = five
    assert len(widths) == 5 and gaps == [layout.BLOCK_GAP_LEAST] * 4
    assert sum(widths) + sum(gaps) <= layout.STRIP_WIDTH
    assert min(widths) >= layout.BLOCK_NARROWEST
    six = rows([layout.BLOCK_LEAST] * 6)
    assert [len(w) for w, _gap in six] == [5, 1]
    # a narrowed row gives each arrow the run its word needs
    (wide,) = rows([layout.BLOCK_MOST] * 5, [layout.BLOCK_GAP, 0, 0, 0])
    assert wide[1][0] == layout.BLOCK_GAP and sum(wide[0]) + sum(wide[1]) <= layout.STRIP_WIDTH


def test_an_arrow_word_and_a_block_title_are_never_cut() -> None:
    lines, short = layout.arrow_lines("controlled by")
    assert " ".join(lines) == "controlled by" and not short
    lines, short = layout.arrow_lines("XR05:CTL1:PORTA1_CH01")
    assert lines == ["PORTA1_CH01"] and short  # the whole word on hover
    for line in layout.arrow_lines("an arrow word too long for any one line")[0]:
        assert layout.text_width(line, layout.SMALL_SIZE) <= layout.ARROW_RUN_MOST - layout.GAP


def test_a_long_tile_name_breaks_at_its_separators_and_is_whole() -> None:
    from osprey.facility.panels.layout import _wrap_label, text_width

    name = "XR01:Vacuum-Controller"
    width = text_width("XR01:Vacuum-", layout.SMALL_SIZE) + 1
    lines = _wrap_label(name, width, layout.SMALL_SIZE, 2)
    assert lines == ["XR01:Vacuum-", "Controller"]
    assert all(text_width(line, layout.SMALL_SIZE) <= width for line in lines)
    assert _wrap_label("120 without a rule", 1e6, layout.SMALL_SIZE, 2) == ["120 without a rule"]


def test_every_size_constant_is_named_once() -> None:
    source = Path(layout.__file__).read_text()
    names = re.findall(r"^([A-Z][A-Z0-9_]*) = ", source, flags=re.MULTILINE)
    twice = sorted({n for n in names if names.count(n) > 1})
    assert twice == []  # a second binding silently changes every use of the first
