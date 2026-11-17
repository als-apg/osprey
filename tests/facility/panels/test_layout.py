"""Layout: view descriptions drawn as Screen-IR documents.

The schema check of every rendered view needs CANOPY and is in
``test_layout_canopy.py``; everything here runs without it.
"""

from __future__ import annotations

import ast
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
    ladder,
    needs,
    render,
    rule_length,
    text_width,
    unexpressed,
)
from osprey.facility.panels.model import load_graph
from osprey.facility.panels.view import Light, Node, Section, Target, View
from tests.facility.panels._fixture import copy_tables, write_fixture_facility
from tests.facility.panels._views import (
    ABOVE,
    RULED,
    SYMBOLS,
    Screens,
    card,
    fixture_views,
    invented_views,
    links,
    nodes,
    rollup_view,
    strip_view,
)

#: Widget types the pinned viewer's catalog (its ir-map.json) maps.
PINNED_WIDGETS = {
    "absolute-canvas",
    "text-label",
    "pv-label",
    "pv-text-input",
    "related-display-button",
    "rectangle",
    "ellipse",
    "line",
    "polygon",
}
VALUES = (-3, 0, 0.5, 1, 2, 2.5, 3, 4.5, 5, 6, 7)
TOKEN = re.compile(r"\{(\d+)\}")


def draw(view: View, graph) -> tuple[dict, Screens]:
    screens = Screens()
    return render(view, graph=graph, screens=screens), screens


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
    """The engine's state as a light draws it: a flagged light is never normal."""
    if state != "normal":
        return state
    if item.disagree:
        return "disagree"
    return "unknown" if item.unsupported else state


def _by_type(doc: dict, kind: str) -> list[dict]:
    return [n for n in nodes(doc) if n["type"] == kind]


def _texts(doc: dict) -> list[str]:
    return [n["props"]["text"] for n in _by_type(doc, "text-label")]


def _beam_lines(doc: dict) -> list[dict]:
    return [n for n in _by_type(doc, "line") if n["props"]["lineColor"] == symbols.BEAM]


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
    rule = ladder(panel_graph, [light(panel_graph, "XR01:PS1:On-Sts")])
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


def test_flagged_lights_are_never_normal(panel_graph) -> None:
    for channel in ("XR01:IP1:State-Sts", "XR02:QF1:Current-RB"):
        item = light(panel_graph, channel)
        assert item.disagree or item.unsupported
        rule = ladder(panel_graph, [item])
        assert all(state != "normal" for _, state in rule.arms)


def test_a_text_label_row_cannot_be_written_and_the_light_is_unknown(tmp_path) -> None:
    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as rows:
        rows.write("XR01:PS1:On-Sts,ON,normal,screen,stated,xr-screens,4d5e6f7,ps.screen:1\n")
    graph = load_graph(write_fixture_facility(tmp_path / "facility.json"), tables)
    assert ladder(graph, [light(graph, "XR01:PS1:On-Sts")]) == "label condition"
    doc, _ = draw(card("label", light(graph, "XR01:PS1:On-Sts")), graph)
    assert unexpressed(doc) == 1
    assert "label condition" in needs(doc)
    (shape,) = _by_type(doc, "polygon")
    assert "rules" not in shape
    assert shape["props"]["lineStyle"] == "dashed" and shape["props"]["fill"] is False
    assert "1 light(s) drawn unknown" in doc["metadata"]["description"]


def _limit_between_one_channel_and_three(graph) -> int:
    singles = [rule_length(ladder(graph, [light(graph, c)])) for c in RULED]
    three = ladder(graph, [light(graph, c) for c in RULED[1:4]])
    assert rule_length(three) > max(singles)
    return max(singles)


def test_the_rule_limit_is_the_formula_engines_expression_limit() -> None:
    assert layout.RULE_EXPRESSION_LIMIT == 2048
    assert "max_expression_len" in Path(layout.__file__).read_text()


def test_a_roll_up_over_the_limit_is_an_aggregate_over_a_mosaic(panel_graph, monkeypatch):
    monkeypatch.setattr(
        layout, "RULE_EXPRESSION_LIMIT", _limit_between_one_channel_and_three(panel_graph)
    )
    assert ladder(panel_graph, [light(panel_graph, c) for c in RULED]) == "aggregate state"
    doc, _ = draw(rollup_view(), panel_graph)
    assert aggregates(doc) == 2  # the root and the three-channel child
    assert unexpressed(doc) == 0  # an aggregate is not an unknown light
    assert "aggregate state" in needs(doc)
    texts = _texts(doc)
    assert texts.count(f"{len(RULED)} channels, open for detail") == 1
    assert texts.count("3 channels, open for detail") == 1
    shapes = _by_type(doc, "polygon")
    look = symbols.look("aggregate")
    aggregate = [
        n for n in shapes if n["props"]["fillColor"] == look["fillColor"] and "rules" not in n
    ]
    # root row, three-child row, and the three-child's tile in the mosaic
    assert len(aggregate) == 3
    tiles = [n for n in shapes if n["geometry"]["width"] == layout.MOSAIC_LIGHT]
    assert len(tiles) == len(RULED) + 1  # one per ruled child; none for the child with no rule
    assert symbols.AGGREGATE_MARK in texts


def test_a_roll_up_screen_draws_one_level(panel_graph) -> None:
    doc, _ = draw(rollup_view(), panel_graph)
    assert "deep" not in _texts(doc)
    assert aggregates(doc) == 0


def test_one_channel_over_the_limit_is_unknown_and_counted(panel_graph, monkeypatch) -> None:
    monkeypatch.setattr(layout, "RULE_EXPRESSION_LIMIT", 10)
    assert ladder(panel_graph, [light(panel_graph, "XR02:IP1:Pressure-RB")]) == "long rule"
    doc, _ = draw(card("long", light(panel_graph, "XR02:IP1:Pressure-RB")), panel_graph)
    assert unexpressed(doc) == 1 and "long rule" in needs(doc)


def test_every_fixture_rule_fits_the_limit(panel_graph) -> None:
    for view in fixture_views(panel_graph):
        assert unexpressed(draw(view, panel_graph)[0]) == 0, view.id


def test_a_lights_rules_carry_the_looks_of_their_states(panel_graph) -> None:
    item = light(panel_graph, "XR02:IP1:Pressure-RB")
    doc, _ = draw(card("one", item), panel_graph)
    (shape,) = _by_type(doc, "polygon")
    rule = ladder(panel_graph, [item])
    assert [r["targetProperty"] for r in shape["rules"]] == list(symbols.LIGHT_PROPS)
    for r in shape["rules"]:
        assert r["pvs"] == [{"name": "XR02:IP1:Pressure-RB", "trigger": True}]
        assert r["default"] == symbols.look("unknown")[r["targetProperty"]]
        for condition, (expression, state) in zip(r["conditions"], rule.arms, strict=True):
            assert condition["expression"] == expression
            assert condition["value"] == symbols.look(state)[r["targetProperty"]]
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
    bound = [
        tuple(p["name"] for p in n["rules"][0]["pvs"])
        for n in _by_type(doc, "polygon")
        if n.get("rules")
    ]
    assert ("A:SP", "A:RB") in bound and ("B:SP", "B:RB") in bound
    assert not any("C:RB" in names for names in bound)
    absolute = layout._tolerance_ladder(invented_views()[3].sections[0].items[0])
    assert evaluate(absolute, [10.0, 10.04]) == "normal"
    assert evaluate(absolute, [10.0, 10.06]) == "warn"
    relative = layout._tolerance_ladder(invented_views()[3].sections[0].items[1])
    assert evaluate(relative, [100.0, 100.9]) == "normal"
    assert evaluate(relative, [100.0, 101.1]) == "warn"


def test_an_inferred_item_carries_the_mark_and_a_stated_one_does_not(panel_graph) -> None:
    marked, _ = draw(card("x", Light("XR01:PS1:On-Sts", inferred=True)), panel_graph)
    plain, _ = draw(card("x", Light("XR01:PS1:On-Sts")), panel_graph)
    assert _texts(marked).count(symbols.INFERRED_MARK) == 1
    assert symbols.INFERRED_MARK not in _texts(plain)
    doc, _ = draw(invented_views()[3], panel_graph)
    # the inferred value row, light, fact and link
    assert _texts(doc).count(symbols.INFERRED_MARK) == 4


def test_a_strip_draws_quadrupoles_above_or_below_the_axis_by_polarity(panel_graph) -> None:
    doc, _ = draw(invented_views()[2], panel_graph)
    (axis,) = [n for n in _beam_lines(doc) if n["geometry"]["height"] == 0]
    y = axis["geometry"]["y"]
    shape = {n["props"]["tooltip"]: n["geometry"] for n in nodes(doc) if "tooltip" in n["props"]}
    assert shape["QF"]["y"] + shape["QF"]["height"] == pytest.approx(y, abs=0.1)
    assert shape["QD"]["y"] == pytest.approx(y, abs=0.1)
    assert shape["QX"]["y"] < y < shape["QX"]["y"] + shape["QX"]["height"]
    assert shape["IP"]["y"] + shape["IP"]["height"] < y  # the lane above
    assert shape["ODD"]["width"] == pytest.approx(symbols.WIDTH_UNITS["generic"] * SYMBOL_UNIT)


def test_a_locator_puts_ticks_and_devices_in_proportion_to_s(panel_graph) -> None:
    doc, _ = draw(invented_views()[1], panel_graph)
    ticks = [
        (float(n["props"]["text"][: -len(" m")]), n["geometry"]["x"] + n["geometry"]["width"] / 2)
        for n in _by_type(doc, "text-label")
        if n["props"]["text"].endswith(" m")
    ]
    assert [m for m, _ in ticks] == [0, 5, 10, 15, 20, 25, 30]
    (m0, x0), (m1, x1) = ticks[0], ticks[-1]
    per_metre = (x1 - x0) / (m1 - m0)
    for m, x in ticks:
        assert x == pytest.approx(x0 + (m - m0) * per_metre, abs=0.2)
    marks = sorted(n["geometry"]["x"] for n in _beam_lines(doc) if n["geometry"]["width"] == 0)
    for s, x in zip((0, 7, 12.5, 30), marks, strict=True):
        assert x == pytest.approx(x0 + s * per_metre, abs=0.2)


def test_a_dense_strip_breaks_into_rows_instead_of_shrinking(panel_graph) -> None:
    doc, _ = draw(invented_views()[0], panel_graph)
    axes = [n for n in _beam_lines(doc) if n["geometry"]["height"] == 0]
    assert len(axes) > 1
    for n in nodes(doc):
        thing = n["props"].get("tooltip", "")
        if thing.startswith("D") and n["type"] == "rectangle":
            kind = ("quadrupole_f", "quadrupole_d", "bend")[int(thing[1:]) % 3]
            natural = symbols.WIDTH_UNITS[kind] * SYMBOL_UNIT
            assert n["geometry"]["width"] >= natural * MIN_STRIP_SCALE - 0.1


def test_the_tray_is_a_dashed_box_of_its_entries_with_their_reason(panel_graph) -> None:
    view = place_view(panel_graph, "XR/XR02", symbols=SYMBOLS, above=ABOVE)
    (section,) = [s for s in view.sections if s.title == "strip"]
    (item,) = section.items
    assert item.tray
    doc, _ = draw(view, panel_graph)
    dashed = [n for n in _by_type(doc, "polygon") if n["props"].get("lineStyle") == "dashed"]
    assert dashed
    texts = _texts(doc)
    assert "not placed" in texts
    for entry in item.tray:
        assert entry.reason in texts


def test_a_conflicting_rack_unit_is_marked(panel_graph) -> None:
    view = rack_elevation(panel_graph, "XR/XR01/RK1")
    (elevation,) = view.sections[0].items
    conflicting = [u for u in elevation.units if u.conflicting]
    assert conflicting
    doc, _ = draw(view, panel_graph)
    assert _texts(doc).count(symbols.CONFLICT_MARK) == len(conflicting)
    assert "no unit" in _texts(doc)


def test_a_rack_unit_is_as_tall_as_its_size(panel_graph) -> None:
    view = rack_elevation(panel_graph, "XR/XR01/RK1")
    doc, _ = draw(view, panel_graph)
    heights = sorted({n["geometry"]["height"] for n in _by_type(doc, "rectangle")})
    assert layout.RACK_U in heights and 2 * layout.RACK_U in heights


def test_a_node_without_a_rule_reads_no_rule_not_unknown(panel_graph) -> None:
    view = health_rollup(panel_graph, "XR/XR01")
    (root,) = view.sections[0].items

    def all_nodes(node):
        yield node
        for child in node.children:
            yield from all_nodes(child)

    found = list(all_nodes(root))
    without = [n for n in found if n.no_rule]
    assert without and len(without) < len(found)
    doc, _ = draw(view, panel_graph)
    assert _texts(doc).count("no rule") == len(without)
    assert len(_by_type(doc, "polygon")) == len(found) - len(without)


def test_a_picture_part_is_a_marker_unless_its_row_names_a_channel(tmp_path) -> None:
    def describe_switch(document) -> None:
        for channel in document["channels"]:
            if channel["id"] == "XR01:QF1:Current-RB":
                channel["description"] = "Thermal switch 1"  # no matching by text

    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as rows:
        rows.write("XR01:QF1:Current-RB,1,bad,screen,stated,xr-screens,4d5e6f7,m.screen:1\n")
    graph = load_graph(write_fixture_facility(tmp_path / "f.json", describe_switch), tables)
    from osprey.facility.panels.describe import device_card

    view = device_card(graph, "XR01:QF1")
    (section,) = [s for s in view.sections if s.title == "picture"]
    (picture,) = section.items
    assert all(part.channel is None for part in picture.parts)
    doc, _ = draw(View(view.id, view.title, (), (section,)), graph)
    assert not _by_type(doc, "polygon")
    assert [n["props"]["tooltip"] for n in _by_type(doc, "ellipse")] == [
        "pole 1",
        "thermal switch 1",
    ]

    named = replace(picture, parts=(replace(picture.parts[1], channel="XR01:QF1:Current-RB"),))
    doc, _ = draw(View(view.id, view.title, (), (Section("picture", (named,)),)), graph)
    (shape,) = _by_type(doc, "polygon")
    assert shape["rules"][0]["pvs"] == [{"name": "XR01:QF1:Current-RB", "trigger": True}]
    assert not _by_type(doc, "ellipse")


# --------------------------------------------------------------------------- whole documents


def test_every_fixture_view_renders_with_unique_ids_and_pinned_widgets(panel_graph) -> None:
    views = fixture_views(panel_graph)
    assert {v.id.view for v in views} == {"card", "set", "place", "rack", "rollup", "checkout"}
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
        files = [n["props"]["file"] for n in _by_type(doc, "related-display-button")]
        files += [a["file"] for n in nodes(doc) for a in n["props"].get("actions", ())]
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
    hit |= set(needs(draw(card("long", Light("XR02:IP1:Pressure-RB")), panel_graph)[0]))
    assert FORMAT_REQUESTS
    # the label condition is hit in test_a_text_label_row_cannot_be_written...
    assert {name for name, _ in FORMAT_REQUESTS} - {"label condition"} == hit
    assert all(why for _, why in FORMAT_REQUESTS)


def test_the_document_is_a_screen_of_the_targets_id(panel_graph) -> None:
    doc, _ = draw(card("XR01:QF1"), panel_graph)
    assert doc["schemaVersion"] == "1.1" and doc["kind"] == "screen"
    assert doc["id"] == layout.screen_id(Target("card", "device", "XR01:QF1"))
    assert re.fullmatch(r"[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)*", doc["id"])
    rollup = Target("rollup", "place", "XR/XR01", by="system")
    assert layout.screen_id(rollup) != layout.screen_id(
        Target("rollup", "place", "XR/XR01", "place")
    )
    named = render(card("x"), graph=panel_graph, screens=Screens(), doc_id="a/b")
    assert named["id"] == "a/b"


def test_an_empty_strip_with_only_a_tray_renders(panel_graph) -> None:
    from osprey.facility.panels.view import TrayEntry

    doc, _ = draw(strip_view([], (TrayEntry("T", None, "no position", None),)), panel_graph)
    assert "no position" in _texts(doc)
    assert not _beam_lines(doc)


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
        render(view, graph=panel_graph, screens=Screens())


def test_a_node_light_reads_its_channels(panel_graph) -> None:
    node = Node("n", None, ("XR01:PS1:On-Sts", "XR02:IP1:Pressure-RB"), False)
    doc, _ = draw(card("n", node), panel_graph)
    (shape,) = _by_type(doc, "polygon")
    assert [p["name"] for p in shape["rules"][0]["pvs"]] == list(node.channels)


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
    assert texts.count("X") == layout.MAX_TABLE_ROWS
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
        rows.write("XR01:PS1:On-Sts,1,normal,screen,stated,xr-screens,4d5e6f7,ps.screen:1\n")
        rows.write("XR01:PS1:On-Sts,0,warn,screen,stated,xr-screens,4d5e6f7,ps.screen:2\n")
    graph = load_graph(write_fixture_facility(tmp_path / "facility.json"), tables)
    item = light(graph, "XR01:PS1:On-Sts")
    rule = ladder(graph, [item])
    assert rule.arms == (("{0} == 0", "warn"), ("{0} == 1", "normal"))
    for value in VALUES:
        assert evaluate(rule, [value]) == channel_state(graph, item.channel, value)


def test_a_one_channel_roll_up_node_over_the_limit_is_an_aggregate(panel_graph, monkeypatch):
    monkeypatch.setattr(layout, "RULE_EXPRESSION_LIMIT", 10)
    node = Node("one", None, ("XR02:IP1:Pressure-RB",), False)
    doc, _ = draw(card("one", node), panel_graph)
    assert aggregates(doc) == 1
    assert unexpressed(doc) == 0
    assert "1 channels, open for detail" in _texts(doc)


def test_an_unwritable_channel_leaves_its_siblings_arms_and_never_normal(tmp_path) -> None:
    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as rows:
        rows.write("XR01:PS1:On-Sts,ON,normal,screen,stated,xr-screens,4d5e6f7,ps.screen:1\n")
    graph = load_graph(write_fixture_facility(tmp_path / "facility.json"), tables)
    items = [light(graph, "XR01:PS1:On-Sts"), light(graph, "XR02:IP1:Pressure-RB")]
    assert ladder(graph, items[:1]) == "label condition"  # alone, its light is unknown
    rule = ladder(graph, items)
    assert not isinstance(rule, str)
    assert rule.unwritten == ("XR01:PS1:On-Sts",)
    assert all(state != "normal" for _, state in rule.arms)
    assert evaluate(rule, [1, 7]) == "bad"  # the sibling's bad still shows
    assert evaluate(rule, [1, 0.5]) == "warn"
    assert evaluate(rule, [1, 3]) == "unknown"  # the sibling is normal: unknown, not normal
    doc, _ = draw(card("node", Node("n", None, tuple(i.channel for i in items), False)), graph)
    assert unexpressed(doc) == 0
    assert "label condition" in needs(doc)
