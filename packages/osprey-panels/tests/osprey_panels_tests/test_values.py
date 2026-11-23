"""Live values as the viewer prints them: precision, one unit, and room on the strip."""

from __future__ import annotations

import itertools
import math

import pytest

from osprey_panels import layout
from osprey_panels.describe import device_card, place_view
from osprey_panels.view import StripElement
from osprey_panels_tests._fixture import PANELS_DIR, copy_tables
from osprey_panels_tests._views import SYMBOLS, nodes, strip_view, stub_render, with_symbols

# -- the precision rule ------------------------------------------------------------


def _decimals(value: float) -> int:
    """The precision expression evaluated as Python: its grammar is a subset."""
    expression = layout.precision_expression("{v}").replace("{v}", f"({value!r})")
    names = {"abs": abs, "max": max, "min": min, "floor": math.floor, "log10": math.log10}
    return eval(expression, {"__builtins__": {}}, names)


@pytest.mark.parametrize(
    ("value", "decimals", "shown"),
    [
        (1.9448, 2, "1.94"),
        (92.1234, 1, "92.1"),
        (-0.3127, 3, "-0.313"),
        (0.0, 3, "0.000"),
        (0.00123, 5, "0.00123"),
        (12345.6, 0, "12346"),
    ],
)
def test_the_precision_rule_gives_three_significant_digits(value, decimals, shown) -> None:
    assert _decimals(value) == decimals
    assert f"{value:.{decimals}f}" == shown
    assert len(shown) <= len(layout.VALUE_SAMPLE)


def test_a_value_too_small_for_fixed_digits_keeps_the_viewers_own_format() -> None:
    assert _decimals(3.2e-9) < 0  # outside 0..100: the viewer ignores it


def _rules(_doc: dict, node: dict) -> list[dict]:
    return [r for r in node.get("rules", []) if r["targetProperty"] == "precision"]


def _formula(doc: dict, rule: dict) -> dict:
    name = rule["formula"].removeprefix("formula://")
    (found,) = [f for f in doc["formulas"] if f["name"] == name]
    return found


# -- a card ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bpm_graph(tmp_path_factory):
    from osprey_panels_tests._fixture import graph_with_bpms

    return graph_with_bpms(tmp_path_factory.mktemp("bpm"))


def test_a_float_value_on_a_card_is_printed_to_three_significant_digits(bpm_graph) -> None:
    doc = stub_render(device_card(bpm_graph, "XR01:BPM1"), bpm_graph)
    channel = "XR01:BPM1:position_x_readback"
    labels = [n for n in nodes(doc) if n["type"] == "pv-label" and n["props"]["pv"] == channel]
    assert labels
    for label in labels:
        assert label["props"]["precision"] == layout.PRECISION_FALLBACK
        (rule,) = _rules(doc, label)
        formula = _formula(doc, rule)
        assert formula["bindings"] == {"v": channel}
        assert formula["expression"] == layout.precision_expression("{v}")


def test_a_setpoint_input_keeps_the_channels_own_precision(bpm_graph) -> None:
    doc = stub_render(device_card(bpm_graph, "XR01:BPM1"), bpm_graph)
    for n in nodes(doc):
        if n["type"] == "pv-text-input":
            assert "precision" not in n["props"] and not _rules(doc, n)


def test_a_pickup_readout_prints_its_unit_once(bpm_graph) -> None:
    doc = stub_render(device_card(bpm_graph, "XR01:BPM1"), bpm_graph)
    texts = [n["props"]["text"] for n in nodes(doc) if n["type"] == "text-label"]
    assert "mm" not in texts  # the value prints the channel's unit; no label beside it
    readouts = [
        n
        for n in nodes(doc)
        if n["type"] == "pv-label"
        and n["props"]["pv"] == "XR01:BPM1:position_x_readback"
        and n["props"].get("showUnits") is True
    ]
    assert readouts
    room = layout.value_width("mm", layout.TEXT_SIZE)
    assert all(n["geometry"]["width"] >= room for n in readouts)


# -- the band ----------------------------------------------------------------------


def _band_numbers(doc: dict) -> list[dict]:
    return [
        n
        for n in nodes(doc)
        if n["type"] == "pv-label" and n["props"]["fontSize"] == layout.BIG_SIZE
    ]


def test_a_key_number_is_printed_to_three_significant_digits_and_a_count_whole(
    panel_graph,
) -> None:
    doc = stub_render(place_view(with_symbols(panel_graph, SYMBOLS), "XR/XR01"), panel_graph)
    numbers = _band_numbers(doc)
    assert numbers
    for number in numbers:
        rules = _rules(doc, number)
        pv = number["props"]["pv"]
        if number["props"]["precision"] == 0:
            assert not rules  # a count
            continue
        assert number["props"]["precision"] == layout.PRECISION_FALLBACK
        (rule,) = rules
        formula = _formula(doc, rule)
        if pv.startswith("formula://"):
            (value,) = [f for f in doc["formulas"] if f["name"] == pv.removeprefix("formula://")]
            assert formula["bindings"] == value["bindings"]
            assert formula["expression"] == layout.precision_expression(f"({value['expression']})")
        else:
            assert formula["bindings"] == {"v": pv}
        room = layout.value_width(None, layout.BIG_SIZE)
        assert number["geometry"]["width"] >= room


def test_a_stated_precision_wins_and_needs_no_rule(tmp_path) -> None:
    from osprey_panels import load_graph
    from osprey_panels_tests._fixture import write_fixture_facility

    tables = copy_tables(tmp_path / "panels")
    path = tables / "metrics.csv"
    lines = path.read_text().splitlines()
    rows = [lines[0] + ",precision"] + [line + ("," if i != 2 else ",4") for i, line in
                                        enumerate(lines[1:], start=1)]  # fmt: skip
    path.write_text("\n".join(rows) + "\n")
    graph = load_graph(write_fixture_facility(tmp_path / "f.json"), tables)
    doc = stub_render(place_view(with_symbols(graph, SYMBOLS), "XR/XR01"), graph)
    numbers = [n for n in nodes(doc) if n["type"] == "pv-label"]
    stated = [n for n in numbers if n["props"]["precision"] == 4]
    assert len(stated) == 1 and not _rules(doc, stated[0])
    assert PANELS_DIR != tables


# -- the strip ---------------------------------------------------------------------


CROWD = 30  # pickups on one strip: more value blocks than two lanes hold


def _crowd(document) -> None:
    from osprey_panels_tests._fixture import add_bpm

    for i in range(CROWD):
        add_bpm(document, f"XR01:BPM{i}", "XR/XR01")


def _crowded_strip(tmp_path):
    from osprey_panels import load_graph
    from osprey_panels_tests._fixture import write_fixture_facility

    graph = load_graph(write_fixture_facility(tmp_path / "f.json", _crowd), PANELS_DIR)
    elements = []
    for i in range(CROWD):
        channels = (f"XR01:BPM{i}:position_x_readback", f"XR01:BPM{i}:position_y_readback")
        annotations = tuple((c, "mm") for c in channels)
        elements.append(
            StripElement(
                f"XR01:BPM{i}", "BPM", i * 0.05, "bpm", "beam", None, (), f"P{i}", annotations
            )  # fmt: skip
        )
    return graph, stub_render(strip_view(elements), graph)


def _box(n: dict) -> tuple[float, float, float, float]:
    g = n["geometry"]
    return g["x"], g["y"], g["x"] + g["width"], g["y"] + g["height"]


def _annotations(doc: dict) -> dict[str, dict]:
    """The strip's values by channel: static labels whose text a formula rule sets."""
    formulas = {f["name"]: f for f in doc.get("formulas", ())}
    found = {}
    for n in nodes(doc):
        for rule in n.get("rules", ()):
            if n["type"] == "text-label" and rule["targetProperty"] == "text" and "formula" in rule:
                formula = formulas[rule["formula"].removeprefix("formula://")]
                found[formula["bindings"]["v"]] = n
    return found


def _annotation_labels(doc: dict) -> list[dict]:
    return list(_annotations(doc).values())


def _strip_parts(doc: dict) -> tuple[dict[str, dict], list[dict]]:
    values = _annotations(doc)
    names = [
        n for n in nodes(doc)
        if n["type"] == "text-label" and n["props"]["text"][:1] == "P" and not n.get("rules")
    ]  # fmt: skip
    return values, names


def test_strip_values_are_whole_in_two_lanes_and_no_box_overlaps_another(tmp_path) -> None:
    _graph, doc = _crowded_strip(tmp_path)
    values, names = _strip_parts(doc)
    assert values and len(names) == CROWD  # a symbol with no room keeps its bare name
    room = layout.text_width(layout.VALUE_SAMPLE, layout.SMALL_SIZE)  # no unit appended
    boxes = [_box(n) for n in [*values.values(), *names]]
    for n in values.values():
        assert n["geometry"]["width"] >= room
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1], (a, b)
    blocks = {n["geometry"]["y"] for n in names if n["props"]["text"] in _shown(values)}
    assert len(blocks) == layout.STRIP_VALUE_LANES  # close neighbours take the second lane
    left_out = 2 * CROWD - len(values)
    assert left_out > 0
    texts = [n["props"]["text"] for n in nodes(doc) if n["type"] == "text-label"]
    assert layout.not_shown(left_out, layout._CROWDED) in texts


def _shown(values: dict[str, dict]) -> set[str]:
    return {f"P{channel.split(':')[1].removeprefix('BPM')}" for channel in values}


def test_each_value_sits_under_its_own_name_and_a_lower_lane_has_a_leader(tmp_path) -> None:
    _graph, doc = _crowded_strip(tmp_path)
    values, names = _strip_parts(doc)
    lines = [n for n in nodes(doc) if n["type"] == "line"]
    top_lane = min(n["geometry"]["y"] for n in names if n["props"]["text"] in _shown(values))
    for shown in sorted(_shown(values)):
        i = shown.removeprefix("P")
        (name,) = [n for n in names if n["props"]["text"] == shown]
        x, y = (f"XR01:BPM{i}:position_{a}_readback" for a in "xy")
        vx, vy = values[x], values[y]
        centre = name["geometry"]["x"] + name["geometry"]["width"] / 2
        for value in (vx, vy):
            g = value["geometry"]
            assert abs(g["x"] + g["width"] / 2 - centre) < 1
        assert vx["geometry"]["y"] == name["geometry"]["y"] + layout.VALUE_STEP
        assert vy["geometry"]["y"] == vx["geometry"]["y"] + layout.VALUE_STEP
        if name["geometry"]["y"] > top_lane:  # the lower lane: a leader down to it
            g = name["geometry"]
            ends = [_points(n)[-1] for n in lines]
            assert (centre, g["y"]) in [
                (pytest.approx(x, abs=0.5), pytest.approx(y, abs=0.5)) for x, y in ends
            ]  # on whole pixels


def _points(line: dict) -> list[tuple[float, float]]:
    """A line's points on the screen."""
    g = line["geometry"]
    return [(g["x"] + p["x"], g["y"] + p["y"]) for p in line["props"]["points"]]


def _crosses(a: tuple[float, float], b: tuple[float, float], box) -> bool:
    """Whether the axis-parallel segment *a*-*b* runs inside *box* (its edge is outside)."""
    left, top, right, bottom = box
    x0, x1 = sorted((a[0], b[0]))
    y0, y1 = sorted((a[1], b[1]))
    return x0 < right and x1 > left and y0 < bottom and y1 > top


def test_no_leader_runs_through_a_block_of_the_lane_above(tmp_path) -> None:
    _graph, doc = _crowded_strip(tmp_path)
    values, names = _strip_parts(doc)
    shown = _shown(values)
    blocks = [_box(n) for n in values.values()] + [
        _box(n) for n in names if n["props"]["text"] in shown
    ]
    lane_two = min(n["geometry"]["y"] for n in names if n["props"]["text"] in shown)
    leaders = [
        n for n in nodes(doc)
        if n["type"] == "line" and n["geometry"]["y"] + n["geometry"]["height"] > lane_two
    ]  # fmt: skip
    assert leaders
    for leader in leaders:
        points = _points(leader)
        for a, b in itertools.pairwise(points):
            assert not [box for box in blocks if _crosses(a, b, box)], points


def test_value_lanes_keep_a_kind_in_its_lane_and_alternate_only_on_a_collision() -> None:
    spans = [(0, 50, "q"), (100, 50, "q"), (200, 50, "b"), (230, 50, "q"), (400, 50, "b")]
    assert layout.value_lanes(spans) == [0, 0, 0, 1, 0]
    crowded = [(0, 100, "q"), (10, 100, "q"), (20, 100, "q")]
    assert layout.value_lanes(crowded) == [0, 1, None]


def test_values_in_card_columns_have_room_for_the_number_and_its_unit(tmp_path) -> None:
    from osprey_panels import load_graph
    from osprey_panels_tests._fixture import write_fixture_facility

    def crowd_with_knobs(document) -> None:
        _crowd(document)
        for n in range(20):
            channel = {"id": f"XR01:BPM0:knob{n:02d}", "signal": f"knob_{n:02d}"}
            document["channels"].append({**channel, "on": {"device": "XR01:BPM0"}})

    graph = load_graph(write_fixture_facility(tmp_path / "f.json", crowd_with_knobs), PANELS_DIR)
    doc = stub_render(device_card(graph, "XR01:BPM0"), graph)
    texts = [n["props"]["text"] for n in nodes(doc) if n["type"] == "text-label"]
    assert any("· ALL CHANNELS" in t for t in texts)  # in the grid's columns
    room = layout.value_width("mm", layout.TEXT_SIZE)
    for n in nodes(doc):
        if n["type"] == "pv-label" and n["props"]["pv"].startswith("XR01:BPM0:position"):
            assert n["geometry"]["width"] >= room


def test_every_formula_takes_min_and_max_over_one_tuple(bpm_graph, panel_graph) -> None:
    """The formula engine's min and max are reductions: two arguments fail there."""
    import re

    docs = [
        stub_render(device_card(bpm_graph, "XR01:BPM1"), bpm_graph),
        stub_render(place_view(with_symbols(panel_graph, SYMBOLS), "XR/XR01"), panel_graph),
    ]
    expressions = [f["expression"] for doc in docs for f in doc["formulas"]]
    assert any("min(" in e for e in expressions) and any("max(" in e for e in expressions)
    for expression in expressions:
        assert not re.search(r"\b(min|max)\((?!\()", expression), expression


# -- whole units, whole states -----------------------------------------------------

LONG_UNITS = ("1/Meter^3", "Radian", "mAmps", "Torr", "Ampere or Counts?")


def _units_device(document) -> None:
    document["devices"].append({"id": "XR01:UNITS1", "class": "Quadrupole", "place": "XR/XR01"})
    for i, unit in enumerate(LONG_UNITS):
        document["channels"].append(
            {"id": f"XR01-UNITS1:V{i}-RB", "signal": f"units_{i}", "unit": unit,
             "on": {"device": "XR01:UNITS1"}}
        )  # fmt: skip


def _overlap(a: dict, b: dict) -> bool:
    (ax, ay, ar, ab), (bx, by, br, bb) = _box(a), _box(b)
    return not (ar <= bx or br <= ax or ab <= by or bb <= ay)


def test_a_value_box_holds_the_number_and_the_whole_unit(tmp_path) -> None:
    from osprey_panels import load_graph
    from osprey_panels_tests._fixture import write_fixture_facility

    graph = load_graph(write_fixture_facility(tmp_path / "f.json", _units_device), PANELS_DIR)
    doc = stub_render(device_card(graph, "XR01:UNITS1"), graph)
    labels = {n["props"]["pv"]: n for n in nodes(doc) if n["type"] == "pv-label"}
    for i, unit in enumerate(LONG_UNITS):
        if len(unit) > layout.CHANNEL_ACCESS_UNIT_CHARS:
            continue  # a static number and unit: the next test
        label = labels[f"XR01-UNITS1:V{i}-RB"]
        assert label["geometry"]["width"] >= layout.value_width(unit, label["props"]["fontSize"])
    drawn = [n for n in nodes(doc) if n["type"] in ("pv-label", "text-label")]
    for label in labels.values():
        others = [n for n in drawn if n is not label]
        assert not [n for n in others if _overlap(label, n)], label["props"]["pv"]
        row = _row_of(doc, label["props"]["pv"])
        assert label["geometry"]["x"] + label["geometry"]["width"] <= _box(row)[2]


def test_a_key_number_with_a_long_unit_fits_its_tile_whole(tmp_path) -> None:
    from osprey_panels import load_graph
    from osprey_panels_tests._fixture import write_fixture_facility

    tables = copy_tables(tmp_path / "panels")
    with (tables / "metrics.csv").open("a") as handle:
        handle.write("place,,9,Long unit,value,units_2,,,one channel,XR-METRICS-001\n")
    graph = load_graph(write_fixture_facility(tmp_path / "f.json", _units_device), tables)
    doc = stub_render(place_view(with_symbols(graph, SYMBOLS), "XR/XR01"), graph)
    (number,) = [
        n for n in nodes(doc) if n["type"] == "pv-label" and n["props"]["pv"] == "XR01-UNITS1:V2-RB"
    ]
    size = number["props"]["fontSize"]
    assert number["geometry"]["width"] >= layout.value_width("mAmps", size)
    assert size >= layout.SMALL_SIZE
    frames = [
        n for n in nodes(doc)
        if n["type"] == "rectangle"
        and n.get("meta", {}).get("comment", "").startswith("tile: Long unit")
    ]  # fmt: skip
    (frame,) = frames
    assert number["geometry"]["x"] + number["geometry"]["width"] <= (
        frame["geometry"]["x"] + frame["geometry"]["width"]
    )


def _row_of(doc: dict, channel: str) -> dict:
    """The grid row of *channel*: the box that shows its full name on hover."""
    (row,) = [
        n for n in nodes(doc) if n["type"] == "rectangle" and n["props"].get("tooltip") == channel
    ]
    return row


def _unit_beside(doc: dict, number: dict, unit: str) -> dict:
    """The static label of *unit* on *number*'s row, right of it."""
    x, top, right, bottom = _box(number)
    found = [
        n for n in nodes(doc)
        if n["type"] == "text-label" and n["props"]["text"] == unit and not n.get("rules")
        and n["geometry"]["x"] >= right and n["geometry"]["y"] < bottom
        and n["geometry"]["y"] + n["geometry"]["height"] > top
    ]  # fmt: skip
    assert len(found) == 1, unit
    return found[0]


def test_the_channel_access_unit_limit_is_a_named_constant() -> None:
    assert layout.CHANNEL_ACCESS_UNIT_CHARS == 7


def test_a_unit_longer_than_channel_access_carries_is_printed_whole_beside_a_formula_number(
    tmp_path,
) -> None:
    from osprey_panels import load_graph
    from osprey_panels_tests._fixture import write_fixture_facility

    graph = load_graph(write_fixture_facility(tmp_path / "f.json", _units_device), PANELS_DIR)
    doc = stub_render(device_card(graph, "XR01:UNITS1"), graph)
    gauges = {n["props"]["pv"] for n in nodes(doc) if n["type"] == "pv-label"}
    numbers = _annotations(doc)
    drawn = [n for n in nodes(doc) if n["type"] in ("pv-label", "text-label")]
    long = [(i, u) for i, u in enumerate(LONG_UNITS) if len(u) > layout.CHANNEL_ACCESS_UNIT_CHARS]
    assert long
    for i, unit in long:
        channel = f"XR01-UNITS1:V{i}-RB"
        assert channel not in gauges  # the viewer would append the unit cut
        number = numbers[channel]
        assert number["props"]["text"] == layout.NO_VALUE
        (rule,) = number["rules"]
        assert rule["default"] == layout.NO_VALUE
        assert _formula(doc, rule)["expression"] == layout.significant_expression("{v}")
        beside = _unit_beside(doc, number, unit)
        assert beside["geometry"]["width"] >= layout.text_width(unit, beside["props"]["fontSize"])
        assert (
            beside["geometry"]["x"] + beside["geometry"]["width"] <= _box(_row_of(doc, channel))[2]
        )
        for label in (number, beside):
            others = [n for n in drawn if n is not label]
            assert not [n for n in others if _overlap(label, n)], channel
    for i, unit in enumerate(LONG_UNITS):
        if len(unit) <= layout.CHANNEL_ACCESS_UNIT_CHARS:
            assert f"XR01-UNITS1:V{i}-RB" in gauges  # the unit arrives whole: pv-label


def test_a_key_number_with_a_unit_channel_access_cuts_prints_the_unit_whole_beside_it(
    tmp_path,
) -> None:
    from osprey_panels import load_graph
    from osprey_panels_tests._fixture import write_fixture_facility

    tables = copy_tables(tmp_path / "panels")
    with (tables / "metrics.csv").open("a") as handle:
        handle.write("place,,9,Cut unit,value,units_0,,,one channel,XR-METRICS-001\n")
    graph = load_graph(write_fixture_facility(tmp_path / "f.json", _units_device), tables)
    doc = stub_render(place_view(with_symbols(graph, SYMBOLS), "XR/XR01"), graph)
    channel = "XR01-UNITS1:V0-RB"
    assert not [n for n in nodes(doc) if n["type"] == "pv-label" and n["props"]["pv"] == channel]
    number = _annotations(doc)[channel]
    assert number["props"]["text"] == layout.NO_VALUE
    beside = _unit_beside(doc, number, LONG_UNITS[0])
    (frame,) = [
        n for n in nodes(doc)
        if n["type"] == "rectangle"
        and n.get("meta", {}).get("comment", "").startswith("tile: Cut unit")
    ]  # fmt: skip
    assert beside["geometry"]["x"] + beside["geometry"]["width"] <= (
        frame["geometry"]["x"] + frame["geometry"]["width"]
    )


def test_a_state_the_health_rules_name_by_whole_values_prints_whole(panel_graph) -> None:
    doc = stub_render(device_card(panel_graph, "XR01:IP1"), panel_graph)
    states = [
        n for n in nodes(doc)
        if n["type"] == "pv-label" and n["props"]["pv"] == "XR01-IP1:State-Sts"
    ]  # fmt: skip
    assert states
    for state in states:
        assert state["props"]["precision"] == 0 and not _rules(doc, state)


def test_no_value_field_on_a_card_overlaps_another(bpm_graph, panel_graph) -> None:
    for graph, device in ((bpm_graph, "XR01:BPM1"), (panel_graph, "XR01:QF1")):
        doc = stub_render(device_card(graph, device), graph)
        fields = [n for n in nodes(doc) if n["type"] in ("pv-label", "pv-text-input")]
        for i, a in enumerate(fields):
            for b in fields[i + 1 :]:
                assert not _overlap(a, b), (a["props"]["pv"], b["props"]["pv"])


def test_a_channel_compared_by_equality_with_whole_values_is_a_state_a_threshold_is_not(
    tmp_path,
) -> None:
    from osprey_panels import load_graph
    from osprey_panels_tests._fixture import write_fixture_facility

    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as handle:
        for value, state in (("!=0", "normal"), ("0", "bad"), ("==1", "normal")):
            handle.write(f"XR01-PS1:On-Sts,{value},{state},screen,stated,xr-screens,4d5e6f7,ps:1\n")
    graph = load_graph(write_fixture_facility(tmp_path / "f.json"), tables)
    doc = stub_render(device_card(graph, "XR01:PS1"), graph)
    shown = [
        n for n in nodes(doc) if n["type"] == "pv-label" and n["props"]["pv"] == "XR01-PS1:On-Sts"
    ]
    assert shown and all(on["props"]["precision"] == 0 for on in shown)
    doc = stub_render(device_card(graph, "XR02:IP1"), graph)
    pressures = [
        n for n in nodes(doc)
        if n["type"] == "pv-label" and n["props"]["pv"] == "XR02-IP1:Pressure-RB"
    ]  # fmt: skip
    assert pressures and all(_rules(doc, n) for n in pressures)  # >=5 and <1 are thresholds


def _compact(value: float) -> int:
    expression = layout.precision_expression("{v}", compact=True).replace("{v}", f"({value!r})")
    names = {"abs": abs, "max": max, "floor": math.floor, "log10": math.log10}
    return eval(expression, {"__builtins__": {}}, names)  # the engine's own text, no builtins


def test_a_long_value_takes_the_compact_rule_that_names_it_once(bpm_graph, monkeypatch) -> None:
    full, compact = (layout.precision_expression("{v}", compact=c) for c in (False, True))
    assert compact.count("{v}") == 1 and len(compact) < len(full)
    assert [_compact(v) for v in (1.299, 0.0, 0.05, 123.4)] == [2, 3, 3, 0]
    monkeypatch.setattr(layout, "RULE_EXPRESSION_LIMIT", len(compact))
    doc = stub_render(device_card(bpm_graph, "XR01:BPM1"), bpm_graph)
    precision = [f for f in doc["formulas"] if f["name"].startswith("precision-")]
    assert precision and all(f["expression"] == compact for f in precision)


# -- the strip's annotations: a static label, its text a formula ---------------------


def _rounded(value: float) -> float:
    """The annotation formula evaluated as Python with the formula engine's functions."""
    expression = layout.significant_expression("{v}").replace("{v}", f"({value!r})")
    names = {"abs": abs, "floor": math.floor, "log10": math.log10, "pow": pow}
    return eval(expression, {"__builtins__": {}}, names)  # the engine's own text, no builtins


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        (1.9448, 1.94),
        (92.1234, 92.1),
        (-0.3127, -0.313),
        (0.0, 0.0),
        (0.00123456, 0.00123),
        (12345.6, 12300.0),
        (3.2179e-9, 3.22e-9),
    ],
)
def test_an_annotation_formula_rounds_to_three_significant_digits(value, shown) -> None:
    assert _rounded(value) == pytest.approx(shown, rel=1e-12)
    assert "round(" not in layout.significant_expression("{v}")  # the server has no round


def test_a_strip_annotation_is_a_static_label_whose_text_a_formula_sets(tmp_path) -> None:
    _graph, doc = _crowded_strip(tmp_path)
    assert not [n for n in nodes(doc) if n["type"] == "pv-label"]  # no unit appended
    values = _annotation_labels(doc)
    assert values
    for label in values:
        assert label["props"]["text"] == layout.NO_VALUE  # until it delivers, or disconnected
        (rule,) = label["rules"]
        assert rule["targetProperty"] == "text"
        formula = _formula(doc, rule)
        assert formula["expression"] == layout.significant_expression("{v}")
        assert set(formula["bindings"]) == {"v"}
    texts = [n["props"]["text"] for n in nodes(doc) if n["type"] == "text-label"]
    assert any(t.endswith("values not shown: open the card") for t in texts)


def test_the_values_line_counts_one_value_in_the_singular() -> None:
    assert layout.not_shown(1, layout._CROWDED) == "1 value not shown: open the card"
    assert layout.not_shown(3, layout._CROWDED) == "3 values not shown: open the card"


def test_a_stated_precision_rounds_a_static_number_to_those_decimals() -> None:
    expression = layout.fixed_expression("{v}", 2).replace("{v}", "(1.23456)")
    names = {"floor": math.floor, "pow": pow}
    assert eval(expression, {"__builtins__": {}}, names) == pytest.approx(1.23)


# -- a value display as wide as what the facility states it can show ----------------

LONG_WORD = "Pending Off Standby Mode"


def _stated(document) -> None:
    for record in document["channels"]:
        if record["id"] == "XR01-PS1:On-Sts":
            record["options"] = ["Normal", LONG_WORD]
    document["limits"] = {
        "records": [
            {"address": "XR01-IP2:Pressure-RB", "min_value": -123456.0, "max_value": 1.0},
            {"address": "XR01-IP1:Pressure-RB", "min_value": -1.0, "max_value": 1.0},
        ]
    }


@pytest.fixture(scope="module")
def stated_graph(tmp_path_factory):
    from osprey_panels import load_graph
    from osprey_panels_tests._fixture import write_fixture_facility

    tmp = tmp_path_factory.mktemp("stated")
    return load_graph(write_fixture_facility(tmp / "facility.json", _stated), PANELS_DIR)


def _displays(graph, device: str, channel: str) -> list[dict]:
    from osprey_panels.describe import channel_list

    doc = stub_render(channel_list(graph, device), graph)
    return [n for n in nodes(doc) if n["type"] == "pv-label" and n["props"]["pv"] == channel]


def test_the_graph_reads_a_channels_options_and_limits(stated_graph) -> None:
    assert stated_graph.channel("XR01-PS1:On-Sts").options == ("Normal", LONG_WORD)
    assert stated_graph.channel("XR01-IP2:Pressure-RB").limits == (-123456.0, 1.0)
    assert stated_graph.channel("XR02-IP1:Pressure-RB").limits is None


def test_a_display_is_as_wide_as_its_longest_stated_word(stated_graph) -> None:
    displays = _displays(stated_graph, "XR01:PS1", "XR01-PS1:On-Sts")
    assert displays
    for display in displays:
        assert display["geometry"]["width"] >= layout.shown_width(LONG_WORD, layout.TEXT_SIZE)


def test_a_display_is_as_wide_as_its_limits_print_when_they_are_wider(stated_graph) -> None:
    (wide,) = _displays(stated_graph, "XR01:IP2", "XR01-IP2:Pressure-RB")
    assert wide["geometry"]["width"] >= layout.shown_width("-123456", layout.TEXT_SIZE)
    # limits narrower than the stand-in leave the stand-in's width, as with none
    (narrow,) = _displays(stated_graph, "XR01:IP1", "XR01-IP1:Pressure-RB")
    (none,) = _displays(stated_graph, "XR02:IP1", "XR02-IP1:Pressure-RB")
    assert narrow["geometry"]["width"] == none["geometry"]["width"]
    assert none["geometry"]["width"] == max(
        layout.VALUE_WIDTH, layout.shown_width(layout.VALUE_SAMPLE, layout.TEXT_SIZE)
    )
