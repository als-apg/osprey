"""Live values as the viewer prints them: precision, one unit, and room on the strip."""

from __future__ import annotations

import math

import pytest

from osprey.facility.panels import layout
from osprey.facility.panels.describe import device_card, place_view
from osprey.facility.panels.view import StripElement
from tests.facility.panels._fixture import PANELS_DIR, copy_tables
from tests.facility.panels._views import SYMBOLS, nodes, strip_view, stub_render

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
    from tests.facility.panels._fixture import graph_with_bpms

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
    assert len(readouts) == 1
    room = layout.value_width("mm", layout.TEXT_SIZE)
    assert readouts[0]["geometry"]["width"] >= room


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
    doc = stub_render(place_view(panel_graph, "XR/XR01", symbols=SYMBOLS), panel_graph)
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
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import write_fixture_facility

    tables = copy_tables(tmp_path / "panels")
    path = tables / "metrics.csv"
    lines = path.read_text().splitlines()
    rows = [lines[0] + ",precision"] + [line + ("," if i != 2 else ",4") for i, line in
                                        enumerate(lines[1:], start=1)]  # fmt: skip
    path.write_text("\n".join(rows) + "\n")
    graph = load_graph(write_fixture_facility(tmp_path / "f.json"), tables)
    doc = stub_render(place_view(graph, "XR/XR01", symbols=SYMBOLS), graph)
    stated = [n for n in _band_numbers(doc) if n["props"]["precision"] == 4]
    assert len(stated) == 1 and not _rules(doc, stated[0])
    assert PANELS_DIR != tables


# -- the strip ---------------------------------------------------------------------


CROWD = 30  # pickups on one strip: more value blocks than two lanes hold


def _crowd(document) -> None:
    from tests.facility.panels._fixture import add_bpm

    for i in range(CROWD):
        add_bpm(document, f"XR01:BPM{i}", "XR/XR01")


def _crowded_strip(tmp_path):
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import write_fixture_facility

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


def _strip_parts(doc: dict) -> tuple[list[dict], list[dict]]:
    values = [n for n in nodes(doc) if n["type"] == "pv-label"]
    names = [n for n in nodes(doc) if n["type"] == "text-label" and n["props"]["text"][:1] == "P"]
    return values, names


def test_strip_values_are_whole_in_two_lanes_and_no_box_overlaps_another(tmp_path) -> None:
    _graph, doc = _crowded_strip(tmp_path)
    values, names = _strip_parts(doc)
    assert values and len(names) == CROWD  # a symbol with no room keeps its bare name
    room = layout.value_width("mm", layout.SMALL_SIZE)
    boxes = [_box(n) for n in values + names]
    for n in values:
        assert n["geometry"]["width"] >= room
        assert _rules(doc, n)
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1], (a, b)
    blocks = {n["geometry"]["y"] for n in names if n["props"]["text"] in _shown(values)}
    assert len(blocks) == layout.STRIP_VALUE_LANES  # close neighbours take the second lane
    left_out = 2 * CROWD - len(values)
    assert left_out > 0
    texts = [n["props"]["text"] for n in nodes(doc) if n["type"] == "text-label"]
    assert layout._CROWDED.format(left_out) in texts


def _shown(values: list[dict]) -> set[str]:
    return {f"P{v['props']['pv'].split(':')[1].removeprefix('BPM')}" for v in values}


def test_each_value_sits_under_its_own_name_and_a_lower_lane_has_a_leader(tmp_path) -> None:
    _graph, doc = _crowded_strip(tmp_path)
    values, names = _strip_parts(doc)
    lines = [n for n in nodes(doc) if n["type"] == "line"]
    top_lane = min(n["geometry"]["y"] for n in names if n["props"]["text"] in _shown(values))
    for shown in sorted(_shown(values)):
        i = shown.removeprefix("P")
        (name,) = [n for n in names if n["props"]["text"] == shown]
        x, y = (f"XR01:BPM{i}:position_{a}_readback" for a in "xy")
        (vx,) = [n for n in values if n["props"]["pv"] == x]
        (vy,) = [n for n in values if n["props"]["pv"] == y]
        centre = name["geometry"]["x"] + name["geometry"]["width"] / 2
        for value in (vx, vy):
            g = value["geometry"]
            assert abs(g["x"] + g["width"] / 2 - centre) < 1
        assert vx["geometry"]["y"] == name["geometry"]["y"] + layout.VALUE_STEP
        assert vy["geometry"]["y"] == vx["geometry"]["y"] + layout.VALUE_STEP
        if name["geometry"]["y"] > top_lane:  # the lower lane: a leader down to it
            g = name["geometry"]
            leaders = [
                n for n in lines
                if g["x"] <= n["geometry"]["x"] <= g["x"] + g["width"]
                and n["geometry"]["y"] + n["geometry"]["height"] == g["y"]
            ]  # fmt: skip
            assert leaders, shown


def test_value_lanes_keep_a_kind_in_its_lane_and_alternate_only_on_a_collision() -> None:
    spans = [(0, 50, "q"), (100, 50, "q"), (200, 50, "b"), (230, 50, "q"), (400, 50, "b")]
    assert layout.value_lanes(spans) == [0, 0, 0, 1, 0]
    crowded = [(0, 100, "q"), (10, 100, "q"), (20, 100, "q")]
    assert layout.value_lanes(crowded) == [0, 1, None]


def test_values_in_card_columns_have_room_for_the_number_and_its_unit(tmp_path) -> None:
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import write_fixture_facility

    def crowd_with_knobs(document) -> None:
        _crowd(document)
        for n in range(20):
            channel = {"id": f"XR01:BPM0:knob{n:02d}", "signal": f"knob_{n:02d}"}
            document["channels"].append({**channel, "on": {"device": "XR01:BPM0"}})

    graph = load_graph(write_fixture_facility(tmp_path / "f.json", crowd_with_knobs), PANELS_DIR)
    doc = stub_render(device_card(graph, "XR01:BPM0"), graph)
    texts = [n["props"]["text"] for n in nodes(doc) if n["type"] == "text-label"]
    assert texts.count("READBACK") >= 2  # in columns
    room = layout.value_width("mm", layout.TEXT_SIZE)
    for n in nodes(doc):
        if n["type"] == "pv-label" and n["props"]["pv"].startswith("XR01:BPM0:position"):
            assert n["geometry"]["width"] >= room


def test_every_formula_takes_min_and_max_over_one_tuple(bpm_graph, panel_graph) -> None:
    """The formula engine's min and max are reductions: two arguments fail there."""
    import re

    docs = [
        stub_render(device_card(bpm_graph, "XR01:BPM1"), bpm_graph),
        stub_render(place_view(panel_graph, "XR/XR01", symbols=SYMBOLS), panel_graph),
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
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import write_fixture_facility

    graph = load_graph(write_fixture_facility(tmp_path / "f.json", _units_device), PANELS_DIR)
    doc = stub_render(device_card(graph, "XR01:UNITS1"), graph)
    labels = {n["props"]["pv"]: n for n in nodes(doc) if n["type"] == "pv-label"}
    for i, unit in enumerate(LONG_UNITS):
        label = labels[f"XR01-UNITS1:V{i}-RB"]
        assert label["geometry"]["width"] >= layout.value_width(unit, label["props"]["fontSize"])
    drawn = [n for n in nodes(doc) if n["type"] in ("pv-label", "text-label")]
    for label in labels.values():
        others = [n for n in drawn if n is not label]
        assert not [n for n in others if _overlap(label, n)], label["props"]["pv"]
        assert label["geometry"]["x"] + label["geometry"]["width"] <= (
            layout.MARGIN + layout.CARD_LEFT - layout.COLUMN_GAP
        )


def test_a_key_number_with_a_long_unit_fits_its_tile_whole(tmp_path) -> None:
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import write_fixture_facility

    tables = copy_tables(tmp_path / "panels")
    with (tables / "metrics.csv").open("a") as handle:
        handle.write("place,,9,Long unit,value,units_2,,,one channel,XR-METRICS-001\n")
    graph = load_graph(write_fixture_facility(tmp_path / "f.json", _units_device), tables)
    doc = stub_render(place_view(graph, "XR/XR01", symbols=SYMBOLS), graph)
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
