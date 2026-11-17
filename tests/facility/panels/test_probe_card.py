"""The probe picture on a pickup's card: the drawing, its readouts and plots."""

from __future__ import annotations

import pytest

from osprey.facility.panels import layout
from osprey.facility.panels.describe import device_card
from osprey.facility.panels.view import Probe
from tests.facility.panels._fixture import copy_tables, graph_with_bpms
from tests.facility.panels._views import nodes, stub_render

PROBE = "probe_pictures.csv"
TITLE = "pickup, looking downstream"


@pytest.fixture(scope="module")
def bpm_graph(tmp_path_factory):
    return graph_with_bpms(tmp_path_factory.mktemp("bpm"))


# -- the card ----------------------------------------------------------------------


def _probe(graph, device: str) -> Probe:
    view = device_card(graph, device)
    (section,) = [s for s in view.sections if s.title == TITLE]
    (probe,) = section.items
    return probe


def test_the_card_holds_the_probe_before_its_calibration(bpm_graph) -> None:
    titles = [s.title for s in device_card(bpm_graph, "XR01:BPM1").sections]
    assert TITLE in titles
    probe = _probe(bpm_graph, "XR01:BPM1")
    channels = {p.element.element: p.channel for p in probe.parts}
    assert channels["A"] == "XR01:BPM1-E:P1:amplitude_a"  # the port device's own record
    assert channels["sum"] == "XR01:BPM1-E:sum_readback"
    marker = next(p for p in probe.parts if p.element.shape == "marker")
    assert (marker.channel, marker.channel_y) == (
        "XR01:BPM1:position_x_readback", "XR01:BPM1:position_y_readback"
    )  # fmt: skip
    assert not probe.inferred
    assert TITLE not in [s.title for s in device_card(bpm_graph, "XR01:QF1").sections]


def _doc(graph, device: str) -> dict:
    return stub_render(device_card(graph, device), graph)


def _block(doc: dict) -> list[dict]:
    (head,) = [
        n for n in nodes(doc) if n["type"] == "text-label" and n["props"]["text"] == TITLE.upper()
    ]
    return [n for n in nodes(doc) if n["geometry"]["y"] > head["geometry"]["y"]]


def test_the_drawing_is_to_scale_with_lettered_buttons_and_bars(bpm_graph) -> None:
    block = _block(_doc(bpm_graph, "XR01:BPM1"))
    bore = [n for n in block if n["type"] == "ellipse" and n["props"].get("tooltip") == "bore"]
    (bore,) = bore
    assert bore["geometry"]["width"] == bore["geometry"]["height"] == layout.PROBE_BORE
    buttons = [n for n in block if n["type"] == "ellipse" and n["props"].get("tooltip") in "ABCD"]
    assert len(buttons) == 4
    size = buttons[0]["geometry"]["width"]
    assert size == pytest.approx(layout.PROBE_BORE * 8 / (2 * 14.2), abs=0.1)
    a = next(n for n in buttons if n["props"]["tooltip"] == "A")
    centre_x = bore["geometry"]["x"] + layout.PROBE_BORE / 2
    centre_y = bore["geometry"]["y"] + layout.PROBE_BORE / 2
    assert a["geometry"]["x"] + size / 2 > centre_x and a["geometry"]["y"] + size / 2 < centre_y
    letters = {n["props"]["text"] for n in block if n["type"] == "text-label"}
    assert {"A", "B", "C", "D"} <= letters
    bars = {n["props"]["pv"]: n for n in block if n["type"] == "pv-progress-bar"}
    assert set(bars) == {f"XR01:BPM1-E:P1:amplitude_{c}" for c in "abcd"}
    assert (
        bars["XR01:BPM1-E:P1:amplitude_a"]["props"]["min"],
        bars["XR01:BPM1-E:P1:amplitude_a"]["props"]["max"],
    ) == (0, 32767)
    assert not [n for n in block if "button placement inferred" in str(n["props"].get("text"))]


def test_an_elliptical_bore_under_its_place(bpm_graph) -> None:
    block = _block(_doc(bpm_graph, "XR02:BPM1"))
    (bore,) = [n for n in block if n["type"] == "ellipse" and n["props"].get("tooltip") == "bore"]
    assert bore["geometry"]["width"] == layout.PROBE_BORE
    assert bore["geometry"]["height"] == pytest.approx(layout.PROBE_BORE * 16 / 60)


def test_the_beam_dot_follows_x_and_y_and_hides_while_either_is_unread(bpm_graph) -> None:
    doc = _doc(bpm_graph, "XR01:BPM1")
    (dot,) = [n for n in _block(doc) if n["props"].get("tooltip") == "beam"]
    assert dot["geometry"]["width"] == dot["geometry"]["height"] == 0  # hidden until read
    rules = {r["targetProperty"]: r for r in dot["rules"]}
    assert set(rules) == {"x", "y", "width", "height"}
    formulas = {f["name"]: f for f in doc["formulas"]}
    for prop, rule in rules.items():
        formula = formulas[rule["formula"].removeprefix("formula://")]
        assert set(formula["bindings"].values()) == {
            "XR01:BPM1:position_x_readback", "XR01:BPM1:position_y_readback"
        }, prop  # fmt: skip
    assert "min(" in formulas[rules["x"]["formula"].removeprefix("formula://")]["expression"]
    texts = {n["props"]["text"] for n in _block(doc) if n["type"] == "text-label"}
    assert "±2 mm" in texts


def test_readouts_and_plots_sit_with_the_drawing(bpm_graph) -> None:
    block = _block(_doc(bpm_graph, "XR01:BPM1"))
    labels = {n["props"]["pv"]: n for n in block if n["type"] == "pv-label"}
    for name in (
        "XR01:BPM1:position_x_readback",
        "XR01:BPM1:position_y_readback",
        "XR01:BPM1-E:sum_readback",
    ):
        assert labels[name]["props"]["fontSize"] == layout.TEXT_SIZE
    (plot,) = [n for n in block if n["type"] == "waveform-plot"]
    assert [c["y_channel"] for c in plot["props"]["curves"]] == [
        "XR01:BPM1:tbt_x",
        "XR01:BPM1:tbt_y",
    ]
    assert [c["name"] for c in plot["props"]["curves"]] == ["x", "y"]
    assert "1,000" in plot["props"]["title"]


def test_inferred_buttons_are_muted_with_a_note_and_an_empty_label_has_no_letter(tmp_path):
    directory = copy_tables(tmp_path / "panels")
    text = (directory / PROBE).read_text()
    text = text.replace("A,Released,sheet 1,\nBPM,B", "A,inferred,sheet 1,\nBPM,B")
    text = text.replace("BPM,D,anchor,D,", "BPM,D,anchor,,")
    (directory / PROBE).write_text(text)
    graph = graph_with_bpms(tmp_path, directory)
    assert _probe(graph, "XR01:BPM1").inferred
    block = _block(stub_render(device_card(graph, "XR01:BPM1"), graph))
    letters = {n["props"]["text"]: n for n in block if n["type"] == "text-label"}
    assert "D" not in letters
    from osprey.facility.panels import symbols

    assert letters["A"]["props"]["foregroundColor"] == symbols.MUTED_TEXT
    assert letters["B"]["props"]["foregroundColor"] == symbols.PAGE_TEXT
    assert "button placement inferred, not stated" in letters


def test_the_probe_block_keeps_the_looks_rules(bpm_graph) -> None:
    from itertools import combinations

    from tests.facility.panels.test_looks import ALLOWED, _box

    for device in ("XR01:BPM1", "XR02:BPM1"):
        doc = _doc(bpm_graph, device)
        block = _block(doc)
        texts = [_box(n) for n in block if n["type"] == "text-label" and n["props"]["text"]]
        for a, b in combinations(texts, 2):
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1], (a, b)
        size = doc["metadata"]["size"]
        for n in block:
            left, top, right, bottom = _box(n)
            assert left >= layout.MARGIN - layout.PANEL_PAD and top >= 0, n
            assert (
                right <= size["width"] - layout.MARGIN + layout.PANEL_PAD
                and bottom <= size["height"]
            ), n
            for (kind, prop), roles in ALLOWED.items():
                if n["type"] == kind and isinstance(n["props"].get(prop), dict):
                    assert n["props"][prop]["$role"] in roles, (kind, prop)


def test_on_a_pickup_card_the_picture_and_calibration_come_first(tmp_path) -> None:
    tables = copy_tables(tmp_path / "panels")
    with (tables / "calibration_kinds.csv").open("a") as handle:
        handle.write("gains,Gains,BPM,scalar,1,beam,,,amplitude_a,1,\n")
    graph = graph_with_bpms(tmp_path, tables)
    doc = stub_render(device_card(graph, "XR01:BPM1"), graph)
    heads = {
        n["props"]["text"]: n["geometry"]["y"]
        for n in nodes(doc)
        if n["type"] == "text-label"
        and n["props"]["text"] in (TITLE.upper(), "CALIBRATION", "SIGNALS")
    }
    assert heads[TITLE.upper()] < heads["CALIBRATION"] < heads["SIGNALS"]


def _many_signals(document) -> None:
    from tests.facility.panels._fixture import add_bpm

    add_bpm(document, "XR01:BPM1", "XR/XR01")
    for n in range(30):
        document["channels"].append(
            {
                "id": f"XR01:BPM1:knob{n:02d}",
                "signal": f"knob_{n:02d}",
                "on": {"device": "XR01:BPM1"},
            }
        )


def test_a_group_of_more_than_twelve_rows_fills_columns_across_the_card(tmp_path) -> None:
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import PANELS_DIR, write_fixture_facility

    graph = load_graph(write_fixture_facility(tmp_path / "f.json", _many_signals), PANELS_DIR)
    doc = stub_render(device_card(graph, "XR01:BPM1"), graph)
    values = [
        n for n in nodes(doc)
        if n["type"].startswith("pv-") and str(n["props"].get("pv", "")).startswith("XR01:BPM1:knob")
    ]  # fmt: skip
    xs = sorted({n["geometry"]["x"] for n in values})
    columns = int(
        (layout.STRIP_WIDTH + layout.COLUMN_GAP) // (layout.CARD_COLUMN_MIN + layout.COLUMN_GAP)
    )
    assert len(xs) == columns >= 2
    first = sorted(
        (n for n in values if n["geometry"]["x"] == xs[0]), key=lambda n: n["geometry"]["y"]
    )
    assert first[0]["props"]["pv"].endswith("knob00")  # filled column by column
    assert first[1]["props"]["pv"].endswith("knob01")
    right = max(n["geometry"]["x"] + n["geometry"]["width"] for n in values)
    assert right <= layout.MARGIN + layout.STRIP_WIDTH
    heads = [n for n in nodes(doc) if n["type"] == "text-label" and n["props"]["text"] == "SIGNALS"]
    assert len(heads) == 1 and heads[0]["geometry"]["y"] < min(n["geometry"]["y"] for n in values)
    texts = [n["props"]["text"] for n in nodes(doc) if n["type"] == "text-label"]
    assert texts.count("READBACK") == columns  # each column has its heads


def test_a_card_of_few_rows_keeps_its_two_columns(panel_graph) -> None:
    doc = stub_render(device_card(panel_graph, "XR01:QF1"), panel_graph)
    texts = [n["props"]["text"] for n in nodes(doc) if n["type"] == "text-label"]
    assert texts.count("READBACK") == 1


def test_with_every_left_group_in_columns_the_right_column_moves_over(tmp_path) -> None:
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import PANELS_DIR, write_fixture_facility

    graph = load_graph(write_fixture_facility(tmp_path / "f.json", _many_signals), PANELS_DIR)
    doc = stub_render(device_card(graph, "XR01:BPM1"), graph)
    (where,) = [
        n for n in nodes(doc) if n["type"] == "text-label" and n["props"]["text"] == "WHERE"
    ]
    assert where["geometry"]["x"] == layout.MARGIN


def _long_signals(document) -> None:
    from tests.facility.panels._fixture import add_bpm

    add_bpm(document, "XR01:BPM1", "XR/XR01")
    for n in range(14):
        document["channels"].append(
            {
                "id": f"XR01:BPM1:long{n}",
                "signal": f"button_{n}_amplitude_reference_average",
                "on": {"device": "XR01:BPM1"},
            }
        )


def test_a_long_signal_name_in_a_column_wraps_to_a_second_line_and_is_never_cut(
    tmp_path,
) -> None:
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import PANELS_DIR, write_fixture_facility

    graph = load_graph(write_fixture_facility(tmp_path / "f.json", _long_signals), PANELS_DIR)
    doc = stub_render(device_card(graph, "XR01:BPM1"), graph)
    labels = [n for n in nodes(doc) if n["type"] == "text-label"]
    named = [n for n in labels if n["props"]["text"].startswith("button")]
    assert named and not any("…" in n["props"]["text"] for n in named)
    for first in named:
        x, y = first["geometry"]["x"], first["geometry"]["y"]
        (second,) = [
            n for n in labels
            if n["geometry"]["x"] == x and n["geometry"]["y"] == y + layout.TITLE_LINE
        ]  # fmt: skip
        whole = f"{first['props']['text']} {second['props']['text']}"
        assert whole.startswith("button") and "average" in whole
    rows = sorted(
        (n for n in labels if n["geometry"]["x"] == named[0]["geometry"]["x"]),
        key=lambda n: n["geometry"]["y"],
    )
    tops = [n["geometry"]["y"] for n in rows]
    assert len(tops) == len(set(tops))  # nothing drawn over the second line
