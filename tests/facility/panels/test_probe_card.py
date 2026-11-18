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
    (picture,) = [
        n["geometry"]["y"] for n in nodes(doc)
        if n["type"] == "text-label" and n["props"]["text"] == TITLE.upper()
    ]  # fmt: skip
    channels = [
        n["geometry"]["y"] for n in nodes(doc)
        if n["type"] == "text-label" and "· ALL CHANNELS" in n["props"]["text"]
    ]  # fmt: skip
    assert picture < _head_y(doc, "CALIBRATION") < channels[0]


def _head_y(doc: dict, word: str) -> float:
    """The y of a drawn card's numbered head ending in *word*."""
    (y,) = [
        n["geometry"]["y"] for n in nodes(doc)
        if n["type"] == "text-label" and n["props"]["text"].endswith("· " + word)
    ]  # fmt: skip
    return y


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


def test_many_channels_fill_the_grid_column_by_column_inside_the_page(tmp_path) -> None:
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import PANELS_DIR, write_fixture_facility

    graph = load_graph(write_fixture_facility(tmp_path / "f.json", _many_signals), PANELS_DIR)
    doc = stub_render(device_card(graph, "XR01:BPM1"), graph)
    rows = [
        n for n in nodes(doc)
        if n["type"] == "rectangle" and str(n["props"].get("tooltip", "")).startswith("XR01:BPM1:knob")
    ]  # fmt: skip
    assert len(rows) == 30
    xs = sorted({n["geometry"]["x"] for n in rows})
    assert len(xs) == 4  # the default recipe's four columns
    first = sorted(
        (n for n in rows if n["geometry"]["x"] == xs[0]), key=lambda n: n["geometry"]["y"]
    )
    assert [n["props"]["tooltip"][-6:] for n in first[:2]] == ["knob00", "knob01"]
    right = max(n["geometry"]["x"] + n["geometry"]["width"] for n in nodes(doc) if n in rows)
    assert right <= layout.MARGIN + layout.STRIP_WIDTH


def test_a_card_of_few_rows_has_no_signal_table(panel_graph) -> None:
    doc = stub_render(device_card(panel_graph, "XR01:QF1"), panel_graph)
    texts = [n["props"]["text"] for n in nodes(doc) if n["type"] == "text-label"]
    assert "READBACK" not in texts and "SIGNALS" not in texts


def test_where_stands_full_width_after_the_channels(tmp_path) -> None:
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import PANELS_DIR, write_fixture_facility

    graph = load_graph(write_fixture_facility(tmp_path / "f.json", _many_signals), PANELS_DIR)
    doc = stub_render(device_card(graph, "XR01:BPM1"), graph)
    (where,) = [
        n for n in nodes(doc)
        if n["type"] == "text-label" and n["props"]["text"].startswith("Where:")
    ]  # fmt: skip
    assert where["geometry"]["x"] == layout.MARGIN
    channels = [
        n["geometry"]["y"] for n in nodes(doc)
        if n["type"] == "text-label" and "· ALL CHANNELS" in n["props"]["text"]
    ]  # fmt: skip
    assert where["geometry"]["y"] > channels[0]


def _long_signals(document) -> None:
    from tests.facility.panels._fixture import add_bpm

    add_bpm(document, "XR01:BPM1", "XR/XR01")
    for n in range(14):
        document["channels"].append(
            {
                "id": f"XR01:BPM1:long{n}",
                "signal": f"button_{n}_amplitude_reference_average_over_the_last_turns",
                "on": {"device": "XR01:BPM1"},
            }
        )


def test_a_long_channel_name_in_the_grid_is_cut_and_whole_on_hover(tmp_path) -> None:
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import PANELS_DIR, write_fixture_facility

    graph = load_graph(write_fixture_facility(tmp_path / "f.json", _raw_channels), PANELS_DIR)
    doc = stub_render(device_card(graph, "XR01:BPM1"), graph)
    for n in range(14):
        name = f"XR01:BPM1:LONGER:PART{n:02d}:segment:of:a:much:longer:name:tail{n:02d}"
        (hover,) = [m for m in nodes(doc) if m["props"].get("tooltip") == name]
        shown = [
            m for m in nodes(doc)
            if m["type"] == "text-label" and m["geometry"]["y"] == hover["geometry"]["y"]
            and m["props"]["text"].startswith(f"LONGER:PART{n:02d}")
        ]  # fmt: skip
        assert len(shown) == 1  # the short name, without the device's prefix
        assert shown[0]["geometry"]["x"] + shown[0]["geometry"]["width"] <= (
            hover["geometry"]["x"] + hover["geometry"]["width"]
        )


def _raw_channels(document) -> None:
    from tests.facility.panels._fixture import add_bpm

    add_bpm(document, "XR01:BPM1", "XR/XR01")
    for n in range(14):
        document["channels"].append(
            {
                "id": f"XR01:BPM1:LONGER:PART{n:02d}:segment:of:a:much:longer:name:tail{n:02d}",
                "on": {"device": "XR01:BPM1"},
            }
        )
