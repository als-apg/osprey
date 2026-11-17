"""The device card's looks: two columns, a signals table, the fed-by chain, chips."""

from __future__ import annotations

from dataclasses import replace

from osprey.facility.panels import layout
from osprey.facility.panels.describe import device_card
from osprey.facility.panels.view import Section, ValueRow
from tests.facility.panels._views import nodes, stub_render


def _by_type(doc: dict, kind: str) -> list[dict]:
    return [n for n in nodes(doc) if n["type"] == kind]


def _text(doc: dict, text: str) -> dict:
    (node,) = [n for n in _by_type(doc, "text-label") if n["props"]["text"] == text]
    return node


def _fields(doc: dict) -> dict[str, dict]:
    return {n["props"]["pv"]: n for n in nodes(doc) if n["type"].startswith("pv-")}


def test_a_device_card_has_its_signals_left_and_its_relations_right(panel_graph) -> None:
    doc = stub_render(device_card(panel_graph, "XR01:QF1"), panel_graph)
    signals, fed = _text(doc, "SIGNALS"), _text(doc, "FED BY")
    assert signals["geometry"]["x"] == layout.MARGIN
    assert fed["geometry"]["x"] >= layout.MARGIN + layout.CARD_LEFT
    assert fed["geometry"]["y"] == signals["geometry"]["y"]  # side by side, from the top
    left_edge = layout.MARGIN + layout.CARD_LEFT
    for n in nodes(doc):  # nothing of the left column reaches into the right one
        g = n["geometry"]
        if g["x"] < left_edge and n["type"] in ("text-label", "pv-label", "pv-text-input"):
            assert g["x"] + g["width"] <= left_edge, n


def test_the_signals_are_a_table_with_heads_equal_fields_and_the_light_after(panel_graph):
    doc = stub_render(device_card(panel_graph, "XR01:QF1"), panel_graph)
    heads = [_text(doc, word) for word in ("SIGNAL", "SETPOINT", "READBACK")]
    assert len({h["geometry"]["y"] for h in heads}) == 1
    fields = _fields(doc)
    sp, rb = fields["XR01-QF1:Current-SP"], fields["XR01-QF1:Current-RB"]
    assert sp["geometry"]["x"] == heads[1]["geometry"]["x"]
    assert rb["geometry"]["x"] == heads[2]["geometry"]["x"]
    assert sp["geometry"]["width"] == rb["geometry"]["width"]
    assert sp["geometry"]["y"] == rb["geometry"]["y"]
    assert rb["props"]["align"] == "right"
    lights = [
        n
        for n in _by_type(doc, "polygon")
        if n["geometry"]["y"] <= rb["geometry"]["y"] + layout.ROW
        and n["geometry"]["y"] >= rb["geometry"]["y"]
        and n["geometry"]["x"] >= rb["geometry"]["x"] + rb["geometry"]["width"]
    ]
    assert lights  # the deviation light after the readback
    label = _text(doc, "current setpoint (A)")  # the unit with the signal's name
    assert label["geometry"]["x"] == layout.MARGIN


def test_a_readable_channel_sits_in_the_readback_column(panel_graph) -> None:
    doc = stub_render(device_card(panel_graph, "XR01:IP1"), panel_graph)
    rb = _fields(doc)["XR01-IP1:Pressure-RB"]
    assert rb["geometry"]["x"] == _text(doc, "READBACK")["geometry"]["x"]
    # its rule's light is after it, so no health row repeats the channel
    assert "XR01-IP1:State-Sts" not in [n["props"]["text"] for n in _by_type(doc, "text-label")]


def test_the_fed_by_chain_is_boxes_with_arrows_into_the_device(panel_graph) -> None:
    doc = stub_render(device_card(panel_graph, "XR01:QF1"), panel_graph)
    arrows = [n for n in _by_type(doc, "line") if n["props"].get("arrowEnd")]
    assert arrows
    feeder = [n for n in nodes(doc) if n["props"].get("tooltip") == "XR01:PS1"]
    assert feeder and feeder[0]["props"]["actions"]
    # the device's own box carries its short name (the header's chip is the other "F1")
    mine = [
        n
        for n in _by_type(doc, "text-label")
        if n["props"]["text"] == "F1" and n["geometry"]["x"] >= layout.MARGIN + layout.CARD_LEFT
    ]
    assert len(mine) == 1
    (arrow,) = arrows
    # the arrow points from the feeder (right) to the device (left)
    pts = arrow["props"]["points"]
    assert pts[-1]["x"] < pts[0]["x"]


def test_other_relations_are_chips_under_also(panel_graph) -> None:
    doc = stub_render(device_card(panel_graph, "XR01:IP1"), panel_graph)
    also = _text(doc, "ALSO")
    chip = _text(doc, "read out by: XR01:CH2")
    assert chip["geometry"]["y"] > also["geometry"]["y"]
    assert chip["geometry"]["x"] >= layout.MARGIN + layout.CARD_LEFT


def test_a_card_with_many_signals_draws_the_pairs_and_counts_the_rest_by_type(panel_graph):
    view = device_card(panel_graph, "XR01:QF1")
    (pair,) = view.sections[0].items
    many = [
        ValueRow(f"bit {i}", "XR01-QF1:Current-RB", None, None, False, drawn_as="state")
        for i in range(layout.CARD_ROWS + 5)
    ]
    signals = Section("signals", (pair, *many))
    doc = stub_render(replace(view, sections=(signals, *view.sections[1:])), panel_graph)
    texts = [n["props"]["text"] for n in _by_type(doc, "text-label")]
    assert "current setpoint (A)" in texts  # the pair first
    drawn = [t for t in texts if t.startswith("bit ")]
    assert len(drawn) == layout.CARD_ROWS - 1
    assert "6 more state channels not drawn" in texts


def test_a_card_with_many_health_lights_draws_the_limit_and_counts_the_rest(panel_graph):
    from osprey.facility.panels.view import Light

    view = device_card(panel_graph, "XR01:PS1")
    lights = tuple(Light("XR01-IP1:State-Sts") for _ in range(layout.CARD_ROWS + 3))
    sections = tuple(Section("health", lights) if s.title == "health" else s for s in view.sections)
    doc = stub_render(replace(view, sections=sections), panel_graph)
    texts = [n["props"]["text"] for n in _by_type(doc, "text-label")]
    assert "3 more not drawn" in texts


def test_a_row_named_by_its_raw_channel_shows_the_last_part_and_the_name_muted(
    panel_graph,
) -> None:
    from osprey.facility.panels.view import Target, View

    raw = ValueRow("XR01-QF1:Current-RB", "XR01-QF1:Current-RB", None, None, False)
    named = ValueRow("current readback", "XR01-QF1:Current-RB", None, None, False)
    view = View(
        Target("card", "device", "XR01:QF1"), "XR01:QF1", (),
        (Section("signals", (raw, named)),),
    )  # fmt: skip
    doc = stub_render(view, panel_graph)
    short = _text(doc, "Current-RB")
    full = _text(doc, "XR01-QF1:Current-RB")
    assert full["props"]["fontSize"] == layout.SMALL_SIZE
    assert full["props"]["foregroundColor"] == layout.MUTED_TEXT
    assert full["geometry"]["y"] >= short["geometry"]["y"] + layout.ROW
    assert _text(doc, "current readback")["geometry"]["y"] >= full["geometry"]["y"] + layout.ROW_GAP


def test_a_value_field_is_framed_once_by_the_viewer(panel_graph) -> None:
    doc = stub_render(device_card(panel_graph, "XR01:QF1"), panel_graph)
    for field in _fields(doc).values():
        if field["type"] in ("pv-label", "pv-text-input"):
            assert field["props"]["alarmSensitiveBorder"] is False
            assert "alarmBorder" not in field["props"]
