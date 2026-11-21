"""The geometry every written screen holds to, read from the files the engine writes.

Nothing is rendered: text is measured in the viewer's font (:mod:`metrics`), as
the layout measures it. Every screen of the fixture facility is checked:

- a text label's text fits its box, in width and in line height, so the viewer
  never shrinks or clips it;
- no widget reaches outside the canvas, but the Apply buttons parked away by
  design (:data:`layout.APPLY_AWAY`);
- no two text labels draw over each other;
- the page never asks for a face the viewer's font set lacks: a plot carries no
  title of its own.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.panels import layout, load_graph, metrics, symbols
from osprey.facility.panels.emit import load_draw, write
from tests._by_path import load_module
from tests.facility.panels._fixture import PANELS_DIR, write_fixture_facility

#: The script that writes :mod:`metrics` from the viewer's font file.
FONT_METRICS = Path(__file__).parents[3] / "scripts" / "panel_font_metrics.py"

#: Below this many px of shared width or height, two texts only touch.
TOUCH = 2


def _nodes(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        if "type" in node and "geometry" in node:
            yield node
        for value in node.values():
            yield from _nodes(value)
    elif isinstance(node, list):
        for value in node:
            yield from _nodes(value)


def _ink(text: str, size: float) -> float:
    """The width of *text*'s letters at *size*, and the viewer's inset either side."""
    em = sum(metrics.WIDTHS.get(c, metrics.UNLISTED) for c in text)
    return em * size / metrics.PER_EM + 2 * layout.LABEL_INSET


@pytest.fixture(scope="module")
def screens(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    tmp = tmp_path_factory.mktemp("geometry")
    graph = load_graph(write_fixture_facility(tmp / "facility.json"), PANELS_DIR)
    write(graph, tmp / "out", draw=load_draw(PANELS_DIR / "draw.json"))
    return {
        str(p.relative_to(tmp / "out")): json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((tmp / "out").rglob("*.screen.json"))
    }


def _texts(doc: dict[str, Any]) -> list[tuple[str, str, float, float, float, float]]:
    """Each text label's id, text and the box its letters take: x, y, width, height."""
    found = []
    for node in _nodes(doc["root"]):
        props, g = node["props"], node["geometry"]
        if node["type"] != "text-label" or not str(props.get("text", "")).strip():
            continue
        size = props.get("fontSize", layout.TEXT_SIZE)
        w = min(_ink(props["text"], size), g["width"])
        align = props.get("align", "left")
        x = {"left": g["x"], "right": g["x"] + g["width"] - w}.get(
            align, g["x"] + (g["width"] - w) / 2
        )
        y = g["y"] + (g["height"] - size) / 2  # one line, centred in its box
        found.append((node["id"], props["text"], x, y, w, size))
    return found


def test_every_text_fits_its_box(screens):
    cut = [
        (name, node["id"], node["props"]["text"])
        for name, doc in screens.items()
        for node in _nodes(doc["root"])
        if node["type"] == "text-label"
        and node["props"].get("text")
        and _ink(node["props"]["text"], node["props"].get("fontSize", layout.TEXT_SIZE))
        > node["geometry"]["width"] + 0.01
    ]
    assert cut == []


def test_every_text_label_is_one_line_high(screens):
    low = [
        (name, node["props"]["text"], node["geometry"]["height"], node["props"]["fontSize"])
        for name, screen in screens.items()
        for node in _nodes(screen["root"])
        if node["type"] == "text-label"
        and node["geometry"]["height"] < layout.line_height(node["props"]["fontSize"])
    ]
    assert low == []


def test_a_lights_mark_gets_a_line_of_its_size():
    _shape, mark = symbols.light(10, 30, layout.LIGHT_SIZE, "unknown")
    drawn = layout._Screen("card-x", "x").add(mark)
    size = mark["props"]["fontSize"]
    assert drawn["geometry"]["height"] >= layout.line_height(size) > layout.LIGHT_SIZE
    # grown about the light's middle, so the mark stays centred in it
    middle = drawn["geometry"]["y"] + drawn["geometry"]["height"] / 2
    assert middle == pytest.approx(30 + layout.LIGHT_SIZE / 2, abs=0.5)
    # and every mark a light shows is measured and fits its width
    for glyph in (symbols.UNKNOWN_MARK, symbols.DISAGREE_MARK, symbols.AGGREGATE_MARK):
        assert glyph in metrics.WIDTHS
        ink = metrics.WIDTHS[glyph] * size / metrics.PER_EM + 2 * layout.LABEL_INSET
        assert ink <= drawn["geometry"]["width"]


def test_no_widget_reaches_outside_the_canvas(screens):
    outside = []
    for name, doc in screens.items():
        size = doc["metadata"]["size"]
        for node in _nodes(doc["root"]):
            g = node["geometry"]
            if node["type"] == "absolute-canvas" or g["x"] == layout.APPLY_AWAY:
                continue
            if (
                g["width"] < 0
                or g["height"] < 0
                or g["x"] < 0
                or g["y"] < 0
                or g["x"] + g["width"] > size["width"] + 0.01
                or g["y"] + g["height"] > size["height"] + 0.01
            ):
                outside.append((name, node["id"], node["type"]))
    assert outside == []


def test_no_text_lies_over_another(screens):
    over = []
    for name, doc in screens.items():
        texts = _texts(doc)
        for i, (a_id, a_text, ax, ay, aw, ah) in enumerate(texts):
            for b_id, b_text, bx, by, bw, bh in texts[i + 1 :]:
                if (
                    ax < bx + bw - TOUCH
                    and bx < ax + aw - TOUCH
                    and ay < by + bh - TOUCH
                    and by < ay + ah - TOUCH
                ):
                    over.append((name, a_id, a_text, b_id, b_text))
    assert over == []


def test_a_plot_carries_no_title_of_its_own(screens):
    titled = [
        (name, node["id"])
        for name, doc in screens.items()
        for node in _nodes(doc["root"])
        if node["type"] == "waveform-plot" and "title" in node["props"]
    ]
    assert titled == []


def test_the_metrics_measure_the_font_the_layout_names():
    assert metrics.FONT
    assert metrics.UNLISTED == max(metrics.WIDTHS.values())
    # every printable character a panel writes is measured, not guessed
    assert all(chr(c) in metrics.WIDTHS for c in range(0x20, 0x7F))
    assert layout.text_width("ab", 10) < layout.text_width("AB", 10)
    assert layout.text_width("ab", 10, caps=True) == layout.text_width("AB", 10)


def test_the_metrics_table_is_the_font_file_it_names():
    script = load_module("panel_font_metrics", FONT_METRICS)
    family, version, widths, line = script.measure()
    # regenerate the module from the font file: it is the module in the tree
    assert metrics.FONT == family
    assert metrics.WIDTHS == widths
    assert metrics.LINE == line
    assert f"from {family}, {version};" in (metrics.__doc__ or "")
    assert (script.FONT_FILE.parent / "OFL.txt").is_file()  # the font's licence beside it


def test_a_cut_text_fits_and_is_counted():
    cuts: list[str] = []
    token = layout._CUTS.set(cuts)
    try:
        text = "a long name that a narrow box cannot hold"
        cut = layout._cut(text, 80, layout.SMALL_SIZE)
    finally:
        layout._CUTS.reset(token)
    assert cut.endswith(layout._ELLIPSIS)
    assert layout.text_width(cut, layout.SMALL_SIZE) <= 80
    assert cuts == [text]


def test_a_checkout_s_name_columns_take_their_longest_names_and_stay_on_the_page():
    from osprey.facility.panels.view import CheckRow

    def row(channel: str, thing: str | None) -> CheckRow:
        return CheckRow(channel, thing, True, True, True, True)

    short = [row("XR01-PS1:On-Sts", "XR01:PS1")]
    assert layout.check_columns(short) == (layout.NAME_WIDTH, layout.NAME_WIDTH)
    long_name = "XR01-PS1:" + "VeryLongReadbackName" * 2 + "-RB"
    channel_w, thing_w = layout.check_columns([*short, row(long_name, "XR01:PS1")])
    assert channel_w >= layout.text_width(long_name, layout.TEXT_SIZE)  # whole, no cut
    assert thing_w == layout.NAME_WIDTH
    room = layout.STRIP_WIDTH - layout.CHECK_FLAGS * layout.CELL_WIDTH
    assert channel_w + thing_w <= room  # the flag columns still fit the page
    # both longer than the room: the thing column keeps its least, the channel the rest
    huge = "X" * 120
    channel_w, thing_w = layout.check_columns([row(huge, huge)])
    assert thing_w == layout.NAME_LEAST and channel_w + thing_w == room


def test_a_line_of_prose_takes_the_lines_it_needs_and_loses_no_text():
    text = (
        "source XR-0001-0002 B.5 (drawing) released in XR-0003-0004 A.12 feature 7, "
        "path XR-0005-0006/XR-0003-0004/XR-0001-0002; the rest of the note follows"
    )
    width = 240
    lines = layout._prose(text, width, layout.SMALL_SIZE)
    assert len(lines) > layout.TILE_LINES  # more than the old limit of two
    assert all(
        layout.text_width(line, layout.SMALL_SIZE) <= width + layout.TEXT_PAD for line in lines
    )
    assert not any(line.endswith("…") for line in lines)  # nothing cut
    # breaking at a space, a slash, a comma or a semicolon keeps every character
    assert "".join(lines).replace(" ", "") == text.replace(" ", "")
