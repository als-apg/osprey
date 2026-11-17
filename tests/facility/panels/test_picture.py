"""A device class's drawing on its card, from ``pictures.csv``: the body, the parts, the legend.

Every class, number and name here is invented.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from osprey.facility.panels import layout, symbols
from osprey.facility.panels.describe import device_card
from osprey.facility.panels.model import load_graph
from osprey.facility.panels.tables import COLUMNS, PanelTableError, read_tables
from tests.facility.panels._fixture import copy_tables, write_fixture_facility
from tests.facility.panels._views import nodes, stub_render

PICTURES = "pictures.csv"
HEADER = ",".join(COLUMNS[PICTURES])
CITE = "XR-DRAWING,B,released,sheet 2"
COLOUR = "#2f9e52"  # the fixture's quadrupole colour


def _row(part, kind, x, y, *, w="", h="", angle="", signal="", cls="Quadrupole") -> str:
    return f"{cls},end,{part},{kind},{x},{y},0,{CITE},{w},{h},{angle},{signal}"


def _six_poles() -> list[str]:
    """An invented six-pole body: a yoke, poles at 30 + 60 k degrees, a coil on each,
    two corrector windings, a bore."""
    rows = [_row("yoke", "yoke", 0, 0, w=300)]
    for k in range(6):
        angle = 30 + 60 * k
        t = math.radians(angle)
        rows.append(_row(f"pole {k + 1}", "pole", 60 * math.cos(t), 60 * math.sin(t), w=30, h=80, angle=angle))  # fmt: skip
        rows.append(_row(f"coil {k + 1}", "coil", 75 * math.cos(t), 75 * math.sin(t), w=12, h=40, angle=angle))  # fmt: skip
    rows.append(_row("corrector winding 1", "corrector_winding", 0, 120, w=4, h=30, angle=0))
    rows.append(_row("corrector winding 2", "corrector_winding", 0, -120, w=4, h=30, angle=0))
    rows.append(_row("bore", "bore", 0, 0, w=24))
    return rows


def _tables(tmp_path: Path, rows: list[str]) -> Path:
    tables = copy_tables(tmp_path / "panels")
    (tables / PICTURES).write_text("\n".join([HEADER, *rows]) + "\n")
    return tables


def _graph(tmp_path: Path, rows: list[str], rule: bool = False):
    tables = _tables(tmp_path, rows)
    if rule:
        with (tables / "health.csv").open("a") as handle:
            handle.write("XR01-QF1:Current-SP,1,bad,screen,stated,xr-screens,4d5e6f7,m.screen:1\n")
    return load_graph(write_fixture_facility(tmp_path / "f.json"), tables)


def _card(graph) -> dict:
    view = device_card(graph, "XR01:QF1", symbols={"Quadrupole": "quadrupole"})
    return stub_render(view, graph, colours={"quadrupole": COLOUR})


def _by_type(doc: dict, kind: str) -> list[dict]:
    return [n for n in nodes(doc) if n["type"] == kind]


def _tipped(doc: dict, tip: str) -> list[dict]:
    return [n for n in nodes(doc) if n["props"].get("tooltip") == tip]


def _texts(doc: dict) -> list[str]:
    return [n["props"]["text"] for n in _by_type(doc, "text-label")]


def _centre(node: dict) -> tuple[float, float]:
    g = node["geometry"]
    return g["x"] + g["width"] / 2, g["y"] + g["height"] / 2


# -- the table ---------------------------------------------------------------------


def test_the_old_eleven_column_file_still_reads(tmp_path) -> None:
    tables = copy_tables(tmp_path / "panels")
    header = "device_class,view,part,kind,x_mm,y_mm,z_mm,document,version,state,locator"
    (tables / PICTURES).write_text(f"{header}\nQuadrupole,end,pole 1,pole,1,2,3,d,v,s,l\n")
    (row,) = read_tables(tables).pictures
    assert (row.kind, row.w_mm, row.h_mm, row.angle_deg, row.signal) == ("pole", *[None] * 4)
    assert not row.shape  # a pole with no size: a part at a point, as before


def test_the_drawing_columns_read(tmp_path) -> None:
    rows = read_tables(_tables(tmp_path, _six_poles())).pictures
    pole = next(r for r in rows if r.part == "pole 2")
    assert (pole.w_mm, pole.h_mm, pole.angle_deg, pole.shape) == (30, 80, 90, True)
    assert next(r for r in rows if r.kind == "bore").h_mm is None


@pytest.mark.parametrize(
    ("row", "word"),
    [
        (_row("yoke", "yoke", 0, 0), "a yoke needs w_mm"),
        (_row("pole 1", "pole", 0, 50, w=0), "w_mm is 0"),
        (_row("coil 1", "coil", 0, 50, w=5, h=-1), "h_mm is -1"),
        (_row("pole 1", "pole", 0, 50, w="wide"), "not a number"),
    ],
)
def test_a_shape_row_states_a_positive_size(tmp_path, row: str, word: str) -> None:
    with pytest.raises(PanelTableError, match=word) as caught:
        read_tables(_tables(tmp_path, [row]))
    assert caught.value.file == PICTURES and caught.value.line == 2


# -- the signal --------------------------------------------------------------------


def test_a_signal_resolves_by_name_or_by_pattern_on_the_device(tmp_path) -> None:
    rows = [
        _row("switch 1", "thermal_switch", 50, 50, signal="current_readback"),
        _row("switch 2", "thermal_switch", -50, 50, signal="*:Current-SP"),
        _row("switch 3", "thermal_switch", -50, -50, signal="*:Nothing-*"),
        _row("switch 4", "thermal_switch", 50, -50),
    ]
    graph = _graph(tmp_path, rows)
    assert graph.picture_channel("XR01:QF1", "*:Current-SP") == "XR01-QF1:Current-SP"
    (section,) = [s for s in device_card(graph, "XR01:QF1").sections if s.title == "picture"]
    (picture,) = section.items
    assert [p.channel for p in picture.parts] == [
        "XR01-QF1:Current-RB",
        "XR01-QF1:Current-SP",
        None,
        None,
    ]


# -- the drawing -------------------------------------------------------------------


def test_six_poles_are_drawn_at_their_angles_each_with_its_coils_two_sides(tmp_path) -> None:
    doc = _card(_graph(tmp_path, _six_poles()))
    (yoke,) = _by_type_tip(doc, "ellipse", "yoke")
    cx, cy = _centre(yoke)
    for k in range(6):
        (pole,) = _tipped(doc, f"pole {k + 1}")
        assert pole["type"] == "polygon"
        px, py = _centre(pole)
        angle = math.degrees(math.atan2(cy - py, px - cx)) % 360
        assert angle == pytest.approx(30 + 60 * k, abs=1)
        sides = _tipped(doc, f"coil {k + 1}")
        assert len(sides) == 2 and all(s["type"] == "polygon" for s in sides)
        (ax, ay), (bx, by) = map(_centre, sides)
        # the two sides lie either side of the pole's axis, at the same reach
        assert math.dist((ax, ay), (cx, cy)) == pytest.approx(math.dist((bx, by), (cx, cy)), 0.01)
    assert len(_tipped(doc, "corrector winding 1")) == 1
    (bore,) = _tipped(doc, "bore")
    (beam,) = _tipped(doc, "beam")
    assert _centre(beam) == pytest.approx(_centre(bore), abs=0.5)
    assert _centre(bore) == pytest.approx((cx, cy), abs=0.5)


def _by_type_tip(doc: dict, kind: str, tip: str) -> list[dict]:
    return [n for n in _tipped(doc, tip) if n["type"] == kind]


def test_the_body_is_in_the_class_colour_faint_and_state_is_the_only_other_colour(
    tmp_path,
) -> None:
    doc = _card(_graph(tmp_path, _six_poles()))
    yoke, pole, coil = (_tipped(doc, t)[0]["props"] for t in ("yoke", "pole 1", "coil 1"))
    assert yoke["fillColor"] == COLOUR + layout.BODY_ALPHA["yoke"]
    assert pole["fillColor"] == COLOUR + layout.BODY_ALPHA["pole"]
    assert coil["fillColor"] == COLOUR + layout.BODY_ALPHA["coil"]
    assert yoke["lineColor"] == COLOUR + layout.BODY_LINE_ALPHA
    winding = _tipped(doc, "corrector winding 1")[0]["props"]
    assert winding["lineColor"] == symbols.MUTED_TEXT and winding["fill"] is False
    assert "outline not stated" not in _texts(doc)


def test_a_body_with_no_class_colour_is_outlined_only(tmp_path) -> None:
    graph = _graph(tmp_path, _six_poles())
    doc = stub_render(device_card(graph, "XR01:QF1"), graph)
    pole = _tipped(doc, "pole 1")[0]["props"]
    assert pole["fill"] is False and pole["lineColor"] == symbols.MUTED_TEXT


def test_a_part_shows_its_state_where_it_sits_and_one_without_a_signal_is_plain(
    tmp_path,
) -> None:
    rows = [
        *_six_poles(),
        _row("switch 1", "thermal_switch", 90, 90, signal="*:Current-SP"),  # ruled: a light
        _row("gauge 1", "thermocouple", -90, 90, signal="current_readback"),  # a value
        _row("switch 2", "thermal_switch", -90, -90),  # no signal: plain
    ]
    doc = _card(_graph(tmp_path, rows, rule=True))
    top = next(n for n in _by_type(doc, "text-label") if n["props"]["text"] == "END VIEW")
    below = top["geometry"]["y"]  # on the drawing, not in the signals table

    def on_drawing(kind: str, tip_or_pv: str, key: str) -> list[dict]:
        return [
            n
            for n in _by_type(doc, kind)
            if n["props"].get(key) == tip_or_pv and n["geometry"]["y"] > below
        ]

    (light,) = on_drawing("polygon", "XR01-QF1:Current-SP", "tooltip")
    assert light.get("rules")
    (number,) = on_drawing("pv-label", "XR01-QF1:Current-RB", "pv")
    (gauge,) = _tipped(doc, "gauge 1")  # no rule: a plain ring, its number beside it
    gx, gy = _centre(gauge)
    nx, ny = _centre(number)
    assert abs(ny - gy) < 1 and abs(nx - gx) < layout.PART_VALUE_WIDTH + layout.LIGHT_SIZE
    (plain,) = _tipped(doc, "switch 2")
    assert plain["type"] == "ellipse" and not plain.get("rules")
    assert plain["props"]["fill"] is False


def test_the_picture_stands_in_the_cards_left_column_under_the_signals(tmp_path) -> None:
    doc = _card(_graph(tmp_path, _six_poles()))
    heads = {n["props"]["text"]: n["geometry"] for n in _by_type(doc, "text-label")}
    view, settings = heads["END VIEW"], heads["SIGNALS"]
    assert view["x"] == settings["x"] == layout.MARGIN
    assert view["y"] > settings["y"]
    drawn = [n for t in ("yoke", "pole 1", "coil 1", "bore") for n in _tipped(doc, t)]
    assert all(
        n["geometry"]["x"] + n["geometry"]["width"] <= layout.MARGIN + layout.CARD_LEFT
        for n in drawn
    )
    assert "PARTS" not in _texts(doc) and "PICTURE" not in _texts(doc)


def test_parts_with_no_body_are_drawn_in_a_frame_that_says_so_with_the_scale(
    tmp_path,
) -> None:
    rows = [
        _row("switch 1", "thermal_switch", 120, 40),
        _row("switch 2", "thermal_switch", -80, 40),
        _row("manifold 1", "manifold", 200, -60),
    ]
    doc = _card(_graph(tmp_path, rows))
    texts = _texts(doc)
    assert "outline not stated" in texts
    assert any(t.endswith(" mm") for t in texts)  # the scale bar's length
    assert [n for n in _by_type(doc, "polygon") if n["props"].get("lineStyle") == "dashed"]
    assert _tipped(doc, "beam axis")  # the frame's one reference
    for name in ("switch 1", "switch 2", "manifold 1"):
        assert name in texts and _tipped(doc, name)


def test_the_legend_lists_each_part_once_by_kind_with_its_source(tmp_path) -> None:
    rows = [
        *_six_poles(),
        _row("thermal switch 1", "thermal_switch", 90, 90),
        _row("thermal switch 2", "thermal_switch", -90, 90),
        _row("odd one", "thermal_switch", 0, -140),
    ]
    doc = _card(_graph(tmp_path, rows))
    texts = _texts(doc)
    assert "1, 2, odd one" in texts  # a numbered member by its number, the rest whole
    assert "1, 2, 3, 4, 5, 6" in texts  # the coils
    assert "1, 2" in texts  # the corrector windings
    for kind in ("thermal switch", "coil", "corrector winding"):
        assert texts.count(kind) == 1
    assert "pole" not in texts and "yoke" not in texts  # the iron is the drawing, not a part
    assert texts.count("source XR-DRAWING B released") == 1
    chips = [n for n in _by_type(doc, "polygon") if n["props"].get("tooltip") == "no rule"]
    assert not chips  # no second block of chips


def test_a_class_with_no_rows_has_no_picture(tmp_path) -> None:
    doc = _card(_graph(tmp_path, [_row("yoke", "yoke", 0, 0, w=10, cls="Dipole")]))
    assert "END VIEW" not in _texts(doc)


# -- check -------------------------------------------------------------------------


def test_check_notes_a_signal_that_resolves_to_no_channel_and_sorts_the_classes(
    tmp_path,
) -> None:
    from osprey.facility.panels.emit import check, picture_classes, write

    rows = [
        *_six_poles(),
        _row("switch 1", "thermal_switch", 90, 90, signal="ghost_status"),
        _row("switch 1", "thermal_switch", 9, 9, cls="Dipole"),
    ]
    graph = _graph(tmp_path, rows)
    out = tmp_path / "out"
    write(graph, out, only=lambda t: t.view == "card" and t.id == "XR01:QF1")
    found = [p for p in check(graph, out) if p.kind == "unbound_picture"]
    assert len(found) == 1 and not found[0].failing
    assert "ghost_status" in found[0].detail and "XR01:QF1" in found[0].detail
    classes = picture_classes(graph)
    assert classes["drawing"] == ["Quadrupole"]
    assert classes["parts only"] == ["Dipole"]
    assert "Pump" in classes["none"]
