"""A device class's drawing on its card, from ``pictures.csv``: the body, the parts, the legend.

Every class, number and name here is invented.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from osprey.facility.panels import layout, symbols
from osprey.facility.panels.describe import device_card
from osprey.facility.panels.model import load_graph
from osprey.facility.panels.tables import COLUMNS, PanelTableError, read_tables
from tests.facility.panels._fixture import add_part, copy_tables, write_fixture_facility
from tests.facility.panels._views import bound, nodes, stub_render

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


def _graph(tmp_path: Path, rows: list[str], rule: bool = False, change=None):
    tables = _tables(tmp_path, rows)
    if rule:
        with (tables / "health.csv").open("a") as handle:
            handle.write("XR01-QF1:Current-SP,1,bad,screen,stated,xr-screens,4d5e6f7,m.screen:1\n")
    return load_graph(write_fixture_facility(tmp_path / "f.json", change), tables)


def _switches(document) -> None:
    """Two thermal-switch devices, each ``part_of`` the magnet, each with a status channel."""
    for n in (1, 2):
        add_part(document, f"XR01:QF1-TS{n}", "XR01:QF1", f"XR01-QF1-TS{n}:Open-Sts")


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
    yoke, _window = _by_type_tip(doc, "ellipse", "yoke")
    cx, cy = _centre(yoke)
    for k in range(6):
        (pole,) = _tipped(doc, f"pole {k + 1}")
        assert pole["type"] == "polygon"
        g, pts = pole["geometry"], pole["props"]["points"]
        px = g["x"] + sum(q["x"] for q in pts) / len(pts)
        py = g["y"] + sum(q["y"] for q in pts) / len(pts)
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


def test_the_body_is_painted_by_material_and_the_class_colour_is_not_used(tmp_path) -> None:
    doc = _card(_graph(tmp_path, _six_poles()))
    yoke, window = (n["props"] for n in _by_type_tip(doc, "ellipse", "yoke"))
    assert yoke["fillColor"] == symbols.STEEL_DARK
    assert window["fillColor"] == symbols.PANEL_BACKGROUND  # a ring: its inside is clear
    pole, coil = (_tipped(doc, t)[0]["props"] for t in ("pole 1", "coil 1"))
    assert pole["fillColor"] == symbols.STEEL and coil["fillColor"] == symbols.COPPER
    assert coil["lineColor"] == symbols.COPPER_EDGE and not _tipped(doc, "coil 1")[0].get("rules")
    assert _tipped(doc, "corrector winding 1")[0]["props"]["fillColor"] == symbols.WINDING
    bore = next(n for n in _tipped(doc, "bore") if n["type"] == "ellipse")
    assert bore["props"]["fillColor"] == symbols.FIELD_BACKGROUND
    assert "beam" in _texts(doc)
    assert COLOUR not in json.dumps(doc)  # material, not the class's strip colour


def test_a_coil_whose_row_reads_a_ruled_channel_has_a_state_outline(tmp_path) -> None:
    rows = [r.replace("coil 2,coil", "coil 2,coil") for r in _six_poles()]
    rows = [r + "*:Current-SP" if r.startswith("Quadrupole,end,coil 2,") else r for r in rows]
    doc = _card(_graph(tmp_path, rows, rule=True))
    sides = _tipped(doc, "coil 2")
    assert len(sides) == 2
    for side in sides:
        targets = {r["targetProperty"] for r in side.get("rules", ())}
        assert targets == {"lineColor", "lineWidth"}  # its fill stays copper
        assert bound(doc, side) == ("XR01-QF1:Current-SP",)
    assert not _tipped(doc, "coil 1")[0].get("rules")


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

    (light,) = on_drawing("polygon", "switch 1", "tooltip")
    assert light.get("rules") and bound(doc, light) == ("XR01-QF1:Current-SP",)
    (number,) = on_drawing("pv-label", "XR01-QF1:Current-RB", "pv")
    (gauge,) = _tipped(doc, "gauge 1")  # a plain dot, its number beside it, outward
    gx, gy = _centre(gauge)
    assert _centre(number)[1] == pytest.approx(gy, abs=1)
    assert number["geometry"]["x"] + number["geometry"]["width"] <= gx
    (plain,) = _tipped(doc, "switch 2")
    assert plain["type"] == "ellipse" and not plain.get("rules")
    assert plain["props"]["fillColor"] == symbols.MUTED_TEXT  # plain: no state colour
    for name in ("switch 1", "gauge 1", "switch 2"):
        assert name not in _texts(doc)  # no names on the drawing: the legend has them
    leaders = [
        n
        for n in _by_type(doc, "line")
        if n["geometry"]["y"] > below
        and n["geometry"]["x"] < layout.MARGIN + layout.CARD_LEFT
        and "tooltip" not in n["props"]
    ]
    assert not leaders


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
        assert _tipped(doc, name)  # at its position, named by its tooltip


def test_parts_off_the_body_are_blocks_with_one_word_per_group(tmp_path) -> None:
    rows = [
        *_six_poles(),  # a yoke of radius 150
        _row("manifold 1", "manifold", 180, 20),
        _row("manifold 2", "manifold", 180, -20),
        _row("manifold 3", "manifold", 180, -20),  # at manifold 2's point: one block
        _row("terminal 1", "terminal", -190, 0),
        _row("switch 1", "thermal_switch", 90, 90),  # on the body: a dot
    ]
    doc = _card(_graph(tmp_path, rows))
    blocks = _by_type(doc, "rectangle")
    tips = [n["props"].get("tooltip") for n in blocks]
    assert tips.count("manifold 1") == tips.count("manifold 2") == tips.count("terminal 1") == 1
    assert "manifold 3" not in tips
    (water,) = [n for n in blocks if n["props"].get("tooltip") == "manifold 1"]
    assert water["props"]["fillColor"] == symbols.WATER and water["props"]["cornerRadius"] > 0
    (steel,) = [n for n in blocks if n["props"].get("tooltip") == "terminal 1"]
    assert steel["props"]["fillColor"] == symbols.STEEL
    texts = _texts(doc)
    assert texts.count("manifold") == 2  # the word above the group, and the legend's row
    assert texts.count("terminal") == 2
    (dot,) = _tipped(doc, "switch 1")
    assert dot["type"] == "ellipse"


def test_each_coil_number_stands_outside_the_yoke_at_its_angle(tmp_path) -> None:
    doc = _card(_graph(tmp_path, _six_poles()))
    yoke, _window = _by_type_tip(doc, "ellipse", "yoke")
    cx, cy = _centre(yoke)
    radius = yoke["geometry"]["width"] / 2
    for k in range(6):
        (label,) = [
            n
            for n in _by_type(doc, "text-label")
            if n["props"]["text"] == str(k + 1)
            and n["props"]["fontSize"] == layout.TEXT_SIZE
            and math.dist(_centre(n), (cx, cy)) < 2 * radius
        ]
        x, y = _centre(label)
        assert math.dist((x, y), (cx, cy)) > radius
        angle = math.degrees(math.atan2(cy - y, x - cx)) % 360
        assert angle == pytest.approx(30 + 60 * k, abs=2)


def test_the_legend_lists_each_part_once_by_kind_with_its_source(tmp_path) -> None:
    rows = [
        *_six_poles(),
        _row("thermal switch 1", "thermal_switch", 90, 90),
        _row("thermal switch 2", "thermal_switch", -90, 90),
        _row("odd one", "thermal_switch", 0, -140),
    ]
    doc = _card(_graph(tmp_path, rows))
    texts = _texts(doc)
    assert "1, 2, 3 (odd one)" in texts  # a numbered member by its number, the rest named
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


def test_a_signal_resolves_on_a_part_device_of_the_magnet(tmp_path) -> None:
    rows = [_row("switch 1", "thermal_switch", 90, 90, signal="*TS1:Open-Sts")]
    graph = _graph(tmp_path, rows, change=_switches)
    assert graph.picture_matches("XR01:QF1", "*TS1:Open-Sts") == [
        ("XR01:QF1-TS1", "XR01-QF1-TS1:Open-Sts")
    ]
    assert graph.picture_channel("XR01:QF1", "*:Open-Sts") == "XR01-QF1-TS1:Open-Sts"
    assert len(graph.picture_matches("XR01:QF1", "*:Open-Sts")) == 2  # two parts: no pick
    assert graph.picture_channel("XR01:QF1", "*:Current-SP") == "XR01-QF1:Current-SP"  # own first


# -- shapes with no size, other pole counts, the yoke as a polygon -----------------


def _unsized(poles: int, first: float, *, bore: str = "bore", gap: float = 30) -> list[str]:
    """An invented class of *poles* poles from *first* degrees, no pole or coil sized;
    its bore (or its gap) sized."""
    rows = [_row(bore, bore, 0, 0, w=gap)]
    for k in range(poles):
        angle = first + 360 / poles * k
        rows.append(_row(f"pole {k + 1}", "pole", 0, 0, angle=angle))
        rows.append(_row(f"coil {k + 1}", "coil", 0, 0, angle=angle))
    return rows


@pytest.mark.parametrize(("poles", "first"), [(2, 90), (4, 45), (6, 30)])
def test_poles_with_an_angle_and_no_size_draw_from_the_bore_and_say_so(
    tmp_path, poles: int, first: float
) -> None:
    from osprey.facility.panels.picture import body
    from osprey.facility.panels.tables import read_tables

    rows = _unsized(poles, first, bore="gap" if poles == 2 else "bore")
    pictures = read_tables(_tables(tmp_path, rows)).pictures
    drawn = body([r for r in pictures if r.shape])
    assert not drawn.sized and drawn.beam == (0, 0)
    for k in range(poles):
        (pole,) = [s for s in drawn.shapes if s.row.part == f"pole {k + 1}"]
        tip = min(math.hypot(a, b) for a, b in pole.points)
        assert tip >= 15 * 0.99  # no pole reaches into the bore (radius 15)
        sides = [s for s in drawn.shapes if s.row.part == f"coil {k + 1}"]
        assert len(sides) == 2
        for side in sides:  # beside the pole, not across it
            assert not _overlap(side.points, pole.points)
    assert [c.number for c in drawn.coils] == [str(k + 1) for k in range(poles)]
    doc = _card(_graph(tmp_path / "g", rows))
    texts = _texts(doc)
    assert any(t.startswith("size not stated") for t in texts)
    for k in range(poles):
        assert str(k + 1) in texts  # each coil's number at the coil


def _overlap(p: tuple, q: tuple) -> bool:
    """Whether two convex polygons overlap by more than a touch (separating axes)."""
    for poly in (p, q):
        for i in range(len(poly)):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
            nx, ny = y1 - y2, x2 - x1
            a = [nx * x + ny * y for x, y in p]
            b = [nx * x + ny * y for x, y in q]
            if max(a) <= min(b) + 1e-6 or max(b) <= min(a) + 1e-6:
                return False
    return True


def test_a_yoke_stated_as_a_polygon_is_drawn_filled_and_bounds_the_poles(tmp_path) -> None:
    from osprey.facility.panels.picture import body
    from osprey.facility.panels.tables import read_tables

    hexagon = [
        _row(
            f"o{k}",
            "outline",
            100 * math.cos(math.radians(60 * k)),
            100 * math.sin(math.radians(60 * k)),
        )
        for k in range(6)
    ]
    rows = [*_unsized(4, 45), *hexagon]
    pictures = read_tables(_tables(tmp_path, rows)).pictures
    drawn = body([r for r in pictures if r.shape], [r for r in pictures if r.kind == "outline"])
    yoke, _window = [s for s in drawn.shapes if s.row.kind == "outline"]
    assert len(yoke.points) == 6 and yoke.role == "yoke" and _window.role == "window"
    inner = 100 * math.cos(math.radians(30))  # the hexagon's nearest edge
    for shape in drawn.shapes:
        if shape.row.kind == "pole":
            assert max(math.hypot(a, b) for a, b in shape.points) < inner
    doc = _card(_graph(tmp_path / "g", rows))
    polygon, window = _by_type_tip(doc, "polygon", "yoke")
    assert polygon["props"]["fillColor"] == symbols.STEEL_DARK
    assert window["props"]["fillColor"] == symbols.PANEL_BACKGROUND  # convex: the outside


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


def test_check_notes_a_signal_that_matches_two_part_devices(tmp_path) -> None:
    from osprey.facility.panels.emit import check, write

    rows = [_row("switch 1", "thermal_switch", 90, 90, signal="*:Open-Sts")]
    graph = _graph(tmp_path, rows, change=_switches)
    out = tmp_path / "out"
    write(graph, out, only=lambda t: t.view == "card" and t.id == "XR01:QF1")
    (found,) = [p for p in check(graph, out) if p.kind == "ambiguous_picture"]
    assert not found.failing
    assert "XR01:QF1-TS1" in found.detail and "XR01:QF1-TS2" in found.detail
