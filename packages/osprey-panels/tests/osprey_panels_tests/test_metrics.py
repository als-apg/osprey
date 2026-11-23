"""The band of key numbers: rows of metrics.csv, selected by stated signals and
computed by screen formulas the viewer evaluates. Formulas are evaluated here
as the formula engine would: its grammar and functions."""

from __future__ import annotations

import ast
import math
import re

import pytest

from osprey_panels import layout, symbols
from osprey_panels.describe import health_rollup, place_view
from osprey_panels.tables import PanelTableError
from osprey_panels.view import Band
from osprey_panels_tests._fixture import copy_tables, read_fixture_tables
from osprey_panels_tests._views import ABOVE, SYMBOLS, nodes, stub_render, with_symbols

#: The formula engine's grammar (canopy formula_engine/language/grammar.py) and the
#: functions the band uses from its namespace.
GRAMMAR = (
    ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.BoolOp, ast.IfExp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub, ast.Not, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.And,
    ast.Or, ast.Call, ast.Tuple,
)  # fmt: skip
FUNCTIONS = {"sqrt": math.sqrt, "abs": abs, "max": max, "min": min}


def evaluate(formula: dict, values: dict[str, float]) -> float:
    """A screen formula's value for channel *values*, as the engine computes it."""
    text = re.sub(r"\{(\w+)\}", lambda m: m.group(1), formula["expression"])
    tree = ast.parse(text, mode="eval")
    assert all(isinstance(node, GRAMMAR) for node in ast.walk(tree)), text
    names = {key: values[channel] for key, channel in formula["bindings"].items()}
    return eval(compile(tree, "formula", "eval"), {"__builtins__": {}}, {**FUNCTIONS, **names})


def _band(view) -> Band:
    (band,) = [i for s in view.sections for i in s.items if isinstance(i, Band)]
    return band


def _tiles(doc: dict) -> dict[str, dict]:
    """Each tile's widgets by its title (the tile's first text label)."""
    out: dict[str, dict] = {}
    for n in doc["root"]["children"]:
        meta = n.get("meta", {}).get("comment", "")
        match = re.search(r"tile: (.+?)(;|$)", meta)
        if match:
            out.setdefault(match.group(1), {}).setdefault(n["type"], []).append(n)
    return out


@pytest.fixture
def xr01(panel_graph):
    view = place_view(with_symbols(panel_graph, SYMBOLS, ABOVE), "XR/XR01")
    return view, stub_render(view, panel_graph)


def test_the_metrics_table_reads_and_refuses_an_unknown_measure(tmp_path, panel_tables):
    rows = read_fixture_tables(panel_tables).metrics
    assert [(r.view, r.level, r.order, r.measure) for r in rows][:2] == [
        ("place", "sector", 1.0, "count_in_tolerance"),
        ("place", None, 2.0, "rms"),
    ]
    assert rows[1].cls == "Quadrupole" and rows[1].unit == "A"
    directory = copy_tables(tmp_path / "panels")
    path = directory / "metrics.csv"
    path.write_text(path.read_text() + "place,,9,Odd,median,current_readback,,,,s\n")
    with pytest.raises(PanelTableError, match="measure is 'median'"):
        read_fixture_tables(directory)


def test_a_place_view_opens_with_its_band_selected_by_stated_signals(xr01) -> None:
    view, _doc = xr01
    assert isinstance(view.sections[0].items[0], Band)  # before the strip
    tiles = {t.title: t for t in _band(view).tiles}
    assert "Ring only" not in tiles  # a row for another level
    assert tiles["Currents"].channels == ("XR01-QD1:Current-RB", "XR01-QF1:Current-RB")
    assert tiles["Vacuum"].channels == ("XR01-IP1:Pressure-RB", "XR01-IP2:Pressure-RB")
    assert tiles["Magnets"].pairs == (
        ("XR01-QF1:Current-SP", "XR01-QF1:Current-RB", "absolute", 0.05),
    )
    assert tiles["Temperature"].channels == () and tiles["Temperature"].pairs == ()


def test_every_measure_is_a_formula_the_engine_computes_right(xr01) -> None:
    _view, doc = xr01
    formulas = {f["name"]: f for f in doc["formulas"]}
    tiles = _tiles(doc)

    def formula_of(title: str) -> dict:
        (label,) = tiles[title]["pv-label"]
        return formulas[label["props"]["pv"].removeprefix("formula://")]

    qf, qd = "XR01-QF1:Current-RB", "XR01-QD1:Current-RB"
    assert evaluate(formula_of("Currents"), {qf: 3.0, qd: 4.0}) == pytest.approx(math.sqrt(12.5))
    ip1, ip2 = "XR01-IP1:Pressure-RB", "XR01-IP2:Pressure-RB"
    assert evaluate(formula_of("Vacuum"), {ip1: 2e-9, ip2: 5e-10}) == 2e-9
    assert evaluate(formula_of("Vacuum best"), {ip1: 2e-9, ip2: 5e-10}) == 5e-10
    status = "XR01-IP1:State-Sts"
    assert evaluate(formula_of("Pumps"), {status: 1}) == 1
    assert evaluate(formula_of("Pumps"), {status: 0}) == 0
    assert evaluate(formula_of("Pumps lost"), {status: 0}) == 1
    sp, rb = "XR01-QF1:Current-SP", "XR01-QF1:Current-RB"
    assert evaluate(formula_of("Magnets"), {sp: 10.0, rb: 10.04}) == 1
    assert evaluate(formula_of("Magnets"), {sp: 10.0, rb: 10.2}) == 0
    (bend,) = tiles["Bend"]["pv-label"]  # one channel: bound as itself
    assert bend["props"]["pv"] == "XR01-BEND1:Current-RB"


def test_a_count_reads_n_of_n_and_turns_its_border_bad_on_a_loss(xr01) -> None:
    _view, doc = xr01
    formulas = {f["name"]: f for f in doc["formulas"]}
    tiles = _tiles(doc)
    texts = [n["props"]["text"] for n in tiles["Magnets"]["text-label"]]
    assert "of 1" in texts
    for title, values, bad in (
        ("Pumps lost", {"XR01-IP1:State-Sts": 0}, True),
        ("Pumps lost", {"XR01-IP1:State-Sts": 1}, False),
        ("Magnets", {"XR01-QF1:Current-SP": 1.0, "XR01-QF1:Current-RB": 2.0}, True),
        ("Magnets", {"XR01-QF1:Current-SP": 1.0, "XR01-QF1:Current-RB": 1.0}, False),
    ):
        (frame,) = [n for n in tiles[title]["rectangle"] if n.get("rules")]
        (rule,) = frame["rules"]
        assert rule["targetProperty"] == "lineColor"
        (pv,) = rule["pvs"]
        value = evaluate(formulas[pv["name"].removeprefix("formula://")], values)
        (condition,) = rule["conditions"]
        holds = eval(condition["expression"].replace("{0}", repr(value)), {"__builtins__": {}})
        assert holds == bad and condition["value"] == symbols.ALARM_MAJOR
        assert rule["default"] == symbols.BOX_BORDER
    assert not [n for n in tiles["Pumps"]["rectangle"] if n.get("rules")]  # nothing to lose


def test_tiles_with_nothing_selected_fold_into_one_last_line_that_says_so_once(xr01) -> None:
    _view, doc = xr01
    texts = [n for n in nodes(doc) if n["type"] == "text-label"]
    (said,) = [n for n in texts if n["props"]["text"] == "NO DATA YET"]
    folded = [n for n in texts if n["props"]["text"] == "TEMPERATURE"]
    assert folded and folded[0]["geometry"]["y"] == said["geometry"]["y"]  # on its line
    assert "Temperature" not in _tiles(doc)  # no tile of its own
    live = [t["rectangle"][0]["geometry"] for t in _tiles(doc).values()]
    assert all(g["y"] + g["height"] <= said["geometry"]["y"] for g in live)  # after the live


def test_the_fold_shares_the_band_s_last_row_where_it_fits(panel_graph, monkeypatch) -> None:
    """Tiles that leave room at the right of their last row: the fold stands on
    that row, right of the last tile, and adds no line under the band."""
    whole = layout.spread
    monkeypatch.setattr(layout, "spread", lambda n, x0, w, gap: whole(n, x0, w / 2, gap))
    view = place_view(with_symbols(panel_graph, SYMBOLS, ABOVE), "XR/XR01")
    doc = stub_render(view, panel_graph)
    texts = [n for n in nodes(doc) if n["type"] == "text-label"]
    (said,) = [n["geometry"] for n in texts if n["props"]["text"] == "NO DATA YET"]
    live = [t["rectangle"][0]["geometry"] for t in _tiles(doc).values()]
    row = [g for g in live if g["y"] == max(g["y"] for g in live)]
    top, bottom = row[0]["y"], row[0]["y"] + row[0]["height"]
    assert top <= said["y"] and said["y"] + said["height"] <= bottom  # on the last row
    assert said["x"] > max(g["x"] + g["width"] for g in row)  # right of its last tile


def test_a_tile_too_long_for_the_engine_needs_a_server_side_aggregate(
    panel_graph, monkeypatch
) -> None:
    monkeypatch.setattr(layout, "RULE_EXPRESSION_LIMIT", 20)
    view = place_view(with_symbols(panel_graph, SYMBOLS, ABOVE), "XR/XR01")
    doc = stub_render(view, panel_graph)
    tile = _tiles(doc)["Currents"]
    texts = [n["props"]["text"] for n in tile["text-label"]]
    assert "—" in texts and "needs a server-side aggregate" in " ".join(texts)
    assert "server-side aggregate" in layout.needs(doc)
    assert all(len(f["expression"]) <= 20 for f in doc.get("formulas", ()))


def test_no_formula_is_longer_than_the_engine_takes(xr01) -> None:
    _view, doc = xr01
    assert doc["formulas"]
    for formula in doc["formulas"]:
        assert len(formula["expression"]) <= layout.RULE_EXPRESSION_LIMIT


def test_a_roll_up_root_carries_the_rollup_rows_of_the_band(panel_graph) -> None:
    view = health_rollup(panel_graph, "XR/XR01")
    assert [t.title for t in _band(view).tiles] == ["Currents"]
    assert len(_band(view).tiles[0].channels) == 3  # every magnet readback in the sector


def _band_doc(panel_graph, tiles):
    from osprey_panels.view import Section, Target, View

    view = View(
        Target("place", "place", "XR/TEST"),
        "test band",
        (),
        (Section("key numbers", (Band(tiles),)),),
    )
    return stub_render(view, panel_graph)


def _frames(doc: dict) -> list[dict]:
    tiles = _tiles(doc)
    return [g for t in tiles.values() for g in (t["rectangle"][0]["geometry"],)]


@pytest.mark.parametrize(("count", "rows"), [(12, [6, 6]), (7, [7]), (9, [5, 4]), (3, [3])])
def test_the_band_is_an_even_grid_whose_rows_span_the_content_width(panel_graph, count, rows):
    from osprey_panels.view import Tile

    tiles = tuple(
        Tile(f"T{i}", "value", None, None, ("XR01-QF1:Current-RB",)) for i in range(count)
    )
    frames = _frames(_band_doc(panel_graph, tiles))
    by_row: dict[float, list[dict]] = {}
    for g in frames:
        by_row.setdefault(g["y"], []).append(g)
    assert [len(by_row[y]) for y in sorted(by_row)] == rows
    for row in by_row.values():
        row.sort(key=lambda g: g["x"])
        assert len({g["width"] for g in row}) == 1 and row[0]["width"] >= layout.MIN_TILE_WIDTH
        assert row[0]["x"] == layout.MARGIN
        assert abs(row[-1]["x"] + row[-1]["width"] - (layout.MARGIN + layout.STRIP_WIDTH)) < 1


def test_the_number_is_the_biggest_text_and_the_count_sits_on_its_baseline(xr01) -> None:
    _view, doc = xr01
    tile = _tiles(doc)["Magnets"]
    (number,) = tile["pv-label"]
    n = number["geometry"]
    assert number["props"]["fontSize"] == layout.BIG_SIZE
    # the viewer shrinks a value whose line overflows its box: room for the line
    assert n["height"] >= layout.BIG_SIZE * layout.LINE_HEIGHT + 2
    assert all(t["props"]["fontSize"] < layout.BIG_SIZE for t in tile["text-label"])
    (count,) = [t for t in tile["text-label"] if t["props"]["text"] == "of 1"]
    c = count["geometry"]
    assert n["x"] + n["width"] <= c["x"] <= n["x"] + n["width"] + layout.GAP
    # the small text's middle sits below the number's middle: on the baseline
    assert c["y"] + c["height"] / 2 > n["y"] + n["height"] / 2
    assert c["y"] + c["height"] <= n["y"] + n["height"]


def test_a_tile_text_never_needs_shrinking_and_a_long_caption_wraps(panel_graph) -> None:
    from osprey_panels.view import Tile

    long = "circuits without cooling-water flow and then some more words"
    tiles = tuple(
        Tile(f"Hottest thermocouple {i}", "value", "C", long, ("XR01-QF1:Current-RB",))
        for i in range(12)
    )
    doc = _band_doc(panel_graph, tiles)
    for tile in _tiles(doc).values():
        for t in tile["text-label"]:
            size, g = t["props"]["fontSize"], t["geometry"]
            assert size >= layout.SMALL_SIZE
            assert layout.text_width(t["props"]["text"], size) <= g["width"] + 0.5, t
        (number,) = tile["pv-label"]
        under = number["geometry"]["y"] + number["geometry"]["height"]
        lines = [t for t in tile["text-label"] if t["geometry"]["y"] >= under]
        assert len(lines) == 2  # wrapped, the rest cut with an ellipsis
        assert lines[0]["props"]["text"] + " " in long + " "
