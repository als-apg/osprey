"""The band of key numbers: rows of metrics.csv, selected by stated signals and
computed by screen formulas the viewer evaluates. Formulas are evaluated here
as the formula engine would: its grammar and functions."""

from __future__ import annotations

import ast
import math
import re

import pytest

from osprey.facility.panels import layout, symbols
from osprey.facility.panels.describe import health_rollup, place_view
from osprey.facility.panels.tables import PanelTableError, read_tables
from osprey.facility.panels.view import Band
from tests.facility.panels._fixture import copy_tables
from tests.facility.panels._views import ABOVE, SYMBOLS, stub_render

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
    view = place_view(panel_graph, "XR/XR01", symbols=SYMBOLS, above=ABOVE)
    return view, stub_render(view, panel_graph)


def test_the_metrics_table_reads_and_refuses_an_unknown_measure(tmp_path, panel_tables):
    rows = read_tables(panel_tables).metrics
    assert [(r.view, r.level, r.order, r.measure) for r in rows][:2] == [
        ("place", "sector", 1.0, "count_in_tolerance"),
        ("place", None, 2.0, "rms"),
    ]
    assert rows[1].cls == "Quadrupole" and rows[1].unit == "A"
    directory = copy_tables(tmp_path / "panels")
    path = directory / "metrics.csv"
    path.write_text(path.read_text() + "place,,9,Odd,median,current_readback,,,,s\n")
    with pytest.raises(PanelTableError, match="measure is 'median'"):
        read_tables(directory)


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


def test_a_tile_with_nothing_selected_is_greyed_and_says_no_data_yet(xr01) -> None:
    _view, doc = xr01
    tile = _tiles(doc)["Temperature"]
    texts = [n["props"]["text"] for n in tile["text-label"]]
    assert "no data yet" in texts and "—" in texts
    assert "pv-label" not in tile


def test_a_tile_too_long_for_the_engine_needs_a_server_side_aggregate(
    panel_graph, monkeypatch
) -> None:
    monkeypatch.setattr(layout, "RULE_EXPRESSION_LIMIT", 20)
    view = place_view(panel_graph, "XR/XR01", symbols=SYMBOLS, above=ABOVE)
    doc = stub_render(view, panel_graph)
    tile = _tiles(doc)["Currents"]
    texts = [n["props"]["text"] for n in tile["text-label"]]
    assert "—" in texts and "needs a server-side aggregate" in texts
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
