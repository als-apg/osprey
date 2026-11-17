"""Symbols: the element shapes, the five looks of a light, and the theme-role table."""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import pytest

from osprey.facility.panels import symbols
from osprey.facility.panels.symbols import (
    ALARM_ROLES,
    KINDS,
    STATES,
    WIDTH_UNITS,
    draw,
    light,
    look,
)

PANELS = Path(symbols.__file__).parent
HEX = re.compile(r"#[0-9a-fA-F]{6}")
BRIEF_KINDS = (
    "bend",
    "quadrupole_f",
    "quadrupole_d",
    "sextupole",
    "skew_quadrupole",
    "corrector",
    "bpm",
    "neg_pump",
    "ion_pump",
    "gauge",
    "gate_valve",
    "bellows",
    "kicker",
    "scraper",
    "generic",
)
DRAWING = {"rectangle", "ellipse", "polygon", "line"}
AXIS = 100.0
UNIT = 10.0


def _colours(value):
    if isinstance(value, dict) and "$role" in value:
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _colours(item)
    elif isinstance(value, list):
        for item in value:
            yield from _colours(item)


def _extent(nodes):
    top = min(n["geometry"]["y"] for n in nodes)
    bottom = max(n["geometry"]["y"] + n["geometry"]["height"] for n in nodes)
    left = min(n["geometry"]["x"] for n in nodes)
    right = max(n["geometry"]["x"] + n["geometry"]["width"] for n in nodes)
    return left, top, right, bottom


def test_every_kind_of_the_brief_is_drawn_and_one_without_polarity_besides() -> None:
    assert set(BRIEF_KINDS) <= set(KINDS)
    assert set(KINDS) - set(BRIEF_KINDS) == {"quadrupole"}


@pytest.mark.parametrize("kind", KINDS)
def test_a_symbol_is_drawing_widgets_centred_on_x_and_as_wide_as_its_kind(kind) -> None:
    nodes = draw(kind, 50.0, AXIS, unit=UNIT)
    assert nodes
    assert {n["type"] for n in nodes} <= DRAWING
    assert all("id" not in n for n in nodes)
    left, _, right, _ = _extent(nodes)
    assert (left + right) / 2 == pytest.approx(50.0)
    assert right - left == pytest.approx(WIDTH_UNITS[kind] * UNIT)


@pytest.mark.parametrize("kind", KINDS)
def test_every_colour_of_a_symbol_is_a_theme_role(kind) -> None:
    for node in draw(kind, 0.0, AXIS, unit=UNIT):
        for key, value in node["props"].items():
            if key.endswith("Color"):
                assert isinstance(value, dict) and "$role" in value, (kind, key)


def test_quadrupoles_sit_above_or_below_the_axis_by_polarity() -> None:
    _, top_f, _, bottom_f = _extent(draw("quadrupole_f", 0.0, AXIS, unit=UNIT))
    _, top_d, _, bottom_d = _extent(draw("quadrupole_d", 0.0, AXIS, unit=UNIT))
    _, top_x, _, bottom_x = _extent(draw("quadrupole", 0.0, AXIS, unit=UNIT))
    assert bottom_f == pytest.approx(AXIS) and top_f < AXIS
    assert top_d == pytest.approx(AXIS) and bottom_d > AXIS
    assert (top_x + bottom_x) / 2 == pytest.approx(AXIS)


def test_an_unknown_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="no symbol"):
        draw("wiggler", 0.0, AXIS)


def test_a_state_sets_the_symbols_outline() -> None:
    (node,) = draw("generic", 0.0, AXIS, "bad", unit=UNIT)
    assert node["props"]["lineColor"] == look("bad")["lineColor"]


def _appearance(state: str) -> tuple:
    shown = look(state)
    return tuple(
        shown[p]["$role"] if isinstance(shown[p], dict) else shown[p]
        for p in (*symbols.LIGHT_PROPS, symbols.MARK_PROP)
    )


def test_the_five_states_and_the_aggregate_look_pairwise_different() -> None:
    assert len(STATES) == 5
    assert symbols.APPEARANCES == (*STATES, "aggregate")
    for a, b in combinations(symbols.APPEARANCES, 2):
        assert _appearance(a) != _appearance(b), (a, b)
    aggregate = look("aggregate")
    assert aggregate["text"] == symbols.AGGREGATE_MARK
    roles = {c["$role"] for c in _colours(dict(aggregate))}
    assert not roles & ALARM_ROLES  # an aggregate is not a state


def test_normal_is_neutral_with_no_state_colour() -> None:
    shown = look("normal")
    roles = {c["$role"] for c in _colours(dict(shown))}
    assert not roles & ALARM_ROLES
    assert "OK" not in roles  # never green
    assert shown["fill"] is True and shown["text"] == ""


def test_unknown_is_hollow_and_dashed_and_disagree_carries_a_mark() -> None:
    unknown, disagree = look("unknown"), look("disagree")
    assert unknown["fill"] is False and unknown["lineStyle"] == "dashed"
    assert unknown["text"] == symbols.UNKNOWN_MARK
    assert disagree["text"] == symbols.DISAGREE_MARK != symbols.UNKNOWN_MARK
    assert disagree["fillColor"]["$role"] not in ALARM_ROLES


def test_a_light_is_a_polygon_and_a_mark_in_the_look_of_its_state() -> None:
    shape, mark = light(10.0, 20.0, 14.0, "unknown")
    assert shape["type"] == "polygon" and len(shape["props"]["points"]) == 8
    assert shape["props"]["lineStyle"] == "dashed" and shape["props"]["fill"] is False
    assert mark["type"] == "text-label" and mark["props"]["text"] == symbols.UNKNOWN_MARK
    assert mark["geometry"] == {"x": 10.0, "y": 20.0, "width": 14.0, "height": 14.0}


def _between(text: str, start: str, end: str) -> tuple[str, str]:
    head, rest = text.split(start, 1)
    inside, tail = rest.split(end, 1)
    return inside, head + tail


def test_no_hex_colour_outside_the_theme_role_table() -> None:
    table, rest = _between(
        (PANELS / "symbols.py").read_text(), "# --- theme roles", "# --- end theme roles"
    )
    assert HEX.search(table)
    assert not HEX.search(rest)
    assert not HEX.search((PANELS / "layout.py").read_text())
