"""The schematic arrangement of a place: true order, compressed spacing."""

from __future__ import annotations

import random
import time

import pytest

from osprey_panels.strip import arrange, arrange_scaled, locator
from osprey_panels.view import StripElement

WIDTHS = {"quad": 20.0, "bpm": 10.0, "dipole": 40.0}


def el(thing: str, s: float, symbol: str = "quad", lane: str = "beam") -> StripElement:
    return StripElement(thing, None, s, symbol, lane, None)


def test_empty_input_gives_empty_output() -> None:
    assert arrange([], WIDTHS) == []
    assert locator([], 10.0) == []


def test_single_element_sits_at_the_centre() -> None:
    (p,) = arrange([el("a", 3.0)], WIDTHS, strip_width=1000.0)
    assert p.x == pytest.approx(500.0)
    assert p.thing == "a"


def test_output_is_in_s_order_and_one_per_element() -> None:
    items = [el("c", 3.0), el("a", 1.0), el("b", 2.0)]
    out = arrange(items, WIDTHS)
    assert [p.thing for p in out] == ["a", "b", "c"]
    assert [p.x for p in out] == sorted(p.x for p in out)
    assert len(out) == len(items)


def test_equal_s_keeps_input_order() -> None:
    items = [el("z", 1.0), el("y", 1.0), el("x", 1.0)]
    assert [p.thing for p in arrange(items, WIDTHS)] == ["z", "y", "x"]


def test_equal_s_on_two_lanes_keeps_both_close() -> None:
    items = [el("q", 5.0), el("m", 5.0, "bpm", "above")]
    out = arrange(items, WIDTHS, strip_width=100.0)
    assert {p.thing for p in out} == {"q", "m"}
    scale = out[0].width / WIDTHS["quad"]
    assert abs(out[0].x - out[1].x) <= WIDTHS["quad"] * scale + 1e-9


def test_gap_is_capped() -> None:
    def span(gap_m: float) -> float:
        out = arrange([el("a", 0.0), el("b", gap_m), el("c", gap_m + 1)], WIDTHS, 1000.0)
        # Compare in units: undo the scale through the width of the first symbol.
        k = out[0].width / WIDTHS["quad"]
        return (out[1].x - out[0].x) / k

    assert span(10.0) == pytest.approx(span(3.0))


def test_gap_grows_with_distance_below_the_cap() -> None:
    near = arrange([el("a", 0.0), el("b", 0.1)], WIDTHS)
    far = arrange([el("a", 0.0), el("b", 1.0)], WIDTHS)
    assert (far[1].x - far[0].x) > (near[1].x - near[0].x)


def test_no_two_beam_symbols_overlap_for_random_spacings() -> None:
    rng = random.Random(7)
    s, items = 0.0, []
    for i in range(200):
        s += rng.choice([0.0, 0.05, 0.5, 2.0, 12.0]) * rng.random()
        items.append(
            el(f"e{i}", s, rng.choice(["quad", "bpm", "dipole"]), rng.choice(["beam", "above"]))
        )
    out = [p for p in arrange(items, WIDTHS, 1000.0) if p.lane == "beam"]
    for a, b in zip(out, out[1:], strict=False):
        assert b.x - b.width / 2 >= a.x + a.width / 2 - 1e-9


def test_above_lane_alternates_two_levels_in_order() -> None:
    items = [el(f"m{i}", float(i), "bpm", "above") for i in range(4)]
    items.insert(2, el("q", 1.5))
    out = arrange(items, WIDTHS)
    assert [p.level for p in out if p.lane == "above"] == [0, 1, 0, 1]


def test_above_symbols_take_a_fraction_of_the_width() -> None:
    a, b = arrange([el("a", 0.0, "dipole", "above"), el("b", 0.0, "dipole")], WIDTHS)
    assert a.width == pytest.approx(0.55 * b.width)


def test_unknown_symbol_gets_the_default_width() -> None:
    (p,) = arrange([el("a", 0.0, "mystery")], {})
    assert p.width > 0


def test_locator_is_proportional_and_clamped() -> None:
    items = [el("a", 0.0), el("b", 5.0), el("c", 10.0), el("d", 99.0), el("e", -4.0)]
    got = dict(locator(items, 10.0, 1000.0))
    assert got["a"] == pytest.approx(0.0)
    assert got["b"] == pytest.approx(500.0)
    assert got["c"] == pytest.approx(1000.0)
    assert got["d"] == pytest.approx(1000.0)
    assert got["e"] == pytest.approx(0.0)


def test_locator_with_zero_extent_centres_everything() -> None:
    got = locator([el("a", 2.0), el("b", 2.0)], 0.0, 1000.0)
    assert [x for _, x in got] == [500.0, 500.0]


def test_five_thousand_elements_arrange_quickly() -> None:
    items = [el(f"e{i}", i * 0.3, "quad", "beam" if i % 3 else "above") for i in range(5000)]
    # CPU time, best of three: a shared CI runner that deschedules the worker
    # must not read as a slow algorithm. The budget itself is unchanged.
    best = float("inf")
    for _ in range(3):
        t = time.process_time()
        assert len(arrange(items, WIDTHS)) == 5000
        best = min(best, time.process_time() - t)
    assert best < 0.5


def test_a_lone_element_keeps_its_natural_width() -> None:
    (p,), scale = arrange_scaled([el("a", 0.0)], WIDTHS, strip_width=1000.0)
    assert p.width == pytest.approx(WIDTHS["quad"])
    assert p.x == pytest.approx(500.0)
    assert scale == 1.0


def test_a_short_strip_keeps_natural_widths_and_is_centred_as_a_block() -> None:
    items = [el("a", 0.0), el("b", 1.0), el("c", 2.0)]
    out, scale = arrange_scaled(items, WIDTHS, strip_width=1000.0)
    assert scale == 1.0
    assert [p.width for p in out] == [20.0, 20.0, 20.0]
    natural = 20.0 * 3 + 2 * 18.0  # gap(1 m) = 4 + 14
    left = out[0].x - out[0].width / 2
    right = out[-1].x + out[-1].width / 2
    assert right - left == pytest.approx(natural)
    assert (left + right) / 2 == pytest.approx(500.0)


def test_a_long_strip_is_compressed_and_still_does_not_overlap() -> None:
    items = [el(f"e{i}", i * 0.5) for i in range(500)]
    out, scale = arrange_scaled(items, WIDTHS, strip_width=1000.0)
    assert scale < 1.0
    assert arrange(items, WIDTHS, 1000.0) == out
    assert out[0].x - out[0].width / 2 >= -1e-9
    assert out[-1].x + out[-1].width / 2 <= 1000.0 + 1e-9
    for a, b in zip(out, out[1:], strict=False):
        assert b.x - b.width / 2 >= a.x + a.width / 2 - 1e-9


def test_empty_input_scales_by_one() -> None:
    assert arrange_scaled([], WIDTHS) == ([], 1.0)


def test_a_long_strip_narrows_its_gaps_before_its_symbols() -> None:
    items = [el(f"e{i}", i * 2.0) for i in range(30)]  # natural: 30 x 20 + 29 x 26 = 1354
    out, scale = arrange_scaled(items, WIDTHS, strip_width=1000.0)
    assert scale == 1.0 and all(p.width == 20.0 for p in out)
    assert out[-1].x + out[-1].width / 2 - (out[0].x - out[0].width / 2) == pytest.approx(1000.0)


def test_a_filling_strip_spans_the_width_and_keeps_the_order_of_its_gaps() -> None:
    items = [el("a", 0.0), el("b", 0.5), el("c", 3.0)]
    out, scale = arrange_scaled(items, WIDTHS, strip_width=1000.0, fill=True)
    assert scale == 1.0
    assert out[0].x - 10.0 == pytest.approx(0.0) and out[-1].x + 10.0 == pytest.approx(1000.0)
    near, far = out[1].x - out[0].x, out[2].x - out[1].x
    assert near < far  # 0.5 m apart stays closer than 2.5 m apart
