"""A button pickup: each button's reading of a beam, and the statement that describes it."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from osprey.simulation.engines import pyat_pickup
from osprey.simulation.engines.pyat_pickup import (
    WALL_SHARE,
    RoundPickup,
    TablePickup,
    parse,
    round_response,
)
from osprey.simulation.engines.pyat_turn_settings import stated_buttons, stated_counts

RADIUS = 0.016
ANGLES = (45.0, 135.0, 225.0, 315.0)
HALF_WIDTH = 14.0

ROUND: dict[str, Any] = {
    "shape": "round",
    "radius": RADIUS,
    "buttons": list(ANGLES),
    "half_width": HALF_WIDTH,
    "x": {"plus": [1, 4], "minus": [2, 3]},
    "y": {"plus": [1, 2], "minus": [3, 4]},
}


class Refused(Exception):
    def __init__(self, detail: str, remedy: str) -> None:
        super().__init__(detail)
        self.detail = detail
        self.remedy = remedy


def read(stated: Any, directory: Path = Path("."), **keywords: Any) -> pyat_pickup.Pickup:
    return parse(stated, "pickup", directory, Refused, **keywords)


def changed(**changes: Any) -> dict[str, Any]:
    stated = {key: value for key, value in ROUND.items() if changes.get(key, 0) is not None}
    stated.update({key: value for key, value in changes.items() if value is not None})
    return stated


def wall_charge(x: float, y: float, angle: float, half_width: float, radius: float) -> float:
    """The mean wall image charge density over a button's arc, 1 for a centred beam.

    A Gauss-Legendre integration of the density of the image charge a line
    charge at ``(x, y)`` induces on a round conducting pipe: ``(a^2 - r^2) /
    |a e^(i psi) - z|^2`` at the wall angle ``psi``.
    """
    nodes, weights = np.polynomial.legendre.leggauss(200)
    psi = angle + half_width * nodes
    wall_x, wall_y = radius * np.cos(psi), radius * np.sin(psi)
    density = (radius**2 - x**2 - y**2) / ((wall_x - x) ** 2 + (wall_y - y) ** 2)
    return float(weights @ density) / 2.0


def series(x: float, y: float, angle: float, half_width: float, radius: float) -> float:
    """The multipole series of the same mean, summed until it no longer changes."""
    rho, theta = math.hypot(x, y) / radius, math.atan2(y, x)
    return 1.0 + 2.0 * sum(
        rho**n * math.cos(n * (theta - angle)) * math.sin(n * half_width) / (n * half_width)
        for n in range(1, 400)
    )


POSITIONS = [
    (0.0, 0.0),
    (0.001, 0.0),
    (0.0, -0.002),
    (0.003, 0.004),
    (-0.006, 0.002),
    (0.007, 0.007),
    (-0.010, -0.009),
    (0.0, 0.0155),
]


class TestTheRoundResponse:
    @pytest.mark.parametrize("position", POSITIONS)
    def test_it_is_the_wall_image_charge_integrated_over_each_button(self, position):
        x, y = position
        angles = np.radians(ANGLES)
        delta = math.radians(HALF_WIDTH)
        got = round_response(x, y, RADIUS, angles, delta)
        expected = [wall_charge(x, y, angle, delta, RADIUS) for angle in angles]
        np.testing.assert_allclose(got, expected, rtol=1.0e-10)

    @pytest.mark.parametrize("position", POSITIONS[:6])
    def test_it_is_the_multipole_series_summed(self, position):
        x, y = position
        angles = np.radians(ANGLES)
        delta = math.radians(HALF_WIDTH)
        got = round_response(x, y, RADIUS, angles, delta)
        expected = [series(x, y, angle, delta, RADIUS) for angle in angles]
        np.testing.assert_allclose(got, expected, rtol=1.0e-10)

    def test_it_takes_arrays_of_positions_button_first(self):
        x = np.array([[0.0, 0.001, 0.002], [0.003, -0.004, 0.005]])
        y = np.array([[0.001, 0.0, -0.002], [0.0, 0.004, 0.005]])
        angles = np.radians(ANGLES)
        got = round_response(x, y, RADIUS, angles, 0.2)
        assert got.shape == (4, 2, 3)
        for index in np.ndindex(x.shape):
            np.testing.assert_allclose(
                got[(slice(None), *index)],
                round_response(float(x[index]), float(y[index]), RADIUS, angles, 0.2),
                rtol=1.0e-13,
            )

    def test_a_centred_beam_reads_one_on_every_button(self):
        got = round_response(0.0, 0.0, RADIUS, np.radians(ANGLES), 0.2)
        np.testing.assert_array_equal(got, np.ones(4))

    def test_a_beam_beyond_the_wall_reads_as_one_just_inside_it(self):
        angles = np.radians(ANGLES)
        direction = np.array([math.cos(0.3), math.sin(0.3)])
        outside = round_response(*(3.0 * RADIUS * direction), RADIUS, angles, 0.2)
        inside = round_response(*(WALL_SHARE * RADIUS * direction), RADIUS, angles, 0.2)
        np.testing.assert_allclose(outside, inside, rtol=1.0e-9)
        assert np.all(np.isfinite(outside)) and np.all(outside >= 0.0)


class TestARoundPickup:
    def test_its_buttons_sum_to_one_for_a_centred_beam(self):
        pickup = read(ROUND)
        assert isinstance(pickup, RoundPickup)
        assert pickup.buttons == 4
        assert pickup.response(0.0, 0.0).sum() == pytest.approx(1.0, rel=1.0e-15)
        np.testing.assert_allclose(pickup.centre, pickup.response(0.0, 0.0), rtol=1.0e-15)

    def test_its_sum_grows_towards_a_button_and_shrinks_on_the_axes(self):
        pickup = read(ROUND)
        r = 0.006
        on_axis = [pickup.response(r * math.cos(t), r * math.sin(t)).sum() for t in AXES]
        to_button = [pickup.response(r * math.cos(t), r * math.sin(t)).sum() for t in DIAGONALS]
        assert all(total < 1.0 for total in on_axis)
        assert all(total > 1.0 for total in to_button)
        np.testing.assert_allclose(on_axis, on_axis[0], rtol=1.0e-12)
        np.testing.assert_allclose(to_button, to_button[0], rtol=1.0e-12)

    def test_its_sum_goes_as_the_fourth_power_of_the_offset_to_leading_order(self):
        pickup = read(ROUND)
        delta = math.radians(HALF_WIDTH)
        # The n = 4 term: 2 (r/a)^4 cos(4 (theta - 45 deg)) sin(4 delta) / (4 delta).
        strength = 2.0 * math.sin(4.0 * delta) / (4.0 * delta)
        for r in (0.001, 0.002, 0.003):
            for theta in (0.0, 0.3, math.pi / 4.0):
                total = pickup.response(r * math.cos(theta), r * math.sin(theta)).sum()
                expected = -strength * (r / RADIUS) ** 4 * math.cos(4.0 * theta)
                assert total - 1.0 == pytest.approx(expected, rel=2.0e-3, abs=1.0e-12)

    def test_the_default_scale_makes_the_reading_exact_to_first_order(self):
        pickup = read(ROUND)
        expected = RADIUS * math.radians(HALF_WIDTH) / math.sin(math.radians(HALF_WIDTH))
        expected /= 2.0 * math.cos(math.radians(45.0))
        assert pickup.scale_x == pytest.approx(expected, rel=1.0e-12)
        assert pickup.scale_y == pytest.approx(expected, rel=1.0e-12)
        x, y = 2.0e-6, -3.0e-6
        buttons = pickup.response(x, y)
        assert pickup.scale_x * (pickup.signs_x @ buttons) / buttons.sum() == pytest.approx(
            x, rel=1.0e-6
        )
        assert pickup.scale_y * (pickup.signs_y @ buttons) / buttons.sum() == pytest.approx(
            y, rel=1.0e-6
        )

    def test_a_large_offset_reads_short_of_where_the_beam_is(self):
        pickup = read(ROUND)
        buttons = pickup.response(0.008, 0.0)
        reading = pickup.scale_x * (pickup.signs_x @ buttons) / buttons.sum()
        assert 0.004 < reading < 0.0075

    def test_a_stated_scale_is_taken_as_it_is(self):
        pickup = read(changed(x={"plus": [1, 4], "minus": [2, 3], "scale": 0.011}))
        assert pickup.scale_x == 0.011
        assert pickup.scale_y != 0.011

    def test_a_diameter_is_an_arc_of_that_width_along_the_wall(self):
        by_width = read(ROUND)
        diameter = 2.0 * RADIUS * math.radians(HALF_WIDTH)
        by_diameter = read(changed(half_width=None, diameter=diameter))
        np.testing.assert_allclose(
            by_diameter.response(0.003, -0.002), by_width.response(0.003, -0.002), rtol=1.0e-12
        )

    def test_two_buttons_read_one_plane_with_a_stated_scale_on_the_other(self):
        pickup = read(
            {
                "shape": "round",
                "radius": RADIUS,
                "buttons": [0.0, 180.0],
                "half_width": 10.0,
                "x": {"plus": [1], "minus": [2]},
                "y": {"plus": [1], "minus": [2], "scale": 1.0},
            }
        )
        assert pickup.buttons == 2
        assert pickup.scale_y == 1.0


AXES = (0.0, math.pi / 2.0, math.pi, -math.pi / 2.0)
DIAGONALS = (math.pi / 4.0, 3.0 * math.pi / 4.0, -3.0 * math.pi / 4.0, -math.pi / 4.0)


def round_table(
    directory: Path, *, half_span: float = 0.008, points: int = 33, name: str = "pickup.json"
) -> Path:
    """A table of the round pickup's own responses, with an arbitrary common factor."""
    grid = np.linspace(-half_span, half_span, points)
    mesh_x, mesh_y = np.meshgrid(grid, grid)
    values = 3.7 * round_response(
        mesh_x, mesh_y, RADIUS, np.radians(ANGLES), math.radians(HALF_WIDTH)
    )
    path = directory / name
    path.write_text(
        json.dumps({"x": grid.tolist(), "y": grid.tolist(), "buttons": values.tolist()}),
        encoding="utf-8",
    )
    return path


def table(name: str = "pickup.json", **changes: Any) -> dict[str, Any]:
    stated = {
        "shape": "table",
        "table": name,
        "buttons": 4,
        "x": ROUND["x"],
        "y": ROUND["y"],
    }
    stated.update(changes)
    return stated


class TestATabulatedPickup:
    @pytest.mark.parametrize(("how", "tolerance"), [("bilinear", 2.0e-3), ("bicubic", 5.0e-5)])
    def test_a_table_of_the_round_formula_agrees_with_it(self, tmp_path, how, tolerance):
        round_table(tmp_path)
        tabulated = read(table(interpolation=how), tmp_path)
        analytic = read(ROUND)
        assert isinstance(tabulated, TablePickup)
        rng = np.random.default_rng(7)
        x = rng.uniform(-0.007, 0.007, 400)
        y = rng.uniform(-0.007, 0.007, 400)
        np.testing.assert_allclose(
            tabulated.response(x, y), analytic.response(x, y), rtol=tolerance
        )

    def test_bilinear_is_the_default(self, tmp_path):
        round_table(tmp_path)
        default = read(table(), tmp_path)
        bilinear = read(table(interpolation="bilinear"), tmp_path)
        np.testing.assert_array_equal(
            default.response(0.0031, -0.0012), bilinear.response(0.0031, -0.0012)
        )

    @pytest.mark.parametrize("how", ["bilinear", "bicubic"])
    def test_it_reads_the_table_itself_on_a_grid_point(self, tmp_path, how):
        round_table(tmp_path)
        tabulated = read(table(interpolation=how), tmp_path)
        analytic = read(ROUND)
        np.testing.assert_allclose(
            tabulated.response(0.004, -0.002), analytic.response(0.004, -0.002), rtol=1.0e-12
        )

    @pytest.mark.parametrize("how", ["bilinear", "bicubic"])
    def test_its_buttons_sum_to_one_for_a_centred_beam(self, tmp_path, how):
        round_table(tmp_path)
        tabulated = read(table(interpolation=how), tmp_path)
        assert tabulated.response(0.0, 0.0).sum() == pytest.approx(1.0, rel=1.0e-14)

    @pytest.mark.parametrize("how", ["bilinear", "bicubic"])
    def test_a_position_off_the_grid_reads_at_its_edge(self, tmp_path, how):
        round_table(tmp_path)
        tabulated = read(table(interpolation=how), tmp_path)
        np.testing.assert_allclose(
            tabulated.response(0.05, 0.001), tabulated.response(0.008, 0.001), rtol=1.0e-12
        )
        np.testing.assert_allclose(
            tabulated.response(-0.05, -0.2), tabulated.response(-0.008, -0.008), rtol=1.0e-12
        )

    @pytest.mark.parametrize("how", ["bilinear", "bicubic"])
    def test_its_default_scale_is_the_round_pickups(self, tmp_path, how, tolerance=2.0e-3):
        # Either way the slope at a grid point is the table's own central
        # difference there, so the scale is as fine as the grid.
        round_table(tmp_path)
        tabulated = read(table(interpolation=how), tmp_path)
        analytic = read(ROUND)
        assert tabulated.scale_x == pytest.approx(analytic.scale_x, rel=tolerance)
        assert tabulated.scale_y == pytest.approx(analytic.scale_y, rel=tolerance)

    def test_the_table_is_found_beside_the_directory_given(self, tmp_path):
        (tmp_path / "tables").mkdir()
        round_table(tmp_path / "tables", name="bpm.json")
        assert read(table("tables/bpm.json"), tmp_path).buttons == 4

    def test_a_grid_need_not_be_square(self, tmp_path):
        grid_x = np.linspace(-0.004, 0.006, 21)
        grid_y = np.linspace(-0.002, 0.002, 9)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        values = round_response(
            mesh_x, mesh_y, RADIUS, np.radians(ANGLES), math.radians(HALF_WIDTH)
        )
        (tmp_path / "pickup.json").write_text(
            json.dumps({"x": grid_x.tolist(), "y": grid_y.tolist(), "buttons": values.tolist()})
        )
        tabulated = read(table(interpolation="bicubic"), tmp_path)
        analytic = read(ROUND)
        np.testing.assert_allclose(
            tabulated.response(0.0052, 0.0013), analytic.response(0.0052, 0.0013), rtol=1.0e-4
        )


class TestAStatementIsRefused:
    @pytest.mark.parametrize(
        ("stated", "says"),
        [
            ("round", "is not a map"),
            (changed(shape="oval"), "shape is 'oval'"),
            (changed(shape=None), "shape is None"),
            (changed(gap=0.001), "unknown keys gap"),
            (changed(interpolation="bicubic"), "unknown keys interpolation"),
            (changed(radius=0.0), "radius is 0.0"),
            (changed(radius=None), "radius is None"),
            (changed(radius="wide"), "radius is 'wide'"),
            (changed(buttons=[45.0]), "buttons is [45.0]"),
            (changed(buttons=4), "buttons is 4"),
            (changed(buttons=[45.0, "left"]), "buttons is [45.0, 'left']"),
            (changed(half_width=None), "neither of half_width and diameter"),
            (changed(diameter=0.006), "both of half_width and diameter"),
            (changed(half_width=0.0), "half_width is 0.0"),
            (changed(half_width=120.0), "half_width is 120.0"),
            (changed(half_width=None, diameter=1.0), "diameter is 1.0"),
            (changed(x=None), "pickup.x is None, not a map"),
            (changed(y=[1, 2]), "pickup.y is [1, 2], not a map"),
            (changed(x={"plus": [1, 4]}), "pickup.x.minus is None"),
            (changed(x={"minus": [1, 4]}), "pickup.x.plus is None"),
            (changed(x={"plus": [], "minus": [2]}), "pickup.x.plus is []"),
            (changed(x={"plus": [1, 5], "minus": [2]}), "names button 5; the pickup has buttons"),
            (changed(x={"plus": [0], "minus": [2]}), "names button 0"),
            (changed(x={"plus": [1.0], "minus": [2]}), "names button 1.0"),
            (changed(x={"plus": [1], "minus": [1, 2]}), "names button 1 twice"),
            (changed(x={"plus": [1], "minus": [2], "gain": 2}), "pickup.x carries unknown keys"),
            (changed(x={"plus": [1], "minus": [2], "scale": 0}), "pickup.x.scale is 0"),
            (changed(x={"plus": [1], "minus": [2], "scale": "k"}), "pickup.x.scale is 'k'"),
            (changed(x={"plus": [1, 2], "minus": [3, 4]}), "does not move with the beam along x"),
        ],
    )
    def test_a_round_statement_that_is_wrong(self, stated, says):
        with pytest.raises(Refused) as refused:
            read(stated)
        assert says in refused.value.detail
        assert refused.value.remedy

    def test_keys_beside_the_pickups_own_are_let_through_by_name(self):
        stated = changed(elements=["BPM1"])
        with pytest.raises(Refused, match="unknown keys elements"):
            read(stated)
        assert read(stated, extra_keys=("elements",)).buttons == 4

    @pytest.mark.parametrize(
        ("changes", "says"),
        [
            ({"buttons": [45.0, 135.0]}, "buttons is [45.0, 135.0]"),
            ({"buttons": 1}, "buttons is 1"),
            ({"interpolation": "spline"}, "interpolation is 'spline'"),
            ({"table": None}, "table is None"),
            ({"radius": 0.01}, "unknown keys radius"),
        ],
    )
    def test_a_table_statement_that_is_wrong(self, tmp_path, changes, says):
        round_table(tmp_path)
        with pytest.raises(Refused) as refused:
            read(table(**changes), tmp_path)
        assert says in refused.value.detail

    def test_a_missing_table_is_named_with_its_resolved_path(self, tmp_path):
        with pytest.raises(Refused) as refused:
            read(table("tables/none.json"), tmp_path)
        assert "tables/none.json" in refused.value.detail
        assert str(tmp_path / "tables" / "none.json") in refused.value.detail
        assert "cannot be read" in refused.value.detail

    def test_a_table_that_is_not_json_is_named_with_its_resolved_path(self, tmp_path):
        (tmp_path / "pickup.json").write_text("x: [0, 1]", encoding="utf-8")
        with pytest.raises(Refused) as refused:
            read(table(), tmp_path)
        assert str(tmp_path / "pickup.json") in refused.value.detail
        assert "is not JSON" in refused.value.detail

    @pytest.mark.parametrize(
        ("spoil", "says"),
        [
            (lambda document: [document], "is not a JSON map"),
            (lambda document: document | {"x": [0.0]}, "x is not a list of two or more numbers"),
            (lambda document: document | {"y": "wide"}, "y is not a list of two or more numbers"),
            (
                lambda document: document | {"x": document["x"][::-1]},
                "x is not strictly increasing",
            ),
            (
                lambda document: document | {"x": [-0.008, -0.002, 0.001, 0.004, 0.008]},
                "x is not a regular grid",
            ),
            (
                lambda document: document | {"y": [0.001, 0.002, 0.003, 0.004, 0.005]},
                "does not hold 0",
            ),
            (
                lambda document: document | {"buttons": document["buttons"][:3]},
                "buttons has shape [3, 5, 5]",
            ),
            (
                lambda document: document | {"buttons": [rows[:4] for rows in document["buttons"]]},
                "buttons has shape [4, 4, 5]",
            ),
            (lambda document: document | {"buttons": "none"}, "buttons is not an array of numbers"),
            (
                lambda document: (
                    document | {"buttons": (-np.asarray(document["buttons"])).tolist()}
                ),
                "negative or not finite",
            ),
            (
                lambda document: (
                    document | {"buttons": np.zeros_like(document["buttons"]).tolist()}
                ),
                "sum to 0 at the centre",
            ),
        ],
    )
    def test_a_malformed_table_is_named_with_its_resolved_path(self, tmp_path, spoil, says):
        path = round_table(tmp_path, points=5)
        document = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(spoil(document)), encoding="utf-8")
        with pytest.raises(Refused) as refused:
            read(table(), tmp_path)
        assert says in refused.value.detail
        assert str(path) in refused.value.detail


class TestTheButtonsAStatementDeclares:
    @pytest.mark.parametrize(
        ("stated", "count"),
        [
            (ROUND, 4),
            ({"shape": "round", "buttons": [0.0, 180.0]}, 2),
            ({"shape": "table", "buttons": 6}, 6),
            ({"buttons": [0.0, 90.0, 180.0]}, 3),
            ({"shape": "table", "buttons": [0.0, 180.0]}, None),
            ({"shape": "round", "buttons": 4}, None),
            ({"shape": "table", "buttons": True}, None),
            ({"shape": "round", "buttons": [45.0]}, None),
            ({"shape": "round"}, None),
            ("round", None),
            (None, None),
        ],
    )
    def test_they_are_read_without_checking_the_rest(self, stated, count):
        assert stated_buttons(stated) == count

    def test_a_blocks_counts_are_its_rings_and_its_items(self):
        block = {
            "pickup": ROUND,
            "pickups": [
                {"elements": ["BPM1", "BPM2"], "shape": "table", "buttons": 6},
                {"elements": ["BPM2"], "shape": "round", "buttons": [0.0, 180.0]},
                {"elements": "BPM3", "shape": "table", "buttons": 4},
                {"elements": ["BPM4"], "shape": "table"},
                "BPM5",
            ],
        }
        assert stated_counts({"pyat": {"turn_by_turn": block}}) == (4, {"BPM1": 6, "BPM2": 2})
        assert stated_counts({"pyat": {"turn_by_turn": {"turns": 5}}}) == (None, {})
        assert stated_counts({"pyat": {"turn_by_turn": "on"}}) == (None, {})
        assert stated_counts(None) == (None, {})
