"""The pyat engine's ``response_matrix(..., kind="trajectory")``: the first-turn response.

A row is one wired turn-series record, read through that record's own
calibration: every horizontal record in the order the launched particle meets
the monitors, then every vertical one. A column is one corrector setpoint in
its hardware unit, the horizontal correctors before the vertical ones. Every
row and column comes with its element's index in the lattice as tracked: the
deck rotated to the launch marker for a periodic model, the deck as written
for a line. The oracle is pyAT's own tracking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

at = pytest.importorskip("at")

import numpy as np  # noqa: E402

from osprey.facility.errors import FacilityBuildError  # noqa: E402
from osprey.simulation.engines import pyat as engine  # noqa: E402

N_CELLS = 8
CELLS = tuple(range(1, N_CELLS + 1))
MARKER = "INJ"
#: The cell whose monitor the launch marker follows.
MARKER_AFTER = 3
#: The cells in the order a particle launched at the marker meets them.
ROTATED_CELLS = (*CELLS[MARKER_AFTER:], *CELLS[:MARKER_AFTER])

#: The settings launch from the identity: the matrix is linearised about it,
#: whatever launch a scenario later gives the built model's shots.
SETTINGS = {"pyat": {"injection": {"marker": MARKER}}}

#: Hardware per metre of a turn-series record, and of an orbit record.
SERIES_GAIN = 1.0e9
ORBIT_GAIN = 1.0e3
#: Hardware per radian of a corrector.
KICK_GAIN = 2.0e4

QUADRUPOLE = "QF:01:SP"
STEP = 0.5


def _linear(gain: float) -> dict[str, Any]:
    return {"curve": {"linear": {"gain": gain, "offset": 0.0}}, "energy_scaling": "none"}


def _elements(aperture: float | None = None) -> list[Any]:
    """Eight FODO cells, both correctors and a monitor in each; the marker after the third."""
    elements: list[Any] = []
    for cell in CELLS:
        monitor = at.Monitor(f"BPM_{cell:02d}")
        if aperture is not None:
            monitor.RApertures = [-aperture, aperture, -aperture, aperture]
        elements += [
            at.Drift("DRIFT", 0.4),
            at.Quadrupole(f"QF_{cell:02d}", 0.3, 1.0),
            at.Drift("DRIFT", 0.4),
            at.Corrector(f"HCOR_{cell:02d}", 0.0, [0.0, 0.0]),
            at.Corrector(f"VCOR_{cell:02d}", 0.0, [0.0, 0.0]),
            at.Dipole(f"BEND_{cell:02d}", 1.0, 2 * np.pi / N_CELLS),
            monitor,
        ]
        if cell == MARKER_AFTER:
            elements.append(at.Marker(MARKER))
        elements += [at.Drift("DRIFT", 0.8), at.Quadrupole(f"QD_{cell:02d}", 0.3, -1.0)]
    return elements


def _save(path: Path, elements: list[Any]) -> Path:
    lattice = at.Lattice(elements, name="TEST", energy=3.0e9, particle="electron", periodicity=1)
    lattice.disable_6d()
    at.save_lattice(lattice, str(path))
    return path


@pytest.fixture
def deck(tmp_path: Path) -> Path:
    return _save(tmp_path / "periodic.json", _elements())


def series(cell: int, axis: str) -> str:
    return f"BPM:{cell:02d}:TBT:{axis.upper()}"


def hcor(cell: int) -> str:
    return f"HCOR:{cell:02d}:SP"


def vcor(cell: int) -> str:
    return f"VCOR:{cell:02d}:SP"


def wiring(series_gain: float = SERIES_GAIN, kick_gain: float = KICK_GAIN) -> list[dict[str, Any]]:
    """Orbit and turn-series records of every monitor, every corrector, one quadrupole.

    The records are written against the lattice order and with the vertical
    plane first, so no row or column order can come from the order they are
    written in.
    """
    records: list[dict[str, Any]] = []
    for cell in reversed(CELLS):
        for axis in ("sum", "y", "x"):
            records.append(
                {
                    "address": series(cell, axis),
                    "element": f"BPM_{cell:02d}",
                    "direction": "read",
                    "engine": {"axis": axis, "series": "turn"},
                    **({} if axis == "sum" else {"calibration": _linear(1.0 / series_gain)}),
                }
            )
        for axis in ("y", "x"):
            records.append(
                {
                    "address": f"BPM:{cell:02d}:{axis.upper()}",
                    "element": f"BPM_{cell:02d}",
                    "direction": "read",
                    "engine": {"axis": axis},
                    "calibration": _linear(1.0 / ORBIT_GAIN),
                }
            )
        for address, element, index in (
            (vcor(cell), f"VCOR_{cell:02d}", 1),
            (hcor(cell), f"HCOR_{cell:02d}", 0),
        ):
            records.append(
                {
                    "address": address,
                    "element": element,
                    "direction": "write",
                    "engine": {"attribute": "KickAngle", "index": index},
                    "default": 0.0,
                    "calibration": _linear(1.0 / kick_gain),
                }
            )
    records.append(
        {
            "address": QUADRUPOLE,
            "element": "QF_01",
            "direction": "write",
            "engine": {"attribute": "PolynomB", "index": 1},
            "default": 100.0,
            "calibration": _linear(1.0e-2),
        }
    )
    return records


def _response(deck: Path, correctors: list[str], **changes: Any) -> Any:
    return engine.response_matrix(
        deck,
        changes.get("wiring", wiring()),
        changes.get("settings", SETTINGS),
        correctors,
        dict.fromkeys(correctors, changes.get("step", STEP)),
        kind="trajectory",
    )


def _rotated(deck: Path) -> Any:
    """The deck as a particle launched at the marker meets it."""
    lattice = at.load_lattice(str(deck))
    return lattice.rotate(int(lattice.get_uint32_index(MARKER)[0]))


def _index(lattice: Any, name: str) -> int:
    return int(lattice.get_uint32_index(name)[0])


def _row_index(response: Any) -> list[int]:
    return [response.index[row] for row in response.rows]


def _column_index(response: Any) -> list[int]:
    return [response.index[column] for column in response.columns]


def _first_turn(deck: Path, element: str, plane: int, kick: float) -> np.ndarray:
    """The turn pyAT itself tracks from the marker, launched on axis: x then y, in metres."""
    lattice = _rotated(deck)
    angles = [0.0, 0.0]
    angles[plane] = kick
    lattice[_index(lattice, element)].KickAngle = angles
    monitors = lattice.get_uint32_index(at.Monitor)
    start = np.zeros((6, 1))
    r_out = at.lattice_track(lattice, start, nturns=1, refpts=monitors)[0]
    return np.concatenate([r_out[0, 0, :, 0], r_out[2, 0, :, 0]])


class TestRowsAndColumns:
    def test_rows_are_the_series_records_x_then_y_from_the_marker_on(self, deck):
        response = _response(deck, [hcor(1)])

        assert response.rows == [series(cell, axis) for axis in "xy" for cell in ROTATED_CELLS]
        assert response.matrix.shape == (2 * N_CELLS, 1)

    def test_the_answer_is_the_response_matrix_carrying_an_index(self, deck):
        response = _response(deck, [hcor(1), vcor(1)])

        assert isinstance(response, engine.ResponseMatrix)
        assert response.unsolved == {}
        assert response.index is not None
        assert list(response.index) == [*response.rows, *response.columns]

    def test_every_row_carries_its_index_in_the_rotated_lattice(self, deck):
        response = _response(deck, [hcor(1)])
        rotated = _rotated(deck)

        block = [_index(rotated, f"BPM_{cell:02d}") for cell in ROTATED_CELLS]
        assert _row_index(response) == block + block
        assert block == sorted(block)
        assert rotated[0].FamName == MARKER

    def test_columns_are_the_horizontal_correctors_then_the_vertical_ones(self, deck):
        named = [vcor(2), hcor(5), vcor(7), hcor(1)]
        response = _response(deck, named)
        rotated = _rotated(deck)

        assert response.columns == [hcor(5), hcor(1), vcor(2), vcor(7)]
        assert _column_index(response) == [
            _index(rotated, "HCOR_05"),
            _index(rotated, "HCOR_01"),
            _index(rotated, "VCOR_02"),
            _index(rotated, "VCOR_07"),
        ]
        assert response.matrix.shape[1] == len(response.columns)

    def test_no_corrector_is_left_out(self, deck):
        named = [*(hcor(cell) for cell in CELLS), *(vcor(cell) for cell in CELLS)]

        response = _response(deck, named)

        assert response.columns == named
        assert response.matrix.shape == (2 * N_CELLS, 2 * N_CELLS)

    def test_an_unwired_series_record_has_no_row(self, deck):
        dropped = series(5, "y")
        records = [record for record in wiring() if record["address"] != dropped]

        response = _response(deck, [hcor(1)], wiring=records)

        assert dropped not in response.rows
        assert len(response.rows) == len(_row_index(response)) == 2 * N_CELLS - 1

    def test_a_model_wiring_no_series_record_has_no_row(self, deck):
        records = [record for record in wiring() if "series" not in record["engine"]]

        response = _response(deck, [hcor(1)], wiring=records)

        assert response.rows == [] and _row_index(response) == []
        assert response.matrix.shape == (0, 1)


class TestMatrix:
    def test_a_column_is_the_tracked_central_difference(self, deck):
        response = _response(deck, [hcor(1), vcor(6)])

        for column, (element, plane) in enumerate((("HCOR_01", 0), ("VCOR_06", 1))):
            kick = 0.5 * STEP / KICK_GAIN
            metres = _first_turn(deck, element, plane, kick) - _first_turn(
                deck, element, plane, -kick
            )
            np.testing.assert_allclose(
                response.matrix[:, column], metres * SERIES_GAIN / STEP, rtol=1e-9, atol=1e-6
            )

    def test_a_corrector_moves_only_the_monitors_downstream_of_it(self, deck):
        response = _response(deck, [hcor(1)])
        horizontal = dict(zip(response.rows[:N_CELLS], response.matrix[:N_CELLS, 0], strict=True))

        # Launched after the third monitor, the particle meets cells 4..8
        # before the first cell's corrector.
        for cell in CELLS[MARKER_AFTER:]:
            assert horizontal[series(cell, "x")] == 0.0
        for cell in CELLS[:MARKER_AFTER]:
            assert abs(horizontal[series(cell, "x")]) > 0.0

    def test_rows_go_through_the_series_calibration_not_the_orbit_one(self, deck):
        base = _response(deck, [hcor(1), vcor(1)])
        scaled = _response(deck, [hcor(1), vcor(1)], wiring=wiring(series_gain=4.0 * SERIES_GAIN))

        assert np.max(np.abs(base.matrix)) > 0.0
        np.testing.assert_allclose(scaled.matrix, 4.0 * base.matrix, rtol=1e-12)

    def test_a_column_is_per_hardware_unit_of_its_corrector(self, deck):
        base = _response(deck, [hcor(1), vcor(1)])
        scaled = _response(
            deck,
            [hcor(1), vcor(1)],
            wiring=wiring(kick_gain=8.0 * KICK_GAIN),
            step=8.0 * STEP,
        )

        np.testing.assert_allclose(scaled.matrix, base.matrix / 8.0, rtol=1e-12)

    def test_the_deck_is_left_as_it_was(self, deck):
        setpoints = [record for record in wiring() if record["direction"] == "write"]
        before = engine.start_values(deck, setpoints, SETTINGS)

        _response(deck, [hcor(1), vcor(1)])

        assert engine.start_values(deck, setpoints, SETTINGS) == before


class TestRefusals:
    def test_a_periodic_model_without_an_injection_block_is_refused(self, deck):
        with pytest.raises(FacilityBuildError) as raised:
            _response(deck, [hcor(1)], settings=None)

        assert raised.value.kind == "engine-invalid"
        assert "pyat.injection" in str(raised.value)

    def test_a_kind_the_engine_does_not_compute_is_refused(self, deck):
        with pytest.raises(ValueError, match="dispersion"):
            engine.response_matrix(
                deck, wiring(), SETTINGS, [hcor(1)], {hcor(1): STEP}, kind="dispersion"
            )

    def test_a_setpoint_that_steers_no_plane_is_refused(self, deck):
        with pytest.raises(ValueError, match=QUADRUPOLE):
            _response(deck, [hcor(1), QUADRUPOLE])

    def test_a_corrector_that_is_not_a_wired_setpoint_is_refused(self, deck):
        with pytest.raises(ValueError, match="NOT:A:SETPOINT"):
            _response(deck, ["NOT:A:SETPOINT"])

    def test_a_corrector_without_a_step_is_refused(self, deck):
        with pytest.raises(ValueError, match=vcor(2)):
            engine.response_matrix(deck, wiring(), SETTINGS, [vcor(2)], {}, kind="trajectory")

    def test_a_step_that_loses_the_particle_is_an_unsolved_corrector(self, tmp_path):
        """Only the corrector whose kick loses the particle has no column; it keeps its index."""
        tight = _save(tmp_path / "tight.json", _elements(aperture=5.0e-3))

        named = [vcor(4), hcor(4)]
        response = engine.response_matrix(
            tight,
            wiring(),
            SETTINGS,
            named,
            {vcor(4): STEP, hcor(4): 0.2 * KICK_GAIN},
            kind="trajectory",
        )

        assert list(response.unsolved) == [hcor(4)]
        text = response.unsolved[hcor(4)]
        assert text.startswith("lost at element")
        assert text.endswith(f"(while stepping {hcor(4)})")
        assert response.columns == [vcor(4)]
        assert response.matrix.shape == (len(response.rows), 1)
        assert response.index is not None and hcor(4) in response.index


LINE_CELLS = (1, 2, 3)
LINE_SETTINGS = {
    "pyat": {
        "solve": "single_pass",
        "twiss_in": {
            "beta": [5.0, 3.0],
            "alpha": [0.1, -0.2],
            "closed_orbit": [1.0e-4, 2.0e-5, -5.0e-5, 1.0e-5],
        },
    }
}


@pytest.fixture
def line(tmp_path: Path) -> Path:
    """Three FODO cells, both correctors and a monitor in each, saved as a line."""
    elements: list[Any] = []
    for cell in LINE_CELLS:
        elements += [
            at.Drift(f"DA_{cell}", 0.5),
            at.Quadrupole(f"QF_{cell}", 0.3, 1.2),
            at.Drift(f"DB_{cell}", 0.5),
            at.Corrector(f"HCOR_{cell}", 0.0, [0.0, 0.0]),
            at.Corrector(f"VCOR_{cell}", 0.0, [0.0, 0.0]),
            at.Monitor(f"BPM_{cell}"),
            at.Drift(f"DC_{cell}", 0.5),
            at.Quadrupole(f"QD_{cell}", 0.3, -1.1),
        ]
    lattice = at.Lattice(elements, energy=3e9, particle="electron", periodicity=1)
    path = tmp_path / "line.json"
    at.save_lattice(lattice, str(path))
    return path


def line_wiring() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for cell in reversed(LINE_CELLS):
        for axis in ("y", "x"):
            records.append(
                {
                    "address": f"L:BPM_{cell}:TBT:{axis.upper()}",
                    "element": f"BPM_{cell}",
                    "direction": "read",
                    "engine": {"axis": axis, "series": "turn"},
                }
            )
        for plane, index in (("V", 1), ("H", 0)):
            records.append(
                {
                    "address": f"L:{plane}COR_{cell}:SP",
                    "element": f"{plane}COR_{cell}",
                    "direction": "write",
                    "engine": {"attribute": "KickAngle", "index": index},
                    "default": 0.0,
                }
            )
    return records


def _pass(deck: Path, element: str, plane: int, kick: float) -> np.ndarray:
    """The pass pyAT itself tracks down the line: every x, then every y."""
    lattice = at.load_lattice(str(deck))
    angles = [0.0, 0.0]
    angles[plane] = kick
    lattice[_index(lattice, element)].KickAngle = angles
    monitors = lattice.get_uint32_index(at.Monitor)
    start = engine.prepare(deck, LINE_SETTINGS).twiss_in["closed_orbit"].reshape(6, 1)
    r_out = at.lattice_track(lattice, start, refpts=monitors)[0]
    return np.concatenate([r_out[0, 0, :, 0], r_out[2, 0, :, 0]])


class TestLine:
    def test_a_line_answers_plane_major_with_the_deck_s_own_indices(self, line):
        named = ["L:VCOR_2:SP", "L:HCOR_1:SP"]
        step = 2.0e-4

        response = engine.response_matrix(
            line, line_wiring(), LINE_SETTINGS, named, dict.fromkeys(named, step), kind="trajectory"
        )

        lattice = at.load_lattice(str(line))
        block = [_index(lattice, f"BPM_{cell}") for cell in LINE_CELLS]
        assert response.rows == [f"L:BPM_{cell}:TBT:{axis}" for axis in "XY" for cell in LINE_CELLS]
        assert _row_index(response) == block + block
        assert response.columns == ["L:HCOR_1:SP", "L:VCOR_2:SP"]
        assert _column_index(response) == [_index(lattice, "HCOR_1"), _index(lattice, "VCOR_2")]

    def test_a_line_column_is_the_tracked_central_difference(self, line):
        named = ["L:HCOR_1:SP", "L:VCOR_2:SP"]
        step = 2.0e-4

        response = engine.response_matrix(
            line, line_wiring(), LINE_SETTINGS, named, dict.fromkeys(named, step), kind="trajectory"
        )

        for column, (element, plane) in enumerate((("HCOR_1", 0), ("VCOR_2", 1))):
            expected = (
                _pass(line, element, plane, step / 2) - _pass(line, element, plane, -step / 2)
            ) / step
            np.testing.assert_allclose(response.matrix[:, column], expected, rtol=1e-12, atol=1e-15)
        assert np.max(np.abs(response.matrix)) > 0.0
