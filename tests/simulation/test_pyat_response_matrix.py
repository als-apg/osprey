"""The pyat engine's ``response_matrix``: the orbit response over wired correctors.

Each column is one corrector stepped ±½ its step in hardware units, each row one
wired monitor readback in its own hardware unit. The SR cases read the built
demo's model as the build renders it; the single-pass case reads a three-cell
synthetic line.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

at = pytest.importorskip("at")

import numpy as np  # noqa: E402

from osprey.facility.records import FacilityRecords  # noqa: E402
from osprey.facility.views.simulator import simulator_wiring  # noqa: E402
from osprey.simulation.engines import pyat as engine  # noqa: E402

MODEL = "SR"
CORRECTOR_X = "SR:MAG:HCM:01:CURRENT:SP"
CORRECTOR_Y = "SR:MAG:VCM:01:CURRENT:SP"
SR_MONITORS = 144


@pytest.fixture(scope="module")
def demo(built_demo: Any) -> dict[str, Any]:
    """The SR model of the built demo: its wiring entries, deck and settings."""
    facility = built_demo.facility
    (model,) = [entry for entry in facility["models"] if entry["name"] == MODEL]
    return {
        "wiring": simulator_wiring(FacilityRecords(facility), MODEL),
        "deck": built_demo.facility_dir / model["deck"],
        "settings": model.get("settings"),
    }


def _is_monitor(record: dict[str, Any]) -> bool:
    return "axis" in record["engine"] and "attribute" not in record["engine"]


def _lattice_ordered_monitors(deck: Path, wiring: list[dict[str, Any]]) -> list[str]:
    """Every wired monitor readback, by (element lattice index, x before y)."""
    lattice = at.load_lattice(str(deck))
    index = {element.FamName: position for position, element in enumerate(lattice)}
    records = [record for record in wiring if _is_monitor(record)]
    records.sort(key=lambda record: (index[record["element"]], record["engine"]["axis"]))
    return [record["address"] for record in records]


def _matrix(demo: dict[str, Any], correctors: list[str], step: float, **changes: Any) -> Any:
    return engine.response_matrix(
        demo["deck"],
        changes.get("wiring", demo["wiring"]),
        demo["settings"],
        correctors,
        dict.fromkeys(correctors, step),
    )


def _linear(gain: float, offset: float) -> dict[str, Any]:
    return {"curve": {"linear": {"gain": gain, "offset": offset}}}


class TestRows:
    def test_the_sr_rows_are_the_144_monitor_readbacks_in_lattice_order(self, demo):
        result = _matrix(demo, [CORRECTOR_X, CORRECTOR_Y], 0.5)

        assert len(result.rows) == SR_MONITORS
        assert result.rows == _lattice_ordered_monitors(demo["deck"], demo["wiring"])
        assert result.matrix.shape == (SR_MONITORS, 2)

    def test_an_unwired_monitor_has_no_row(self, demo):
        dropped = next(record for record in demo["wiring"] if _is_monitor(record))
        wiring = [record for record in demo["wiring"] if record is not dropped]

        result = _matrix(demo, [CORRECTOR_X], 0.5, wiring=wiring)

        assert dropped["address"] not in result.rows
        assert result.matrix.shape == (SR_MONITORS - 1, 1)

    def test_columns_follow_the_corrector_order(self, demo):
        forward = _matrix(demo, [CORRECTOR_X, CORRECTOR_Y], 0.5)
        backward = _matrix(demo, [CORRECTOR_Y, CORRECTOR_X], 0.5)

        np.testing.assert_array_equal(forward.matrix, backward.matrix[:, ::-1])

    def test_a_horizontal_corrector_moves_the_horizontal_rows(self, demo):
        result = _matrix(demo, [CORRECTOR_X], 0.5)
        horizontal = [row for row, address in enumerate(result.rows) if address.endswith(":X")]

        assert np.max(np.abs(result.matrix[horizontal, 0])) > 0.0


def _lattice_index(deck: Path) -> dict[str, int]:
    return {element.FamName: place for place, element in enumerate(at.load_lattice(str(deck)))}


def _lattice_length(deck: Path) -> int:
    """The deck's element count, repeated names included."""
    return len(at.load_lattice(str(deck)))


class TestPositions:
    def test_every_orbit_row_carries_its_monitor_elements_lattice_index(self, demo):
        result = _matrix(demo, [CORRECTOR_X], 0.5)
        index = _lattice_index(demo["deck"])
        element = {record["address"]: record.get("element") for record in demo["wiring"]}

        assert result.row_positions == [index[element[row]] for row in result.rows]
        assert result.row_positions == sorted(result.row_positions)

    def test_every_orbit_column_carries_its_corrector_elements_lattice_index(self, demo):
        result = _matrix(demo, [CORRECTOR_Y, CORRECTOR_X], 0.5)
        index = _lattice_index(demo["deck"])
        element = {record["address"]: record.get("element") for record in demo["wiring"]}

        assert result.column_positions == [index[element[column]] for column in result.columns]

    def test_an_unsolved_corrector_has_no_position(self, demo, monkeypatch):
        clean = _matrix(demo, [CORRECTOR_X, CORRECTOR_Y], 0.5)
        _refuse_solve(monkeypatch, call=1)

        result = _matrix(demo, [CORRECTOR_X, CORRECTOR_Y], 0.5)

        assert result.columns == [CORRECTOR_Y]
        assert result.column_positions == [clean.column_positions[1]]

    def test_positions_are_whole_numbers(self, demo):
        result = _matrix(demo, [CORRECTOR_X], 0.5)

        assert all(type(place) is int for place in result.row_positions)
        assert all(type(place) is int for place in result.column_positions)


class TestCalibration:
    def test_a_linear_calibration_scales_the_physics_matrix_by_the_two_gains(self, demo):
        corrector_gain, monitor_gain = 4.0, 1024.0
        physics: list[dict[str, Any]] = []
        calibrated: list[dict[str, Any]] = []
        for record in demo["wiring"]:
            bare = {key: value for key, value in record.items() if key != "calibration"}
            physics.append(bare)
            scaled = copy.deepcopy(bare)
            if _is_monitor(record):
                # The hardware reading is monitor_gain times metres.
                scaled["calibration"] = _linear(1.0 / monitor_gain, 0.0)
            elif record["address"] == CORRECTOR_X:
                # The hardware setting is corrector_gain times radians.
                scaled["calibration"] = _linear(1.0 / corrector_gain, 0.0)
            calibrated.append(scaled)
        physics_step = 1.0e-5

        expected = _matrix(demo, [CORRECTOR_X], physics_step, wiring=physics)
        served = _matrix(demo, [CORRECTOR_X], physics_step * corrector_gain, wiring=calibrated)

        assert served.rows == expected.rows
        np.testing.assert_allclose(
            served.matrix, expected.matrix * monitor_gain / corrector_gain, rtol=1e-12, atol=1e-12
        )


def _refuse_solve(monkeypatch: pytest.MonkeyPatch, call: int) -> None:
    """Make the ``call``-th closed orbit solve raise; every other one solves."""
    original = at.find_orbit6
    calls = {"count": 0}

    def raiser(*args: Any, **kwargs: Any) -> Any:
        calls["count"] += 1
        if calls["count"] == call:
            raise at.AtError("no synchronous phase")
        return original(*args, **kwargs)

    monkeypatch.setattr(at, "find_orbit6", raiser)


class TestSolveErrors:
    def test_a_failed_step_names_the_corrector_and_leaves_its_column_out(self, demo, monkeypatch):
        _refuse_solve(monkeypatch, 3)

        result = _matrix(demo, [CORRECTOR_X, CORRECTOR_Y], 0.5)

        assert result.columns == [CORRECTOR_X]
        assert result.matrix.shape == (SR_MONITORS, 1)
        assert list(result.unsolved) == [CORRECTOR_Y]
        text = result.unsolved[CORRECTOR_Y]
        assert text.startswith("closed orbit solve raised AtError:")
        assert text.endswith(f"(while stepping {CORRECTOR_Y})")

    def test_a_column_after_an_unsolved_one_equals_a_clean_run(self, demo, monkeypatch):
        clean = _matrix(demo, [CORRECTOR_X, CORRECTOR_Y], 0.5)
        _refuse_solve(monkeypatch, 1)

        result = _matrix(demo, [CORRECTOR_X, CORRECTOR_Y], 0.5)

        assert result.columns == [CORRECTOR_Y]
        np.testing.assert_array_equal(result.matrix[:, 0], clean.matrix[:, 1])

    def test_the_deck_is_left_as_it_was_after_an_unsolved_step(self, demo, monkeypatch):
        setpoints = [item for item in demo["wiring"] if item["direction"] == "write"]
        before = engine.start_values(demo["deck"], setpoints, demo["settings"])
        _refuse_solve(monkeypatch, 1)

        _matrix(demo, [CORRECTOR_X, CORRECTOR_Y], 0.5)

        assert engine.start_values(demo["deck"], setpoints, demo["settings"]) == before

    def test_two_correctors_cost_as_many_lattice_copies_as_one(self, demo, monkeypatch):
        original = at.Lattice.deepcopy
        copies = {"count": 0}

        def counting(self: Any, *args: Any, **kwargs: Any) -> Any:
            copies["count"] += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(at.Lattice, "deepcopy", counting)

        _matrix(demo, [CORRECTOR_X], 0.5)
        one = copies["count"]
        copies["count"] = 0
        _matrix(demo, [CORRECTOR_X, CORRECTOR_Y], 0.5)

        assert one > 0
        assert copies["count"] == one

    def test_the_deck_is_left_as_it_was(self, demo):
        setpoints = [item for item in demo["wiring"] if item["direction"] == "write"]
        before = engine.start_values(demo["deck"], setpoints, demo["settings"])

        _matrix(demo, [CORRECTOR_X, CORRECTOR_Y], 0.5)

        assert engine.start_values(demo["deck"], setpoints, demo["settings"]) == before

    def test_a_corrector_that_is_not_a_wired_setpoint_is_refused(self, demo):
        with pytest.raises(engine.RequestRefused, match="SR:NOT:A:SETPOINT"):
            _matrix(demo, ["SR:NOT:A:SETPOINT"], 0.5)

    def test_a_corrector_without_a_step_is_refused(self, demo):
        with pytest.raises(engine.RequestRefused, match=CORRECTOR_Y):
            engine.response_matrix(
                demo["deck"], demo["wiring"], demo["settings"], [CORRECTOR_Y], {}
            )


LINE_CELLS = (1, 2, 3)
LINE_TWISS = {
    "beta": [5.0, 3.0],
    "alpha": [0.1, -0.2],
    "closed_orbit": [1.0e-4, 2.0e-5, -5.0e-5, 1.0e-5],
}
LINE_SETTINGS = {"pyat": {"solve": "single_pass", "twiss_in": LINE_TWISS}}
LINE_KICK = "COR_1:X:SP"


@pytest.fixture
def line(tmp_path: Path) -> Path:
    """Three FODO cells, a corrector and a monitor in each, saved as a line."""
    elements: list[Any] = []
    for cell in LINE_CELLS:
        elements += [
            at.Drift(f"DA_{cell}", 0.5),
            at.Quadrupole(f"QF_{cell}", 0.3, 1.2),
            at.Drift(f"DB_{cell}", 0.5),
            at.Corrector(f"COR_{cell}", 0.0, [0.0, 0.0]),
            at.Monitor(f"BPM_{cell}"),
            at.Drift(f"DC_{cell}", 0.5),
            at.Quadrupole(f"QD_{cell}", 0.3, -1.1),
        ]
    lattice = at.Lattice(elements, energy=3e9, particle="electron", periodicity=1)
    path = tmp_path / "line.json"
    at.save_lattice(lattice, str(path))
    return path


LINE_WIRING: list[dict[str, Any]] = [
    *(
        {
            "id": f"LINE/BPM_{cell}:{axis.upper()}",
            "address": f"BPM_{cell}:{axis.upper()}",
            "direction": "read",
            "element": f"BPM_{cell}",
            "engine": {"axis": axis},
        }
        for cell in reversed(LINE_CELLS)
        for axis in ("y", "x")
    ),
    {
        "id": f"LINE/{LINE_KICK}",
        "address": LINE_KICK,
        "direction": "write",
        "element": "COR_1",
        "engine": {"attribute": "KickAngle", "index": 0},
        "default": 0.0,
    },
]


def _tracked(deck: Path, kick: float) -> np.ndarray:
    """The pass pyAT itself tracks: x then y at every monitor, in lattice order."""
    lattice = at.load_lattice(str(deck))
    lattice[lattice.get_uint32_index("COR_1")[0]].KickAngle = [kick, 0.0]
    monitors = lattice.get_uint32_index(at.Monitor)
    start = engine.prepare(deck, LINE_SETTINGS).twiss_in["closed_orbit"].reshape(6, 1)
    r_out = at.lattice_track(lattice, start, refpts=monitors)[0]
    return np.array([r_out[plane, 0, row, 0] for row in range(len(monitors)) for plane in (0, 2)])


class TestSinglePass:
    def test_a_line_column_is_the_tracked_central_difference(self, line):
        step = 2.0e-4

        result = engine.response_matrix(
            line, LINE_WIRING, LINE_SETTINGS, [LINE_KICK], {LINE_KICK: step}
        )
        rows, matrix = result.rows, result.matrix

        assert rows == [f"BPM_{cell}:{axis}" for cell in LINE_CELLS for axis in ("X", "Y")]
        expected = (_tracked(line, step / 2) - _tracked(line, -step / 2)) / step
        np.testing.assert_allclose(matrix[:, 0], expected, rtol=1e-12, atol=1e-15)
        assert abs(matrix[rows.index("BPM_3:X"), 0]) > 0.0


# -- the first-turn trajectory response -----------------------------------------

INJECTION = "INJ"
TRAJECTORY_LAUNCH = {"x": 1.0e-3, "y": -5.0e-4}


def _turn_settings(demo: dict[str, Any], **block: Any) -> dict[str, Any]:
    """The demo SR settings with a turn-by-turn block launching at the injection marker."""
    settings = copy.deepcopy(demo["settings"]) or {}
    stated: dict[str, Any] = {"injection_element": INJECTION, "launch": dict(TRAJECTORY_LAUNCH)}
    stated.update(block)
    settings.setdefault("pyat", {})["turn_by_turn"] = stated
    return settings


def _turn_record(element: str, plane: str) -> dict[str, Any]:
    """A monitor's turn-buffer record, as a turn-by-turn ring wires it."""
    return {
        "id": f"{element}:TBT:{plane.upper()}",
        "address": f"{element}:TBT:{plane.upper()}",
        "element": element,
        "direction": "read",
        "engine": {"signal": f"turn_{plane}"},
    }


def _monitor_elements(demo: dict[str, Any]) -> list[str]:
    """The demo's monitor elements, in deck order."""
    index = _lattice_index(demo["deck"])
    elements = {record["element"] for record in demo["wiring"] if _is_monitor(record)}
    return sorted(elements, key=index.__getitem__)


def _turn_wiring(demo: dict[str, Any], leave_out: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """The demo SR wiring with both turn-buffer records on every monitor element."""
    records = [
        _turn_record(element, plane) for element in _monitor_elements(demo) for plane in ("y", "x")
    ]
    return [*demo["wiring"], *(record for record in records if record["address"] not in leave_out)]


def _trajectory(
    demo: dict[str, Any],
    settings: dict[str, Any],
    steps: dict[str, float],
    wiring: list[dict[str, Any]] | None = None,
) -> engine.ResponseMatrix:
    return engine.response_matrix(
        demo["deck"],
        _turn_wiring(demo) if wiring is None else wiring,
        settings,
        list(steps),
        steps,
        kind="trajectory",
    )


def _first_turn_order(demo: dict[str, Any], start: str = INJECTION) -> list[tuple[int, str]]:
    """Each turn-buffer row's (place counted from ``start``, address), as the bunch meets them."""
    index = _lattice_index(demo["deck"])
    rows = [
        (
            (index[element] - index[start]) % _lattice_length(demo["deck"]),
            plane,
            f"{element}:TBT:{plane.upper()}",
        )
        for element in _monitor_elements(demo)
        for plane in ("x", "y")
    ]
    return [(place, address) for place, _, address in sorted(rows)]


def _first_turn(
    demo: dict[str, Any], settings: dict[str, Any], rows: list[str], kicks: dict[str, float]
) -> np.ndarray:
    """One particle tracked one turn from INJ by pyAT, each turn-buffer row in metres.

    Each corrector in ``kicks`` is moved by that many hardware units from the
    value the deck holds.
    """
    from lume_pyat.simulator import PyATSimulator

    from osprey.simulation.engines.pyat_variables import EV_PER_GEV, variable_from_wiring

    lattice = at.load_lattice(str(demo["deck"]))
    simulator = PyATSimulator(lattice)
    by_address = {record["address"]: record for record in demo["wiring"]}
    for address, kick in kicks.items():
        variable = variable_from_wiring(
            by_address[address], deck_energy_gev=float(lattice.energy) / EV_PER_GEV
        )
        variable._set(simulator, variable._get(simulator) + kick)
    index = {element.FamName: place for place, element in enumerate(lattice)}
    start = index[INJECTION]
    launch = engine.prepare(demo["deck"], settings).turn_by_turn.launch
    readings = [tuple(address.split(":TBT:")) for address in rows]
    places = sorted({index[element] for element, _ in readings})
    rotated = lattice.rotate(start)
    refpts = np.array(sorted((place - start) % len(lattice) for place in places), np.uint32)
    r_out = at.lattice_track(rotated, np.array(launch, dtype=float).reshape(6, 1), refpts=refpts)[0]
    slot = {int(ref): position for position, ref in enumerate(refpts)}
    values = []
    for element, plane in readings:
        ref = (index[element] - start) % len(lattice)
        values.append(float(r_out[0 if plane == "X" else 2, 0, slot[ref], 0]))
    return np.array(values)


class TestTrajectory:
    def test_the_matrix_times_a_small_kick_predicts_the_tracked_first_turn_difference(self, demo):
        settings = _turn_settings(demo)
        result = _trajectory(demo, settings, {CORRECTOR_X: 0.01, CORRECTOR_Y: 0.01})
        kick = np.array([0.004, -0.003])

        predicted = result.matrix @ kick
        tracked = _first_turn(
            demo, settings, result.rows, dict(zip(result.columns, kick, strict=True))
        ) - _first_turn(demo, settings, result.rows, {})

        assert result.unsolved == {}
        assert result.columns == [CORRECTOR_X, CORRECTOR_Y]
        assert np.max(np.abs(tracked)) > 0.0
        np.testing.assert_allclose(
            predicted, tracked, rtol=1e-3, atol=1e-6 * np.max(np.abs(tracked))
        )

    def test_rows_are_the_turn_buffer_records_in_the_order_the_bunch_meets_them(self, demo):
        result = _trajectory(demo, _turn_settings(demo), {CORRECTOR_X: 0.5})
        expected = _first_turn_order(demo)

        assert result.rows == [address for _, address in expected]
        assert result.row_positions == [place for place, _ in expected]
        assert result.matrix.shape == (SR_MONITORS, 1)

    def test_no_row_is_an_orbit_readback(self, demo):
        orbit = _matrix(demo, [CORRECTOR_X], 0.5)
        result = _trajectory(demo, _turn_settings(demo), {CORRECTOR_X: 0.5})

        assert not set(result.rows) & set(orbit.rows)

    def test_row_positions_count_from_the_injection_element_and_rise_within_a_plane(self, demo):
        result = _trajectory(demo, _turn_settings(demo), {CORRECTOR_X: 0.5})
        length = _lattice_length(demo["deck"])

        assert all(0 <= place < length for place in result.row_positions)
        for plane in ("X", "Y"):
            places = [
                place
                for row, place in zip(result.rows, result.row_positions, strict=True)
                if row.endswith(f":TBT:{plane}")
            ]
            assert len(places) == SR_MONITORS // 2
            assert all(later > earlier for earlier, later in zip(places, places[1:], strict=False))

    def test_an_injection_element_mid_ring_wraps_the_monitors_before_it_to_the_end(self, demo):
        # Launching at a monitor: it is met first, the monitor before it last.
        monitors = _monitor_elements(demo)
        start = monitors[len(monitors) // 2]
        before = monitors[len(monitors) // 2 - 1]
        index = _lattice_index(demo["deck"])
        settings = _turn_settings(demo, injection_element=start)

        result = _trajectory(demo, settings, {CORRECTOR_X: 0.5})

        expected = _first_turn_order(demo, start)
        assert result.rows == [address for _, address in expected]
        assert result.row_positions == [place for place, _ in expected]
        assert result.rows[:2] == [f"{start}:TBT:X", f"{start}:TBT:Y"]
        assert result.row_positions[:2] == [0, 0]
        assert result.rows[-2:] == [f"{before}:TBT:X", f"{before}:TBT:Y"]
        assert result.row_positions[-1] == (index[before] - index[start]) % _lattice_length(
            demo["deck"]
        )
        assert result.row_positions == sorted(result.row_positions)

    def test_column_positions_count_from_the_injection_element(self, demo):
        index = _lattice_index(demo["deck"])
        element = {record["address"]: record.get("element") for record in demo["wiring"]}
        monitors = _monitor_elements(demo)
        start = monitors[len(monitors) // 2]

        result = _trajectory(
            demo,
            _turn_settings(demo, injection_element=start),
            {CORRECTOR_Y: 0.5, CORRECTOR_X: 0.5},
        )

        assert result.columns == [CORRECTOR_Y, CORRECTOR_X]
        assert result.column_positions == [
            (index[element[column]] - index[start]) % _lattice_length(demo["deck"])
            for column in result.columns
        ]

    def test_a_corrector_moves_only_the_rows_placed_after_it(self, demo):
        result = _trajectory(demo, _turn_settings(demo), {CORRECTOR_X: 0.5})
        (corrector,) = result.column_positions
        horizontal = [
            (place, abs(float(value)))
            for row, place, value in zip(
                result.rows, result.row_positions, result.matrix[:, 0], strict=True
            )
            if row.endswith(":TBT:X")
        ]

        assert all(value == 0.0 for place, value in horizontal if place < corrector)
        assert any(value > 0.0 for place, value in horizontal if place > corrector)

    def test_a_monitor_without_a_turn_buffer_record_has_no_row_for_it(self, demo):
        first, second = _monitor_elements(demo)[:2]
        gone = (f"{first}:TBT:X", f"{first}:TBT:Y", f"{second}:TBT:Y")

        result = _trajectory(
            demo, _turn_settings(demo), {CORRECTOR_X: 0.5}, _turn_wiring(demo, leave_out=gone)
        )

        assert not set(gone) & set(result.rows)
        assert f"{second}:TBT:X" in result.rows
        assert len(result.rows) == len(result.row_positions) == SR_MONITORS - 3
        assert result.matrix.shape == (SR_MONITORS - 3, 1)

    def test_a_ring_wiring_no_turn_buffer_record_refuses_the_trajectory(self, demo):
        with pytest.raises(engine.RequestRefused, match="turn_x") as refused:
            _trajectory(demo, _turn_settings(demo), {CORRECTOR_X: 0.5}, demo["wiring"])

        assert "turn_y" in str(refused.value)

    def test_the_lost_particle_answer_still_places_its_rows(self, demo):
        settings = _turn_settings(demo, launch={"x": 0.02}, aperture={"x": 0.005, "y": 0.005})

        result = _trajectory(demo, settings, {CORRECTOR_X: 0.01})

        assert result.columns == [] and result.column_positions == []
        assert result.row_positions == [place for place, _ in _first_turn_order(demo)]

    def test_a_monitor_upstream_of_the_corrector_does_not_respond_until_the_turn_wraps(self, demo):
        # The first turn runs from INJ to INJ: a monitor read before the
        # corrector sees no kick, one read after it does.
        result = _trajectory(demo, _turn_settings(demo), {CORRECTOR_X: 0.5})
        column = np.abs(result.matrix[:, 0])

        assert np.count_nonzero(column == 0.0) > 0
        assert np.count_nonzero(column > 0.0) > 0

    def test_a_kicked_particle_lost_names_the_corrector_and_where(self, demo):
        settings = _turn_settings(demo, aperture={"x": 0.01, "y": 0.01})

        result = _trajectory(demo, settings, {CORRECTOR_X: 1.0e5, CORRECTOR_Y: 0.01})

        assert result.columns == [CORRECTOR_Y]
        assert list(result.unsolved) == [CORRECTOR_X]
        text = result.unsolved[CORRECTOR_X]
        assert "lost on the first turn at element" in text
        assert "before monitor" in text
        assert text.endswith(f"(while stepping {CORRECTOR_X})")
        assert result.matrix.shape == (SR_MONITORS, 1)

    def test_an_unkicked_particle_lost_leaves_every_corrector_unsolved(self, demo):
        settings = _turn_settings(demo, launch={"x": 0.02}, aperture={"x": 0.005, "y": 0.005})

        result = _trajectory(demo, settings, {CORRECTOR_X: 0.01, CORRECTOR_Y: 0.01})

        assert result.columns == []
        assert result.matrix.shape == (SR_MONITORS, 0)
        assert list(result.unsolved) == [CORRECTOR_X, CORRECTOR_Y]
        texts = {text.split(" (with no corrector")[0] for text in result.unsolved.values()}
        (text,) = texts
        assert "lost on the first turn at element" in text

    def test_a_rectangular_aperture_also_loses_the_particle(self, demo):
        settings = _turn_settings(
            demo, launch={"x": 0.02}, aperture={"x": 0.005, "y": 0.005, "shape": "rectangular"}
        )

        result = _trajectory(demo, settings, {CORRECTOR_X: 0.01})

        assert result.columns == []
        assert list(result.unsolved) == [CORRECTOR_X]

    def test_a_ring_that_is_not_turn_by_turn_refuses_the_trajectory(self, demo):
        with pytest.raises(engine.RequestRefused, match="turn-by-turn"):
            _trajectory(demo, copy.deepcopy(demo["settings"]), {CORRECTOR_X: 0.5})

    def test_an_unknown_kind_is_refused(self, demo):
        with pytest.raises(engine.RequestRefused, match="'dispersion'"):
            engine.response_matrix(
                demo["deck"],
                demo["wiring"],
                demo["settings"],
                [CORRECTOR_X],
                {CORRECTOR_X: 0.5},
                kind="dispersion",
            )

    def test_a_bad_request_is_still_refused_when_the_particle_is_lost(self, demo):
        settings = _turn_settings(demo, launch={"x": 0.02}, aperture={"x": 0.005, "y": 0.005})

        with pytest.raises(engine.RequestRefused, match="SR:NOT:A:SETPOINT"):
            _trajectory(demo, settings, {"SR:NOT:A:SETPOINT": 0.5})

    def test_the_orbit_answer_of_a_turn_by_turn_ring_keeps_its_rows_and_lattice_positions(
        self, demo
    ):
        plain = _matrix(demo, [CORRECTOR_X], 0.5)

        result = engine.response_matrix(
            demo["deck"],
            _turn_wiring(demo),
            _turn_settings(demo),
            [CORRECTOR_X],
            {CORRECTOR_X: 0.5},
        )

        assert result.rows == plain.rows
        assert result.row_positions == plain.row_positions
        assert result.column_positions == plain.column_positions
        np.testing.assert_array_equal(result.matrix, plain.matrix)

    def test_the_orbit_kind_is_the_default(self, demo):
        default = _matrix(demo, [CORRECTOR_X], 0.5)
        orbit = engine.response_matrix(
            demo["deck"],
            demo["wiring"],
            demo["settings"],
            [CORRECTOR_X],
            {CORRECTOR_X: 0.5},
            kind="orbit",
        )

        np.testing.assert_array_equal(default.matrix, orbit.matrix)
