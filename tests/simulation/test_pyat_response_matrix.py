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
from lume_pyat.exceptions import OrbitSolveError  # noqa: E402

from osprey.facility.views.simulator import simulator_wiring  # noqa: E402
from osprey.simulation.engines import pyat as engine  # noqa: E402

MODEL = "SR"
CORRECTOR_X = "SR:MAG:HCM:01:CURRENT:SP"
CORRECTOR_Y = "SR:MAG:VCM:01:CURRENT:SP"
SR_MONITORS = 144


@pytest.fixture(scope="module")
def demo(built_control_assistant: Any) -> dict[str, Any]:
    """The SR model of the built demo: its wiring entries, deck and settings."""
    facility = built_control_assistant.facility
    (model,) = [entry for entry in facility["models"] if entry["name"] == MODEL]
    return {
        "wiring": simulator_wiring(facility, MODEL),
        "deck": built_control_assistant.facility_dir / model["deck"],
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
        rows, matrix = _matrix(demo, [CORRECTOR_X, CORRECTOR_Y], 0.5)

        assert len(rows) == SR_MONITORS
        assert rows == _lattice_ordered_monitors(demo["deck"], demo["wiring"])
        assert matrix.shape == (SR_MONITORS, 2)

    def test_an_unwired_monitor_has_no_row(self, demo):
        dropped = next(record for record in demo["wiring"] if _is_monitor(record))
        wiring = [record for record in demo["wiring"] if record is not dropped]

        rows, matrix = _matrix(demo, [CORRECTOR_X], 0.5, wiring=wiring)

        assert dropped["address"] not in rows
        assert matrix.shape == (SR_MONITORS - 1, 1)

    def test_columns_follow_the_corrector_order(self, demo):
        _, forward = _matrix(demo, [CORRECTOR_X, CORRECTOR_Y], 0.5)
        _, backward = _matrix(demo, [CORRECTOR_Y, CORRECTOR_X], 0.5)

        np.testing.assert_array_equal(forward, backward[:, ::-1])

    def test_a_horizontal_corrector_moves_the_horizontal_rows(self, demo):
        rows, matrix = _matrix(demo, [CORRECTOR_X], 0.5)
        horizontal = [row for row, address in enumerate(rows) if address.endswith(":X")]

        assert np.max(np.abs(matrix[horizontal, 0])) > 0.0


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
                # hardware = monitor_gain * metres
                scaled["calibration"] = _linear(1.0 / monitor_gain, 0.0)
            elif record["address"] == CORRECTOR_X:
                # hardware = corrector_gain * radians
                scaled["calibration"] = _linear(1.0 / corrector_gain, 0.0)
            calibrated.append(scaled)
        physics_step = 1.0e-5

        rows, expected = _matrix(demo, [CORRECTOR_X], physics_step, wiring=physics)
        same_rows, served = _matrix(
            demo, [CORRECTOR_X], physics_step * corrector_gain, wiring=calibrated
        )

        assert same_rows == rows
        np.testing.assert_allclose(
            served, expected * monitor_gain / corrector_gain, rtol=1e-12, atol=1e-12
        )


class TestSolveErrors:
    def test_a_failed_step_names_the_corrector_and_returns_no_matrix(self, demo, monkeypatch):
        original = at.find_orbit6
        calls = {"count": 0}

        def raiser(*args: Any, **kwargs: Any) -> Any:
            calls["count"] += 1
            if calls["count"] == 3:
                raise at.AtError("no synchronous phase")
            return original(*args, **kwargs)

        monkeypatch.setattr(at, "find_orbit6", raiser)

        with pytest.raises(OrbitSolveError) as raised:
            _matrix(demo, [CORRECTOR_X, CORRECTOR_Y], 0.5)

        text = str(raised.value)
        assert text.startswith("closed orbit solve raised AtError:")
        assert text.endswith(f"(while stepping {CORRECTOR_Y})")

    def test_the_deck_is_left_as_it_was(self, demo):
        setpoints = [item for item in demo["wiring"] if item["direction"] == "write"]
        before = engine.start_values(demo["deck"], setpoints, demo["settings"])

        _matrix(demo, [CORRECTOR_X, CORRECTOR_Y], 0.5)

        assert engine.start_values(demo["deck"], setpoints, demo["settings"]) == before

    def test_a_corrector_that_is_not_a_wired_setpoint_is_refused(self, demo):
        with pytest.raises(ValueError, match="SR:NOT:A:SETPOINT"):
            _matrix(demo, ["SR:NOT:A:SETPOINT"], 0.5)

    def test_a_corrector_without_a_step_is_refused(self, demo):
        with pytest.raises(ValueError, match=CORRECTOR_Y):
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

        rows, matrix = engine.response_matrix(
            line, LINE_WIRING, LINE_SETTINGS, [LINE_KICK], {LINE_KICK: step}
        )

        assert rows == [f"BPM_{cell}:{axis}" for cell in LINE_CELLS for axis in ("X", "Y")]
        expected = (_tracked(line, step / 2) - _tracked(line, -step / 2)) / step
        np.testing.assert_allclose(matrix[:, 0], expected, rtol=1e-12, atol=1e-15)
        assert abs(matrix[rows.index("BPM_3:X"), 0]) > 0.0
