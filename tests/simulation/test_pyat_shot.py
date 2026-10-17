"""The pyat engine's shot: arm, request, one tracked particle, the turn series it fills.

These run against two small purpose-built lattices, a periodic one with a
marker to launch at and a line, each wiring the shot's records beside a
corrector. What a shot leaves in a series is compared with pyAT tracking the
same particle directly.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

at = pytest.importorskip("at")

from lume_pyat.exceptions import OrbitSolveError  # noqa: E402

from osprey.simulation.engines import pyat as engine  # noqa: E402
from osprey.simulation.engines import pyat_model  # noqa: E402
from osprey.simulation.engines.pyat_model import (  # noqa: E402
    PyATLatticeModel,
    SeriesVariable,
)

MODEL = "SR"
N_CELLS = 8
MARKER = "INJ"
#: The deck index of the marker: the lattice starts two cells before it.
MARKER_CELL = 3

ARM = "BPM:01:ARM"
OTHER_ARM = "BPM:02:ARM"
ARMED = "BPM:01:ARMED"
OTHER_ARMED = "BPM:02:ARMED"
REQUEST = "INJ:REQUEST"
CORRECTOR = "HCOR:01:SP"
QUAD = "QF:01:SP"

#: The samples each series holds, and the ones before the launch.
LENGTH = 16
PRE_TRIGGER = 5
TURNS = LENGTH - PRE_TRIGGER

#: Metres per published millimetre, and radians per corrector ampere.
MM = 1.0e-3
KICK_PER_AMP = 1.0e-5
#: The quadrupole current the lattice is built at, and one it cannot close at.
QUAD_NOMINAL = 100.0
QUAD_UNSTABLE = 1.0e4

#: A launch off axis in both planes, inside the aperture.
LAUNCH = {"x": 1.0e-3, "px": 1.0e-4, "y": -5.0e-4}
SETTINGS = {"pyat": {"injection": {"marker": MARKER, "pre_trigger_samples": PRE_TRIGGER, **LAUNCH}}}
#: The half aperture the periodic lattice carries, and a launch that leaves it.
HALF_APERTURE = 0.01
LOST_SETTINGS = {
    "pyat": {
        "injection": {
            "marker": MARKER,
            "pre_trigger_samples": PRE_TRIGGER,
            "x": 5.0e-3,
            "px": 8.0e-3,
        }
    }
}


def _linear(gain: float, offset: float = 0.0) -> dict[str, Any]:
    return {"curve": {"linear": {"gain": gain, "offset": offset}}, "energy_scaling": "none"}


def series(address: str, element: str, axis: str, **more: Any) -> dict[str, Any]:
    return {
        "address": address,
        "direction": "read",
        "element": element,
        "engine": {"axis": axis, "series": "turn"},
        "value_type": "waveform",
        "shape": [LENGTH],
        **more,
    }


def command(address: str, attribute: str) -> dict[str, Any]:
    return {
        "address": address,
        "direction": "write",
        "engine": {"attribute": attribute},
        "default": 0,
        "value_type": "int",
    }


def armed(address: str) -> dict[str, Any]:
    return {
        "address": address,
        "direction": "read",
        "engine": {"attribute": "armed"},
        "value_type": "int",
    }


def tbt(cell: int, axis: str) -> str:
    return f"BPM:{cell:02d}:TBT:{axis.upper()}"


def periodic_lattice() -> Any:
    """An eight-cell 4D FODO lattice; the marker opens its third cell, an aperture its last."""
    elements: list[Any] = []
    for cell in range(1, N_CELLS + 1):
        if cell == MARKER_CELL:
            elements.append(at.Marker(MARKER))
        if cell == N_CELLS:
            elements.append(at.Aperture("LIMIT", [-HALF_APERTURE, HALF_APERTURE, -1.0, 1.0]))
        elements += [
            at.Drift("DRIFT", 0.4),
            at.Quadrupole(f"QF_{cell:02d}", 0.3, 1.0),
            at.Drift("DRIFT", 0.4),
            at.Corrector(f"HCOR_{cell:02d}", 0.0, [0.0, 0.0]),
            at.Dipole(f"BEND_{cell:02d}", 1.0, 2 * np.pi / N_CELLS),
            at.Monitor(f"BPM_{cell:02d}"),
            at.Drift("DRIFT", 0.8),
            at.Quadrupole(f"QD_{cell:02d}", 0.3, -1.0),
        ]
    built = at.Lattice(elements, name="TEST", energy=3.0e9, periodicity=1)
    built.disable_6d()
    return built


def periodic_wiring() -> list[dict[str, Any]]:
    """Every monitor's three series in millimetres, two arm/armed pairs, the request, two magnets."""
    records: list[dict[str, Any]] = []
    for cell in range(1, N_CELLS + 1):
        element = f"BPM_{cell:02d}"
        records += [
            series(tbt(cell, "x"), element, "x", unit="mm", calibration=_linear(MM)),
            series(tbt(cell, "y"), element, "y", unit="mm", calibration=_linear(MM)),
            series(tbt(cell, "sum"), element, "sum"),
        ]
    records += [
        command(ARM, "arm"),
        command(OTHER_ARM, "arm"),
        armed(ARMED),
        armed(OTHER_ARMED),
        command(REQUEST, "inject"),
        {
            "address": CORRECTOR,
            "element": "HCOR_01",
            "direction": "write",
            "engine": {"attribute": "KickAngle", "index": 0},
            "unit": "A",
            "default": 0.0,
            "calibration": _linear(KICK_PER_AMP),
        },
        {
            "address": QUAD,
            "element": "QF_01",
            "direction": "write",
            "engine": {"attribute": "PolynomB", "index": 1},
            "unit": "A",
            "default": QUAD_NOMINAL,
            "calibration": _linear(1.0e-2),
        },
    ]
    return records


@pytest.fixture
def deck(tmp_path: Path) -> Path:
    path = tmp_path / "periodic.json"
    at.save_lattice(periodic_lattice(), str(path))
    return path


@pytest.fixture
def built(deck: Path) -> PyATLatticeModel:
    return engine.build(MODEL, periodic_wiring(), deck, SETTINGS)  # type: ignore[return-value]


def launch(settings: dict[str, Any]) -> np.ndarray:
    block = settings["pyat"]["injection"]
    return np.array([block.get(key, 0.0) for key in engine.LAUNCH_KEYS]).reshape(6, 1)


def turns_at(model: PyATLatticeModel, cell: int, settings: dict[str, Any] = SETTINGS) -> np.ndarray:
    """pyAT's own track of the launch from the marker: ``[coordinate, turn]`` at one monitor."""
    lattice = model.lattice
    start = lattice.get_uint32_index(MARKER)[0]
    rotated = lattice.rotate(int(start))
    refpts = rotated.get_uint32_index(f"BPM_{cell:02d}")
    tracked, _, _ = at.lattice_track(
        rotated, launch(settings), nturns=TURNS, refpts=refpts, losses=True
    )
    return tracked[:, 0, 0, :]


def shot_state(model: PyATLatticeModel) -> dict[str, Any]:
    """The four things a shot changes: the series, the flag, the count, the last request seen."""
    names = [
        name
        for name, variable in model.supported_variables.items()
        if isinstance(variable, SeriesVariable)
    ]
    return {
        "series": model.get(names),
        "armed": model.armed,
        "count": model.shot_count,
        "seen": dict(model._seen),
    }


def assert_same_state(now: dict[str, Any], before: dict[str, Any]) -> None:
    assert now["armed"] == before["armed"]
    assert now["count"] == before["count"]
    assert now["seen"] == before["seen"]
    assert now["series"].keys() == before["series"].keys()
    for name, samples in before["series"].items():
        np.testing.assert_array_equal(now["series"][name], samples)


def assert_tracked(model: PyATLatticeModel, settings: dict[str, Any] = SETTINGS) -> None:
    """Every series holds zeros before the launch and pyAT's own turns after it."""
    for cell in range(1, N_CELLS + 1):
        oracle = turns_at(model, cell, settings)
        lost = np.isnan(oracle[0])
        for axis, row in (("x", 0), ("y", 2)):
            samples = model.get(tbt(cell, axis))
            np.testing.assert_array_equal(samples[:PRE_TRIGGER], 0.0)
            np.testing.assert_allclose(samples[PRE_TRIGGER:][~lost], oracle[row][~lost] / MM)
            assert np.isnan(samples[PRE_TRIGGER:][lost]).all()
        total = model.get(tbt(cell, "sum"))
        np.testing.assert_array_equal(total[:PRE_TRIGGER], 0.0)
        np.testing.assert_array_equal(total[PRE_TRIGGER:], np.where(lost, 0.0, 1.0))


class TestAtRest:
    def test_a_built_model_holds_zero_series_no_flag_and_no_shot(self, built: PyATLatticeModel):
        for cell in range(1, N_CELLS + 1):
            for axis in ("x", "y", "sum"):
                samples = built.get(tbt(cell, axis))
                assert samples.shape == (LENGTH,)
                assert samples.dtype == np.float64
                np.testing.assert_array_equal(samples, 0.0)
        assert built.get([ARMED, OTHER_ARMED, ARM, REQUEST]) == {
            ARMED: 0.0,
            OTHER_ARMED: 0.0,
            ARM: 0.0,
            REQUEST: 0.0,
        }
        assert built.armed is False
        assert built.shot_count == 0

    def test_a_read_series_is_a_copy(self, built: PyATLatticeModel):
        built.get(tbt(1, "x"))[:] = 7.0
        np.testing.assert_array_equal(built.get(tbt(1, "x")), 0.0)

    def test_a_series_on_a_model_with_no_injection_is_refused(self):
        channels = [SeriesVariable.from_wiring(series(tbt(1, "x"), "BPM_01", "x"), LENGTH)]
        with pytest.raises(ValueError, match="no injection"):
            PyATLatticeModel(periodic_lattice(), channels)

    def test_a_series_of_an_unknown_axis_is_refused(self, deck: Path):
        wiring = [*periodic_wiring(), series("BPM:01:TBT:Q", "BPM_01", "q")]
        with pytest.raises(ValueError, match="axis 'q'"):
            engine.build(MODEL, wiring, deck, SETTINGS)


class TestArm:
    def test_an_arm_write_of_one_sets_the_one_flag_every_armed_record_reads(
        self, built: PyATLatticeModel
    ):
        built.set({OTHER_ARM: 1})
        assert built.armed is True
        assert built.get([ARMED, OTHER_ARMED]) == {ARMED: 1.0, OTHER_ARMED: 1.0}
        assert built.shot_count == 0

    @pytest.mark.parametrize("value", [0, 2])
    def test_an_arm_write_of_anything_else_leaves_the_flag(
        self, built: PyATLatticeModel, value: int
    ):
        built.set({ARM: value})
        assert built.armed is False
        built.set({ARM: 1})
        built.set({ARM: value})
        assert built.armed is True

    def test_arming_alone_fills_nothing(self, built: PyATLatticeModel):
        built.set({ARM: 1})
        np.testing.assert_array_equal(built.get(tbt(1, "x")), 0.0)


class TestShot:
    def test_an_armed_increment_fills_every_series_with_the_tracked_turns(
        self, built: PyATLatticeModel
    ):
        built.set({ARM: 1})
        built.set({REQUEST: 1})
        assert_tracked(built)
        assert np.abs(built.get(tbt(1, "x"))[PRE_TRIGGER:]).min() > 0.0
        assert built.armed is False
        assert built.get(ARMED) == 0.0
        assert built.shot_count == 1

    def test_the_first_sample_after_the_launch_is_the_first_monitor_past_the_marker(
        self, built: PyATLatticeModel
    ):
        """The marker opens cell 3, so cell 3's monitor is reached before cell 1's on a turn."""
        lattice = periodic_lattice()
        start = int(lattice.get_uint32_index(MARKER)[0])
        refpts = lattice.get_uint32_index("BPM_03")
        tracked, *_ = at.lattice_track(lattice[start:], launch(SETTINGS), refpts=refpts - start)
        built.set({ARM: 1, REQUEST: 1})
        assert built.get(tbt(3, "x"))[PRE_TRIGGER] == pytest.approx(tracked[0, 0, 0, 0] / MM)

    def test_arm_and_request_in_one_batch_take_the_shot(self, built: PyATLatticeModel):
        built.set({REQUEST: 1, ARM: 1})
        assert_tracked(built)
        assert built.armed is False
        assert built.shot_count == 1

    def test_the_shot_sees_the_setpoints_of_its_own_batch(self, built: PyATLatticeModel):
        built.set({ARM: 1, REQUEST: 1})
        straight = built.get(tbt(2, "x"))
        built.set({ARM: 1})
        built.set({REQUEST: 2, CORRECTOR: 20.0})
        assert built.lattice[built.element_index("HCOR_01")].KickAngle[0] == pytest.approx(
            20.0 * KICK_PER_AMP
        )
        assert_tracked(built)
        assert not np.allclose(built.get(tbt(2, "x")), straight)

    def test_an_unarmed_increment_counts_and_keeps_the_series(self, built: PyATLatticeModel):
        built.set({ARM: 1, REQUEST: 1})
        held = built.get(tbt(1, "x"))
        built.set({CORRECTOR: 20.0})
        built.set({REQUEST: 2})
        assert built.shot_count == 2
        assert built.armed is False
        np.testing.assert_array_equal(built.get(tbt(1, "x")), held)

    def test_only_an_increment_over_the_last_value_seen_is_a_shot(self, built: PyATLatticeModel):
        built.set({REQUEST: 5})
        assert built.shot_count == 1
        built.set({REQUEST: 5})
        built.set({REQUEST: 2})
        assert built.shot_count == 1
        built.set({ARM: 1})
        built.set({REQUEST: 3})
        assert built.shot_count == 2
        assert built.armed is False
        assert_tracked(built)

    def test_a_write_that_names_no_request_is_no_shot(self, built: PyATLatticeModel):
        built.set({ARM: 1})
        built.set({CORRECTOR: 5.0})
        assert built.armed is True
        assert built.shot_count == 0
        np.testing.assert_array_equal(built.get(tbt(1, "x")), 0.0)

    def test_a_write_of_the_default_is_never_a_shot(self, built: PyATLatticeModel):
        built.set({REQUEST: -1})
        built.set({ARM: 1})
        built.set({REQUEST: 0})
        assert built.shot_count == 0
        assert built.armed is True
        np.testing.assert_array_equal(built.get(tbt(1, "x")), 0.0)

    def test_a_lost_particle_reads_nan_positions_and_no_sum(self, deck: Path):
        lossy = engine.build(MODEL, periodic_wiring(), deck, LOST_SETTINGS)
        lossy.set({ARM: 1, REQUEST: 1})
        assert_tracked(lossy, LOST_SETTINGS)  # type: ignore[arg-type]
        first, last = lossy.get(tbt(3, "sum")), lossy.get(tbt(1, "sum"))
        assert first[PRE_TRIGGER] == 1.0
        assert last[-1] == 0.0
        assert np.isnan(lossy.get(tbt(1, "y"))[-1])

    def test_a_lattice_with_no_closed_orbit_still_takes_the_shot(self, built: PyATLatticeModel):
        built.set({QUAD: QUAD_UNSTABLE})
        assert built.no_closed_orbit
        built.set({ARM: 1, REQUEST: 1})
        assert built.shot_count == 1
        assert_tracked(built)
        assert built.get(tbt(3, "sum"))[PRE_TRIGGER] == 1.0

    def test_values_pass_the_records_calibration(self, deck: Path):
        wiring = [record for record in periodic_wiring() if record["address"] != tbt(1, "sum")] + [
            series(tbt(1, "sum"), "BPM_01", "sum", calibration=_linear(0.5, 0.25))
        ]
        scaled = engine.build(MODEL, wiring, deck, SETTINGS)
        scaled.set({ARM: 1, REQUEST: 1})
        total = scaled.get(tbt(1, "sum"))
        np.testing.assert_array_equal(total[:PRE_TRIGGER], 0.0)
        np.testing.assert_allclose(total[PRE_TRIGGER:], (1.0 - 0.25) / 0.5)
        oracle = turns_at(scaled, 1)  # type: ignore[arg-type]
        np.testing.assert_allclose(scaled.get(tbt(1, "x"))[PRE_TRIGGER:], oracle[0] * 1.0e3)


class TestCommit:
    def test_a_refused_batch_holding_an_increment_changes_nothing(
        self, built: PyATLatticeModel, monkeypatch: pytest.MonkeyPatch
    ):
        built.set({ARM: 1, REQUEST: 1})
        built.set({ARM: 1})
        before = shot_state(built)

        def refuse() -> None:
            raise OrbitSolveError("no solve")

        monkeypatch.setattr(built.simulator, "solve", refuse)
        with pytest.raises(OrbitSolveError):
            built.set({REQUEST: 2, CORRECTOR: 20.0})
        monkeypatch.undo()

        assert_same_state(shot_state(built), before)
        assert built.get([REQUEST, CORRECTOR, ARMED]) == {REQUEST: 1.0, CORRECTOR: 0.0, ARMED: 1.0}
        assert built.lattice[built.element_index("HCOR_01")].KickAngle[0] == 0.0
        built.set({REQUEST: 2})
        assert built.shot_count == 2

    def test_a_shot_that_cannot_be_tracked_refuses_its_whole_batch(
        self, built: PyATLatticeModel, monkeypatch: pytest.MonkeyPatch
    ):
        built.set({ARM: 1})
        before = shot_state(built)

        def fail(*_: Any, **__: Any) -> None:
            raise RuntimeError("no track")

        monkeypatch.setattr(pyat_model.at, "lattice_track", fail)
        with pytest.raises(RuntimeError, match="no track"):
            built.set({REQUEST: 1, CORRECTOR: 20.0})
        monkeypatch.undo()

        assert_same_state(shot_state(built), before)
        assert built.get([REQUEST, CORRECTOR]) == {REQUEST: 0.0, CORRECTOR: 0.0}
        assert built.lattice[built.element_index("HCOR_01")].KickAngle[0] == 0.0

    def test_a_batch_of_commands_alone_takes_its_shot_without_a_solve(
        self, built: PyATLatticeModel, monkeypatch: pytest.MonkeyPatch
    ):
        def refuse() -> None:
            raise AssertionError("a batch of commands alone solved the lattice")

        monkeypatch.setattr(built.simulator, "solve", refuse)
        built.set({ARM: 1})
        built.set({OTHER_ARM: 1})
        built.set({REQUEST: 1})
        monkeypatch.undo()

        assert built.get([ARM, OTHER_ARM, REQUEST]) == {ARM: 1.0, OTHER_ARM: 1.0, REQUEST: 1.0}
        assert built.armed is False
        assert built.shot_count == 1
        assert_tracked(built)

    def test_a_command_shot_that_cannot_be_tracked_changes_nothing(
        self, built: PyATLatticeModel, monkeypatch: pytest.MonkeyPatch
    ):
        built.set({ARM: 1})
        before = shot_state(built)

        def fail(*_: Any, **__: Any) -> None:
            raise RuntimeError("no track")

        monkeypatch.setattr(pyat_model.at, "lattice_track", fail)
        with pytest.raises(RuntimeError, match="no track"):
            built.set({REQUEST: 1})
        monkeypatch.undo()

        assert_same_state(shot_state(built), before)
        assert built.get(REQUEST) == 0.0
        built.set({REQUEST: 1})
        assert built.shot_count == 1

    def test_a_batch_refused_for_a_name_changes_nothing(self, built: PyATLatticeModel):
        built.set({ARM: 1})
        before = shot_state(built)
        with pytest.raises(Exception, match="read-only|settable"):
            built.set({REQUEST: 1, ARMED: 1})
        assert_same_state(shot_state(built), before)


class TestReset:
    def test_reset_returns_the_shot_to_rest_and_takes_none(self, built: PyATLatticeModel):
        built.set({ARM: 1, REQUEST: 3})
        built.set({ARM: 1})
        built.reset()
        assert built.armed is False
        assert built.shot_count == 0
        assert built.get([REQUEST, ARM, ARMED]) == {REQUEST: 0.0, ARM: 0.0, ARMED: 0.0}
        for cell in range(1, N_CELLS + 1):
            for axis in ("x", "y", "sum"):
                np.testing.assert_array_equal(built.get(tbt(cell, axis)), 0.0)

    def test_reset_is_no_shot_even_from_below_the_default(self, built: PyATLatticeModel):
        built.set({REQUEST: -2})
        built.set({ARM: 1})
        built.reset()
        assert built.shot_count == 0
        np.testing.assert_array_equal(built.get(tbt(1, "x")), 0.0)

    def test_the_first_request_after_a_reset_is_a_shot_again(self, built: PyATLatticeModel):
        built.set({ARM: 1, REQUEST: 3})
        built.reset()
        built.set({ARM: 1, REQUEST: 1})
        assert built.shot_count == 1
        assert_tracked(built)

    def test_a_refused_reset_keeps_the_shot(
        self, built: PyATLatticeModel, monkeypatch: pytest.MonkeyPatch
    ):
        built.set({ARM: 1, REQUEST: 1})
        built.set({ARM: 1})
        before = shot_state(built)

        def refuse() -> None:
            raise OrbitSolveError("no solve")

        monkeypatch.setattr(built.simulator, "solve", refuse)
        with pytest.raises(OrbitSolveError):
            built.reset()
        monkeypatch.undo()
        assert_same_state(shot_state(built), before)
        built.set({REQUEST: 2})
        assert built.shot_count == 2

    def test_a_rebuild_starts_at_rest(self, deck: Path, built: PyATLatticeModel):
        built.set({ARM: 1, REQUEST: 1})
        again = engine.build(MODEL, periodic_wiring(), deck, SETTINGS)
        assert again.armed is False
        assert again.shot_count == 0
        assert again.get(REQUEST) == 0.0
        np.testing.assert_array_equal(again.get(tbt(1, "x")), 0.0)


def series_names(model: PyATLatticeModel) -> set[str]:
    return {
        name
        for name, variable in model.supported_variables.items()
        if isinstance(variable, SeriesVariable)
    }


class TestShotTime:
    def test_a_model_no_shot_has_filled_names_no_shot_time(self, built: PyATLatticeModel):
        assert built.shot_times == {}
        built.set({ARM: 1})
        built.set({CORRECTOR: 20.0})
        assert built.shot_times == {}

    def test_an_armed_shot_stamps_every_series_with_the_time_it_was_taken(
        self, built: PyATLatticeModel
    ):
        built.set({ARM: 1})
        before = time.time()
        built.set({REQUEST: 1})
        after = time.time()
        times = built.shot_times
        assert set(times) == series_names(built)
        assert len(set(times.values())) == 1
        assert before <= next(iter(times.values())) <= after

    def test_each_shot_is_later_than_the_one_before(self, built: PyATLatticeModel):
        built.set({ARM: 1, REQUEST: 1})
        first = built.shot_times
        built.set({ARM: 1, REQUEST: 2})
        second = built.shot_times
        assert all(second[name] > first[name] for name in first)

    def test_a_shot_on_a_clock_that_has_not_moved_is_still_later(
        self, built: PyATLatticeModel, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(pyat_model.time, "time", lambda: 1_760_000_000.0)
        built.set({ARM: 1, REQUEST: 1})
        first = built.shot_times[tbt(1, "x")]
        built.set({ARM: 1, REQUEST: 2})
        assert built.shot_times[tbt(1, "x")] > first

    def test_a_write_that_takes_no_tracked_shot_keeps_the_shot_time(self, built: PyATLatticeModel):
        built.set({ARM: 1, REQUEST: 1})
        held = built.shot_times
        built.set({CORRECTOR: 20.0})
        assert built.shot_times == held
        built.set({REQUEST: 2})
        assert built.shot_count == 2
        assert built.shot_times == held

    def test_a_refused_shot_keeps_the_shot_time(
        self, built: PyATLatticeModel, monkeypatch: pytest.MonkeyPatch
    ):
        built.set({ARM: 1, REQUEST: 1})
        held = built.shot_times
        built.set({ARM: 1})

        def fail(*_: Any, **__: Any) -> None:
            raise RuntimeError("no track")

        monkeypatch.setattr(pyat_model.at, "lattice_track", fail)
        with pytest.raises(RuntimeError, match="no track"):
            built.set({REQUEST: 2})
        monkeypatch.undo()
        assert built.shot_times == held

    def test_reset_and_rebuild_clear_the_shot_time(self, deck: Path, built: PyATLatticeModel):
        built.set({ARM: 1, REQUEST: 1})
        assert built.shot_times
        again = engine.build(MODEL, periodic_wiring(), deck, SETTINGS)
        assert again.shot_times == {}  # type: ignore[attr-defined]
        built.reset()
        assert built.shot_times == {}


# -- a line: one pass from its start, never rotated ---------------------------

LINE_TWISS = {"beta": [5.0, 3.0], "alpha": [0.0, 0.0], "closed_orbit": [2e-3, 1e-4, -1e-3, 0.0]}
LINE_SETTINGS = {"pyat": {"solve": "single_pass", "twiss_in": LINE_TWISS}}
LINE_CORRECTOR = "COR:1:SP"
#: The samples before the launch a line's series holds: the engine's own number.
LINE_PRE_TRIGGER = engine.PRE_TRIGGER_SAMPLES
#: A corrector current that throws the particle out of the line's aperture.
LINE_LOSING = 5.0e3


def line_lattice() -> Any:
    elements: list[Any] = []
    for cell in (1, 2, 3):
        elements += [
            at.Corrector(f"COR_{cell}", 0.0, [0.0, 0.0]),
            at.Quadrupole(f"QF_{cell}", 0.3, 1.2),
            at.Drift("DRIFT", 1.0),
            at.Monitor(f"BPM_{cell}"),
            at.Quadrupole(f"QD_{cell}", 0.3, -1.1),
            at.Drift("DRIFT", 1.0),
        ]
    elements.append(at.Aperture("LIMIT", [-0.02, 0.02, -0.02, 0.02]))
    return at.Lattice(elements, name="line", energy=3.0e9, particle="electron", periodicity=1)


def line_wiring() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for cell in (1, 2, 3):
        element = f"BPM_{cell}"
        records += [
            series(f"L:{cell}:TBT:X", element, "x"),
            series(f"L:{cell}:TBT:Y", element, "y"),
            series(f"L:{cell}:TBT:SUM", element, "sum"),
        ]
    records += [
        command(ARM, "arm"),
        armed(ARMED),
        command(REQUEST, "inject"),
        {
            "address": LINE_CORRECTOR,
            "element": "COR_1",
            "direction": "write",
            "engine": {"attribute": "KickAngle", "index": 0},
            "default": 0.0,
            "calibration": _linear(KICK_PER_AMP),
        },
    ]
    return records


@pytest.fixture
def line(tmp_path: Path) -> PyATLatticeModel:
    path = tmp_path / "line.json"
    at.save_lattice(line_lattice(), str(path))
    return engine.build("LINE", line_wiring(), path, LINE_SETTINGS)  # type: ignore[return-value]


def passed(model: PyATLatticeModel) -> np.ndarray:
    """pyAT's own single pass down the line as it stands: ``[coordinate, monitor]``."""
    lattice = model.lattice
    start = np.array([*LINE_TWISS["closed_orbit"], 0.0, 0.0]).reshape(6, 1)
    refpts = lattice.get_uint32_index(at.Monitor)
    tracked, _, _ = at.lattice_track(lattice, start, refpts=refpts, losses=True)
    return tracked[:, 0, :, 0]


def assert_one_pass(model: PyATLatticeModel) -> None:
    oracle = passed(model)
    for place, cell in enumerate((1, 2, 3)):
        for axis, expected in (("X", oracle[0, place]), ("Y", oracle[2, place]), ("SUM", 1.0)):
            samples = model.get(f"L:{cell}:TBT:{axis}")
            assert samples[LINE_PRE_TRIGGER] == pytest.approx(expected, rel=1e-12)
            np.testing.assert_array_equal(np.delete(samples, LINE_PRE_TRIGGER), 0.0)


class TestLine:
    def test_a_shot_is_one_pass_from_the_start_of_the_line(self, line: PyATLatticeModel):
        line.set({ARM: 1, REQUEST: 1})
        assert_one_pass(line)
        assert line.get("L:1:TBT:X")[LINE_PRE_TRIGGER] != 0.0
        assert line.armed is False
        assert line.shot_count == 1

    def test_the_pass_sees_the_setpoints_of_its_own_batch(self, line: PyATLatticeModel):
        line.set({ARM: 1, REQUEST: 1})
        straight = line.get("L:3:TBT:X")[LINE_PRE_TRIGGER]
        line.set({ARM: 1, REQUEST: 2, LINE_CORRECTOR: 50.0})
        assert_one_pass(line)
        assert line.get("L:3:TBT:X")[LINE_PRE_TRIGGER] != pytest.approx(straight)

    def test_a_refused_batch_holding_an_increment_changes_nothing(self, line: PyATLatticeModel):
        line.set({ARM: 1, REQUEST: 1})
        line.set({ARM: 1})
        before = shot_state(line)
        with pytest.raises(OrbitSolveError):
            line.set({REQUEST: 2, LINE_CORRECTOR: LINE_LOSING})
        assert_same_state(shot_state(line), before)
        assert line.get([REQUEST, LINE_CORRECTOR, ARMED]) == {
            REQUEST: 1.0,
            LINE_CORRECTOR: 0.0,
            ARMED: 1.0,
        }
        line.set({REQUEST: 2})
        assert line.shot_count == 2
        assert_one_pass(line)

    def test_reset_returns_the_line_to_rest(self, line: PyATLatticeModel):
        line.set({ARM: 1, REQUEST: 1})
        line.reset()
        assert line.shot_count == 0
        np.testing.assert_array_equal(line.get("L:1:TBT:X"), 0.0)
