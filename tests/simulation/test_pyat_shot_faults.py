"""What a faulty monitor leaves in a turn series.

A small periodic lattice with a marker to launch at. The first monitor past
the marker wires an orbit reading per plane in millimetres, which is what
carries the readout faults, beside turn series in millimetres and one in
nanometres; the next monitor wires a turn series alone. What a shot leaves is
compared with pyAT tracking the same particle directly, run through the fault
chain by hand. A sample with no beam -- before the launch, or once the
particle is lost -- is read as a beam on axis at the sum's floor, so no sample
is NaN.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

at = pytest.importorskip("at")

from osprey.simulation.engines import pyat as engine  # noqa: E402
from osprey.simulation.engines.pyat_faults import SUM_FLOOR, readout  # noqa: E402
from osprey.simulation.engines.pyat_model import PyATLatticeModel  # noqa: E402

MODEL = "SR"
N_CELLS = 8
MARKER = "INJ"
MONITOR = "BPM_03"
BARE_MONITOR = "BPM_04"

ARM = "BPM:03:ARM"
REQUEST = "INJ:REQUEST"
ORBIT_X = "BPM:03:X"
ORBIT_Y = "BPM:03:Y"
TBT_X = "BPM:03:TBT:X"
TBT_Y = "BPM:03:TBT:Y"
TBT_SUM = "BPM:03:TBT:SUM"
#: The same samples as ``TBT_X``, published a million times finer.
TBT_X_FINE = "BPM:03:TBT:X:NM"
BARE_X = "BPM:04:TBT:X"

LENGTH = 24
PRE_TRIGGER = 4
TURNS = LENGTH - PRE_TRIGGER

#: Metres per published millimetre, and per published nanometre.
MM = 1.0e-3
NM = 1.0e-9

SETTINGS = {"pyat": {"injection": {"marker": MARKER, "trigger_sample": PRE_TRIGGER}}}
#: The scenario's launch, stated as its model-wide launch faults.
LAUNCH = {"launch_x": 1.0e-3, "launch_px": 1.0e-4, "launch_y": -5.0e-4}
HALF_APERTURE = 0.01
#: A launch that leaves the aperture within the series.
LOST_LAUNCH = {"launch_x": 5.0e-3, "launch_px": 8.0e-3, "launch_y": -5.0e-4}


def _linear(gain: float) -> dict[str, Any]:
    return {"curve": {"linear": {"gain": gain, "offset": 0.0}}, "energy_scaling": "none"}


def orbit(address: str, element: str, axis: str) -> dict[str, Any]:
    return {
        "address": address,
        "direction": "read",
        "element": element,
        "engine": {"axis": axis},
        "unit": "mm",
        "calibration": _linear(MM),
    }


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


def lattice() -> Any:
    """An eight-cell 4D FODO lattice; the marker opens its third cell, an aperture its last."""
    elements: list[Any] = []
    for cell in range(1, N_CELLS + 1):
        if cell == 3:
            elements.append(at.Marker(MARKER))
        if cell == N_CELLS:
            elements.append(at.Aperture("LIMIT", [-HALF_APERTURE, HALF_APERTURE, -1.0, 1.0]))
        elements += [
            at.Drift("DRIFT", 0.4),
            at.Quadrupole(f"QF_{cell:02d}", 0.3, 1.0),
            at.Drift("DRIFT", 0.4),
            at.Dipole(f"BEND_{cell:02d}", 1.0, 2 * np.pi / N_CELLS),
            at.Monitor(f"BPM_{cell:02d}"),
            at.Drift("DRIFT", 0.8),
            at.Quadrupole(f"QD_{cell:02d}", 0.3, -1.0),
        ]
    built = at.Lattice(elements, name="TEST", energy=3.0e9, periodicity=1)
    built.disable_6d()
    return built


def wiring() -> list[dict[str, Any]]:
    return [
        orbit(ORBIT_X, MONITOR, "x"),
        orbit(ORBIT_Y, MONITOR, "y"),
        series(TBT_X, MONITOR, "x", unit="mm", calibration=_linear(MM)),
        series(TBT_Y, MONITOR, "y", unit="mm", calibration=_linear(MM)),
        series(TBT_SUM, MONITOR, "sum"),
        series(TBT_X_FINE, MONITOR, "x", unit="nm", calibration=_linear(NM)),
        series(BARE_X, BARE_MONITOR, "x", unit="mm", calibration=_linear(MM)),
        command(ARM, "arm"),
        command(REQUEST, "inject"),
    ]


@pytest.fixture
def deck(tmp_path: Path) -> Path:
    path = tmp_path / "periodic.json"
    at.save_lattice(lattice(), str(path))
    return path


def build(deck: Path, active: dict[str, Any] = LAUNCH) -> PyATLatticeModel:
    return engine.build(MODEL, wiring(), deck, SETTINGS, active)  # type: ignore[return-value]


@pytest.fixture
def model(deck: Path) -> PyATLatticeModel:
    return build(deck)


def truth(
    model: PyATLatticeModel, monitor: str = MONITOR, active: dict[str, Any] = LAUNCH
) -> tuple[np.ndarray, np.ndarray]:
    """pyAT's own track of the launch: the tracked ``(x, y)`` per turn at one monitor, in metres."""
    particle = np.array([active.get(f"launch_{key}", 0.0) for key in engine.LAUNCH_KEYS]).reshape(
        6, 1
    )
    rotated = model.lattice.rotate(int(model.lattice.get_uint32_index(MARKER)[0]))
    tracked, _, _ = at.lattice_track(
        rotated, particle, nturns=TURNS, refpts=rotated.get_uint32_index(monitor), losses=True
    )
    return tracked[0, 0, 0, :], tracked[2, 0, 0, :]


def shoot(model: PyATLatticeModel, request: int = 1, **more: Any) -> None:
    model.set({ARM: 1, REQUEST: request, **more})


def tracked(model: PyATLatticeModel, address: str) -> np.ndarray:
    """A series' samples from the launch on; every sample of the series is a number."""
    samples = model.get(address)
    assert np.isfinite(samples).all()
    return samples[PRE_TRIGGER:]


def before_launch(model: PyATLatticeModel, address: str) -> np.ndarray:
    """A series' samples before the launch: what the monitor reads with no beam."""
    return model.get(address)[:PRE_TRIGGER]


class TestUnseeded:
    def test_a_monitor_with_no_fault_reads_the_tracked_turns(self, model: PyATLatticeModel):
        shoot(model)
        x, y = truth(model)
        np.testing.assert_allclose(tracked(model, TBT_X), x / MM, rtol=1e-12)
        np.testing.assert_allclose(tracked(model, TBT_Y), y / MM, rtol=1e-12)
        np.testing.assert_array_equal(tracked(model, TBT_SUM), 1.0)

    def test_a_monitor_with_no_orbit_reading_carries_no_fault(self, model: PyATLatticeModel):
        shoot(model, **{f"{ORBIT_X}/offset": 0.4, f"{ORBIT_X}/noise": 0.1})
        x, _ = truth(model, BARE_MONITOR)
        np.testing.assert_allclose(tracked(model, BARE_X), x / MM, rtol=1e-12)


class TestSystematicFaults:
    def test_an_offset_is_subtracted_from_every_tracked_sample(self, model: PyATLatticeModel):
        model.set({f"{ORBIT_X}/offset": 0.2})
        shoot(model)
        x, y = truth(model)
        np.testing.assert_allclose(tracked(model, TBT_X), x / MM - 0.2, rtol=1e-10)
        np.testing.assert_allclose(tracked(model, TBT_Y), y / MM, rtol=1e-12)

    def test_a_finer_series_shows_the_same_physical_offset(self, model: PyATLatticeModel):
        model.set({f"{ORBIT_X}/offset": 0.2})
        shoot(model)
        x, _ = truth(model)
        coarse = x - tracked(model, TBT_X) * MM
        fine = x - tracked(model, TBT_X_FINE) * NM
        np.testing.assert_allclose(coarse, 0.2 * MM, rtol=1e-9)
        np.testing.assert_allclose(fine, 0.2 * MM, rtol=1e-9)
        np.testing.assert_allclose(tracked(model, TBT_X_FINE), tracked(model, TBT_X) * 1.0e6)

    def test_gain_and_polarity_follow_the_orbit_chain(self, model: PyATLatticeModel):
        model.set({f"{ORBIT_X}/offset": 0.2, f"{ORBIT_X}/gain": 1.25, f"{ORBIT_X}/polarity": -1.0})
        shoot(model)
        x, _ = truth(model)
        np.testing.assert_allclose(tracked(model, TBT_X), -1.25 * (x / MM - 0.2), rtol=1e-10)

    def test_a_roll_turns_the_tracked_position(self, model: PyATLatticeModel):
        roll = 0.05
        model.set({f"{ORBIT_X}/roll": roll})
        shoot(model)
        x, y = truth(model)
        np.testing.assert_allclose(
            tracked(model, TBT_X), (math.cos(roll) * x - math.sin(roll) * y) / MM, rtol=1e-10
        )
        np.testing.assert_allclose(
            tracked(model, TBT_Y), (math.sin(roll) * x + math.cos(roll) * y) / MM, rtol=1e-10
        )

    def test_a_series_reads_what_the_orbit_reading_would(self, model: PyATLatticeModel):
        """One turn's position, handed to the orbit readout, is that turn's sample."""
        faults = {
            f"{ORBIT_X}/offset": 0.15,
            f"{ORBIT_X}/gain": 0.9,
            f"{ORBIT_Y}/offset": -0.05,
            f"{ORBIT_Y}/polarity": -1.0,
            f"{ORBIT_X}/roll": -0.03,
        }
        shoot(model, **faults)
        x, y = truth(model)
        for turn in (0, TURNS - 1):
            read = readout(model, {ORBIT_X: x[turn] / MM, ORBIT_Y: y[turn] / MM}, 0)
            assert tracked(model, TBT_X)[turn] == pytest.approx(read[ORBIT_X], rel=1e-10)
            assert tracked(model, TBT_Y)[turn] == pytest.approx(read[ORBIT_Y], rel=1e-10)

    def test_a_fault_written_with_the_shot_is_the_one_it_reads(self, model: PyATLatticeModel):
        shoot(model, **{f"{ORBIT_X}/offset": 0.2})
        x, _ = truth(model)
        np.testing.assert_allclose(tracked(model, TBT_X), x / MM - 0.2, rtol=1e-10)


class TestNoise:
    WIDTH = 0.05

    def noisy(self, deck: Path) -> PyATLatticeModel:
        built = build(deck)
        built.set({f"{ORBIT_X}/noise": self.WIDTH})
        return built

    def test_two_reads_of_one_shot_are_identical(self, deck: Path):
        model = self.noisy(deck)
        shoot(model)
        first = model.get([TBT_X, TBT_Y, TBT_SUM, TBT_X_FINE])
        second = model.get([TBT_X, TBT_Y, TBT_SUM, TBT_X_FINE])
        for name, samples in first.items():
            np.testing.assert_array_equal(second[name], samples)

    def test_noise_moves_every_tracked_sample_and_no_other(self, deck: Path):
        model = self.noisy(deck)
        shoot(model)
        x, y = truth(model)
        draws = (tracked(model, TBT_X) - x / MM) / self.WIDTH
        assert np.all(draws != 0.0)
        assert len(set(np.round(draws, 9))) == TURNS
        assert np.all(np.abs(draws) < 8.0)
        np.testing.assert_allclose(tracked(model, TBT_Y), y / MM, rtol=1e-12)

    def test_the_next_shot_differs(self, deck: Path):
        model = self.noisy(deck)
        shoot(model, 1)
        first = tracked(model, TBT_X)
        shoot(model, 2)
        second = tracked(model, TBT_X)
        x, _ = truth(model)
        assert np.all(first != second)
        # The particle is the same one; only the draws moved.
        np.testing.assert_allclose(first, x / MM, atol=8.0 * self.WIDTH)
        np.testing.assert_allclose(second, x / MM, atol=8.0 * self.WIDTH)

    def test_a_shot_is_frozen_until_the_next_refill(self, deck: Path):
        model = self.noisy(deck)
        shoot(model)
        held = model.get(TBT_X)
        model.set({f"{ORBIT_X}/noise": 4 * self.WIDTH, f"{ORBIT_X}/offset": 0.3})
        np.testing.assert_array_equal(model.get(TBT_X), held)
        model.set({REQUEST: 2})  # unarmed: counted, not tracked
        np.testing.assert_array_equal(model.get(TBT_X), held)

    def test_the_draw_is_a_function_of_the_shot_count(self, deck: Path):
        """A rebuilt model repeats a shot of the same count, and no other."""
        one, other = self.noisy(deck), self.noisy(deck)
        shoot(one)
        shoot(other)
        np.testing.assert_array_equal(other.get(TBT_X), one.get(TBT_X))
        other.set({REQUEST: 2})
        shoot(other, 3)
        assert other.shot_count == 3
        assert np.all(tracked(other, TBT_X) != tracked(one, TBT_X))

    def test_each_series_draws_on_its_own_address(self, deck: Path):
        model = self.noisy(deck)
        shoot(model)
        x, _ = truth(model)
        coarse = tracked(model, TBT_X) * MM - x
        fine = tracked(model, TBT_X_FINE) * NM - x
        assert np.all(np.abs(coarse - fine) > 0.0)
        # The width is the monitor's, in metres, whatever unit the series publishes.
        assert np.all(np.abs(fine) < 8.0 * self.WIDTH * MM)
        assert np.max(np.abs(fine)) > 0.1 * self.WIDTH * MM

    def test_the_orbit_reading_keeps_its_per_read_noise(self, deck: Path):
        model = self.noisy(deck)
        values = {ORBIT_X: 0.5, ORBIT_Y: -0.25, TBT_X: 7.0}
        before = readout(model, values, 1_000)
        shoot(model)
        after = readout(model, values, 1_000)
        assert after == before
        assert before[ORBIT_X] != values[ORBIT_X]
        assert readout(model, values, 1_001)[ORBIT_X] != before[ORBIT_X]
        assert before[TBT_X] == 7.0


class TestNoBeam:
    def test_an_unseeded_monitor_reads_no_beam_on_axis_at_the_floor(self, model: PyATLatticeModel):
        shoot(model)
        np.testing.assert_array_equal(before_launch(model, TBT_X), 0.0)
        np.testing.assert_array_equal(before_launch(model, TBT_Y), 0.0)
        np.testing.assert_array_equal(before_launch(model, TBT_SUM), SUM_FLOOR)

    def test_the_faults_act_on_the_no_beam_reading(self, model: PyATLatticeModel):
        shoot(model, **{f"{ORBIT_X}/offset": 0.2, f"{ORBIT_X}/gain": 1.5})
        np.testing.assert_allclose(before_launch(model, TBT_X), -1.5 * 0.2, rtol=1e-12)
        np.testing.assert_allclose(before_launch(model, TBT_X_FINE), -1.5 * 0.2e6, rtol=1e-9)
        np.testing.assert_array_equal(before_launch(model, TBT_SUM), SUM_FLOOR)

    def test_the_noise_fault_reaches_the_no_beam_samples(self, deck: Path):
        model = build(deck)
        shoot(model, **{f"{ORBIT_X}/noise": 0.05})
        assert np.all(before_launch(model, TBT_X) != 0.0)
        assert np.all(np.abs(before_launch(model, TBT_X)) < 8.0 * 0.05)


class TestLost:
    def test_lost_samples_read_no_beam_and_the_sum_its_floor(self, deck: Path):
        model = build(deck, LOST_LAUNCH)
        shoot(
            model,
            **{
                f"{ORBIT_X}/noise": 0.5,
                f"{ORBIT_Y}/noise": 0.5,
                f"{ORBIT_X}/offset": 2.0,
                f"{ORBIT_X}/gain": 3.0,
                f"{ORBIT_X}/roll": 0.05,
            },
        )
        x, _ = truth(model, active=LOST_LAUNCH)
        lost = np.isnan(x)
        assert lost.any() and not lost.all()
        for address in (TBT_X, TBT_Y, TBT_X_FINE):
            samples = tracked(model, address)
            assert np.all(samples[lost] != 0.0)
        # A lost sample of x is the offset read off axis, under the noise.
        np.testing.assert_allclose(tracked(model, TBT_X)[lost], -3.0 * 2.0, atol=3.0 * 8 * 0.5)
        total = tracked(model, TBT_SUM)
        np.testing.assert_array_equal(total[~lost], 1.0)
        np.testing.assert_array_equal(total[lost], SUM_FLOOR)

    def test_an_unseeded_monitor_reads_a_lost_sample_on_axis(self, deck: Path):
        model = build(deck, LOST_LAUNCH)
        shoot(model)
        x, y = truth(model, active=LOST_LAUNCH)
        lost = np.isnan(x)
        np.testing.assert_array_equal(tracked(model, TBT_X)[lost], 0.0)
        np.testing.assert_allclose(tracked(model, TBT_X)[~lost], x[~lost] / MM, rtol=1e-12)
        np.testing.assert_allclose(tracked(model, TBT_Y)[~lost], y[~lost] / MM, rtol=1e-12)
