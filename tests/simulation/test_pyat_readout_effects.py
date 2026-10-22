"""What a monitor's electronics add to a turn series: noise over the sum, a tone, a gain.

A monitor element carries four axis-less readout attributes beside its
per-axis faults: the electronic noise at unit sum, a tone amplitude, a signal
gain and a sum scale. Here they are wired as setting records onto the first
monitor past the marker, the way a facility binds its own records to them,
and a shot is compared with pyAT tracking the same particle, run through the
one noise relation by hand: a position's sigma is the electronic noise over
the larger of the sample's sum and the floor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

at = pytest.importorskip("at")

from osprey.simulation.engines import pyat as engine  # noqa: E402
from osprey.simulation.engines.pyat_faults import (  # noqa: E402
    ELECTRONIC_NOISE_SUBKEY,
    READOUT_ATTRIBUTE_IDENTITY,
    SUM_FLOOR,
    TONE_SPIKE_M,
    TONE_SPIKE_PERIOD,
    TONE_SUBKEY,
    shot_read,
)
from osprey.simulation.engines.pyat_model import PyATLatticeModel  # noqa: E402
from osprey_connectors.simulation.series import (  # noqa: E402
    channel_key_bytes,
    keyed_normals,
)

MODEL = "SR"
N_CELLS = 8
MARKER = "INJ"
MONITOR = "BPM_03"
BARE_MONITOR = "BPM_04"

ARM = "BPM:03:ARM"
REQUEST = "INJ:REQUEST"
CLEAR = "SR:CLEAR"
ORBIT_X = "BPM:03:X"
ORBIT_Y = "BPM:03:Y"
TBT_X = "BPM:03:TBT:X"
TBT_Y = "BPM:03:TBT:Y"
TBT_SUM = "BPM:03:TBT:SUM"
BARE_X = "BPM:04:TBT:X"
BARE_SUM = "BPM:04:TBT:SUM"

#: The setting records bound to the monitor's readout attributes.
TONE = "BPM:03:TONE"
NOISE_SCALE = "BPM:03:NOISE"
GAIN = "BPM:03:GAIN"
SUM_SCALE = "BPM:03:SUM_SCALE"
SETTINGS_BY_ATTRIBUTE = {
    "readout_tone_amplitude": TONE,
    "readout_noise_scale": NOISE_SCALE,
    "readout_gain": GAIN,
    "readout_sum_scale": SUM_SCALE,
}

LENGTH = 40
PRE_TRIGGER = 6
TURNS = LENGTH - PRE_TRIGGER
MM = 1.0e-3

SETTINGS = {"pyat": {"injection": {"marker": MARKER, "trigger_sample": PRE_TRIGGER}}}
LAUNCH = {"launch_x": 1.0e-3, "launch_px": 1.0e-4, "launch_y": -5.0e-4}
HALF_APERTURE = 0.01
LOST_LAUNCH = {"launch_x": 5.0e-3, "launch_px": 8.0e-3, "launch_y": -5.0e-4}

#: Electronic noise at unit sum, in metres.
SIGMA_E = 2.0e-6


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


def setting(address: str, attribute: str) -> dict[str, Any]:
    return {
        "address": address,
        "direction": "write",
        "element": MONITOR,
        "engine": {"attribute": attribute},
        "default": READOUT_ATTRIBUTE_IDENTITY[attribute],
        "calibration": _linear(1.0),
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


def wiring(*, with_settings: bool = True) -> list[dict[str, Any]]:
    records = [
        orbit(ORBIT_X, MONITOR, "x"),
        orbit(ORBIT_Y, MONITOR, "y"),
        series(TBT_X, MONITOR, "x", unit="mm", calibration=_linear(MM)),
        series(TBT_Y, MONITOR, "y", unit="mm", calibration=_linear(MM)),
        series(TBT_SUM, MONITOR, "sum"),
        series(BARE_X, BARE_MONITOR, "x", unit="mm", calibration=_linear(MM)),
        series(BARE_SUM, BARE_MONITOR, "sum"),
        command(ARM, "arm"),
        command(REQUEST, "inject"),
        command(CLEAR, "clear_stored_charge"),
    ]
    if with_settings:
        records += [setting(address, name) for name, address in SETTINGS_BY_ATTRIBUTE.items()]
    return records


@pytest.fixture
def deck(tmp_path: Path) -> Path:
    path = tmp_path / "periodic.json"
    at.save_lattice(lattice(), str(path))
    return path


def build(
    deck: Path, active: dict[str, Any] = LAUNCH, *, with_settings: bool = True
) -> PyATLatticeModel:
    records = wiring(with_settings=with_settings)
    return engine.build(MODEL, records, deck, SETTINGS, active)  # type: ignore[return-value]


@pytest.fixture
def model(deck: Path) -> PyATLatticeModel:
    return build(deck)


def truth(
    model: PyATLatticeModel, monitor: str = MONITOR, active: dict[str, Any] = LAUNCH
) -> tuple[np.ndarray, np.ndarray]:
    """The tracked ``(x, y)`` per series sample at one monitor, in metres; NaN with no beam."""
    particle = np.array([active.get(f"launch_{key}", 0.0) for key in engine.LAUNCH_KEYS]).reshape(
        6, 1
    )
    rotated = model.lattice.rotate(int(model.lattice.get_uint32_index(MARKER)[0]))
    tracked, _, _ = at.lattice_track(
        rotated, particle, nturns=TURNS, refpts=rotated.get_uint32_index(monitor), losses=True
    )
    x = np.full(LENGTH, np.nan)
    y = np.full(LENGTH, np.nan)
    x[PRE_TRIGGER:] = tracked[0, 0, 0, :]
    y[PRE_TRIGGER:] = tracked[2, 0, 0, :]
    return x, y


def shoot(model: PyATLatticeModel, request: int = 1, **more: Any) -> None:
    model.set({ARM: 1, REQUEST: request, **more})


def draws(address: str, subkey: bytes, shot: int = 1) -> np.ndarray:
    """The keyed standard normals a series' stream holds for one shot, one per sample."""
    key = channel_key_bytes(address) + subkey + str(shot).encode("ascii")
    return keyed_normals(key, np.arange(LENGTH))


def beam_sum(x: np.ndarray) -> np.ndarray:
    """The physics sum a single particle leaves: 1 where it passed, 0 with no beam."""
    return np.where(np.isnan(x), 0.0, 1.0)


class TestIdentity:
    def test_each_setting_starts_at_its_identity(self, model: PyATLatticeModel):
        for attribute, address in SETTINGS_BY_ATTRIBUTE.items():
            assert model.get(address) == READOUT_ATTRIBUTE_IDENTITY[attribute]

    def test_settings_at_identity_leave_every_series_bit_identical(self, deck: Path):
        wired, bare = build(deck), build(deck, with_settings=False)
        shoot(wired, **{f"{ORBIT_X}/noise": 0.05, f"{ORBIT_X}/offset": 0.1})
        shoot(bare, **{f"{ORBIT_X}/noise": 0.05, f"{ORBIT_X}/offset": 0.1})
        for address in (TBT_X, TBT_Y, TBT_SUM, BARE_X, BARE_SUM):
            np.testing.assert_array_equal(wired.get(address), bare.get(address))

    def test_positions_at_identity_are_the_tracked_ones(self, model: PyATLatticeModel):
        shoot(model)
        x, y = truth(model)
        np.testing.assert_allclose(model.get(TBT_X)[PRE_TRIGGER:], x[PRE_TRIGGER:] / MM)
        np.testing.assert_allclose(model.get(TBT_Y)[PRE_TRIGGER:], y[PRE_TRIGGER:] / MM)
        np.testing.assert_array_equal(model.get(TBT_SUM)[PRE_TRIGGER:], 1.0)


class TestNoiseRelation:
    def test_sigma_is_the_electronic_noise_over_the_floored_sum(self, model: PyATLatticeModel):
        shoot(model, **{NOISE_SCALE: SIGMA_E})
        x, y = truth(model)
        sigma = SIGMA_E / np.maximum(beam_sum(x), SUM_FLOOR)
        for address, position in ((TBT_X, x), (TBT_Y, y)):
            expected = np.nan_to_num(position, nan=0.0) + sigma * draws(
                address, ELECTRONIC_NOISE_SUBKEY
            )
            np.testing.assert_allclose(model.get(address) * MM, expected, rtol=1e-9, atol=1e-15)

    def test_no_beam_is_noisier_by_the_floor(self, model: PyATLatticeModel):
        shoot(model, **{NOISE_SCALE: SIGMA_E})
        x, _ = truth(model)
        residual = model.get(TBT_X) * MM - np.nan_to_num(x, nan=0.0)
        normals = draws(TBT_X, ELECTRONIC_NOISE_SUBKEY)
        np.testing.assert_allclose(
            residual[:PRE_TRIGGER], SIGMA_E / SUM_FLOOR * normals[:PRE_TRIGGER], rtol=1e-9
        )
        np.testing.assert_allclose(
            residual[PRE_TRIGGER:], SIGMA_E * normals[PRE_TRIGGER:], rtol=1e-6, atol=1e-15
        )

    def test_the_series_has_no_exact_zero_and_no_nan(self, deck: Path):
        model = build(deck, LOST_LAUNCH)
        shoot(model, **{NOISE_SCALE: SIGMA_E})
        x, _ = truth(model, active=LOST_LAUNCH)
        assert np.isnan(x[PRE_TRIGGER:]).any()
        for address in (TBT_X, TBT_Y, TBT_SUM):
            samples = model.get(address)
            assert np.isfinite(samples).all()
            assert np.all(samples != 0.0)

    def test_the_noise_fault_composes_on_top(self, deck: Path):
        both, electronic, fault = build(deck), build(deck), build(deck)
        shoot(both, **{NOISE_SCALE: SIGMA_E, f"{ORBIT_X}/noise": 0.01})
        shoot(electronic, **{NOISE_SCALE: SIGMA_E})
        shoot(fault, **{f"{ORBIT_X}/noise": 0.01})
        x, _ = truth(both)
        position = np.nan_to_num(x, nan=0.0) / MM
        combined = (electronic.get(TBT_X) - position) + (fault.get(TBT_X) - position)
        np.testing.assert_allclose(both.get(TBT_X) - position, combined, rtol=1e-9, atol=1e-12)

    def test_a_smaller_sum_reads_noisier_down_to_the_floor(self, model: PyATLatticeModel):
        model.set({NOISE_SCALE: SIGMA_E})
        x = np.zeros(4)
        y = np.zeros(4)
        sums = np.array([1.0, 0.5, SUM_FLOOR, SUM_FLOOR / 10.0])
        read = shot_read(model, MONITOR, "x", x, y, address=TBT_X, shot=7, first=0, sums=sums)
        key = channel_key_bytes(TBT_X) + ELECTRONIC_NOISE_SUBKEY + b"7"
        normals = keyed_normals(key, np.arange(4))
        sigma = SIGMA_E / np.array([1.0, 0.5, SUM_FLOOR, SUM_FLOOR])
        np.testing.assert_allclose(read, sigma * normals, rtol=1e-9)

    def test_the_noise_follows_the_shot_and_holds_between_shots(self, model: PyATLatticeModel):
        shoot(model, 1, **{NOISE_SCALE: SIGMA_E})
        first = model.get(TBT_X)
        np.testing.assert_array_equal(model.get(TBT_X), first)
        shoot(model, 2)
        assert np.all(model.get(TBT_X) != first)


class TestTone:
    AMPLITUDE = 4.0

    def test_a_tone_raises_the_sum_with_and_without_beam(self, deck: Path):
        model = build(deck, LOST_LAUNCH)
        shoot(model, **{TONE: self.AMPLITUDE})
        x, _ = truth(model, active=LOST_LAUNCH)
        expected = np.maximum(beam_sum(x), SUM_FLOOR) + self.AMPLITUDE
        np.testing.assert_allclose(model.get(TBT_SUM), expected, rtol=1e-12)

    def test_a_tone_spikes_x_and_y_every_period(self, model: PyATLatticeModel):
        shoot(model, **{TONE: self.AMPLITUDE})
        x, y = truth(model)
        on = np.arange(LENGTH) % TONE_SPIKE_PERIOD == 0
        for address, position in ((TBT_X, x), (TBT_Y, y)):
            spikes = model.get(address) * MM - np.nan_to_num(position, nan=0.0)
            expected = self.AMPLITUDE * TONE_SPIKE_M * draws(address, TONE_SUBKEY)
            np.testing.assert_allclose(spikes[on], expected[on], rtol=1e-6)
            np.testing.assert_allclose(spikes[~on], 0.0, atol=1e-15)
            assert np.all(np.abs(spikes[on]) > 0.0)

    def test_a_tone_reaches_only_its_own_monitor(self, model: PyATLatticeModel):
        shoot(model, **{TONE: self.AMPLITUDE})
        x, _ = truth(model, BARE_MONITOR)
        np.testing.assert_allclose(model.get(BARE_X)[PRE_TRIGGER:], x[PRE_TRIGGER:] / MM)
        np.testing.assert_array_equal(model.get(BARE_SUM)[PRE_TRIGGER:], 1.0)


class TestSumScales:
    def test_gain_and_scale_multiply_the_sum_and_leave_the_positions(self, model: PyATLatticeModel):
        shoot(model, **{GAIN: 0.5, SUM_SCALE: 8000.0, TONE: 1.0})
        x, _ = truth(model)
        expected = (np.maximum(beam_sum(x), SUM_FLOOR) + 1.0) * 0.5 * 8000.0
        np.testing.assert_allclose(model.get(TBT_SUM), expected, rtol=1e-12)

    def test_gain_and_scale_alone_leave_the_positions_exact(self, deck: Path):
        scaled, bare = build(deck), build(deck, with_settings=False)
        shoot(scaled, **{GAIN: 0.25, SUM_SCALE: 3.0})
        shoot(bare)
        np.testing.assert_array_equal(scaled.get(TBT_X), bare.get(TBT_X))
        np.testing.assert_array_equal(scaled.get(TBT_Y), bare.get(TBT_Y))


class TestClearStoredCharge:
    def test_the_command_is_built_and_takes_no_shot(self, model: PyATLatticeModel):
        shoot(model)
        held = {address: model.get(address) for address in (TBT_X, TBT_SUM)}
        count = model.shot_count
        model.set({CLEAR: 1})
        assert model.shot_count == count
        assert model.get(CLEAR) == 1
        for address, samples in held.items():
            np.testing.assert_array_equal(model.get(address), samples)

    def test_the_command_is_classified_as_one(self):
        record = command(CLEAR, "clear_stored_charge")
        description = engine.describe(record)
        assert description[engine.COMMAND_MARK] is True
