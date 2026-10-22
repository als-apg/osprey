"""The pyat engine's injection step: a bunch's shots accumulate a stored charge.

A bunch injects ``bunch_charge`` (Q_i) and the model holds a stored charge
(Q_s), the scenario's ``stored_charge`` at rest. One sum formula holds on
every path: a sample's physics sum is ``f + Q_s/Q_i``, the share of the bunch
alive there plus the stored charge in bunches; the single particle has no
stored term. With a closed orbit the particles alive at the end of the
tracked window are captured and add their share of Q_i to Q_s, served as a
current; the monitors see the stored beam in every sample's sum and in the
positions as the charge-weighted centroid of the survivors and of the stored
beam on the unfaulted closed orbit, read through the monitor's faults once.
With every cavity off nothing is captured, the stored charge keeps its
surviving fraction per shot down to the floor, and it shows in no series; a
closed orbit lost with RF on loses it. Clearing empties it; reset restores
the scenario's.

The capture and centroid rules run on an eight-cell 4D FODO ring with an
aperture; the RF rules on a 6D ring with one cavity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

at = pytest.importorskip("at")

from osprey.simulation.engines import pyat as engine  # noqa: E402
from osprey.simulation.engines.pyat_faults import SUM_FLOOR  # noqa: E402
from osprey.simulation.engines.pyat_model import (  # noqa: E402
    PyATLatticeModel,
    SeriesVariable,
    StoredCurrentVariable,
)

MODEL = "SR"
N_CELLS = 8
MARKER = "INJ"
MARKER_CELL = 3
HALF_APERTURE = 0.01
CAVITY = "CAV"
ENERGY_EV = 3.0e9

ARM = "BPM:ARM"
REQUEST = "INJ:REQUEST"
CLEAR = "BEAM:CLEAR"
CURRENT = "BEAM:CURRENT"
CORRECTOR = "HCOR:01:SP"
QUADRUPOLE = "QF:01:SP"
RF_ON = "RF:ON:SP"
ORBIT_X = "BPM:01:X"

LENGTH = 16
PRE_TRIGGER = 5
TURNS = LENGTH - PRE_TRIGGER
MM = 1.0e-3
#: The stored-current record publishes milliamperes.
MA = 1.0e-3

SETTINGS = {"pyat": {"injection": {"marker": MARKER, "trigger_sample": PRE_TRIGGER}}}
SHAPE = {
    "bunch_emittance_x": 2.0e-7,
    "bunch_emittance_y": 5.0e-8,
    "bunch_sigma_delta": 1.0e-3,
    "bunch_sigma_z": 3.0e-3,
}
LAUNCH = {"launch_x": 1.0e-3, "launch_px": 1.0e-4, "launch_y": -5.0e-4}
#: A bunch of sixteen particles, charge 2, onto a stored charge of 1: half a bunch.
BUNCH = {"bunch_particles": 16, "bunch_charge": 2.0, "stored_charge": 1.0, **SHAPE}
#: A launch whose bunch straddles the aperture, so part of it is lost.
EDGE_LAUNCH = {"launch_x": 8.0e-3}
#: A launch outside the aperture: the whole bunch is lost on its first pass of it.
LOST_LAUNCH = {"launch_x": 3.0 * HALF_APERTURE}

SURVIVING = 0.5
FLOOR = 0.1


def _linear(gain: float) -> dict[str, Any]:
    return {"curve": {"linear": {"gain": gain, "offset": 0.0}}, "energy_scaling": "none"}


def tbt(cell: int, axis: str) -> str:
    return f"BPM:{cell:02d}:TBT:{axis.upper()}"


def fodo() -> Any:
    """An eight-cell 4D FODO ring; the marker opens its third cell, an aperture its last."""
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
    ring = at.Lattice(elements, name="TEST", energy=ENERGY_EV, periodicity=1)
    ring.disable_6d()
    return ring


def ring_6d() -> Any:
    """A 6D ring without radiation: a marker, a cavity, then a FODO cell per bend."""
    elements: list[Any] = [at.Marker(MARKER)]
    for cell in range(1, N_CELLS + 1):
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
    ring = at.Lattice(elements, name="TEST", energy=ENERGY_EV, periodicity=1)
    frequency = at.constants.clight / ring.circumference * 10
    ring.insert(1, at.RFCavity(CAVITY, 0.0, 1.0e6, frequency, 10, ENERGY_EV))
    ring = at.Lattice(ring, energy=ENERGY_EV, periodicity=1)
    ring.enable_6d(dipole_pass=None, quadrupole_pass=None)
    return ring


def stored_current(**engine_block: Any) -> dict[str, Any]:
    return {
        "address": CURRENT,
        "direction": "read",
        "engine": {"attribute": "stored_current", **engine_block},
        "unit": "mA",
        "calibration": _linear(MA),
    }


def command(address: str, attribute: str) -> dict[str, Any]:
    return {
        "address": address,
        "direction": "write",
        "engine": {"attribute": attribute},
        "default": 0,
        "value_type": "int",
    }


def wiring(*, current: dict[str, Any] | None = None, rf: bool = False) -> list[dict[str, Any]]:
    """Every monitor's three series, an orbit reading, the commands, a corrector, a quadrupole."""
    records: list[dict[str, Any]] = []
    for cell in range(1, N_CELLS + 1):
        for axis in ("x", "y", "sum"):
            record: dict[str, Any] = {
                "address": tbt(cell, axis),
                "direction": "read",
                "element": f"BPM_{cell:02d}",
                "engine": {"axis": axis, "series": "turn"},
                "value_type": "waveform",
                "shape": [LENGTH],
            }
            if axis != "sum":
                record |= {"unit": "mm", "calibration": _linear(MM)}
            records.append(record)
    records += [
        {
            "address": ORBIT_X,
            "direction": "read",
            "element": "BPM_01",
            "engine": {"axis": "x"},
            "unit": "mm",
            "calibration": _linear(MM),
        },
        command(ARM, "arm"),
        command(REQUEST, "inject"),
        command(CLEAR, "clear_stored_charge"),
        current if current is not None else stored_current(),
        {
            "address": CORRECTOR,
            "element": "HCOR_01",
            "direction": "write",
            "engine": {"attribute": "KickAngle", "index": 0},
            "unit": "A",
            "default": 0.0,
            "calibration": _linear(1.0e-5),
        },
        {
            "address": QUADRUPOLE,
            "element": "QF_01",
            "direction": "write",
            "engine": {"attribute": "PolynomB", "index": 1},
            "default": 1.0,
        },
    ]
    if rf:
        records.append(
            {
                "address": RF_ON,
                "element": CAVITY,
                "direction": "write",
                "engine": {"attribute": "Voltage"},
                "value_type": "bool",
            }
        )
    return records


@pytest.fixture
def deck(tmp_path: Path) -> Path:
    path = tmp_path / "fodo.json"
    at.save_lattice(fodo(), str(path))
    return path


@pytest.fixture
def deck_6d(tmp_path: Path) -> Path:
    path = tmp_path / "ring6d.json"
    at.save_lattice(ring_6d(), str(path))
    return path


def build(deck: Path, active: dict[str, Any], **kwargs: Any) -> PyATLatticeModel:
    return engine.build(MODEL, wiring(**kwargs), deck, SETTINGS, active)  # type: ignore[return-value]


def series_names(model: PyATLatticeModel) -> list[str]:
    return [
        name
        for name, variable in model.supported_variables.items()
        if isinstance(variable, SeriesVariable)
    ]


def shoot(model: PyATLatticeModel, *, armed: bool = True) -> dict[str, np.ndarray]:
    """Take one shot and return every series the model holds after it."""
    batch: dict[str, Any] = {REQUEST: model.shot_count + 1}
    if armed:
        batch[ARM] = 1
    model.set(batch)
    return model.get(series_names(model))


def amperes(model: PyATLatticeModel, charge: float) -> float:
    """The current ``charge`` makes circulating once per turn of the model's deck."""
    return charge * at.constants.clight / model.lattice.circumference


def captured_share(model: PyATLatticeModel, particles: np.ndarray) -> float:
    """pyAT's own count of ``particles`` alive at the end of the tracked window."""
    rotated = model.lattice.rotate(int(model.lattice.get_uint32_index(MARKER)[0]))
    tracked, _, _ = at.lattice_track(
        rotated, particles, nturns=TURNS, refpts=[len(rotated)], losses=True
    )
    alive = ~(np.isnan(tracked[0, :, 0, -1]) | np.isnan(tracked[2, :, 0, -1]))
    return float(alive.mean())


class TestSumFormula:
    def test_the_single_particle_carries_no_stored_term(self, deck: Path):
        plain = shoot(build(deck, LAUNCH))
        stored = build(deck, {**LAUNCH, "stored_charge": 5.0})
        held = shoot(stored)
        for name, samples in plain.items():
            np.testing.assert_array_equal(held[name], samples)
        assert stored.stored_charge == 5.0

    def test_the_stored_charge_is_counted_in_bunches_in_every_sample(self, deck: Path):
        model = build(deck, {**LAUNCH, **BUNCH})
        sums = shoot(model)[tbt(1, "sum")]
        share = BUNCH["stored_charge"] / BUNCH["bunch_charge"]
        np.testing.assert_allclose(sums[:PRE_TRIGGER], share, rtol=0, atol=1e-15)
        np.testing.assert_allclose(sums[PRE_TRIGGER:], 1.0 + share, rtol=0, atol=1e-15)

    def test_with_nothing_stored_the_bunch_path_reads_the_survivors_alone(self, deck: Path):
        model = build(deck, {**LAUNCH, **BUNCH, "stored_charge": 0.0})
        sums = shoot(model)[tbt(1, "sum")]
        np.testing.assert_array_equal(sums[:PRE_TRIGGER], SUM_FLOOR)
        np.testing.assert_array_equal(sums[PRE_TRIGGER:], 1.0)

    def test_the_sum_record_calibration_publishes_the_physics_sum(self, deck: Path):
        records = wiring()
        for record in records:
            if record["address"] == tbt(1, "sum"):
                record["calibration"] = _linear(1.0e-4)
        model = engine.build(MODEL, records, deck, SETTINGS, {**LAUNCH, **BUNCH})
        sums = shoot(model)[tbt(1, "sum")]  # type: ignore[arg-type]
        np.testing.assert_allclose(sums[:PRE_TRIGGER], 0.5e4, rtol=1e-12)
        np.testing.assert_allclose(sums[PRE_TRIGGER:], 1.5e4, rtol=1e-12)


class TestCapture:
    def test_a_surviving_bunch_adds_its_whole_charge(self, deck: Path):
        model = build(deck, {**LAUNCH, **BUNCH})
        shoot(model)
        assert model.stored_charge == pytest.approx(1.0 + 2.0)
        current = model.get(CURRENT)
        assert current == pytest.approx(amperes(model, 3.0) / MA, rel=1e-12)

    def test_the_share_alive_at_the_end_of_the_window_is_captured(self, deck: Path):
        model = build(deck, {**EDGE_LAUNCH, **BUNCH, "stored_charge": 0.0})
        particles = model._bunch_at(1)
        share = captured_share(model, particles)
        assert 0.0 < share < 1.0
        shoot(model)
        assert model.stored_charge == pytest.approx(share * BUNCH["bunch_charge"], abs=1e-15)

    def test_a_lost_bunch_captures_nothing(self, deck: Path):
        model = build(deck, {**LOST_LAUNCH, **BUNCH})
        shoot(model)
        assert model.stored_charge == BUNCH["stored_charge"]

    def test_an_unarmed_shot_injects_too_and_leaves_the_series(self, deck: Path):
        model = build(deck, {**LAUNCH, **BUNCH})
        before = model.get(series_names(model))
        after = shoot(model, armed=False)
        assert model.stored_charge == pytest.approx(3.0)
        for name, samples in before.items():
            np.testing.assert_array_equal(after[name], samples)

    def test_the_next_shot_sees_what_the_last_one_stored(self, deck: Path):
        model = build(deck, {**LAUNCH, **BUNCH})
        shoot(model)
        sums = shoot(model)[tbt(1, "sum")]
        np.testing.assert_allclose(sums[:PRE_TRIGGER], 3.0 / 2.0, rtol=1e-15)
        assert model.stored_charge == pytest.approx(5.0)

    def test_a_scenario_without_a_bunch_charge_counts_in_bunches(self, deck: Path):
        active = {**LAUNCH, **BUNCH}
        del active["bunch_charge"]
        model = build(deck, active)
        sums = shoot(model)[tbt(1, "sum")]
        np.testing.assert_allclose(sums[:PRE_TRIGGER], 1.0, rtol=1e-15)
        assert model.stored_charge == pytest.approx(2.0)


class TestCentroid:
    def test_positions_are_the_charge_weighted_centroid(self, deck: Path):
        active = {**EDGE_LAUNCH, **BUNCH, CORRECTOR: 20.0}
        injected = build(deck, {**active, "stored_charge": 0.0})
        mixed = build(deck, active)
        alone = shoot(injected)
        both = shoot(mixed)
        orbit_x = mixed.simulator.last_solution["BPM_01"][0] / MM
        assert orbit_x != 0.0
        share = BUNCH["stored_charge"] / BUNCH["bunch_charge"]
        fraction = np.zeros(LENGTH)
        fraction[PRE_TRIGGER:] = alone[tbt(1, "sum")][PRE_TRIGGER:]
        fraction = np.where(fraction > SUM_FLOOR, fraction, 0.0)
        x_injected = np.where(fraction > 0.0, alone[tbt(1, "x")], 0.0)
        expected = (fraction * x_injected + share * orbit_x) / (fraction + share)
        np.testing.assert_allclose(both[tbt(1, "x")], expected, rtol=1e-12, atol=1e-12)

    def test_an_offset_fault_reaches_the_stored_beam_once(self, deck: Path):
        offset = 0.5
        active = {**LAUNCH, **BUNCH, CORRECTOR: 20.0, f"{ORBIT_X}/offset": offset}
        model = build(deck, active)
        x = shoot(model)[tbt(1, "x")]
        orbit_x = model.simulator.last_solution["BPM_01"][0] / MM
        np.testing.assert_allclose(x[:PRE_TRIGGER], orbit_x - offset, rtol=1e-12)
        assert not np.allclose(x[:PRE_TRIGGER], orbit_x - 2 * offset)


class TestClearAndReset:
    def test_clearing_empties_the_stored_charge_and_takes_no_shot(self, deck: Path):
        model = build(deck, {**LAUNCH, **BUNCH})
        shoot(model)
        count = model.shot_count
        model.set({CLEAR: 1})
        assert model.stored_charge == 0.0
        assert model.get(CURRENT) == 0.0
        assert model.shot_count == count

    def test_a_write_of_the_clearing_default_clears_nothing(self, deck: Path):
        model = build(deck, {**LAUNCH, **BUNCH})
        model.set({CLEAR: 0})
        assert model.stored_charge == BUNCH["stored_charge"]

    def test_clearing_before_a_shot_in_one_batch_stores_only_that_shot(self, deck: Path):
        model = build(deck, {**LAUNCH, **BUNCH})
        model.set({CLEAR: 1, ARM: 1, REQUEST: 1})
        assert model.stored_charge == pytest.approx(BUNCH["bunch_charge"])
        sums = model.get(tbt(1, "sum"))
        np.testing.assert_array_equal(sums[:PRE_TRIGGER], SUM_FLOOR)

    def test_reset_restores_the_scenario_stored_charge(self, deck: Path):
        model = build(deck, {**LAUNCH, **BUNCH})
        shoot(model)
        model.set({CLEAR: 1})
        shoot(model)
        model.reset()
        assert model.stored_charge == BUNCH["stored_charge"]
        assert model.get(CURRENT) == pytest.approx(amperes(model, 1.0) / MA, rel=1e-12)


class TestNoRf:
    def active(self) -> dict[str, Any]:
        return {
            "bunch_particles": 8,
            "bunch_emittance_x": 2.0e-8,
            "bunch_emittance_y": 1.0e-8,
            "bunch_sigma_delta": 1.0e-4,
            "bunch_sigma_z": 1.0e-3,
            "bunch_charge": 1.0,
            "stored_charge": 1.0,
        }

    def test_with_rf_off_the_stored_charge_decays_per_shot_to_the_floor(self, deck_6d: Path):
        current = stored_current(surviving_fraction=SURVIVING, charge_floor=FLOOR)
        model = build(deck_6d, self.active(), current=current, rf=True)
        model.set({RF_ON: 0})
        assert model.no_closed_orbit
        assert model.stored_charge == 1.0
        seen = []
        for _ in range(5):
            shoot(model)
            seen.append(model.stored_charge)
        assert seen == [0.5, 0.25, 0.125, 0.0, 0.0]

    def test_with_rf_off_the_stored_beam_shows_in_no_series(self, deck_6d: Path):
        current = stored_current(surviving_fraction=SURVIVING, charge_floor=FLOOR)
        stored = build(deck_6d, self.active(), current=current, rf=True)
        empty = build(deck_6d, {**self.active(), "stored_charge": 0.0}, current=current, rf=True)
        for model in (stored, empty):
            model.set({RF_ON: 0})
        held, bare = shoot(stored), shoot(empty)
        for name, samples in bare.items():
            assert not np.isnan(held[name]).any()
            np.testing.assert_array_equal(held[name], samples)

    def test_rf_on_again_resumes_the_capture(self, deck_6d: Path):
        model = build(deck_6d, self.active(), rf=True)
        model.set({RF_ON: 0})
        shoot(model)
        assert model.stored_charge == 0.0
        model.set({RF_ON: 1})
        assert not model.no_closed_orbit
        shoot(model)
        assert model.stored_charge > 0.0

    def test_without_a_stored_current_reading_one_shot_loses_it_all(self, deck_6d: Path):
        records = [record for record in wiring(rf=True) if record["address"] != CURRENT]
        model = engine.build(MODEL, records, deck_6d, SETTINGS, self.active())
        model.set({RF_ON: 0})
        shoot(model)  # type: ignore[arg-type]
        assert model.stored_charge == 0.0  # type: ignore[attr-defined]

    def test_a_closed_orbit_lost_with_rf_on_loses_the_stored_charge(self, deck_6d: Path):
        model = build(deck_6d, self.active(), rf=True)
        model.set({QUADRUPOLE: 3.0})
        assert model.no_closed_orbit
        assert model.stored_charge == 0.0
        assert model.get(CURRENT) == 0.0


class TestBuild:
    def test_a_bunch_charge_of_zero_is_refused(self, deck: Path):
        with pytest.raises(ValueError, match="bunch_charge"):
            build(deck, {**LAUNCH, **BUNCH, "bunch_charge": 0.0})

    @pytest.mark.parametrize(
        ("block", "word"),
        [({"surviving_fraction": 1.5}, "surviving_fraction"), ({"charge_floor": -1.0}, "floor")],
    )
    def test_decay_data_out_of_range_are_refused(self, deck: Path, block: dict, word: str):
        with pytest.raises(ValueError, match=word):
            build(deck, BUNCH, current=stored_current(**block))

    def test_two_readings_stating_different_decay_data_are_refused(self, deck: Path):
        records = [
            *wiring(current=stored_current(surviving_fraction=0.5)),
            {**stored_current(surviving_fraction=0.25), "address": "BEAM:CURRENT:2"},
        ]
        with pytest.raises(ValueError, match="decay data"):
            engine.build(MODEL, records, deck, SETTINGS, BUNCH)

    def test_the_reading_is_built_from_its_record(self, deck: Path):
        current = stored_current(surviving_fraction=SURVIVING, charge_floor=FLOOR)
        model = build(deck, BUNCH, current=current)
        reading = model.supported_variables[CURRENT]
        assert isinstance(reading, StoredCurrentVariable)
        assert (reading.surviving_fraction, reading.charge_floor) == (SURVIVING, FLOOR)

    def test_the_stored_current_is_never_voided_by_a_lost_closed_orbit(self, deck_6d: Path):
        model = build(deck_6d, BUNCH, rf=True)
        model.set({RF_ON: 0})
        voided = engine.condition(model)
        assert voided is not None
        assert CURRENT not in voided[1]
        assert ORBIT_X in voided[1]
