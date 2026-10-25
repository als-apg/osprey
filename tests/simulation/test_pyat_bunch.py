"""The pyat engine's bunch: N particles per shot, read as survivor fraction and centroid.

A scenario that states ``bunch_particles`` declares a bunch. Each shot then
tracks that many particles in one pass of pyAT and every monitor reads, per
sample, the share of the bunch still alive as its physics sum and the
survivors' centroid as its positions; a sample no particle reaches reads the
no-beam value. The bunch's shape is matched to the deck's optics at the
injection, computed once when the model is built. Its only randomness is a
centroid jitter keyed to model and shot, so two engines built alike replay
the same shots. Without a bunch the single particle is tracked as before, and
a one-particle bunch with no jitter is that same particle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

at = pytest.importorskip("at")

from osprey.simulation.engines import pyat as engine  # noqa: E402
from osprey.simulation.engines.pyat_faults import SUM_FLOOR, bunch_key  # noqa: E402
from osprey.simulation.engines.pyat_model import (  # noqa: E402
    PyATLatticeModel,
    SeriesVariable,
)
from osprey_connectors.simulation.series import keyed_normals  # noqa: E402

MODEL = "SR"
N_CELLS = 8
MARKER = "INJ"
MARKER_CELL = 3
HALF_APERTURE = 0.01

ARM = "BPM:01:ARM"
REQUEST = "INJ:REQUEST"
CORRECTOR = "HCOR:01:SP"

LENGTH = 16
PRE_TRIGGER = 5
TURNS = LENGTH - PRE_TRIGGER
MM = 1.0e-3

SETTINGS = {"pyat": {"injection": {"marker": MARKER, "trigger_sample": PRE_TRIGGER}}}
LAUNCH = {"launch_x": 1.0e-3, "launch_px": 1.0e-4, "launch_y": -5.0e-4}
SHAPE = {
    "bunch_emittance_x": 2.0e-7,
    "bunch_emittance_y": 5.0e-8,
    "bunch_sigma_delta": 1.0e-3,
    "bunch_sigma_z": 3.0e-3,
}
JITTER = {"centroid_jitter_x": 2.0e-4, "centroid_jitter_y": 1.0e-4}
#: A launch whose bunch straddles the aperture, so part of it is lost.
EDGE_LAUNCH = {"launch_x": 8.0e-3}
#: A launch outside the aperture: the whole bunch is lost on its first pass of it.
LOST_LAUNCH = {"launch_x": 3.0 * HALF_APERTURE}


def _linear(gain: float) -> dict[str, Any]:
    return {"curve": {"linear": {"gain": gain, "offset": 0.0}}, "energy_scaling": "none"}


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


def wiring(length: int = LENGTH) -> list[dict[str, Any]]:
    """Every monitor's three series (positions in millimetres), the arm, the request, a corrector."""
    records: list[dict[str, Any]] = []
    for cell in range(1, N_CELLS + 1):
        for axis in ("x", "y", "sum"):
            record: dict[str, Any] = {
                "address": tbt(cell, axis),
                "direction": "read",
                "element": f"BPM_{cell:02d}",
                "engine": {"axis": axis, "series": "turn"},
                "value_type": "waveform",
                "shape": [length],
            }
            if axis != "sum":
                record |= {"unit": "mm", "calibration": _linear(MM)}
            records.append(record)
    records += [
        {
            "address": ARM,
            "direction": "write",
            "engine": {"attribute": "arm"},
            "default": 0,
            "value_type": "int",
        },
        {
            "address": REQUEST,
            "direction": "write",
            "engine": {"attribute": "inject"},
            "default": 0,
            "value_type": "int",
        },
        {
            "address": CORRECTOR,
            "element": "HCOR_01",
            "direction": "write",
            "engine": {"attribute": "KickAngle", "index": 0},
            "unit": "A",
            "default": 0.0,
            "calibration": _linear(1.0e-5),
        },
    ]
    return records


@pytest.fixture
def deck(tmp_path: Path) -> Path:
    path = tmp_path / "periodic.json"
    at.save_lattice(periodic_lattice(), str(path))
    return path


def build(deck: Path, active: dict[str, Any], length: int = LENGTH) -> PyATLatticeModel:
    return engine.build(MODEL, wiring(length), deck, SETTINGS, active)  # type: ignore[return-value]


def shoot(model: PyATLatticeModel) -> dict[str, np.ndarray]:
    """Take one shot and return every series it left."""
    model.set({ARM: 1, REQUEST: model.shot_count + 1})
    names = [
        name
        for name, variable in model.supported_variables.items()
        if isinstance(variable, SeriesVariable)
    ]
    return model.get(names)


def oracle(model: PyATLatticeModel, particles: np.ndarray, cell: int) -> np.ndarray:
    """pyAT's own track of ``particles`` from the marker: ``[coordinate, particle, turn]``."""
    lattice = model.lattice
    start = int(lattice.get_uint32_index(MARKER)[0])
    rotated = lattice.rotate(start)
    refpts = rotated.get_uint32_index(f"BPM_{cell:02d}")
    tracked, _, _ = at.lattice_track(rotated, particles, nturns=TURNS, refpts=refpts, losses=True)
    return tracked[:, :, 0, :]


def launch(active: dict[str, Any]) -> np.ndarray:
    return np.array([active.get(f"launch_{key}", 0.0) for key in engine.LAUNCH_KEYS])


class TestNoBunch:
    def test_the_bunch_shape_without_a_particle_count_is_the_single_particle(self, deck: Path):
        plain = shoot(build(deck, LAUNCH))
        shaped = shoot(build(deck, {**LAUNCH, **SHAPE, **JITTER}))
        for name, samples in plain.items():
            np.testing.assert_array_equal(shaped[name], samples)

    def test_one_particle_with_no_jitter_is_the_single_particle(self, deck: Path):
        plain = shoot(build(deck, LAUNCH))
        single = shoot(build(deck, {**LAUNCH, **SHAPE, "bunch_particles": 1}))
        for name, samples in plain.items():
            np.testing.assert_array_equal(single[name], samples)

    def test_a_single_particle_tracks_bit_for_bit_with_pyat_over_100_turns(self, deck: Path):
        length = PRE_TRIGGER + 100
        model = build(deck, LAUNCH, length)
        series = shoot(model)
        lattice = model.lattice
        rotated = lattice.rotate(int(lattice.get_uint32_index(MARKER)[0]))
        for cell in (1, 4, 8):
            refpts = rotated.get_uint32_index(f"BPM_{cell:02d}")
            tracked, _, _ = at.lattice_track(
                rotated, launch(LAUNCH).reshape(6, 1), nturns=100, refpts=refpts, losses=True
            )
            np.testing.assert_array_equal(
                series[tbt(cell, "x")][PRE_TRIGGER:], tracked[0, 0, 0, :] / MM
            )
            np.testing.assert_array_equal(
                series[tbt(cell, "y")][PRE_TRIGGER:], tracked[2, 0, 0, :] / MM
            )


class TestBunchReading:
    @pytest.mark.parametrize("active", [LAUNCH, EDGE_LAUNCH], ids=["inside", "edge"])
    def test_sum_is_the_survivor_fraction_and_positions_the_survivors_centroid(
        self, deck: Path, active: dict[str, Any]
    ):
        n = 64
        model = build(deck, {**active, **SHAPE, **JITTER, "bunch_particles": n})
        series = shoot(model)
        particles = model._bunch_at(1)
        assert particles.shape == (6, n)
        for cell in range(1, N_CELLS + 1):
            tracked = oracle(model, particles, cell)
            alive = ~(np.isnan(tracked[0]) | np.isnan(tracked[2]))
            count = alive.sum(axis=0)
            fraction = count / n
            total = series[tbt(cell, "sum")]
            np.testing.assert_array_equal(total[:PRE_TRIGGER], SUM_FLOOR)
            np.testing.assert_allclose(
                total[PRE_TRIGGER:], np.maximum(fraction, SUM_FLOOR), rtol=0, atol=1e-15
            )
            for axis, row in (("x", 0), ("y", 2)):
                with np.errstate(invalid="ignore", divide="ignore"):
                    centroid = np.where(alive, tracked[row], 0.0).sum(axis=0) / count
                samples = series[tbt(cell, axis)][PRE_TRIGGER:]
                seen = count > 0
                np.testing.assert_allclose(samples[seen], centroid[seen] / MM, rtol=1e-12)
                np.testing.assert_array_equal(samples[~seen], 0.0)

    def test_a_bunch_at_the_aperture_loses_part_of_itself(self, deck: Path):
        n = 64
        series = shoot(build(deck, {**EDGE_LAUNCH, **SHAPE, "bunch_particles": n}))
        fractions = np.concatenate(
            [series[tbt(cell, "sum")][PRE_TRIGGER:] for cell in range(1, N_CELLS + 1)]
        )
        partial = fractions[(fractions > SUM_FLOOR) & (fractions < 1.0)]
        assert partial.size > 0
        np.testing.assert_allclose(partial * n, np.round(partial * n), atol=1e-9)

    def test_a_bunch_with_no_survivors_reads_the_no_beam_value(self, deck: Path):
        series = shoot(build(deck, {**LOST_LAUNCH, **SHAPE, "bunch_particles": 16}))
        assert series[tbt(1, "sum")][-1] == SUM_FLOOR
        assert series[tbt(1, "x")][-1] == 0.0
        assert series[tbt(1, "y")][-1] == 0.0
        for samples in series.values():
            assert not np.isnan(samples).any()


class TestBunchShape:
    def test_the_bunch_is_matched_to_the_optics_at_the_injection(self, deck: Path):
        n = 2048
        model = build(deck, {**LAUNCH, **SHAPE, "bunch_particles": n})
        particles = model._bunch_at(1)
        np.testing.assert_allclose(particles.mean(axis=1), launch(LAUNCH), atol=1e-12)
        four = periodic_lattice()
        _, _, optics = four.get_optics(refpts=four.get_uint32_index(MARKER))
        beta, alpha = optics["beta"][0], optics["alpha"][0]
        dispersion = optics["dispersion"][0]
        sigma_delta = SHAPE["bunch_sigma_delta"]
        for plane, (pos, mom) in enumerate(((0, 1), (2, 3))):
            emittance = SHAPE[("bunch_emittance_x", "bunch_emittance_y")[plane]]
            eta, eta_p = dispersion[2 * plane], dispersion[2 * plane + 1]
            expected_x = emittance * beta[plane] + (eta * sigma_delta) ** 2
            expected_xp = (
                emittance * (1 + alpha[plane] ** 2) / beta[plane] + (eta_p * sigma_delta) ** 2
            )
            expected_xxp = -emittance * alpha[plane] + eta * eta_p * sigma_delta**2
            covariance = np.cov(particles[pos], particles[mom], bias=True)
            np.testing.assert_allclose(covariance[0, 0], expected_x, rtol=0.05)
            np.testing.assert_allclose(covariance[1, 1], expected_xp, rtol=0.05)
            np.testing.assert_allclose(
                covariance[0, 1], expected_xxp, atol=0.05 * np.sqrt(expected_x * expected_xp)
            )
        np.testing.assert_allclose(particles[4].std(), sigma_delta, rtol=0.02)
        np.testing.assert_allclose(particles[5].std(), SHAPE["bunch_sigma_z"], rtol=0.02)

    def test_the_shape_holds_no_randomness(self, deck: Path):
        model = build(deck, {**LAUNCH, **SHAPE, "bunch_particles": 32})
        np.testing.assert_array_equal(model._bunch_at(1), model._bunch_at(2))

    def test_the_matching_optics_are_computed_once_at_build(
        self, deck: Path, monkeypatch: pytest.MonkeyPatch
    ):
        model = build(deck, {**LAUNCH, **SHAPE, **JITTER, "bunch_particles": 8})

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("optics solved during a shot")

        monkeypatch.setattr(at.Lattice, "get_optics", refuse)
        monkeypatch.setattr(at.Lattice, "linopt6", refuse, raising=False)
        before = model._bunch_at(1)
        model.set({CORRECTOR: 30.0})
        shoot(model)
        shoot(model)
        np.testing.assert_array_equal(model._bunch_at(1), before)


class TestJitter:
    def test_the_centroid_jitter_is_the_keyed_stream_of_model_and_shot(self, deck: Path):
        model = build(deck, {**LAUNCH, **SHAPE, **JITTER, "bunch_particles": 16})
        for shot in (1, 2, 7):
            normals = keyed_normals(bunch_key(MODEL, shot), np.array([0, 1]))
            expected = launch(LAUNCH) + np.array(
                [
                    JITTER["centroid_jitter_x"] * normals[0],
                    0.0,
                    JITTER["centroid_jitter_y"] * normals[1],
                    0.0,
                    0.0,
                    0.0,
                ]
            )
            np.testing.assert_allclose(model._bunch_at(shot).mean(axis=1), expected, atol=1e-12)

    def test_two_engines_built_alike_replay_the_same_shots(self, deck: Path):
        active = {**LAUNCH, **SHAPE, **JITTER, "bunch_particles": 16}
        one, two = build(deck, active), build(deck, active)
        first = [shoot(one) for _ in range(3)]
        second = [shoot(two) for _ in range(3)]
        for mine, theirs in zip(first, second, strict=True):
            for name, samples in mine.items():
                np.testing.assert_array_equal(theirs[name], samples)
        assert not np.array_equal(first[0][tbt(4, "x")], first[1][tbt(4, "x")])

    def test_reset_replays_the_shot_sequence(self, deck: Path):
        model = build(deck, {**LAUNCH, **SHAPE, **JITTER, "bunch_particles": 16})
        before = shoot(model)
        shoot(model)
        model.reset()
        again = shoot(model)
        for name, samples in before.items():
            np.testing.assert_array_equal(again[name], samples)


class TestRefusal:
    def test_a_bunch_on_a_model_without_injection_is_refused(self, deck: Path):
        setpoints = [record for record in wiring() if record["address"] == CORRECTOR]
        with pytest.raises(ValueError, match="bunch_particles"):
            engine.build(MODEL, setpoints, deck, {"pyat": {}}, {"bunch_particles": 4})
