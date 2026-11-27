"""The pyat engine's turn tracking: injected particles read per monitor and turn.

The functions take a lattice and plain numbers. These tests track on the
example ring's deck: the same shot replays identically, two shots differ by
the jitter, a corrector kick reaches only the monitors after it on the first
turn, and the share of particles alive never grows.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import at
import numpy as np
import pytest

from osprey.simulation.engines import pyat_turns as turns
from osprey_connectors.simulation.series import keyed_normals

DECK = (
    Path(__file__).resolve().parents[2]
    / "src/osprey/templates/facilities/example/simulation/decks/SR.json"
)
MODEL = "SR"
MARKER = "INJ"
TURNS = 4
PARTICLES = 64
SHAPE = {
    "emittance_x": 2.0e-8,
    "emittance_y": 2.0e-9,
    "sigma_delta": 1.0e-3,
    "sigma_z": 3.0e-3,
}
# Sigmas in pyAT's order: x, px, y, py, delta, ct.
JITTER = (2.0e-4, 0.0, 1.0e-4, 0.0, 0.0, 0.0)
JITTER_6D = (2.0e-4, 3.0e-5, 1.0e-4, 2.0e-5, 4.0e-4, 5.0e-3)
LAUNCH = (1.0e-3, 0.0, -5.0e-4, 0.0, 0.0, 0.0)
KICK = 1.0e-4


@pytest.fixture(scope="module")
def ring() -> Any:
    return at.load_lattice(str(DECK))


@pytest.fixture(scope="module")
def injection(ring: Any) -> int:
    (index,) = ring.get_uint32_index(MARKER)
    return int(index)


@pytest.fixture(scope="module")
def monitors(ring: Any) -> np.ndarray:
    return np.asarray(ring.get_uint32_index(at.Monitor), dtype=np.int64)


@pytest.fixture(scope="module")
def shape(ring: Any, injection: int) -> np.ndarray:
    return turns.declared_bunch(ring, injection, PARTICLES, **SHAPE)


def _shot(ring: Any, injection: int, monitors: np.ndarray, shape: np.ndarray, shot: int) -> Any:
    return turns.track_shot(
        ring,
        injection,
        monitors,
        LAUNCH,
        TURNS,
        shape=shape,
        key=MODEL,
        shot=shot,
        jitter=JITTER,
    )


# -- the bunch ---------------------------------------------------------------


def test_the_bunch_shape_is_deterministic_and_centred(ring: Any, injection: int) -> None:
    first = turns.declared_bunch(ring, injection, PARTICLES, **SHAPE)
    again = turns.declared_bunch(ring, injection, PARTICLES, **SHAPE)
    assert first.shape == (6, PARTICLES)
    np.testing.assert_array_equal(first, again)
    np.testing.assert_allclose(first.mean(axis=1), 0.0, atol=1e-15)


def test_the_bunch_shape_matches_the_declared_sizes() -> None:
    shape = turns.bunch_shape(
        beta=(4.0, 9.0),
        alpha=(0.0, 0.0),
        dispersion=(0.0, 0.0, 0.0, 0.0),
        particles=4096,
        **SHAPE,
    )
    assert np.std(shape[0]) == pytest.approx(np.sqrt(SHAPE["emittance_x"] * 4.0), rel=0.02)
    assert np.std(shape[2]) == pytest.approx(np.sqrt(SHAPE["emittance_y"] * 9.0), rel=0.02)
    assert np.std(shape[4]) == pytest.approx(SHAPE["sigma_delta"], rel=0.02)
    assert np.std(shape[5]) == pytest.approx(SHAPE["sigma_z"], rel=0.02)


def test_an_unstated_size_is_zero() -> None:
    shape = turns.bunch_shape(
        beta=(4.0, 9.0), alpha=(0.0, 0.0), dispersion=(0.0,) * 4, particles=16
    )
    np.testing.assert_array_equal(shape, 0.0)


def test_a_bunch_takes_at_least_one_particle(ring: Any, injection: int) -> None:
    with pytest.raises(ValueError, match="particles"):
        turns.declared_bunch(ring, injection, 0, **SHAPE)


def test_the_launched_bunch_is_launch_plus_shape_plus_keyed_jitter(shape: np.ndarray) -> None:
    launched = turns.launched_bunch(LAUNCH, shape, key=MODEL, shot=7, jitter=JITTER)
    normals = keyed_normals(turns.shot_key(MODEL, 7), np.arange(2))
    centre = np.asarray(LAUNCH, dtype=np.float64)
    centre[0] += JITTER[0] * normals[0]
    centre[2] += JITTER[2] * normals[1]
    expected = centre.reshape(6, 1) + shape
    np.testing.assert_array_equal(launched, expected)


def test_the_jitter_reaches_all_six_coordinates(shape: np.ndarray) -> None:
    launched = turns.launched_bunch(LAUNCH, shape, key=MODEL, shot=7, jitter=JITTER_6D)
    centre = launched.mean(axis=1) - np.asarray(LAUNCH) - shape.mean(axis=1)
    normals = keyed_normals(turns.shot_key(MODEL, 7), np.arange(6))
    # x and y keep the first two draws; the other coordinates take the next four.
    order = (0, 2, 1, 3, 4, 5)
    np.testing.assert_allclose(centre, np.asarray(JITTER_6D) * normals[list(order)], atol=1e-18)


def test_a_position_jitter_draws_the_same_with_or_without_the_other_four(
    shape: np.ndarray,
) -> None:
    plain = turns.launched_bunch(LAUNCH, shape, key=MODEL, shot=7, jitter=JITTER)
    full = turns.launched_bunch(LAUNCH, shape, key=MODEL, shot=7, jitter=JITTER_6D)
    np.testing.assert_array_equal(plain[[0, 2]], full[[0, 2]])


def test_the_jitter_has_its_sigma_over_many_shots(shape: np.ndarray) -> None:
    centres = np.array(
        [
            turns.launched_bunch(LAUNCH, shape, key=MODEL, shot=shot, jitter=JITTER_6D).mean(axis=1)
            for shot in range(2000)
        ]
    )
    np.testing.assert_allclose(centres.std(axis=0), JITTER_6D, rtol=0.06)


def test_a_static_offset_moves_the_whole_bunch(shape: np.ndarray) -> None:
    offset = (1.0e-4, -2.0e-5, 3.0e-4, 4.0e-5, 1.0e-3, 2.0e-3)
    plain = turns.launched_bunch(LAUNCH, shape, key=MODEL, shot=7, jitter=JITTER)
    moved = turns.launched_bunch(LAUNCH, shape, key=MODEL, shot=7, jitter=JITTER, offset=offset)
    np.testing.assert_allclose(moved - plain, np.asarray(offset).reshape(6, 1) + 0 * plain)


# -- the arriving optics -----------------------------------------------------


def _moments(shape: np.ndarray, plane: int) -> tuple[float, float, float]:
    """(emittance, beta, alpha) of one plane of a bunch without dispersion."""
    x, px = shape[2 * plane], shape[2 * plane + 1]
    xx, pp, xp = np.mean(x * x), np.mean(px * px), np.mean(x * px)
    emittance = float(np.sqrt(xx * pp - xp * xp))
    return emittance, float(xx / emittance), float(-xp / emittance)


def test_a_mismatch_scales_beta_and_keeps_the_emittance(ring: Any, injection: int) -> None:
    sizes = {"emittance_x": 2.0e-8, "emittance_y": 2.0e-9}
    matched = turns.declared_bunch(ring, injection, 4096, **sizes)
    wide = turns.declared_bunch(ring, injection, 4096, mismatch=(1.5, 0.5), **sizes)
    for plane, factor in enumerate((1.5, 0.5)):
        emittance, beta, alpha = _moments(matched, plane)
        emittance_m, beta_m, alpha_m = _moments(wide, plane)
        assert emittance_m == pytest.approx(emittance, rel=1e-9)
        assert beta_m == pytest.approx(factor * beta, rel=1e-9)
        assert alpha_m == pytest.approx(alpha, abs=1e-9)


def test_a_mismatch_of_one_is_the_matched_bunch(ring: Any, injection: int) -> None:
    np.testing.assert_array_equal(
        turns.declared_bunch(ring, injection, PARTICLES, **SHAPE),
        turns.declared_bunch(ring, injection, PARTICLES, mismatch=(1.0, 1.0), **SHAPE),
    )


def test_stated_twiss_replace_the_rings(ring: Any, injection: int) -> None:
    shape = turns.declared_bunch(
        ring,
        injection,
        4096,
        emittance_x=2.0e-8,
        emittance_y=2.0e-9,
        beta=(7.0, 3.0),
        alpha=(0.4, -0.2),
        dispersion=(0.0, 0.0, 0.0, 0.0),
    )
    for plane, (beta, alpha) in enumerate(((7.0, 0.4), (3.0, -0.2))):
        _, found_beta, found_alpha = _moments(shape, plane)
        # The quantile sequences are uncorrelated to a few parts in 1e4, not exactly.
        assert found_beta == pytest.approx(beta, rel=2e-3)
        assert found_alpha == pytest.approx(alpha, abs=2e-3)


def test_an_unstated_twiss_key_is_the_rings(ring: Any, injection: int) -> None:
    _, _, optics = ring.disable_6d(copy=True).get_optics(refpts=[injection])
    stated = turns.declared_bunch(ring, injection, PARTICLES, beta=(7.0, 3.0), **SHAPE)
    by_hand = turns.bunch_shape(
        (7.0, 3.0), optics["alpha"][0], optics["dispersion"][0], PARTICLES, **SHAPE
    )
    np.testing.assert_array_equal(stated, by_hand)


def test_fully_stated_twiss_need_no_periodic_optics() -> None:
    line = at.Lattice([at.Drift("D", 1.0), at.Quadrupole("Q", 0.2, 30.0)], energy=1e9)
    with pytest.raises(ValueError, match="no periodic optics"):
        turns.declared_bunch(line, 0, 4, emittance_x=1e-9)
    shape = turns.declared_bunch(
        line, 0, 4, emittance_x=1e-9, beta=(2.0, 2.0), alpha=(0.0, 0.0), dispersion=(0.0,) * 4
    )
    assert shape.shape == (6, 4)


def test_a_mismatch_factor_is_above_zero(ring: Any, injection: int) -> None:
    with pytest.raises(ValueError, match="mismatch"):
        turns.declared_bunch(ring, injection, 4, emittance_x=1e-9, mismatch=(0.0, 1.0))


def test_a_mismatched_bunch_beats_in_size_and_a_matched_one_does_not(
    ring: Any, injection: int
) -> None:
    """The mismatch is seen where the ring sees it: the size at injection beats turn by turn."""
    sizes = {"emittance_x": 2.0e-8}
    lattice = ring.disable_6d(copy=True).rotate(injection)

    def size_per_turn(mismatch: tuple[float, float]) -> np.ndarray:
        bunch = turns.declared_bunch(ring, injection, 256, mismatch=mismatch, **sizes)
        out, _, _ = at.lattice_track(lattice, np.asfortranarray(bunch), nturns=12, refpts=[0])
        return np.asarray(out[0, :, 0, :].std(axis=0))

    matched, wide = size_per_turn((1.0, 1.0)), size_per_turn((2.0, 1.0))
    assert np.ptp(matched) / matched.mean() < 0.02
    assert np.ptp(wide) / wide.mean() > 0.2


def test_the_shot_key_names_the_model_and_the_shot() -> None:
    assert turns.shot_key(MODEL, 3) != turns.shot_key(MODEL, 4)
    assert turns.shot_key(MODEL, 3) != turns.shot_key("BR", 3)
    assert turns.shot_key(MODEL, 3).endswith(b":bunch3")


# -- apertures ---------------------------------------------------------------


def test_apertures_are_written_on_every_element_with_a_length(ring: Any) -> None:
    lattice = copy.deepcopy(ring)
    turns.write_apertures(lattice, elliptical=(0.01, 0.005))
    for element in lattice:
        if float(getattr(element, "Length", 0.0)) > 0.0:
            np.testing.assert_array_equal(element.EApertures, [0.01, 0.005])
        else:
            assert not hasattr(element, "EApertures")


def test_a_rectangular_aperture_is_the_four_edges(ring: Any) -> None:
    lattice = copy.deepcopy(ring)
    turns.write_apertures(lattice, rectangular=(0.02, 0.01))
    lengthy = next(element for element in lattice if float(element.Length) > 0.0)
    np.testing.assert_array_equal(lengthy.RApertures, [-0.02, 0.02, -0.01, 0.01])


def test_no_aperture_stated_writes_nothing(ring: Any) -> None:
    lattice = copy.deepcopy(ring)
    turns.write_apertures(lattice)
    assert not any(hasattr(e, "EApertures") or hasattr(e, "RApertures") for e in lattice)


def test_an_aperture_half_axis_must_be_positive(ring: Any) -> None:
    with pytest.raises(ValueError, match="half-axis"):
        turns.write_apertures(copy.deepcopy(ring), elliptical=(0.0, 0.01))


def test_a_rectangle_may_state_its_limits_per_plane(ring: Any) -> None:
    lattice = copy.deepcopy(ring)
    turns.write_apertures(lattice, rectangular=((-0.03, 0.006), 0.01))
    lengthy = next(element for element in lattice if float(element.Length) > 0.0)
    np.testing.assert_array_equal(lengthy.RApertures, [-0.03, 0.006, -0.01, 0.01])


def test_a_rectangles_limits_must_rise(ring: Any) -> None:
    with pytest.raises(ValueError, match="limits"):
        turns.write_apertures(copy.deepcopy(ring), rectangular=((0.006, -0.03), 0.01))


def test_named_indexes_take_the_aperture_whatever_their_length(ring: Any, injection: int) -> None:
    lattice = copy.deepcopy(ring)
    turns.write_apertures(lattice, elliptical=(0.01, 0.005), indexes=(injection,))
    np.testing.assert_array_equal(lattice[injection].EApertures, [0.01, 0.005])
    assert sum(hasattr(element, "EApertures") for element in lattice) == 1


def test_a_later_aperture_replaces_the_one_an_element_holds(ring: Any) -> None:
    lattice = copy.deepcopy(ring)
    bends = tuple(int(i) for i in lattice.get_uint32_index(at.Dipole))
    turns.write_apertures(lattice, elliptical=(0.01, 0.01))
    turns.write_apertures(lattice, rectangular=(0.02, 0.003), indexes=bends)
    for index, element in enumerate(lattice):
        if index in bends:
            np.testing.assert_array_equal(element.RApertures, [-0.02, 0.02, -0.003, 0.003])
            assert not hasattr(element, "EApertures")
        elif float(getattr(element, "Length", 0.0)) > 0.0:
            np.testing.assert_array_equal(element.EApertures, [0.01, 0.01])
            assert not hasattr(element, "RApertures")


def test_declared_apertures_are_the_rings_then_each_item_in_order(ring: Any) -> None:
    from osprey.simulation.engines.pyat_turn_settings import Aperture, ElementAperture

    lattice = copy.deepcopy(ring)
    bends = tuple(int(i) for i in lattice.get_uint32_index(at.Dipole))
    turns.write_declared_apertures(
        lattice,
        Aperture("elliptical", 0.01, 0.01),
        (
            ElementAperture(Aperture("elliptical", 0.02, 0.004), bends),
            ElementAperture(Aperture("rectangular", (-0.03, 0.006), 0.02), bends[:1]),
        ),
    )
    np.testing.assert_array_equal(lattice[bends[0]].RApertures, [-0.03, 0.006, -0.02, 0.02])
    assert not hasattr(lattice[bends[0]], "EApertures")
    np.testing.assert_array_equal(lattice[bends[1]].EApertures, [0.02, 0.004])
    drift = next(e for e in lattice if isinstance(e, at.Drift) and float(e.Length) > 0.0)
    np.testing.assert_array_equal(drift.EApertures, [0.01, 0.01])


def test_no_declared_aperture_leaves_the_decks_own(ring: Any) -> None:
    lattice = copy.deepcopy(ring)
    turns.write_declared_apertures(lattice, None, ())
    assert not any(hasattr(e, "EApertures") or hasattr(e, "RApertures") for e in lattice)


def test_a_tight_aperture_on_one_class_loses_the_beam_there(
    ring: Any, injection: int, monitors: np.ndarray
) -> None:
    """A vertical offset the ring-wide aperture passes is lost at the bends' own aperture."""
    launch = (0.0, 0.0, 1.0e-3, 0.0, 0.0, 0.0)
    wide = copy.deepcopy(ring)
    turns.write_apertures(wide, elliptical=(0.01, 0.01))
    kept = turns.single_particle(wide, injection, monitors, launch)
    np.testing.assert_array_equal(kept.alive, 1.0)

    tight = copy.deepcopy(wide)
    bends = np.asarray(tight.get_uint32_index(at.Dipole), dtype=np.int64)
    turns.write_apertures(tight, elliptical=(0.01, 1.0e-4), indexes=tuple(int(i) for i in bends))
    lost = turns.single_particle(tight, injection, monitors, launch)
    first_bend = int(np.min((bends - injection) % len(ring)))
    before = (monitors - injection) % len(ring) < first_bend
    assert before.any() and (~before).any()
    np.testing.assert_array_equal(lost.alive[before, 0], 1.0)
    np.testing.assert_array_equal(lost.alive[~before, 0], 0.0)


def test_an_aperture_moves_with_its_elements_misalignment() -> None:
    """pyAT checks an aperture in the element's own frame, behind its entrance transform."""

    def lost(shift: float, x: float) -> bool:
        quadrupole = at.Quadrupole("Q", 0.2, 0.0)
        quadrupole.T1 = np.array([-shift, 0.0, 0.0, 0.0, 0.0, 0.0])
        quadrupole.T2 = -quadrupole.T1
        lattice = at.Lattice(
            [at.Marker("INJ"), quadrupole, at.Monitor("BPM")], energy=1e9, periodicity=1
        )
        turns.write_apertures(lattice, elliptical=(0.010, 0.010))
        reading = turns.single_particle(lattice, 0, [2], (x, 0.0, 0.0, 0.0, 0.0, 0.0))
        return bool(reading.alive[0, 0] == 0.0)

    assert not lost(0.0, 0.009) and lost(0.0, 0.011)
    # Moved by +2 mm, the opening is [-8, 12] mm in the ring's frame.
    assert not lost(0.002, 0.011) and lost(0.002, -0.009)


# -- tracking ----------------------------------------------------------------


def test_a_reading_holds_one_row_per_monitor_and_one_column_per_turn(
    ring: Any, injection: int, monitors: np.ndarray, shape: np.ndarray
) -> None:
    reading = _shot(ring, injection, monitors, shape, 1)
    for series in (reading.x, reading.y, reading.alive):
        assert series.shape == (monitors.size, TURNS)
        assert series.dtype == np.float64


def test_the_same_shot_number_tracks_identically(
    ring: Any, injection: int, monitors: np.ndarray, shape: np.ndarray
) -> None:
    first = _shot(ring, injection, monitors, shape, 5)
    again = _shot(ring, injection, monitors, shape, 5)
    np.testing.assert_array_equal(first.x, again.x)
    np.testing.assert_array_equal(first.y, again.y)
    np.testing.assert_array_equal(first.alive, again.alive)


def test_different_shot_numbers_differ_by_the_jitter_scale(
    ring: Any, injection: int, monitors: np.ndarray, shape: np.ndarray
) -> None:
    one = _shot(ring, injection, monitors, shape, 1)
    two = _shot(ring, injection, monitors, shape, 2)
    assert not np.array_equal(one.x, two.x)
    # At the first monitor after the injection on the first turn the centroid
    # has moved by about the launch difference, the jitter draws' difference.
    normals_one = keyed_normals(turns.shot_key(MODEL, 1), np.arange(2))
    normals_two = keyed_normals(turns.shot_key(MODEL, 2), np.arange(2))
    launched_dx = JITTER[0] * (normals_one[0] - normals_two[0])
    launched_dy = JITTER[2] * (normals_one[1] - normals_two[1])
    first = int(np.argmin((monitors - injection) % len(ring)))
    dx = one.x[first, 0] - two.x[first, 0]
    dy = one.y[first, 0] - two.y[first, 0]
    assert 0.1 * abs(launched_dx) < abs(dx) < 10.0 * abs(launched_dx)
    assert 0.1 * abs(launched_dy) < abs(dy) < 10.0 * abs(launched_dy)
    spread = np.nanmax(np.abs(one.x - two.x))
    assert spread < 100.0 * max(JITTER)


def test_no_jitter_makes_every_shot_alike(
    ring: Any, injection: int, monitors: np.ndarray, shape: np.ndarray
) -> None:
    one = turns.track_shot(ring, injection, monitors, LAUNCH, 2, shape=shape, key=MODEL, shot=1)
    two = turns.track_shot(ring, injection, monitors, LAUNCH, 2, shape=shape, key=MODEL, shot=2)
    np.testing.assert_array_equal(one.x, two.x)


def test_a_corrector_kick_changes_only_monitors_downstream_on_the_first_turn(
    ring: Any, injection: int, monitors: np.ndarray
) -> None:
    correctors = [
        int(i) for i in ring.get_uint32_index(at.Corrector) if ring[int(i)].FamName.startswith("H")
    ]
    corrector = correctors[len(correctors) // 2]
    kicked = copy.deepcopy(ring)
    kicked[corrector].KickAngle = np.array([KICK, 0.0])
    plain = turns.single_particle(ring, injection, monitors, LAUNCH)
    moved = turns.single_particle(kicked, injection, monitors, LAUNCH)
    assert plain.x.shape == (monitors.size, 1)
    downstream = (monitors - injection) % len(ring) > (corrector - injection) % len(ring)
    assert downstream.any() and (~downstream).any()
    np.testing.assert_array_equal(moved.x[~downstream, 0], plain.x[~downstream, 0])
    np.testing.assert_array_equal(moved.y[~downstream, 0], plain.y[~downstream, 0])
    assert np.all(moved.x[downstream, 0] != plain.x[downstream, 0])


def test_a_single_particle_reads_its_own_position_and_whole_or_nothing(
    ring: Any, injection: int, monitors: np.ndarray
) -> None:
    reading = turns.single_particle(ring, injection, monitors, LAUNCH, turns=2)
    np.testing.assert_array_equal(reading.alive, 1.0)
    first = int(np.argmin((monitors - injection) % len(ring)))
    assert reading.x[first, 0] == pytest.approx(LAUNCH[0], abs=5e-3)


def test_a_lost_particle_reads_nan_and_zero_alive(
    ring: Any, injection: int, monitors: np.ndarray
) -> None:
    lattice = copy.deepcopy(ring)
    turns.write_apertures(lattice, elliptical=(0.005, 0.005))
    reading = turns.single_particle(lattice, injection, monitors, (0.02, 0, 0, 0, 0, 0))
    np.testing.assert_array_equal(reading.alive, 0.0)
    assert np.isnan(reading.x).all() and np.isnan(reading.y).all()


def test_the_alive_share_never_increases(ring: Any, injection: int, monitors: np.ndarray) -> None:
    lattice = copy.deepcopy(ring)
    turns.write_apertures(lattice, elliptical=(0.0035, 0.003))
    launch = (2.5e-3, 0.0, 0.0, 0.0, 0.0, 0.0)
    wide = turns.declared_bunch(
        ring, injection, 256, emittance_x=5.0e-7, emittance_y=5.0e-8, sigma_delta=1.0e-3
    )
    reading = turns.track_shot(
        lattice, injection, monitors, launch, 6, shape=wide, key=MODEL, shot=3, jitter=JITTER
    )
    order = np.argsort((monitors - injection) % len(ring), kind="stable")
    sequence = reading.alive[order, :].T.ravel()  # turn by turn, monitors in beam order
    assert np.all(np.diff(sequence) <= 0.0)
    assert 0.0 < sequence[-1] < sequence[0] <= 1.0  # some, not all, are lost


def test_tracking_in_chunks_matches_one_call(
    ring: Any, injection: int, monitors: np.ndarray, shape: np.ndarray, monkeypatch: Any
) -> None:
    whole = _shot(ring, injection, monitors, shape, 9)
    monkeypatch.setattr(turns, "TRACK_FLOATS", 6 * monitors.size * TURNS * 5)
    chunked = _shot(ring, injection, monitors, shape, 9)
    np.testing.assert_allclose(chunked.x, whole.x, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(chunked.alive, whole.alive)


def test_tracking_leaves_the_lattice_untouched(
    ring: Any, injection: int, monitors: np.ndarray, shape: np.ndarray
) -> None:
    before = [element.FamName for element in ring]
    _shot(ring, injection, monitors, shape, 1)
    assert [element.FamName for element in ring] == before


# -- pulsed kicks ------------------------------------------------------------

KICKER = "HCM01"  # a corrector after BPM01 and before BPM02


@pytest.fixture(scope="module")
def kicker(ring: Any) -> int:
    (index,) = ring.get_uint32_index(KICKER)
    return int(index)


def _pulsed(ring: Any, injection: int, monitors: np.ndarray, kicks: Any, turns_: int = 5) -> Any:
    return turns.track_shot(ring, injection, monitors, LAUNCH, turns_, pulsed=kicks)


def _kicked_copy(ring: Any, kicker: int, angle: float) -> Any:
    lattice = copy.deepcopy(ring)
    lattice[kicker].KickAngle = np.array([angle, 0.0])
    return lattice


def test_a_pulse_that_lasts_every_turn_is_a_static_kick(
    ring: Any, injection: int, monitors: np.ndarray, kicker: int
) -> None:
    pulsed = _pulsed(ring, injection, monitors, [turns.PulsedKick(kicker, 0, KICK, (1.0,) * 5)])
    static = _pulsed(_kicked_copy(ring, kicker, KICK), injection, monitors, ())
    np.testing.assert_array_equal(pulsed.x, static.x)
    np.testing.assert_array_equal(pulsed.y, static.y)


def test_a_one_turn_pulse_kicks_the_first_turn_only(
    ring: Any, injection: int, monitors: np.ndarray, kicker: int
) -> None:
    pulsed = _pulsed(ring, injection, monitors, [turns.PulsedKick(kicker, 0, KICK, (1.0,))])
    static = _pulsed(_kicked_copy(ring, kicker, KICK), injection, monitors, ())
    plain = _pulsed(ring, injection, monitors, ())
    np.testing.assert_array_equal(pulsed.x[:, 0], static.x[:, 0])
    assert not np.allclose(pulsed.x[:, 1:], static.x[:, 1:], atol=1e-9)
    # The monitor ahead of the kicker reads the unkicked bunch on turn 1.
    ahead = (monitors - injection) % len(ring) < (kicker - injection) % len(ring)
    np.testing.assert_array_equal(pulsed.x[ahead, 0], plain.x[ahead, 0])
    assert np.all(pulsed.x[~ahead, 0] != plain.x[~ahead, 0])


def test_a_one_turn_pulse_is_two_trackings_end_to_end(
    ring: Any, injection: int, monitors: np.ndarray, kicker: int
) -> None:
    pulsed = _pulsed(ring, injection, monitors, [turns.PulsedKick(kicker, 0, KICK, (1.0,))], 3)
    start = np.asfortranarray(np.asarray(LAUNCH, dtype=np.float64).reshape(6, 1))
    kicked = _kicked_copy(ring, kicker, KICK).rotate(injection)
    at.lattice_track(kicked, start, nturns=1, in_place=True)
    places = np.sort((monitors - injection) % len(ring)).astype(np.uint32)
    out, _, _ = at.lattice_track(
        ring.rotate(injection), start, nturns=2, refpts=places, keep_counter=True
    )
    order = np.argsort(np.argsort((monitors - injection) % len(ring)))
    np.testing.assert_allclose(pulsed.x[:, 1:], out[0, 0][order], atol=1e-14)


def test_a_pulse_scales_the_kick_turn_by_turn(
    ring: Any, injection: int, monitors: np.ndarray, kicker: int
) -> None:
    half = _pulsed(ring, injection, monitors, [turns.PulsedKick(kicker, 0, KICK, (0.5,))])
    static = _pulsed(_kicked_copy(ring, kicker, 0.5 * KICK), injection, monitors, ())
    np.testing.assert_array_equal(half.x[:, 0], static.x[:, 0])
    none = _pulsed(ring, injection, monitors, [turns.PulsedKick(kicker, 0, KICK, (0.0,))])
    plain = _pulsed(ring, injection, monitors, ())
    np.testing.assert_array_equal(none.x, plain.x)


def test_a_vertical_pulse_kicks_the_other_plane(
    ring: Any, injection: int, monitors: np.ndarray, kicker: int
) -> None:
    launch = (0.0,) * 6
    reading = turns.track_shot(
        ring, injection, monitors, launch, 1, pulsed=[turns.PulsedKick(kicker, 1, KICK, (1.0,))]
    )
    plain = turns.track_shot(ring, injection, monitors, launch, 1)
    after = int(np.flatnonzero(reading.y[:, 0] != plain.y[:, 0])[0])
    # First order in the kick vertically; horizontally only what the sextupoles couple in.
    assert abs(reading.y[after, 0]) > 1e-5
    assert abs(reading.x[after, 0]) < 1e-3 * abs(reading.y[after, 0])


def test_a_pulsed_kick_adds_to_the_kick_the_element_holds(
    ring: Any, injection: int, monitors: np.ndarray, kicker: int
) -> None:
    held = _kicked_copy(ring, kicker, KICK)
    both = _pulsed(held, injection, monitors, [turns.PulsedKick(kicker, 0, KICK, (1.0,))], 1)
    double = _pulsed(_kicked_copy(ring, kicker, 2 * KICK), injection, monitors, (), 1)
    np.testing.assert_array_equal(both.x, double.x)


def test_a_pulsed_kick_leaves_the_lattice_untouched(
    ring: Any, injection: int, monitors: np.ndarray, kicker: int
) -> None:
    before = np.array(ring[kicker].KickAngle, dtype=np.float64)
    _pulsed(ring, injection, monitors, [turns.PulsedKick(kicker, 0, KICK, (1.0, 0.5))])
    np.testing.assert_array_equal(ring[kicker].KickAngle, before)


def test_a_pulse_tracks_the_same_in_particle_chunks(
    ring: Any,
    injection: int,
    monitors: np.ndarray,
    shape: np.ndarray,
    kicker: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kicks = [turns.PulsedKick(kicker, 0, KICK, (1.0, 0.25))]
    whole = turns.track_shot(ring, injection, monitors, LAUNCH, 4, shape=shape, pulsed=kicks)
    monkeypatch.setattr(turns, "TRACK_FLOATS", 6 * monitors.size * 4 * 8)
    chunked = turns.track_shot(ring, injection, monitors, LAUNCH, 4, shape=shape, pulsed=kicks)
    np.testing.assert_allclose(chunked.x, whole.x, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(chunked.alive, whole.alive)


def test_an_element_that_applies_no_kick_angle_is_named(ring: Any) -> None:
    (drift,) = ring.get_uint32_index("SEPTUM")
    assert turns.applies_kick_angle(ring[int(drift)]) is False
    (corrector,) = ring.get_uint32_index(KICKER)
    assert turns.applies_kick_angle(ring[int(corrector)]) is True
    quadrupole = at.Quadrupole("K", 0.3, 0.0)
    assert turns.applies_kick_angle(quadrupole) is True
    assert not hasattr(quadrupole, "KickAngle")


# -- how far the beam got -------------------------------------------------------


def _reading_of(ring: Any, injection: int, monitors: np.ndarray, alive: list[list[float]], mp):
    """A shot's reading with the alive shares given per monitor, in ring order from the injection."""
    shares = np.array(alive, dtype=np.float64)
    centroid = np.where(shares > 0.0, 0.0, np.nan)
    mp.setattr(turns, "track_particles", lambda *args: (shares, centroid, centroid))
    return turns.track_shot(ring, injection, monitors[: len(alive)], LAUNCH, shares.shape[1])


@pytest.mark.parametrize(
    ("alive", "survived", "reached"),
    [
        ([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], 3, 3),
        ([[1.0, 1.0, 0.5], [1.0, 0.5, 0.0], [1.0, 0.5, 0.0]], 2, 3),
        ([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]], 0, 2),
        ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], 0, 0),
    ],
    ids=["kept", "lost on turn 3", "lost on turn 1", "lost at once"],
)
def test_a_reading_says_how_far_the_beam_got(
    ring: Any,
    injection: int,
    monitors: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    alive: list[list[float]],
    survived: int,
    reached: int,
) -> None:
    reading = _reading_of(ring, injection, monitors, alive, monkeypatch)
    assert reading.turns_survived == survived
    assert reading.monitors_reached == reached


def test_the_last_monitor_is_the_last_one_from_the_injection(
    ring: Any, injection: int, monitors: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The caller's order does not matter: the beam's does."""
    alive = [[1.0, 1.0], [1.0, 0.0]]
    two = monitors[:2]
    forward = _reading_of(ring, injection, two, alive, monkeypatch)
    backward = _reading_of(ring, injection, two[::-1], alive, monkeypatch)
    assert forward.turns_survived == backward.turns_survived == 1
    np.testing.assert_array_equal(forward.alive, backward.alive[::-1])


def test_a_tracked_beam_that_is_kept_survives_every_turn(
    ring: Any, injection: int, monitors: np.ndarray
) -> None:
    reading = turns.track_shot(ring, injection, monitors, LAUNCH, 3)
    assert reading.turns_survived == 3
    assert reading.monitors_reached == len(monitors)


def test_the_bunch_set_up_error_is_a_value_error(ring: Any, injection: int) -> None:
    assert issubclass(turns.ShotSetupError, ValueError)
    with pytest.raises(turns.ShotSetupError, match="mismatch factor"):
        turns.declared_bunch(ring, injection, 4, mismatch=(0.0, 1.0))
