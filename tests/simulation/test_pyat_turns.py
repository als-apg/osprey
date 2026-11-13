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

DECK = Path(__file__).resolve().parents[2] / "src/osprey/templates/facilities/example/decks/SR.json"
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
JITTER = (2.0e-4, 1.0e-4)
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
    centre[2] += JITTER[1] * normals[1]
    expected = centre.reshape(6, 1) + shape
    np.testing.assert_array_equal(launched, expected)


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
    launched_dy = JITTER[1] * (normals_one[1] - normals_two[1])
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
