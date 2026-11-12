"""A shot tracked in turn slices reads exactly as one tracking call over every turn.

pyAT holds the GIL for a whole call, so ``track_particles`` tracks in slices
of a few turns and the threads serving the model's records answer between
them. These tests pin that the slicing is invisible in the result: the same
centroids and survival, bit for bit, whatever the slice length, on a ring
with an RF cavity on (the turn number enters the path length) and apertures
that lose particles part-way through.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

at = pytest.importorskip("at")

from osprey.simulation.engines import pyat_turns as turns  # noqa: E402

DECK = Path(__file__).resolve().parents[2] / "src/osprey/templates/facilities/example/decks/SR.json"
MARKER = "INJ"
TURNS = 40
PARTICLES = 48
SHAPE = {
    "emittance_x": 2.0e-6,
    "emittance_y": 2.0e-7,
    "sigma_delta": 5.0e-3,
    "sigma_z": 3.0e-3,
}
LAUNCH = (2.0e-3, 0.0, -1.0e-3, 0.0, 1.0e-3, 0.0)


@pytest.fixture(scope="module")
def ring() -> Any:
    lattice = at.load_lattice(str(DECK)).deepcopy()
    lattice.enable_6d()
    turns.write_apertures(lattice, elliptical=(4.0e-3, 3.0e-3))
    (start,) = lattice.get_uint32_index(MARKER)
    return lattice.rotate(int(start))


@pytest.fixture(scope="module")
def bunch(ring: Any) -> np.ndarray:
    shape = turns.declared_bunch(ring, 0, PARTICLES, **SHAPE)
    return turns.launched_bunch(LAUNCH, shape, key="SR", shot=1, jitter=(0.0, 0.0))


@pytest.fixture(scope="module")
def refpts(ring: Any) -> np.ndarray:
    return np.asarray(ring.get_uint32_index(at.Monitor), dtype=np.uint32)


def _one_call(ring: Any, bunch: np.ndarray, refpts: np.ndarray) -> tuple[np.ndarray, ...]:
    """The reduction over a single pyAT call for every turn: the reference."""
    tracked, _, _ = at.lattice_track(
        ring, np.ascontiguousarray(bunch), nturns=TURNS, refpts=refpts, losses=True
    )
    x, y = tracked[0], tracked[2]
    live = ~(np.isnan(x) | np.isnan(y))
    alive = live.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = np.where(alive > 0, np.where(live, x, 0.0).sum(axis=0) / alive, np.nan)
        cy = np.where(alive > 0, np.where(live, y, 0.0).sum(axis=0) / alive, np.nan)
    return alive / bunch.shape[1], cx, cy


def test_the_fixture_loses_particles_part_way(
    ring: Any, bunch: np.ndarray, refpts: np.ndarray
) -> None:
    alive, _, _ = _one_call(ring, bunch, refpts)
    assert alive[:, -1].min() < alive[:, 0].max() <= 1.0
    assert alive[:, -1].max() > 0.0


@pytest.mark.parametrize("slice_s", [0.0, 1.0e-4, 10.0])
def test_sliced_tracking_reads_exactly_as_one_call(
    ring: Any,
    bunch: np.ndarray,
    refpts: np.ndarray,
    slice_s: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(turns, "SLICE_S", slice_s)
    expected = _one_call(ring, bunch, refpts)
    got = turns.track_particles(ring, bunch, refpts, TURNS)
    for want, have in zip(expected, got, strict=True):
        np.testing.assert_array_equal(have, want)


def test_slices_track_in_calls_of_one_turn_when_the_budget_is_zero(
    ring: Any, bunch: np.ndarray, refpts: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    track = at.lattice_track

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(int(kwargs["nturns"]))
        return track(*args, **kwargs)

    monkeypatch.setattr(turns, "SLICE_S", 0.0)
    monkeypatch.setattr(turns.at, "lattice_track", counted)
    turns.track_particles(ring, bunch, refpts, TURNS)
    assert calls == [1] * TURNS


def test_no_slice_holds_the_interpreter_much_past_the_budget(
    ring: Any, refpts: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    shape = turns.declared_bunch(ring, 0, 2000, **SHAPE)
    heavy = turns.launched_bunch(LAUNCH, shape, key="SR", shot=2, jitter=(0.0, 0.0))
    spans: list[float] = []
    track = at.lattice_track

    def timed(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = track(*args, **kwargs)
        spans.append(time.perf_counter() - started)
        return result

    monkeypatch.setattr(turns.at, "lattice_track", timed)
    turns.track_particles(ring, heavy, refpts, 200)
    assert len(spans) > 1
    assert max(spans) < 3 * turns.SLICE_S + spans[0]
