"""The pyat engine's turn tracking: injected particles read per monitor and turn.

Pure functions over a lattice and plain numbers. A declared bunch is a fixed
shape of particles matched to the optics at the injection, built from
deterministic normal quantiles, never from random draws. A shot launches it
at the launch coordinates plus a centroid jitter keyed on the model and the
shot number, so the same shot replays identically in any process. The ring is
rotated to start at the injection element, tracked for some turns with
losses, and every monitor reads, per turn, the survivors' centroid in ``x``
and ``y`` and the share of particles alive. One particle without jitter is the
single-particle reading the response matrix uses.

No model state, no records, no clock. pyAT, numpy and scipy are imported
here, so this module is imported only when a pyat model tracks turns.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import at
import numpy as np

__all__ = [
    "SHAPE_BASES",
    "SLICE_S",
    "TRACK_FLOATS",
    "TurnReading",
    "bunch_shape",
    "declared_bunch",
    "launched_bunch",
    "shot_key",
    "single_particle",
    "track_particles",
    "track_shot",
    "write_apertures",
]

#: The prime bases of the six low-discrepancy sequences a bunch's shape is
#: drawn from, one per pyAT coordinate.
SHAPE_BASES: tuple[int, ...] = (2, 3, 5, 7, 11, 13)

#: The most floats one tracking call may return. A bunch whose track would
#: return more is tracked in particle chunks, which track exactly as one call
#: does: particles do not interact.
TRACK_FLOATS = 1 << 24

#: The longest one tracking call should hold the interpreter, in seconds. A
#: pyAT call does not release the GIL, so a shot is tracked in turn slices,
#: each sized from the last one's pace to stay under this, and the threads
#: serving the model's records answer between slices. Slices track exactly as
#: one call does: each carries the coordinates forward in place and keeps
#: pyAT's turn counter (``keep_counter``; pyAT 0.8's ``turn`` keyword raises),
#: which the RF cavities' path length reads. The counter is pyAT's process
#: state, so a shot's slices must not interleave with other tracking: the
#: model runner runs model work one job at a time.
SLICE_S = 0.05

#: The subkey of a bunch's per-shot draw, between the model's key and the shot.
BUNCH_SUBKEY = b":bunch"


@dataclass(frozen=True)
class TurnReading:
    """One tracked shot, per monitor (rows, in the caller's order) and turn (columns).

    Attributes:
        x: The survivors' horizontal centroid in metres, NaN where none is alive.
        y: The survivors' vertical centroid in metres, NaN where none is alive.
        alive: The share of the launched particles alive, from 0 to 1.
    """

    x: np.ndarray
    y: np.ndarray
    alive: np.ndarray


def shot_key(model: str, shot: int) -> bytes:
    """The key of one shot's jitter draw on one model.

    Args:
        model: The model's id.
        shot: The shot number.

    Returns:
        ``channel_key_bytes(model) + b":bunch" + <shot>``.
    """
    from osprey_connectors.simulation.series import channel_key_bytes

    return channel_key_bytes(model) + BUNCH_SUBKEY + str(int(shot)).encode("ascii")


def _radical_inverse(indexes: np.ndarray, base: int) -> np.ndarray:
    """The van der Corput sequence in ``base`` at ``indexes``."""
    value = np.zeros(indexes.shape, dtype=np.float64)
    scale = 1.0 / base
    remaining = indexes.copy()
    while remaining.any():
        value += (remaining % base) * scale
        remaining //= base
        scale /= base
    return value


def _normal_quantiles(particles: int, base: int) -> np.ndarray:
    """``particles`` centred standard-normal quantiles, ordered by the sequence in ``base``.

    Each particle takes one stratum ``(rank + 0.5) / particles`` of the normal
    distribution, its rank the order of its van der Corput value in ``base``:
    every coordinate holds the same quantiles, and the distinct bases leave
    them uncorrelated across coordinates. One particle sits at 0.
    """
    from scipy.special import ndtri

    ranks = np.argsort(np.argsort(_radical_inverse(np.arange(particles), base), kind="stable"))
    quantiles = np.asarray(ndtri((ranks + 0.5) / particles), dtype=np.float64)
    centred: np.ndarray = quantiles - quantiles.mean()
    return centred


def bunch_shape(
    beta: Sequence[float],
    alpha: Sequence[float],
    dispersion: Sequence[float],
    particles: int,
    *,
    emittance_x: float = 0.0,
    emittance_y: float = 0.0,
    sigma_delta: float = 0.0,
    sigma_z: float = 0.0,
) -> np.ndarray:
    """The particles' offsets from the launch, matched to the given optics.

    Each transverse plane is a Gaussian of its emittance in the Twiss
    ``beta`` and ``alpha``, displaced by the dispersion times the momentum
    offset; the momentum offset and the path length are Gaussians of
    ``sigma_delta`` and ``sigma_z``. An unstated size is 0.

    Args:
        beta: ``(beta_x, beta_y)`` in metres.
        alpha: ``(alpha_x, alpha_y)``.
        dispersion: ``(eta_x, eta_px, eta_y, eta_py)``.
        particles: The particle count, 1 or more.
        emittance_x: Horizontal emittance, metre-radians.
        emittance_y: Vertical emittance, metre-radians.
        sigma_delta: Relative energy spread.
        sigma_z: Bunch length, metres.

    Returns:
        ``(6, particles)`` float64 offsets, centred on 0.

    Raises:
        ValueError: ``particles`` is below 1.
    """
    particles = int(particles)
    if particles < 1:
        raise ValueError(f"a bunch takes 1 or more particles, not {particles}")
    normal = np.vstack([_normal_quantiles(particles, base) for base in SHAPE_BASES])
    twiss_beta = np.asarray(beta, dtype=np.float64)
    twiss_alpha = np.asarray(alpha, dtype=np.float64)
    eta = np.asarray(dispersion, dtype=np.float64)
    shape = np.zeros((6, particles), dtype=np.float64)
    shape[4] = float(sigma_delta) * normal[4]
    shape[5] = float(sigma_z) * normal[5]
    for plane, emittance in enumerate((float(emittance_x), float(emittance_y))):
        a, b = normal[2 * plane], normal[2 * plane + 1]
        shape[2 * plane] = math.sqrt(emittance * twiss_beta[plane]) * a
        shape[2 * plane + 1] = math.sqrt(emittance / twiss_beta[plane]) * (
            b - twiss_alpha[plane] * a
        )
        shape[2 * plane] += eta[2 * plane] * shape[4]
        shape[2 * plane + 1] += eta[2 * plane + 1] * shape[4]
    return shape


def declared_bunch(
    lattice: at.Lattice,
    injection: int,
    particles: int,
    *,
    emittance_x: float = 0.0,
    emittance_y: float = 0.0,
    sigma_delta: float = 0.0,
    sigma_z: float = 0.0,
) -> np.ndarray:
    """The declared bunch's shape, matched to the periodic optics at ``injection``.

    The optics are solved on a 4D copy of ``lattice``; ``lattice`` is not
    changed.

    Args:
        lattice: The ring.
        injection: The index of the injection element.
        particles: The particle count, 1 or more.
        emittance_x: Horizontal emittance, metre-radians.
        emittance_y: Vertical emittance, metre-radians.
        sigma_delta: Relative energy spread.
        sigma_z: Bunch length, metres.

    Returns:
        ``(6, particles)`` float64 offsets from the launch.

    Raises:
        ValueError: ``particles`` is below 1, or the ring has no periodic
            optics at ``injection``.
    """
    if int(particles) < 1:
        raise ValueError(f"a bunch takes 1 or more particles, not {int(particles)}")
    try:
        _, _, optics = lattice.disable_6d(copy=True).get_optics(refpts=[int(injection)])
    except Exception as error:
        raise ValueError(
            f"the lattice has no periodic optics at element {int(injection)} to match "
            f"the bunch to: {error}"
        ) from error
    return bunch_shape(
        optics["beta"][0],
        optics["alpha"][0],
        optics["dispersion"][0],
        particles,
        emittance_x=emittance_x,
        emittance_y=emittance_y,
        sigma_delta=sigma_delta,
        sigma_z=sigma_z,
    )


def launched_bunch(
    launch: Sequence[float],
    shape: np.ndarray,
    *,
    key: str,
    shot: int,
    jitter: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """The particles shot ``shot`` launches, ``(6, particles)``.

    The launch, plus the bunch's fixed shape, plus a centroid jitter in ``x``
    and ``y``: the jitter sigmas times the first two normals of the stream
    keyed on ``key`` and ``shot``. Nothing else varies from shot to shot.

    Args:
        launch: ``(x, px, y, py, delta, ct)`` in pyAT's units.
        shape: The bunch's offsets from the launch, ``(6, particles)``.
        key: The model's id.
        shot: The shot number.
        jitter: The centroid jitter's sigma in ``x`` and ``y``, metres.
    """
    from osprey_connectors.simulation.series import keyed_normals

    start = np.asarray(launch, dtype=np.float64).reshape(6, 1)
    offset = np.zeros((6, 1), dtype=np.float64)
    jitter_x, jitter_y = (float(sigma) for sigma in jitter)
    if (jitter_x, jitter_y) != (0.0, 0.0):
        normals = keyed_normals(shot_key(key, shot), np.arange(2))
        offset[0, 0] = jitter_x * normals[0]
        offset[2, 0] = jitter_y * normals[1]
    return np.asarray(start + offset + np.asarray(shape, dtype=np.float64), dtype=np.float64)


def write_apertures(
    lattice: at.Lattice,
    *,
    elliptical: tuple[float, float] | None = None,
    rectangular: tuple[float, float] | None = None,
) -> None:
    """Write the stated apertures on every element of ``lattice`` that has a length.

    An elliptical aperture is pyAT's ``EApertures``, its two half-axes; a
    rectangular one is ``RApertures``, the four edges of a rectangle centred
    on the axis. ``lattice`` is changed in place: pass a copy to keep the
    original.

    Args:
        lattice: The lattice to write on.
        elliptical: The ellipse's half-axes ``(x, y)`` in metres.
        rectangular: The rectangle's half-widths ``(x, y)`` in metres.

    Raises:
        ValueError: a half-axis is not a finite number above 0.
    """
    apertures: dict[str, np.ndarray] = {}
    for attribute, half_axes in (("EApertures", elliptical), ("RApertures", rectangular)):
        if half_axes is None:
            continue
        x, y = (float(value) for value in half_axes)
        if not (0.0 < x < math.inf and 0.0 < y < math.inf):
            raise ValueError(
                f"an aperture half-axis is a finite number of metres above 0, not ({x:g}, {y:g})"
            )
        edges = (x, y) if attribute == "EApertures" else (-x, x, -y, y)
        apertures[attribute] = np.array(edges, dtype=np.float64)
    if not apertures:
        return
    for element in lattice:
        if float(getattr(element, "Length", 0.0)) > 0.0:
            for attribute, values in apertures.items():
                setattr(element, attribute, values.copy())


def track_particles(
    lattice: at.Lattice, particles: np.ndarray, refpts: np.ndarray, turns: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track ``particles`` and reduce them per reference point and turn, each ``(refpts, turns)``.

    A particle is alive where neither tracked transverse coordinate is NaN.
    A bunch whose track would return more than :data:`TRACK_FLOATS` floats is
    tracked in particle chunks, and every chunk in turn slices of about
    :data:`SLICE_S` each, so other threads run while a long shot tracks.

    Args:
        lattice: The lattice, tracked from its first element.
        particles: ``(6, particles)`` launch coordinates.
        refpts: The element indexes read, in the lattice's order.
        turns: The turns tracked, 1 or more.

    Returns:
        The share of ``particles`` alive, and the survivors' centroid in
        ``x`` and ``y``, NaN where none is alive.
    """
    particles = np.asarray(particles, dtype=np.float64)
    refpts = np.asarray(refpts, dtype=np.uint32)
    count = particles.shape[1]
    alive = np.zeros((refpts.size, turns), dtype=np.int64)
    sum_x = np.zeros((refpts.size, turns), dtype=np.float64)
    sum_y = np.zeros((refpts.size, turns), dtype=np.float64)
    chunk = max(1, TRACK_FLOATS // (6 * max(1, refpts.size) * max(1, turns)))
    for begin in range(0, count, chunk):
        coordinates = np.array(particles[:, begin : begin + chunk], dtype=np.float64, order="F")
        done = 0
        step = 1
        while done < turns:
            nturns = min(step, turns - done)
            started = time.perf_counter()
            tracked, _, _ = at.lattice_track(
                lattice,
                coordinates,
                nturns=nturns,
                refpts=refpts,
                in_place=True,
                keep_counter=done > 0,
                losses=True,
            )
            elapsed = time.perf_counter() - started
            x, y = tracked[0], tracked[2]
            live = ~(np.isnan(x) | np.isnan(y))
            span = slice(done, done + nturns)
            alive[:, span] += live.sum(axis=0)
            sum_x[:, span] += np.where(live, x, 0.0).sum(axis=0)
            sum_y[:, span] += np.where(live, y, 0.0).sum(axis=0)
            done += nturns
            step = max(1, int(SLICE_S * nturns / elapsed)) if elapsed > 0.0 else 2 * nturns
    with np.errstate(invalid="ignore", divide="ignore"):
        centroid_x = np.where(alive > 0, sum_x / alive, np.nan)
        centroid_y = np.where(alive > 0, sum_y / alive, np.nan)
    return alive / count, centroid_x, centroid_y


def track_shot(
    lattice: at.Lattice,
    injection: int,
    monitors: Sequence[int] | np.ndarray,
    launch: Sequence[float],
    turns: int,
    *,
    shape: np.ndarray | None = None,
    key: str = "",
    shot: int = 0,
    jitter: tuple[float, float] = (0.0, 0.0),
) -> TurnReading:
    """Track one shot from the injection element for ``turns`` turns.

    The ring is rotated to start at ``injection``, so a turn runs from the
    injection to the injection; a monitor upstream of the injection is read
    at the end of each turn. ``lattice`` is not changed.

    Args:
        lattice: The ring, with whatever apertures it carries.
        injection: The index of the injection element in ``lattice``.
        monitors: The monitors' element indexes in ``lattice``, in the order
            the reading's rows take.
        launch: ``(x, px, y, py, delta, ct)`` the shot is launched at.
        turns: The turns tracked, 1 or more.
        shape: The bunch's offsets from the launch, ``(6, particles)``; one
            particle at the launch when ``None``.
        key: The model's id, which keys the jitter.
        shot: The shot number, which keys the jitter.
        jitter: The centroid jitter's sigma in ``x`` and ``y``, metres.

    Returns:
        The reading per monitor and turn.

    Raises:
        ValueError: ``turns`` is below 1.
    """
    turns = int(turns)
    if turns < 1:
        raise ValueError(f"a shot tracks 1 or more turns, not {turns}")
    start = int(injection)
    length = len(lattice)
    rotated = lattice.rotate(start)
    places = (np.asarray(monitors, dtype=np.int64) - start) % length
    order = np.argsort(places, kind="stable")
    refpts = places[order].astype(np.uint32)
    if shape is None:
        shape = np.zeros((6, 1), dtype=np.float64)
    particles = launched_bunch(launch, shape, key=key, shot=shot, jitter=jitter)
    alive, x, y = track_particles(rotated, particles, refpts, turns)
    rows = np.argsort(order)  # each caller's monitor's row in ring order
    return TurnReading(x=x[rows], y=y[rows], alive=alive[rows])


def single_particle(
    lattice: at.Lattice,
    injection: int,
    monitors: Sequence[int] | np.ndarray,
    launch: Sequence[float],
    turns: int = 1,
) -> TurnReading:
    """One particle at ``launch``, without jitter, tracked from the injection.

    The reading the first-turn response matrix is built from: each monitor
    reads the particle's position, and an alive share of 1 or 0.

    Args:
        lattice: The ring.
        injection: The index of the injection element in ``lattice``.
        monitors: The monitors' element indexes in ``lattice``.
        launch: ``(x, px, y, py, delta, ct)``.
        turns: The turns tracked, 1 by default.
    """
    return track_shot(lattice, injection, monitors, launch, turns)
