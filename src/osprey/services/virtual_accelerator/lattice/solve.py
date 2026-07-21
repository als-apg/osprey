"""Guarded 4D closed-orbit solve for the ALS-U AR ring.

This is the load-bearing safety semantic of the VA ring repoint: on the real
nonlinear AR ring, an unstable/destabilized magnet configuration presents as
NaN-without-exception -- ``at.find_m44``/``at.find_orbit4`` emit an
``at.AtWarning`` and return non-finite garbage rather than raising. A caller
that just checked ``np.isfinite`` on a bare ``find_orbit4`` call, or worse,
didn't check at all, would fail *open*: it would serve a garbage orbit to a
BPM readback instead of refusing to.

:func:`solve_orbit` closes that gap by detecting instability *by value*
(non-finite one-turn-matrix entries, |trace| >= 2.0 in either transverse
plane, or a non-finite closed orbit) and raising :class:`OrbitSolveError`
instead. pyAT's own warnings are expected noise on the failure path here --
they are suppressed inside this module so a guard trip surfaces to the
caller as a single exception, not console spam plus an exception.

Ring-facing: this module only imports ``at``/``numpy`` (plus the stdlib) so
it stays valid for any 4D-canonical ring built the same way (see
:func:`osprey.services.virtual_accelerator.lattice.build_ring`), independent
of how that ring gets served (softioc today, a future LUME-based physics
server later).
"""

from __future__ import annotations

import warnings

import at
import numpy as np

# |trace| at or above this value marks a transverse plane's one-turn map as
# unstable (a stable linear map has both eigenvalues on the unit circle,
# which bounds |trace| = |lambda + 1/lambda| < 2).
_TRACE_INSTABILITY_THRESHOLD = 2.0


class OrbitSolveError(Exception):
    """Raised when a ring's one-turn map or closed orbit is unstable or non-finite.

    Covers three guard conditions, checked in order by :func:`solve_orbit`:
    a non-finite ``find_m44`` one-turn matrix, a transverse-plane one-turn
    trace with ``|trace| >= 2.0``, or a non-finite ``find_orbit4`` closed
    orbit. Re-exported from ``ioc.physics_bridge`` for API stability once
    that module is repointed onto this solve helper.
    """


def _monitor_refpts(ring: at.Lattice) -> np.ndarray:
    """Indices of every `at.Monitor` element in `ring`, selected by type."""
    return np.array([i for i, element in enumerate(ring) if isinstance(element, at.Monitor)])


def monitor_xy(ring: at.Lattice, orbit_at_monitors: np.ndarray) -> list[tuple[str, float, float]]:
    """Per-monitor ``(FamName, x, y)`` readout of a :func:`solve_orbit` result.

    The single shared "read the solved orbit at the BPMs" primitive: both the
    live IOC bridge (`ioc.physics_bridge`) and the model oracle
    (`lattice.response`) key their BPM readings off this, which is what keeps
    their row -> element alignment identical by construction --
    `orbit_at_monitors` rows are ordered by the same :func:`_monitor_refpts`
    selection :func:`solve_orbit` solved at.

    Args:
        ring: The lattice `orbit_at_monitors` was solved on.
        orbit_at_monitors: A :func:`solve_orbit` result for `ring`, shape
            `(n_monitors, 6)`.

    Returns:
        One ``(FamName, x_m, y_m)`` tuple per `at.Monitor` element, in ring
        order (e.g. ``("BPM01", 1.2e-6, -3.4e-6)``), with x/y taken from
        orbit coordinates 0 and 2.
    """
    return [
        (ring[el_idx].FamName, float(orbit_at_monitors[row, 0]), float(orbit_at_monitors[row, 2]))
        for row, el_idx in enumerate(_monitor_refpts(ring))
    ]


def solve_orbit(ring: at.Lattice) -> np.ndarray:
    """Guarded 4D closed-orbit solve at every `at.Monitor` refpt in `ring`.

    Runs `at.find_m44` and `at.find_orbit4` with pyAT's `AtWarning`
    instability warnings suppressed (they are the expected shape of the
    failure this function guards against, not information the caller
    needs), then checks each result by value before trusting it:

    1. `find_m44`'s one-turn matrix must be entirely finite.
    2. Its transverse-plane traces (x-block `m44[0,0] + m44[1,1]`,
       y-block `m44[2,2] + m44[3,3]`) must both satisfy `|trace| < 2.0`.
    3. `find_orbit4`'s closed orbit at the monitor refpts must be entirely
       finite.

    Args:
        ring: A 4D-canonical `at.Lattice` (radiation/cavity disabled, e.g.
            via `build_ring()`).

    Returns:
        The `find_orbit4` closed-orbit array at every `at.Monitor` refpt,
        shape `(n_monitors, 6)` (72 monitors on the ALS-U AR ring).

    Raises:
        OrbitSolveError: if any of the three guard conditions above trips --
            the ring configuration is unstable and no orbit can be trusted.
    """
    refpts = _monitor_refpts(ring)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=at.AtWarning)
        m44 = at.find_m44(ring)[0]

    if not np.all(np.isfinite(m44)):
        raise OrbitSolveError("find_m44 one-turn matrix has non-finite entries")

    trace_x = m44[0, 0] + m44[1, 1]
    trace_y = m44[2, 2] + m44[3, 3]
    if abs(trace_x) >= _TRACE_INSTABILITY_THRESHOLD or abs(trace_y) >= _TRACE_INSTABILITY_THRESHOLD:
        raise OrbitSolveError(
            f"one-turn matrix unstable: |trace_x| = {abs(trace_x):.3f}, "
            f"|trace_y| = {abs(trace_y):.3f} (threshold {_TRACE_INSTABILITY_THRESHOLD})"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=at.AtWarning)
        _, orbit_at_monitors = at.find_orbit4(ring, refpts=refpts)

    if not np.all(np.isfinite(orbit_at_monitors)):
        raise OrbitSolveError("find_orbit4 returned a non-finite closed orbit")

    return np.asarray(orbit_at_monitors)
