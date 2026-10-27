"""The pyat engine's periodic solve with the closed orbit searched for once.

:func:`lume_pyat.solve.solve_orbit` calls ``at.find_m66`` bare, and
``at.find_m66`` left to itself searches for the closed orbit before it
differentiates about it; ``solve_orbit`` then searches again for the orbit it
reads out. The two searches use the same solver and steps and land on the same
orbit. :class:`PeriodicSimulator` searches once and hands that orbit to
``at.find_m66``, so the one-turn matrix, the orbit and the guards that judge
them are the ones ``solve_orbit`` gives, at half the tracking. A response
check solves the lattice twice per corrector, which is where that half shows.

The guards are copies of ``solve_orbit``'s, run in its order with its
messages and tolerance; only the order of the two pyAT calls changes, and
``tests/simulation/test_pyat_periodic.py`` holds the two solves to the same
bits and the same refusals. pyAT, numpy and lume-pyat are imported here, so
this module is imported only when a periodic response check runs.
"""

from __future__ import annotations

import warnings

import at
import numpy as np
from lume_pyat.exceptions import OrbitSolveError
from lume_pyat.simulator import PyATSimulator
from lume_pyat.solve import monitor_xy

__all__ = ["PeriodicSimulator", "solve_orbit_once"]

#: lume-pyat's limit: a one-turn eigenvalue above 1 + this marks its plane unstable.
_EIGENVALUE_TOLERANCE = 1.0e-6

_PLANES = ("horizontal", "vertical", "longitudinal")


def _monitor_refpts(lattice: at.Lattice) -> np.ndarray:
    """Indices of every ``at.Monitor`` element in ``lattice``, as lume-pyat selects them."""
    return np.array([i for i, element in enumerate(lattice) if isinstance(element, at.Monitor)])


def _eigenvalue_magnitude_per_plane(m66: np.ndarray) -> list[float]:
    """Largest |eigenvalue| of the one-turn map, attributed to each plane as lume-pyat does.

    Each eigenvalue goes to the plane whose uncoupled 2x2 diagonal block has
    the nearest eigenvalue, two per plane.
    """
    values = np.linalg.eigvals(m66)
    block_values = [
        np.linalg.eigvals(m66[2 * plane : 2 * plane + 2, 2 * plane : 2 * plane + 2])
        for plane in range(3)
    ]
    distances = sorted(
        (float(np.min(np.abs(value - block_values[plane]))), k, plane)
        for k, value in enumerate(values)
        for plane in range(3)
    )

    assigned: dict[int, int] = {}
    per_plane_count = [0, 0, 0]
    for _, k, plane in distances:
        if k in assigned or per_plane_count[plane] == 2:
            continue
        assigned[k] = plane
        per_plane_count[plane] += 1

    worst = [0.0, 0.0, 0.0]
    for k, plane in assigned.items():
        worst[plane] = max(worst[plane], float(abs(values[k])))
    return worst


# Stand-in until lume-pyat's solve_orbit hands find_m66 the orbit it searched for; this module goes with it.
def solve_orbit_once(lattice: at.Lattice) -> np.ndarray:
    """:func:`lume_pyat.solve.solve_orbit` with the closed orbit searched once.

    Args:
        lattice: A 4D lattice, or a 6D one with its cavity enabled and radiation off.

    Returns:
        The closed orbit at every ``at.Monitor`` refpt, shape
        ``(n_monitors, 6)``.

    Raises:
        OrbitSolveError: the one-turn matrix is non-finite or has an
            eigenvalue off the unit circle, or the orbit is non-finite.
    """
    refpts = _monitor_refpts(lattice)
    find_orbit = at.find_orbit6 if lattice.is_6d else at.find_orbit4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=at.AtWarning)
        orbit, orbit_at_monitors = find_orbit(lattice, refpts=refpts)
        m66 = at.find_m66(lattice, orbit=orbit, keep_lattice=True)[0]

    if not np.all(np.isfinite(m66)):
        raise OrbitSolveError("find_m66 one-turn matrix has non-finite entries")

    worst = _eigenvalue_magnitude_per_plane(m66)
    unstable = [
        plane
        for plane, magnitude in zip(_PLANES, worst, strict=True)
        if magnitude > 1.0 + _EIGENVALUE_TOLERANCE
    ]
    if unstable:
        where = (
            f"{unstable[0]} plane"
            if len(unstable) == 1
            else f"{', '.join(unstable[:-1])} and {unstable[-1]} planes"
        )
        per_plane = ", ".join(
            f"{plane} {magnitude:.3e}" for plane, magnitude in zip(_PLANES, worst, strict=True)
        )
        raise OrbitSolveError(
            f"one-turn map unstable in the {where}: |eigenvalue| = "
            f"{max(worst):.3e} (limit 1 + {_EIGENVALUE_TOLERANCE:.0e}); "
            f"per plane: {per_plane}"
        )

    if not np.all(np.isfinite(orbit_at_monitors)):
        raise OrbitSolveError(f"{find_orbit.__name__} returned a non-finite closed orbit")

    return np.asarray(orbit_at_monitors)


class PeriodicSimulator(PyATSimulator):
    """:class:`~lume_pyat.simulator.PyATSimulator` solved by :func:`solve_orbit_once`.

    Everything but :meth:`solve` is lume-pyat's, and :meth:`solve` keeps its
    contract: a failed solve leaves :attr:`last_solution` alone, and pyAT's own
    exceptions fold into :class:`OrbitSolveError`.
    """

    def solve(self) -> dict[str, tuple[float, float]]:
        """Solve the closed orbit once and read it out at every monitor.

        Returns:
            ``FamName -> (x, y)`` in meters, one entry per monitor.

        Raises:
            OrbitSolveError: the lattice has no trustworthy closed orbit.
        """
        try:
            orbit_at_monitors = solve_orbit_once(self._lattice)
        except (at.AtError, np.linalg.LinAlgError, ValueError) as exc:
            raise OrbitSolveError(f"closed orbit solve raised {type(exc).__name__}: {exc}") from exc

        solution = {
            fam_name: (x, y) for fam_name, x, y in monitor_xy(self._lattice, orbit_at_monitors)
        }
        self._last_solution = solution
        return solution
