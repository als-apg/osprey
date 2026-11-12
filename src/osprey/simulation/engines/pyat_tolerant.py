"""The pyat engine's tolerant periodic solve: a lost closed orbit is a state, not a refusal.

A periodic model that states where a tracked particle is launched can still be
driven when its lattice has no closed orbit: the particle is tracked from the
launch whatever the lattice does to it. :class:`TolerantPeriodicSimulator`
therefore solves the closed orbit exactly as
:class:`~lume_pyat.simulator.PyATSimulator` does, and where that solve fails it
installs a solution that reads NaN at every monitor instead of raising. That
solution is a :class:`NoClosedOrbit`, and it is the flag: the simulator has no
closed orbit exactly while its last solution is one. The flag is therefore
snapshotted and restored with the solution, and a later solve that finds a
closed orbit clears it by replacing the solution.

pyAT, numpy and lume-pyat are imported here, so this module is imported only
when a model is built.
"""

from __future__ import annotations

import math

import at
from lume_pyat.exceptions import OrbitSolveError
from lume_pyat.simulator import PyATSimulator

__all__ = ["NoClosedOrbit", "TolerantPeriodicSimulator"]


class NoClosedOrbit(dict[str, tuple[float, float]]):
    """The solution of a lattice with no closed orbit: NaN at every monitor.

    Attributes:
        reason: What the closed-orbit solve said when it failed.
    """

    def __init__(self, monitors: list[str], reason: str) -> None:
        super().__init__(dict.fromkeys(monitors, (math.nan, math.nan)))
        self.reason: str = reason


class TolerantPeriodicSimulator(PyATSimulator):
    """One persistent periodic lattice whose solve never refuses a lost closed orbit.

    Everything but :meth:`solve` is the closed-orbit simulator's: the lattice
    is owned and mutated in place, element lookup is by ``FamName`` and
    monitor names are unique.
    """

    @property
    def no_closed_orbit(self) -> bool:
        """Whether the last solve found no closed orbit."""
        return isinstance(self._last_solution, NoClosedOrbit)

    def solve(self) -> dict[str, tuple[float, float]]:
        """Solve the closed orbit, or record that the lattice has none.

        Returns:
            ``FamName -> (x, y)`` in meters at every monitor; a
            :class:`NoClosedOrbit`, NaN at every monitor, when the lattice has
            no closed orbit. Either way :attr:`last_solution` is replaced
            with a new object.
        """
        try:
            return super().solve()
        except OrbitSolveError as exc:
            solution = NoClosedOrbit(
                [element.FamName for element in self._lattice if isinstance(element, at.Monitor)],
                str(exc),
            )
            self._last_solution = solution
            return solution
