"""The pyat engine's single-pass solve: one particle tracked once through a line.

A line has no closed orbit and no tunes: the beam enters with the initial
conditions the model's ``pyat.twiss_in`` states and leaves at the far end.
:class:`SinglePassSimulator` replaces the closed-orbit solve of
:class:`~lume_pyat.simulator.PyATSimulator` with one pass of
``at.lattice_track`` from ``twiss_in``'s ``closed_orbit``, then propagates the
optics from ``twiss_in`` with ``at.get_optics``. Its solution has the same
shape as the closed-orbit solve's -- ``FamName -> (x, y)`` at every monitor --
and also carries the beta functions at those monitors, so a solution and the
optics it was solved with are snapshotted and restored together.

A particle the tracking loses is a failed solve: :class:`OrbitSolveError`
names where pyAT lost it. pyAT, numpy and lume-pyat are imported here, so this
module is imported only when a single-pass model is built.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import at
import numpy as np
from lume_pyat.exceptions import OrbitSolveError
from lume_pyat.simulator import PyATSimulator

__all__ = ["SinglePassSimulator", "SinglePassSolution", "loss_text"]


class SinglePassSolution(dict[str, tuple[float, float]]):
    """One single-pass solve: ``FamName -> (x, y)`` at every monitor.

    Attributes:
        beta: ``FamName -> (beta_x, beta_y)`` at every monitor, propagated
            from ``twiss_in`` through the same lattice.
    """

    def __init__(
        self,
        orbit: Mapping[str, tuple[float, float]],
        beta: Mapping[str, tuple[float, float]],
    ) -> None:
        super().__init__(orbit)
        self.beta: dict[str, tuple[float, float]] = dict(beta)


def loss_text(lattice: Any, loss_map: Any) -> str | None:
    """Where pyAT lost the tracked particle, or ``None`` when it survived.

    Args:
        lattice: The lattice the particle was tracked through.
        loss_map: pyAT's ``loss_map`` for one particle (``islost``, ``elem``,
            ``turn``, ``coord``).

    Returns:
        ``lost at element <index> (<FamName>) turn <turn>: <coordinates>``,
        the six coordinates at the loss in pyAT's order.
    """
    if not bool(loss_map.islost[0]):
        return None
    index = int(loss_map.elem[0])
    name = lattice[index].FamName if index < len(lattice) else "end"
    coordinates = ", ".join(f"{float(value):.6g}" for value in loss_map.coord[0])
    return f"lost at element {index} ({name}) turn {int(loss_map.turn[0])}: [{coordinates}]"


class SinglePassSimulator(PyATSimulator):
    """One persistent line, solved by a single tracked pass from ``twiss_in``.

    Everything but :meth:`solve` is the closed-orbit simulator's: the lattice
    is owned and mutated in place, element lookup is by ``FamName``, monitor
    names are unique, and a failed solve leaves the last solution alone.
    """

    def __init__(
        self,
        lattice: at.Lattice,
        *,
        twiss_in: Mapping[str, np.ndarray],
        element_misalignments: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Adopt the line and the initial conditions every pass starts from.

        Args:
            lattice: The line to own. Mutated in place, never copied.
            twiss_in: pyAT's initial conditions, as ``prepare`` normalises
                them: ``beta``, ``alpha``, ``dispersion`` and a six-long
                ``closed_orbit``.
            element_misalignments: As for
                :class:`~lume_pyat.simulator.PyATSimulator`.
        """
        super().__init__(lattice, element_misalignments=element_misalignments)
        self._twiss_in: dict[str, np.ndarray] = {
            key: np.array(value, dtype=np.float64) for key, value in twiss_in.items()
        }
        self._monitors: np.ndarray = np.array(
            [index for index, element in enumerate(lattice) if isinstance(element, at.Monitor)],
            dtype=np.uint32,
        )

    @property
    def twiss_in(self) -> dict[str, np.ndarray]:
        """The initial conditions every pass starts from, as copies."""
        return {key: value.copy() for key, value in self._twiss_in.items()}

    def solve(self) -> SinglePassSolution:
        """Track one particle once through the line and read every monitor.

        The particle starts at ``twiss_in``'s ``closed_orbit``; the beta
        functions at the monitors are propagated from ``twiss_in``. On success
        :attr:`last_solution` is replaced; on failure it is left alone.

        Returns:
            ``FamName -> (x, y)`` in meters at every monitor, carrying the
            beta functions there.

        Raises:
            OrbitSolveError: the particle is lost (the message says where), or
                the optics propagation raised.
        """
        lattice = self._lattice
        names = [lattice[int(index)].FamName for index in self._monitors]
        r_in = self._twiss_in["closed_orbit"].reshape(6, 1)
        try:
            r_out, _, data = at.lattice_track(lattice, r_in, refpts=self._monitors, losses=True)
            lost = loss_text(lattice, data["loss_map"])
            if lost is not None:
                raise OrbitSolveError(lost)
            _, _, element_data = at.get_optics(
                lattice, refpts=self._monitors, twiss_in=self._twiss_in
            )
        except (at.AtError, np.linalg.LinAlgError, ValueError) as exc:
            raise OrbitSolveError(f"optics solve raised {type(exc).__name__}: {exc}") from exc

        orbit = {
            name: (float(r_out[0, 0, row, 0]), float(r_out[2, 0, row, 0]))
            for row, name in enumerate(names)
        }
        beta = {
            name: (float(element_data.beta[row][0]), float(element_data.beta[row][1]))
            for row, name in enumerate(names)
        }
        solution = SinglePassSolution(orbit, beta)
        self._last_solution = solution
        return solution
