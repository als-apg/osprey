"""Guarded 4D closed-orbit solve, re-exported from ``lume_pyat``.

The solve itself -- and the load-bearing safety semantic around it -- lives in
:mod:`lume_pyat.solve`: on a nonlinear ring an unstable magnet configuration
presents as NaN-without-exception (``at.find_m44``/``at.find_orbit4`` emit an
``at.AtWarning`` and return non-finite garbage rather than raising), so
:func:`~lume_pyat.solve.solve_orbit` detects instability *by value* and raises
:class:`~lume_pyat.exceptions.OrbitSolveError` instead of serving a garbage
orbit to a BPM readback.

This module stays as the ring-facing name for those helpers because callers
here reach them without going through the model: the ORM oracle
(:mod:`.response`) and the IOC bridge (:mod:`..ioc.physics_bridge`) both
import from here. Re-exported, never restated -- there is exactly one
``solve_orbit`` and exactly one ``OrbitSolveError`` in the process, so a
caller catching the exception from this module catches the same class a
model driving the lattice through ``lume_pyat`` raises.

``OrbitSolveError`` comes from :mod:`lume_pyat.exceptions`, its canonical
home, rather than through :mod:`lume_pyat.solve`'s namespace.
"""

from __future__ import annotations

from lume_pyat.exceptions import OrbitSolveError
from lume_pyat.solve import monitor_xy, solve_orbit

__all__ = [
    "OrbitSolveError",
    "monitor_xy",
    "solve_orbit",
]
