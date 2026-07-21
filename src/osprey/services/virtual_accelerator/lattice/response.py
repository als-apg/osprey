"""Corrector-kick closed-orbit response for the ALS-U AR lattice.

Provides orbit_response(corrector_name, current) -> {bpm_name: (x_m, y_m)}:
applies one corrector's current through the same current->strength formula
:mod:`ioc.physics_bridge` uses (delegated to
:class:`~osprey.services.virtual_accelerator.lattice.strengths.StrengthMap`,
which sets ``KickAngle[plane] = I / AMPS_PER_RADIAN_KICK`` absolute), re-solves
the closed orbit with the guarded :func:`~.solve.solve_orbit` helper, and
returns every BPM's (x, y) reading in meters, keyed by its flat FamName (e.g.
"BPM01"). Sharing both the ring source (`build_ring`) and the strength-map
code path with the bridge is deliberate: it is what lets the ORM crosscheck
(task 4.3) prove this oracle and the live IOC agree because they run the same
model, not two independently-written ones.

``AMPS_PER_RADIAN_KICK`` is defined in this module and imported by
``strengths`` (its docstring and formula reference it); ``StrengthMap`` is
therefore imported back into this module lazily, inside
:func:`_get_strength_map`, rather than at module scope, to avoid a load-order
cycle between the two modules.

``solve_orbit`` guards against the unstable-lattice failure mode described in
`solve.py`'s docstring (non-finite/unstable closed orbit surfaces as
`OrbitSolveError`, not a silently-returned garbage orbit); this oracle lets
that exception propagate to its caller uncaught -- a corrector current that
destabilizes the ring has no valid model-oracle answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import at

from .ring import build_ring
from .solve import solve_orbit

if TYPE_CHECKING:
    from .strengths import StrengthMap

# Current-to-kick calibration shared with ioc.physics_bridge via
# strengths.StrengthMap: KickAngle[plane] = I / AMPS_PER_RADIAN_KICK.
#
# This is still a deterministic, sign-correct calibration constant, not a
# claim of physical realism -- but its magnitude matters now in a way it
# didn't on the toy ring: the real AR lattice carries much stronger
# sextupoles (low-emittance MBA design), so a corrector kick large enough to
# move the closed orbit by several mm picks up measurable *nonlinear*
# sextupole feed-down (the +I/-I response stops being antisymmetric). Kept
# at 1e6 (rather than the toy ring's 1e4) so a corrector's typical +-10 A
# range stays in the small-signal/quasi-linear regime (tens of microns of
# orbit shift, not millimeters) -- see test_lattice.py's antisymmetry check.
AMPS_PER_RADIAN_KICK = 1_000_000.0

_CORRECTOR_FAMILIES = ("HCM", "VCM")

_RING: at.Lattice | None = None
_STRENGTH_MAP: StrengthMap | None = None


def _get_ring() -> at.Lattice:
    """Return the (lazily-built, process-wide cached) SR lattice."""
    global _RING
    if _RING is None:
        _RING = build_ring()
    return _RING


def _get_strength_map(ring: at.Lattice) -> StrengthMap:
    """Return the (lazily-built, process-wide cached) StrengthMap for `ring`.

    Imported locally rather than at module scope: `strengths` imports
    `AMPS_PER_RADIAN_KICK` from this module, so importing `StrengthMap` back
    at this module's top level would create a load-order cycle.
    """
    global _STRENGTH_MAP
    if _STRENGTH_MAP is None:
        from .strengths import StrengthMap

        _STRENGTH_MAP = StrengthMap(ring)
    return _STRENGTH_MAP


def _split_corrector_name(corrector_name: str) -> tuple[str, str]:
    """Split a corrector's FamName (e.g. "HCM01") into ("HCM", "01")."""
    for family in _CORRECTOR_FAMILIES:
        if corrector_name.startswith(family):
            return family, corrector_name[len(family) :]
    raise ValueError(
        f"unrecognized corrector name '{corrector_name}' (expected 'HCM..' or 'VCM..')"
    )


def _corrector_index(ring: at.Lattice, corrector_name: str) -> int:
    for i, el in enumerate(ring):
        if el.FamName == corrector_name:
            return i
    raise ValueError(f"no corrector element named '{corrector_name}' in the SR lattice")


def orbit_response(corrector_name: str, current: float) -> dict[str, tuple[float, float]]:
    """Set one corrector's current and return the resulting closed-orbit BPM readings.

    Args:
        corrector_name: Flat FamName of the corrector, e.g. "HCM01" or "VCM07"
            (matches the manifest's zero-padded SR:MAG:{HCM,VCM} device ids).
        current: Corrector current in Amps.

    Returns:
        Dict mapping every BPM's FamName (e.g. "BPM01") to its (x, y) closed-orbit
        position in meters. The corrector's kick is reset to zero before this
        function returns, regardless of outcome, so repeated calls are
        independent (no accumulated state).

    Raises:
        ValueError: if corrector_name doesn't match "HCM.."/"VCM.." or doesn't
            match any element in the lattice.
        OrbitSolveError: if the resulting lattice has no stable closed orbit
            (see :func:`~.solve.solve_orbit`).
    """
    ring = _get_ring()
    family, device_id = _split_corrector_name(corrector_name)
    idx = _corrector_index(ring, corrector_name)
    strength_map = _get_strength_map(ring)

    try:
        strength_map.apply(ring, family, device_id, current)
        orbit_at_monitors = solve_orbit(ring)
    finally:
        ring[idx].KickAngle = [0.0, 0.0]

    monitor_indices = ring.get_refpts(at.Monitor)
    return {
        ring[el_idx].FamName: (float(orbit_at_monitors[row, 0]), float(orbit_at_monitors[row, 2]))
        for row, el_idx in enumerate(monitor_indices)
    }
