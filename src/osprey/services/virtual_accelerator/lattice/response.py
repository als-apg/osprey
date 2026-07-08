"""Corrector-kick closed-orbit response for the SR lattice.

Provides orbit_response(corrector_name, current) -> {bpm_name: (x_m, y_m)}:
sets one corrector's kick angle from a current, re-solves the closed orbit with
AT's find_orbit4 (4x4 linear closed-orbit search), and returns every BPM's
(x, y) reading in meters. find_orbit4 solves in low-single-digit milliseconds
at this ring's size (280 elements) -- comfortably inside the FR3 synchronous
recompute contract's <100 ms budget (see test_lattice.py for the measured
figure).

Current-to-kick calibration is a toy constant (AMPS_PER_RADIAN_KICK): this
module only needs *a* deterministic, sign-correct, linear mapping so setpoints
move the lattice by a plausible amount. ioc-physics-bridge (task 3.4) owns the
real current<->field calibration used by the running IOC.
"""

from __future__ import annotations

import at

from .ring import build_ring

# Toy calibration: a corrector in the DB's typical +-10A range produces a
# few-mrad kick, giving mm-scale BPM orbit shifts -- plausible for a light
# source ring, without claiming physical precision.
AMPS_PER_RADIAN_KICK = 10_000.0

_RING: at.Lattice | None = None


def _get_ring() -> at.Lattice:
    """Return the (lazily-built, process-wide cached) SR lattice."""
    global _RING
    if _RING is None:
        _RING = build_ring()
    return _RING


def _corrector_plane(corrector_name: str) -> int:
    """Return 0 for a horizontal corrector (HCM), 1 for vertical (VCM)."""
    if corrector_name.startswith("HCM"):
        return 0
    if corrector_name.startswith("VCM"):
        return 1
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
        corrector_name: Lattice FamName of the corrector, e.g. "HCM01" or "VCM07"
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
    """
    ring = _get_ring()
    plane = _corrector_plane(corrector_name)
    idx = _corrector_index(ring, corrector_name)
    kick = current / AMPS_PER_RADIAN_KICK

    kick_angle = [0.0, 0.0]
    kick_angle[plane] = kick
    ring[idx].KickAngle = kick_angle

    try:
        _, orbit_at_monitors = at.find_orbit4(ring, refpts=at.Monitor)
    finally:
        ring[idx].KickAngle = [0.0, 0.0]

    monitor_indices = ring.get_refpts(at.Monitor)
    return {
        ring[el_idx].FamName: (float(orbit_at_monitors[row, 0]), float(orbit_at_monitors[row, 2]))
        for row, el_idx in enumerate(monitor_indices)
    }
