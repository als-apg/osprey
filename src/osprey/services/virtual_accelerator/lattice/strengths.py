"""Ring-facing current<->strength mapping for the ALS-U AR virtual accelerator.

Owns the per-device nominal-current baseline (read from the scenario-seed
``machine.json``, no hardcoded currents) and the current->strength formulas
for every magnet and corrector family in the real ring. This module is
deliberately decoupled from the softioc -- it never imports anything from
:mod:`osprey.services.virtual_accelerator.ioc` -- so a future LUME-based
physics server can consume it identically.

Apply-current semantics (per family):

- ``HCM``/``VCM`` correctors: ``KickAngle[plane] = I / AMPS_PER_RADIAN_KICK``
  (absolute). Each corrector is its own single-plane element in the real
  ring (see :mod:`osprey.services.virtual_accelerator.lattice.response`):
  an ``HCM`` element writes ``KickAngle[0]``, a ``VCM`` element writes
  ``KickAngle[1]``.
- ``QF``/``QD``/``QFA`` quadrupoles: ``K = K_baked * I / I_nom`` (the baked
  gradient scaled by the fractional current).
- ``DIPOLE``: ``PolynomB[0] = (I / I_nom - 1) * BendingAngle / Length``.
  This is a trim-coil model: the real AR dipoles are combined-function
  bends (nonzero baked ``PolynomB[1]`` gradient), and this only perturbs the
  pure-dipole field-error term (``PolynomB[0]``) -- the baked bending angle
  and gradient are held fixed. At nominal current the field error is
  exactly zero.
- ``SF``/``SD``/``SHF``/``SHD`` sextupoles: ``PolynomB[2] = h_baked * I /
  I_nom`` (the baked sextupole strength scaled by the fractional current).

Baked strengths are snapshotted once, from the ring passed to
:class:`StrengthMap`'s constructor, before any current is applied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import at

from osprey.services.virtual_accelerator.manifest.loaders import (
    load_machine_json_channels,
)

from .response import AMPS_PER_RADIAN_KICK

# Families dispatched by formula (see module docstring). Every magnet/
# corrector family the facility spec declares must appear in exactly one of
# these; BPMs (monitors) never appear here.
QUADRUPOLE_FAMILIES = frozenset({"QF", "QD", "QFA"})
SEXTUPOLE_FAMILIES = frozenset({"SF", "SD", "SHF", "SHD"})
DIPOLE_FAMILY = "DIPOLE"
CORRECTOR_FAMILIES = frozenset({"HCM", "VCM"})

# KickAngle index each corrector family writes (see module docstring).
_CORRECTOR_PLANE = {"HCM": 0, "VCM": 1}

_FAM_NAME_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def _split_fam_name(fam_name: str) -> tuple[str, str]:
    """Split a flat element name (e.g. ``"QF01"``) into ``("QF", "01")``."""
    match = _FAM_NAME_RE.match(fam_name)
    if match is None:
        raise ValueError(f"cannot parse family/device-id from FamName {fam_name!r}")
    return match.group(1), match.group(2)


def current_address(family: str, device_id: str) -> str:
    """Return the SR machine.json ``CURRENT:SP`` address for a device.

    e.g. ``current_address("QF", "01") == "SR:MAG:QF:01:CURRENT:SP"``. The
    ``SR:MAG:`` prefix is fixed -- machine.json is a namespace shared with
    other rings (``BR``, ``BTS``), and this map only ever addresses the
    storage ring this service simulates.
    """
    return f"SR:MAG:{family}:{device_id}:CURRENT:SP"


@dataclass(frozen=True)
class ElementState:
    """Snapshot of one ring element's mutable strength fields.

    ``polynom_b`` and ``kick_angle`` are ``None`` when the element doesn't
    carry that field (correctors have no ``PolynomB``; magnets have no
    ``KickAngle``).
    """

    polynom_b: list[float] | None
    kick_angle: list[float] | None


def snapshot_element(element: at.Element) -> ElementState:
    """Capture ``element``'s current ``PolynomB``/``KickAngle`` state.

    For later write-rollback: pair with :func:`restore_element`.
    """
    return ElementState(
        polynom_b=list(element.PolynomB) if hasattr(element, "PolynomB") else None,
        kick_angle=list(element.KickAngle) if hasattr(element, "KickAngle") else None,
    )


def restore_element(element: at.Element, state: ElementState) -> None:
    """Write a previously captured :class:`ElementState` back onto ``element``."""
    if state.polynom_b is not None:
        element.PolynomB = list(state.polynom_b)
    if state.kick_angle is not None:
        element.KickAngle = list(state.kick_angle)


class StrengthMap:
    """Current<->strength mapping for every magnet/corrector in the AR ring.

    Baked strengths are snapshotted from ``ring`` at construction time.
    :meth:`apply` may then be called against that same ring, or any other
    ring built the same way (see
    :func:`osprey.services.virtual_accelerator.lattice.build_ring`), to
    write a current and update the matching element in place.
    """

    def __init__(self, ring: at.Lattice) -> None:
        self._i_nom_by_address: dict[str, float] = {
            address: float(entry["value"])
            for address, entry in load_machine_json_channels().items()
            if address.endswith(":CURRENT:SP")
        }
        self._baked: dict[str, float] = {}
        for element in ring:
            try:
                family, _device_id = _split_fam_name(element.FamName)
            except ValueError:
                continue
            if family in QUADRUPOLE_FAMILIES:
                self._baked[element.FamName] = float(element.K)
            elif family == DIPOLE_FAMILY:
                self._baked[element.FamName] = float(element.PolynomB[0])
            elif family in SEXTUPOLE_FAMILIES:
                self._baked[element.FamName] = float(element.PolynomB[2])
            elif family in CORRECTOR_FAMILIES:
                plane = _CORRECTOR_PLANE[family]
                self._baked[element.FamName] = float(element.KickAngle[plane])

    def i_nom(self, address: str) -> float:
        """Return the machine.json nominal (baseline) current for ``address``."""
        return self._i_nom_by_address[address]

    def baked(self, fam_name: str) -> float:
        """Return the strength snapshotted at construction for ``fam_name``.

        The relevant strength is family-dependent: ``K`` for quadrupoles,
        ``PolynomB[0]`` for the dipole field error, ``PolynomB[2]`` for
        sextupoles, and the baked ``KickAngle`` component for correctors
        (zero for a freshly built ring -- correctors carry no baked kick).
        """
        return self._baked[fam_name]

    def apply(self, ring: at.Lattice, family: str, device_id: str, current: float) -> None:
        """Write ``current`` (Amps) onto the ``family``+``device_id`` element of ``ring``.

        Args:
            ring: The lattice to mutate (the target element is matched by
                ``FamName``; need not be the ring baked strengths were
                snapshotted from, as long as it was built the same way).
            family: Family token, e.g. ``"QF"``, ``"DIPOLE"``, ``"HCM"``.
            device_id: Zero-padded family-scoped device id, e.g. ``"01"``.
            current: Current to apply, in Amps.

        Raises:
            ValueError: if ``family`` isn't a recognized magnet/corrector
                family, or no matching element exists in ``ring``.
        """
        is_magnet = (
            family in QUADRUPOLE_FAMILIES or family == DIPOLE_FAMILY or family in SEXTUPOLE_FAMILIES
        )
        if not is_magnet and family not in CORRECTOR_FAMILIES:
            raise ValueError(f"unrecognized family {family!r}")

        fam_name = f"{family}{device_id}"
        element = next((el for el in ring if el.FamName == fam_name), None)
        if element is None:
            raise ValueError(f"no ring element named {fam_name!r}")

        if family in CORRECTOR_FAMILIES:
            plane = _CORRECTOR_PLANE[family]
            kick_angle = list(element.KickAngle)
            kick_angle[plane] = current / AMPS_PER_RADIAN_KICK
            element.KickAngle = kick_angle
            return

        i_nom = self._i_nom_by_address[current_address(family, device_id)]
        fraction = current / i_nom

        if family in QUADRUPOLE_FAMILIES:
            element.K = self._baked[fam_name] * fraction
        elif family == DIPOLE_FAMILY:
            element.PolynomB[0] = (fraction - 1.0) * element.BendingAngle / element.Length
        else:
            element.PolynomB[2] = self._baked[fam_name] * fraction
