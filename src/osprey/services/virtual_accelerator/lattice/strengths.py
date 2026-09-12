"""Ring-facing current<->strength mapping for the ALS-U AR virtual accelerator.

Owns the per-device nominal-current baseline -- the manifest's pyat-coupled
setpoints, valued from the scenario-seed ``machine.json``, no hardcoded
currents -- and the current->strength formulas
for every magnet and corrector family in the real ring. This module is
deliberately decoupled from the serving layer -- it never imports anything
from :mod:`osprey.services.virtual_accelerator.ioc` or
:mod:`osprey.services.virtual_accelerator.serving` -- so any physics server
can consume it identically.

Scoped to one ring, and it says so. The families below, their formulas and the
calibration constants are the AR ring's physics, not a framework contract. The
channels they apply to are not stated here at all: which addresses carry a
magnet current, and which device each belongs to, come from the manifest's
pyat-coupled partition, so no address spelling -- the bundled demo tree's six
colon-separated tokens included -- is assumed or reconstructed in this module.

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

The element-level write-rollback primitives (:class:`ElementState`,
:func:`snapshot_element`, :func:`restore_element`) are facility-agnostic and
live in :mod:`lume_pyat.simulator`; they are re-exported here so a caller
already holding this module for the mapping does not need a second import
to roll a write back.
"""

from __future__ import annotations

import re

import at
from lume_pyat.simulator import ElementState, restore_element, snapshot_element

from osprey.services.virtual_accelerator.manifest import (
    build_manifest,
    pyat_coupled_setpoint_addresses,
)
from osprey.services.virtual_accelerator.manifest.loaders import (
    load_machine_json_channels,
)

from .calibration import AMPS_PER_RADIAN_KICK

__all__ = [
    "CORRECTOR_FAMILIES",
    "DIPOLE_FAMILY",
    "QUADRUPOLE_FAMILIES",
    "SEXTUPOLE_FAMILIES",
    "ElementState",
    "StrengthMap",
    "restore_element",
    "snapshot_element",
    "split_fam_name",
]

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


def split_fam_name(fam_name: str) -> tuple[str, str]:
    """Split a flat element name (e.g. ``"QF01"``) into ``("QF", "01")``.

    The single grammar for the ``{FAMILY}{DD}`` flat-name convention every
    ring element's ``FamName`` follows (letters then digits); consumers that
    accept only a subset of families (e.g. :mod:`.response`'s correctors)
    layer their own allow-list check on top rather than re-parsing.

    Raises:
        ValueError: if ``fam_name`` doesn't match the letters-then-digits
            grammar.
    """
    match = _FAM_NAME_RE.match(fam_name)
    if match is None:
        raise ValueError(f"cannot parse family/device-id from FamName {fam_name!r}")
    return match.group(1), match.group(2)


class StrengthMap:
    """Current<->strength mapping for every magnet/corrector in the AR ring.

    Baked strengths are snapshotted from ``ring`` at construction time.
    :meth:`apply` may then be called against that same ring, or any other
    ring built the same way (see
    :func:`osprey.services.virtual_accelerator.lattice.build_ring`), to
    write a current and update the matching element in place.
    """

    def __init__(self, ring: at.Lattice, channels: list[dict] | None = None) -> None:
        """Snapshot the baked strengths of ``ring`` and the nominal-current baseline.

        Args:
            ring: The lattice to snapshot baked strengths from.
            channels: the manifest's ``channels`` list; ``None`` builds the
                manifest. A caller that already has one should pass it.
        """
        if channels is None:
            channels = build_manifest()["channels"]
        machine_channels = load_machine_json_channels()
        coupled_setpoints = pyat_coupled_setpoint_addresses(channels)
        self._i_nom_by_address: dict[str, float] = {
            address: float(machine_channels[address]["value"])
            for address in coupled_setpoints
            if address in machine_channels
        }
        self._i_nom_by_device: dict[tuple[str, str], float] = {
            (channel["family"], channel["device"]): self._i_nom_by_address[channel["address"]]
            for channel in channels
            if channel["address"] in self._i_nom_by_address
            and channel["address"] in coupled_setpoints
        }
        self._baked: dict[str, float] = {}
        for element in ring:
            try:
                family, _device_id = split_fam_name(element.FamName)
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

    def i_nom_for(self, family: str, device_id: str) -> float:
        """Return the nominal current of the ``family``+``device_id`` device.

        The device-keyed half of :meth:`i_nom`, for the callers that hold an
        element's family and id rather than a channel address. The pair is
        resolved through the manifest rows the baseline itself is built from,
        so a facility's address spelling is never reconstructed here.

        Raises:
            ValueError: the manifest declares no lattice-backed current
                setpoint for that pair -- a device outside the partition, or
                one the scenario seed carries no nominal current for.
        """
        try:
            return self._i_nom_by_device[(family, device_id)]
        except KeyError:
            raise ValueError(
                f"no nominal current for family {family!r} device {device_id!r}: the "
                "manifest declares no lattice-backed current setpoint for that device"
            ) from None

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

        i_nom = self.i_nom_for(family, device_id)
        fraction = current / i_nom

        if family in QUADRUPOLE_FAMILIES:
            element.K = self._baked[fam_name] * fraction
        elif family == DIPOLE_FAMILY:
            element.PolynomB[0] = (fraction - 1.0) * element.BendingAngle / element.Length
        else:
            element.PolynomB[2] = self._baked[fam_name] * fraction
