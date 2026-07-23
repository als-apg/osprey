"""Declarative facility specification — the single source of truth for the
``one-facility`` epic's device families, counts, naming scheme, and machine
constants.

Stdlib-only (``dataclasses`` + ``typing``), zero third-party dependencies,
following the import discipline of
:mod:`osprey.services.bluesky_bridge.devices.specs` (plain frozen dataclasses,
no control-system imports) and the catalog-lookup shape of
:mod:`osprey.services.build_artifacts.catalog`.

This module inverts the retired ``manifest -> inventory -> lattice`` direction:
the spec is the authority on families / counts / names, and the hand-ported
ring (:func:`osprey.simulation.lattice.ring.build_ring`) is its *first
consumer*. A drift-guard test binds the ring's actual per-family element counts
and assigned naming to this declaration
(``tests/simulation/test_facility_spec.py``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Family:
    """One device family declared by a :class:`FacilitySpec`.

    Attributes:
        name: Family token as it appears in the ``{fam}{id:02d}`` device
            name (e.g. ``"QF"``, ``"BPM"``, ``"HCM"``).
        count: Number of devices of this family in the whole ring.
        kind: Coarse device role — one of ``"magnet"``, ``"monitor"``, or
            ``"corrector"``.
    """

    name: str
    count: int
    kind: str


@dataclass(frozen=True)
class FacilitySpec:
    """Declarative single source of truth for a facility's device inventory.

    Attributes:
        name: Facility identifier (e.g. ``"ALS-U-AR"``).
        energy_ev: Design beam energy in electron-volts.
        harmonic: RF harmonic number.
        naming: Device naming-scheme template (``str.format`` fields
            ``fam`` / ``id``; flat stack-native convention, ``sup`` is
            accepted by :meth:`device_name` but not rendered).
        families: The declared device families (order-insensitive).
    """

    name: str
    energy_ev: float
    harmonic: int
    naming: str
    families: tuple[Family, ...]

    def family(self, name: str) -> Family:
        """Return the :class:`Family` named ``name``.

        Raises:
            KeyError: If no family with that name is declared.
        """
        for fam in self.families:
            if fam.name == name:
                return fam
        raise KeyError(name)

    def counts(self) -> dict[str, int]:
        """Return a ``{family_name: count}`` mapping over all families."""
        return {fam.name: fam.count for fam in self.families}

    def family_names(self) -> tuple[str, ...]:
        """Return the declared family names, in declaration order."""
        return tuple(fam.name for fam in self.families)

    def device_name(self, sup: str, fam: str, ident: int | str) -> str:
        """Render a device name for family ``fam``, flat stack-native scheme.

        ``sup`` is accepted for API stability (superperiod-scoped callers
        still pass it) but is not part of the rendered name — ids are
        family-scoped across the whole ring, not per-superperiod.

        Args:
            sup: Superperiod token (e.g. ``"01C"``); unused in rendering.
            fam: Family token (e.g. ``"BPM"``).
            ident: Family-scoped device index, zero-padded to ≥2 digits.

        Returns:
            The formatted name, e.g. ``"BPM03"``.
        """
        return self.naming.format(fam=fam, id=int(ident))


# ── The ALS-U Accumulator Ring instance ──────────────────────────────────────
#
# Families and counts declare the *coarse* truth the hand-ported ring must
# satisfy. Magnet families are ported verbatim from ``ALS_U_AR_v6.m``; BPMs are
# promoted to ``at.Monitor`` and named by the port; HCM/VCM correctors are the
# synthetic steerers the source omits (one of each co-located per BPM). See
# PROPOSAL.md FR3/FR4.
# Naming contract (consumed by ring.py and the tier-DB generator): rendered
# names match ``^(HCM|VCM|QF|QD|QFA|DIPOLE|SF|SD|SHF|SHD|BPM)\d{2,}$``; ids are
# zero-padded to >=2 digits and family-scoped, assigned by ascending
# s-position within each family (ring.py owns the s-ordering — this spec only
# declares the rendering template).
ALS_U_AR = FacilitySpec(
    name="ALS-U-AR",
    energy_ev=2.0e9,
    harmonic=304,
    naming="{fam}{id:02d}",
    families=(
        Family("QF", 24, "magnet"),
        Family("QD", 24, "magnet"),
        Family("QFA", 24, "magnet"),
        Family("DIPOLE", 36, "magnet"),
        Family("SF", 24, "magnet"),
        Family("SD", 24, "magnet"),
        Family("SHF", 24, "magnet"),
        Family("SHD", 24, "magnet"),
        Family("BPM", 72, "monitor"),
        Family("HCM", 72, "corrector"),
        Family("VCM", 72, "corrector"),
    ),
)


def als_u_ar() -> FacilitySpec:
    """Return the ALS-U Accumulator Ring facility spec (the epic default)."""
    return ALS_U_AR
