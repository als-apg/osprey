"""Thin adapter over the real ALS-U AR ring for the virtual accelerator.

`build_ring()` delegates to :func:`osprey.simulation.lattice.build_ring` --
the hand-ported ALS-U Accumulator Ring, the source of truth for this
service's physics -- and disables radiation/cavity effects so downstream
closed-orbit and linear-optics calls (`at.find_orbit4`, `orbit_response`) see
a stable 4D ring. Energy comes from the real ring itself, which is built at
`ALS_U_AR.energy_ev`.

Before returning, the manifest's pyat-coupled device inventory (per-family
device counts derived from the namespace-union channel DBs, see
:mod:`inventory`) is validated against :data:`ALS_U_AR`'s declared family
counts -- the facility spec fixes the expected inventory, and any drift
between the manifest and the spec is a configuration error, not a lattice
concern.

No caching here: each consumer builds its own ring (~5 ms), so IOC-driven
strength/current updates are always reflected in the physics model.
"""

from __future__ import annotations

import at

from osprey.simulation.facility_spec import ALS_U_AR
from osprey.simulation.lattice import build_ring as _build_real_ring

from . import inventory


def _validate_inventory() -> None:
    """Validate the manifest's pyat-coupled device counts against ALS_U_AR.

    Raises:
        ValueError: naming the family, expected count, and actual count for
            the first family whose manifest-derived device count doesn't
            match the ALS_U_AR facility spec.
    """
    inv = inventory.pyat_coupled_device_ids()
    for family in ALS_U_AR.family_names():
        expected = ALS_U_AR.family(family).count
        actual = len(inv.get(family, []))
        if actual != expected:
            raise ValueError(
                f"manifest pyat-coupled partition has {actual} '{family}' devices, "
                f"expected {expected} per the ALS_U_AR facility spec"
            )


def build_ring() -> at.Lattice:
    """Build the ALS-U AR ring for the virtual accelerator.

    Returns:
        An `at.Lattice` with radiation and cavity effects disabled (4D
        closed-orbit / linear-optics ring).

    Raises:
        ValueError: if the manifest's pyat-coupled partition's per-family
            device counts don't match the ALS_U_AR facility spec.
    """
    _validate_inventory()
    ring = _build_real_ring()
    ring.disable_6d()
    return ring
