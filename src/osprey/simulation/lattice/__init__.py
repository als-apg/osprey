"""Shared, plain-venv-importable ALS-U Accumulator Ring lattice.

The hand-ported real ALS-U AR ring (:func:`build_ring`), the declarative
:class:`~osprey.simulation.facility_spec.FacilitySpec` it consumes, and the
canonical ``.mat`` artifact helpers. This subpackage imports only ``at`` +
``numpy`` (no ``softioc`` / EPICS stack), so any consumer — dashboard, channels,
logbook, the future virtual-accelerator repoint — can import the ring without
the ``osprey-framework`` control-system extras.
"""

from osprey.simulation.facility_spec import ALS_U_AR, FacilitySpec
from osprey.simulation.lattice.artifact import (
    canonical_mat_path,
    load_canonical_ring,
    save_canonical_mat,
)
from osprey.simulation.lattice.ring import build_ring, superperiod

__all__ = [
    "ALS_U_AR",
    "FacilitySpec",
    "build_ring",
    "canonical_mat_path",
    "load_canonical_ring",
    "save_canonical_mat",
    "superperiod",
]
