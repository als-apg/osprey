"""The in-memory facility build: every stage, then the combined file.

``build_facility`` runs S1 to S5 and the stages after them, in order::

    wiring    the computed slots of every wiring record    (wiring.py)
    compute   positions, places, ordinals, deck checks     (compute.py)

and returns the facility file, or raises the first error of the first stage
that fails. Nothing is written to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

from osprey.facility.compute import check_compute
from osprey.facility.validate import Stage, run_stages
from osprey.facility.wiring import fill_wiring_slots

__all__ = ["LATER_STAGES", "FacilityDocument", "build_facility"]

#: The combined facility file, the shape ``build/facility.json`` holds.
FacilityDocument: TypeAlias = dict[str, Any]

#: The stages after S5, in the order they run.
LATER_STAGES: tuple[tuple[str, Stage], ...] = (
    ("wiring", fill_wiring_slots),
    ("compute", check_compute),
)


def build_facility(facility_dir: Path, *, project_name: str) -> FacilityDocument:
    """Build the facility file in memory.

    Args:
        facility_dir: The ``data/facility`` directory itself; a missing one
            is zero sources.
        project_name: The project's name, folded into the identity ``code``
            when there is no ``identity.yaml``.

    Returns:
        The facility file.

    Raises:
        FacilityBuildError: The first error of the first stage that fails.
    """
    report = run_stages(facility_dir, project_name=project_name, later=LATER_STAGES)
    report.raise_first()
    document = report.validated.document
    if document is None:
        raise RuntimeError("the stages ran clean without producing the document")
    return document
