"""Building a description in memory, as the package alone builds it.

The framework's build passes the description its own checks through
``extra_checks`` (the scenario-server check, ``FRAMEWORK_CHECKS`` in the root
``tests/_builds.py``). The package suite has no framework, so its builds pass
none: a test that needs the framework's checks is a framework test and stays in
the root suite.

Nothing here imports ``osprey`` or the framework's ``tests`` package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from osprey_facility.build import FacilityDocument, build_facility

#: The ``extra_checks`` a package build passes: none.
NO_EXTRA_CHECKS: tuple[()] = ()


def build(facility_dir: Path, **kw: Any) -> FacilityDocument:
    """Build a facility file in memory with no checks beyond the description's own.

    Args:
        facility_dir: The ``facility`` directory.
        **kw: The keywords of :func:`osprey_facility.build.build_facility`;
            ``extra_checks`` defaults to :data:`NO_EXTRA_CHECKS`.

    Returns:
        The facility file.
    """
    kw.setdefault("extra_checks", NO_EXTRA_CHECKS)
    return build_facility(facility_dir, **kw)
