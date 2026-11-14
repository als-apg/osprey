"""The injection fixture facility: one deckless injection chain, built and rendered.

``tests/facility/fixtures/injection`` holds a facility tree with one model of
the ``injection`` engine and no deck: a request, four parameters with non-zero
start values, two counters, two charges, two efficiencies, the slow and the
fast stored current, and a busy signal, under invented channel names. Every
step offset sits within the first tenth of a second of a shot, short enough for
a live test to fire several shots per global tick.

:func:`injection_view` copies the tree under a test's own directory, builds and
renders it there (a checkout mounted read-only still builds it), and returns the
render's simulator view. The constants name the tree's channels by their part
in the chain, so a test states what it checks without repeating the YAML.
"""

from __future__ import annotations

import shutil
from pathlib import Path

__all__ = [
    "BUSY",
    "CHARGES",
    "COUNTERS",
    "CURRENT",
    "CURRENT_FAST",
    "EFFICIENCIES",
    "FIXTURE",
    "MODEL",
    "OFFSETS",
    "PARAMETERS",
    "READINGS",
    "REQUEST",
    "SETPOINTS",
    "injection_view",
]

#: The fixture's facility tree on disk.
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "injection"

#: The chain's model name.
MODEL = "CHAIN"

#: The request channel; it starts at zero.
REQUEST = "INJ:SHOT:REQ"

#: Each parameter and its start value, as the tree states it.
PARAMETERS: dict[str, float] = {
    "INJ:GUN:CHARGE:SP": 1.5,
    "INJ:TRANSFER:EFF:SP": 0.8,
    "INJ:RING:EFF:SP": 0.6,
    "INJ:RING:CURRENT:SP": 20.0,
}

#: The counters a shot steps.
COUNTERS = ("INJ:SHOT:COUNT", "INJ:TRANSFER:COUNT")

#: The per-stage charge readings, stage 0 first.
CHARGES = ("INJ:TRANSFER:CHARGE", "INJ:RING:CHARGE")

#: The per-stage efficiency readings, stage 0 first.
EFFICIENCIES = ("INJ:TRANSFER:EFF", "INJ:RING:EFF")

#: The stored current a shot steps.
CURRENT = "INJ:RING:CURRENT"

#: The stored current read again on every fast pass.
CURRENT_FAST = "INJ:RING:CURRENT:FAST"

#: The busy signal.
BUSY = "INJ:SHOT:BUSY"

#: Each step's offset from its shot's request, in seconds.
OFFSETS: dict[str, float] = {
    "INJ:SHOT:COUNT": 0.01,
    "INJ:TRANSFER:COUNT": 0.02,
    "INJ:TRANSFER:CHARGE": 0.03,
    "INJ:TRANSFER:EFF": 0.04,
    "INJ:RING:CHARGE": 0.05,
    "INJ:RING:EFF": 0.06,
    "INJ:RING:CURRENT": 0.08,
}

#: Every setpoint of the chain: the request, then the parameters.
SETPOINTS = (REQUEST, *PARAMETERS)

#: Every reading of the chain.
READINGS = (*COUNTERS, *CHARGES, *EFFICIENCIES, CURRENT, CURRENT_FAST, BUSY)

#: The views every render of this process has named as not written; each is
#: named once.
_REPORTED: set[str] = set()


def injection_view(root: Path) -> Path:
    """Build and render the injection fixture under ``root``.

    The tree is copied to ``root/data/facility``, built, and rendered into
    ``root/build`` under the in-process render configuration of
    :mod:`tests.facility.served_tree`.

    Args:
        root: A directory holding neither ``data/facility`` nor ``build``.

    Returns:
        The render's simulator view, ``root/build/data/simulator``.
    """
    from osprey.facility.build import build_facility
    from osprey.facility.render import render_facility_outputs
    from tests.facility.served_tree import IN_PROCESS_RENDER_CONFIG

    facility_dir = root / "data" / "facility"
    shutil.copytree(FIXTURE, facility_dir)
    render_dir = root / "build"
    render_dir.mkdir(parents=True)
    document = build_facility(facility_dir, project_name="injection")
    render_facility_outputs(
        render_dir, document, IN_PROCESS_RENDER_CONFIG, facility_dir, omitted_reported=_REPORTED
    )
    return render_dir / "data" / "simulator"
