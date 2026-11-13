"""The turn-by-turn fixture facility: an injection chain feeding the example ring on pyAT.

``tests/facility/fixtures/turn_by_turn`` holds a facility tree with two
models. ``CHAIN`` is the injection fixture's chain with two changes: the
stored-current parameter starts at zero and its band reaches zero, and a ring
shot counter steps after the ring charge. ``RING`` is the example facility's
storage ring on pyAT, its deck copied beside the tree, made a turn-by-turn
ring by its ``turn_by_turn`` settings block. It carries the RF switch, the
pilot tone, two ``stored`` records, every monitor's arm, armed and three turn
arrays, the turns tracked, the orbit monitors and the correctors, and takes
the chain's ring shot counter, ring charge and stored current as its inputs.

The block's aperture and bunch sizes are calibrated with the tracking core:
with the RF off the radiation loss drives the bunch into the aperture between
turns 110 and 210, and with it on the bunch survives every turn.

:func:`turn_by_turn_view` copies the tree under a test's own directory, builds
and renders it there, and returns the render's simulator view. The constants
name the tree's channels by their part, so a test states what it checks
without repeating the YAML.
"""

from __future__ import annotations

import shutil
from pathlib import Path

__all__ = [
    "ARMED",
    "ARMS",
    "CHAIN",
    "CHARGE",
    "CURRENT",
    "CURRENT_SETPOINT",
    "DECK",
    "FIXTURE",
    "INPUTS",
    "MONITORS",
    "PILOT_TONE",
    "REQUEST",
    "RF_SWITCH",
    "RING",
    "SAMPLES",
    "SHOT",
    "STORED",
    "TURNS",
    "TURNS_TRACKED",
    "TURN_SUM",
    "TURN_X",
    "TURN_Y",
    "turn_by_turn_view",
]

#: The fixture's facility tree on disk.
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "turn_by_turn"

#: The ring's lattice file inside the tree.
DECK = FIXTURE / "decks" / "SR.json"

#: The injection chain's model name.
CHAIN = "CHAIN"

#: The ring's model name.
RING = "RING"

#: The chain's request channel.
REQUEST = "INJ:SHOT:REQ"

#: The chain's stored-current parameter; it starts at zero.
CURRENT_SETPOINT = "INJ:RING:CURRENT:SP"

#: The ring shot counter; it steps after :data:`CHARGE`.
SHOT = "INJ:RING:SHOT:COUNT"

#: The charge a shot brings to the ring.
CHARGE = "INJ:RING:CHARGE"

#: The stored current a shot steps.
CURRENT = "INJ:RING:CURRENT"

#: The ring's inputs: each input signal and the chain record it names.
INPUTS: dict[str, str] = {"shot": SHOT, "charge": CHARGE, "stored_current": CURRENT}

#: The turns one shot tracks, and the samples of one turn array.
TURNS = 250
SAMPLES = 10 + TURNS

#: The ring's monitor elements, in deck order.
MONITORS = tuple(f"BPM{n:02d}" for n in range(1, 73))


def _per_monitor(suffix: str) -> dict[str, str]:
    return {element: f"SR:DIAG:BPM:{element[3:]}:{suffix}" for element in MONITORS}


#: Each monitor's arm setpoint.
ARMS = _per_monitor("ARM")
#: Each monitor's armed reading.
ARMED = _per_monitor("ARMED")
#: Each monitor's turn arrays.
TURN_X = _per_monitor("TBT:X")
TURN_Y = _per_monitor("TBT:Y")
TURN_SUM = _per_monitor("TBT:SUM")

#: The ring's RF switch, pilot tone and turns tracked.
RF_SWITCH = "SR:RF:SWITCH"
PILOT_TONE = "SR:DIAG:TBT:PILOT"
TURNS_TRACKED = "SR:DIAG:TBT:TURNS"

#: The two records that hold what is written.
STORED = ("SR:DIAG:TBT:STORED:01", "SR:DIAG:TBT:STORED:02")

#: The views every render of this process has named as not written; each is
#: named once.
_REPORTED: set[str] = set()


def turn_by_turn_view(root: Path) -> Path:
    """Build and render the turn-by-turn fixture under ``root``.

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
    document = build_facility(facility_dir, project_name="turn-by-turn")
    render_facility_outputs(
        render_dir, document, IN_PROCESS_RENDER_CONFIG, facility_dir, omitted_reported=_REPORTED
    )
    return render_dir / "data" / "simulator"
