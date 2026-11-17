"""The panels fixture: an invented two-sector ring and its rule tables.

``fixture/facility/`` is a ``data/facility`` tree built by the framework's own
build; ``fixture/panels/`` holds the rule tables beside it. Every name, number
and document in both is made up for the tests.

The facility: places ``XR`` (machine), sectors ``XR/XR01`` and ``XR/XR02``,
rack ``XR/XR01/RK1``; six magnets (``XR0n:QF1``, ``XR0n:QD1``,
``XR0n:BEND1``), four pumps (``XR0n:IP1``, ``XR0n:IP2``), three chassis
(``XR01:CH1..3``) in the rack, the supply ``XR01:PS1`` feeding ``XR01:QF1``
and ``XR01:QD1``, the excluded thing ``XR02:SAF1`` (with a channel, a group
membership, a relation and a pose), the excluded place ``XR/XR02/SAFE``,
which hides ``XR/XR02/SAFE/CAB1`` and ``XR02:SAF2`` in it (each with a
channel, a relation and a pose), ``XR02:SHIELD1`` with a pose and no
channel, and ``XR01:QF1`` and ``XR01:IP1`` at the same ``s_m``, and ``XR02:IP1`` with a
ring-frame row and no ``s_m``. The channel
``XR01-IP1:State-Sts`` has health rows from two sources that disagree on value
``0``; ``XR02-IP2:Interlock-Sts`` is an excluded channel on a kept thing;
``VA:XR:InjectionKnob`` is a simulation-only channel; ``XR01-QF1:Current-SP``
states a tolerance. The rack ``XR/XR01/RK1`` names units for ``XR01:CH1``
(10, size 2) and ``XR01:CH2`` (20), none for ``XR01:CH3``; ``XR02:BLOCK1`` has
a citation and nothing else. ``XR02:CH5`` has unit claims into two racks
(``XR/XR01/RK1`` unit 40, ``XR/XR02/RK2`` unit 5), which the build marks
contested, so its built place is ``None``. A second claim from another document puts ``XR01:CH2``
at unit 20 too, and a third claim puts ``XR02:CH5`` at unit 41 in ``XR/XR01/RK1``.
``XR09:LOOSE1`` is in no place (no built place, no claim) and has a rule-bearing
channel, ``XR09-LOOSE1:Fault-Sts`` (``1`` is bad).
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

FIXTURE_DIR = Path(__file__).parent / "fixture"
FACILITY_TREE = FIXTURE_DIR / "facility"
PANELS_DIR = FIXTURE_DIR / "panels"
RING_FRAME = "XR-RING"


def build_fixture_document() -> dict[str, Any]:
    """The fixture tree built into its facility document."""
    from osprey.facility.build import build_facility

    return build_facility(FACILITY_TREE, project_name="xr")


def write_fixture_facility(
    target: Path, change: Callable[[dict[str, Any]], None] | None = None
) -> Path:
    """Build the fixture and write its facility file at *target*.

    Args:
        target: The file to write.
        change: Called with the built document before it is written, for a
            test that needs a facility the build would not produce.

    Returns:
        *target*.
    """
    from osprey.facility.render import facility_bytes

    document = build_fixture_document()
    if change is not None:
        change(document)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(facility_bytes(document))
    return target


def copy_tables(target: Path) -> Path:
    """Copy the fixture's rule tables into *target*, for a test that edits them."""
    shutil.copytree(PANELS_DIR, target)
    return target


#: The rule row :func:`graph_with_part` adds: ``1`` on the part's channel is bad.
PART_RULE = "XR01-CARD1:Fault-Sts,1,bad,database-severity,stated,xr-ioc,0a1b2c3,card.db:1\n"


def add_part(document: dict[str, Any], part: str, whole: str, channel: str | None = None) -> None:
    """Add the device *part*, with no place, ``part_of`` the device *whole*."""
    document["devices"].append({"id": part, "class": "ElectronicsModule"})
    document["relations"].append(
        {
            "id": f"{part.lower()}-in-{whole.lower()}",
            "type": "part_of",
            "subject": {"device": part},
            "object": {"device": whole},
            "standing": "accepted",
        }
    )
    if channel is not None:
        document["channels"].append({"id": channel, "on": {"device": part}})


def graph_with_part(tmp_path: Path, *changes: Callable[[dict[str, Any]], None]):
    """The fixture graph with ``XR01:CARD1``, no place, ``part_of`` the racked ``XR01:CH1``.

    The card has the rule channel ``XR01-CARD1:Fault-Sts`` (``1`` is bad);
    *changes* run on the built document after the card is added.
    """
    from osprey.facility.panels import load_graph

    def change(document: dict[str, Any]) -> None:
        add_part(document, "XR01:CARD1", "XR01:CH1", "XR01-CARD1:Fault-Sts")
        for extra in changes:
            extra(document)

    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as rows:
        rows.write(PART_RULE)
    return load_graph(write_fixture_facility(tmp_path / "facility.json", change), tables)
