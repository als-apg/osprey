"""The panels fixture: an invented two-sector ring and its rule tables.

The ``facility`` tree is a description, so it lives in the facility
package suite (``packages/osprey-facility/tests/osprey_facility_tests/panel_fixture/``);
``fixture/facility.json`` is that tree built, checked in, so the panel tests read a document without running the build (the
package suite's ``rebuild_panel_fixture.py`` regenerates it, and a package test
holds it to the build byte for byte); ``fixture/panels/`` holds the rule tables
beside it. Every name, number
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

import functools
import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey_panels_tests._harness import FIXTURE_DIR

if TYPE_CHECKING:
    from osprey_panels.tables import PanelTables

PANELS_DIR = FIXTURE_DIR / "panels"
#: The fixture tree built into its facility file, checked in.
FIXTURE_DOCUMENT = FIXTURE_DIR / "facility.json"
RING_FRAME = "XR-RING"


def facility_bytes(document: Any) -> bytes:
    """*document* as the facility build writes ``facility.json``: sorted keys,
    two-space indent, non-ASCII as itself, NaN refused, one final newline, UTF-8."""
    text = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def load_fixture_document() -> dict[str, Any]:
    """The checked-in fixture document, read fresh: the caller may change it."""
    return json.loads(FIXTURE_DOCUMENT.read_text(encoding="utf-8"))


@functools.cache
def fixture_relation_types() -> tuple[Mapping[str, Any], ...]:
    """The ``relation_types`` rows of the checked-in fixture document."""
    return tuple(load_fixture_document()["relation_types"])


def read_fixture_tables(directory: Path) -> PanelTables:
    """Read the rule tables in *directory* against the fixture's relation types."""
    from osprey_panels.tables import read_tables

    return read_tables(directory, relation_types=fixture_relation_types())


def write_fixture_facility(
    target: Path, change: Callable[[dict[str, Any]], None] | None = None
) -> Path:
    """Write the checked-in fixture document at *target*.

    Args:
        target: The file to write.
        change: Called with the document before it is written, for a
            test that needs a facility the build would not produce.

    Returns:
        *target*.
    """
    document = load_fixture_document()
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
    from osprey_panels import load_graph

    def change(document: dict[str, Any]) -> None:
        add_part(document, "XR01:CARD1", "XR01:CH1", "XR01-CARD1:Fault-Sts")
        for extra in changes:
            extra(document)

    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as rows:
        rows.write(PART_RULE)
    return load_graph(write_fixture_facility(tmp_path / "facility.json", change), tables)


def graph_with_wide(tmp_path: Path, channels: int):
    """The fixture graph with ``XR01:WIDE1`` in ``XR/XR01``: *channels* rule channels.

    Each ``XR01-WIDE1:Sig<i>-Sts`` has ``0`` normal (from two sources), ``1``
    bad, ``2`` warn and ``>=5`` bad.
    """
    from osprey_panels import load_graph

    def wide(document: dict[str, Any]) -> None:
        document["devices"].append(
            {"id": "XR01:WIDE1", "class": "Infrastructure", "place": "XR/XR01"}
        )
        for i in range(channels):
            document["channels"].append(
                {"id": f"XR01-WIDE1:Sig{i}-Sts", "on": {"device": "XR01:WIDE1"}}
            )

    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as rows:
        for i in range(channels):
            name = f"XR01-WIDE1:Sig{i}-Sts"
            for value, state in (("0", "normal"), ("1", "bad"), ("2", "warn"), (">=5", "bad")):
                rows.write(
                    f"{name},{value},{state},database-severity,stated,xr-ioc,0a1b2c3,w.db:{i}\n"
                )
            rows.write(f"{name},0,normal,screen,stated,xr-screens,4d5e6f7,w.screen:{i}\n")
    return load_graph(write_fixture_facility(tmp_path / "facility.json", wide), tables)


#: The signals :func:`add_bpm` gives a pickup, each on a channel ``<id>:<signal>``.
BPM_SIGNALS = (
    "position_x_readback",
    "position_y_readback",
    "amplitude_a",
    "amplitude_b",
    "amplitude_c",
    "amplitude_d",
    "tbt_x",
    "tbt_y",
)


def add_bpm(document: dict[str, Any], device: str, place: str) -> None:
    """Add the pickup *device* (class ``BPM``) in *place*, with a channel per
    :data:`BPM_SIGNALS`, ``read_out_by`` the chassis ``<device>-E`` at the port
    ``<device>-E:P1``, a device of its own. The chassis states ``sum_readback``
    and ``amplitude_a``; the port device states ``amplitude_a`` to ``amplitude_d``.
    So ``amplitude_a`` is stated at all three: pickup, port and chassis."""
    electronics = f"{device}-E"
    port = f"{electronics}:P1"
    document["devices"] += [
        {"id": device, "class": "BPM", "place": place},
        {"id": electronics, "class": "BPMElectronics", "place": place},
        {"id": port, "class": "BPMChannel", "place": place},
    ]
    for signal in ("amplitude_a", "amplitude_b", "amplitude_c", "amplitude_d"):
        document["channels"].append(
            {"id": f"{port}:{signal}", "signal": signal, "on": {"device": port}}
        )
    document["channels"].append(
        {"id": f"{electronics}:amplitude_a", "signal": "amplitude_a", "on": {"device": electronics}}
    )
    for signal in BPM_SIGNALS:
        record: dict[str, Any] = {"id": f"{device}:{signal}", "signal": signal}
        record["on"] = {"device": device}
        if signal.startswith("position"):
            record["unit"] = "mm"
        document["channels"].append(record)
    document["channels"].append(
        {
            "id": f"{electronics}:sum_readback",
            "signal": "sum_readback",
            "on": {"device": electronics},
        }
    )
    document["relations"].append(
        {
            "id": f"{device.lower()}-read-out-by",
            "type": "read_out_by",
            "subject": {"device": device},
            "object": {"device": electronics, "port": port},
            "standing": "accepted",
        }
    )


def graph_with_bpms(tmp_path: Path, tables: Path | None = None):
    """The fixture graph with the pickups ``XR01:BPM1`` in ``XR/XR01`` and ``XR02:BPM1``
    in ``XR/XR02`` (:func:`add_bpm`), read with *tables* (the fixture's by default)."""
    from osprey_panels import load_graph

    def change(document: dict[str, Any]) -> None:
        add_bpm(document, "XR01:BPM1", "XR/XR01")
        add_bpm(document, "XR02:BPM1", "XR/XR02")

    facility = write_fixture_facility(tmp_path / "bpm.json", change)
    return load_graph(facility, tables or PANELS_DIR)
