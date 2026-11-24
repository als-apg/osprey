"""The Bluesky devices view: the facility's channels as the worker's device file.

Written to ``<render>/data/bluesky_devices.yml``::

    schema: osprey.facility.bluesky_devices/1
    # generated-file header
    settables:
      - {name: <setpoint>, setpoint: <setpoint>, readback: <pair>, settle_tolerance: <x>,
         unit: <pair's unit>, setpoint_unit: <setpoint's unit>}
      - {name: <setpoint>, setpoint: <setpoint>, readback: <pair>, motion_band: <band>}
    triggerables:
      - {name: <command>, command: <command>, status: <pair>, idle: <idle>, timeout_s: <bound>}
    readables:
      - {name: <readback>, pv: <readback>, unit: <unit>}
      - {name: <waveform>, pv: <waveform>, unit: <unit>, shape: [<n>]}

One settable per setpoint channel, whose ``readback`` is the channel's ``pair``
when the pair is another channel and is absent otherwise; one readable per
readback channel, paired or not; a channel whose role is ``none`` is no device.
A command whose status states ``idle`` and which states ``timeout_s`` is one
triggerable, a plan fires it and waits for its act, and it is no settable; its
status stays a readable.
A settable carries ``settle_tolerance`` when its setpoint declares a
``tolerance``, a further address of a wiring record (under ``also``) its
record's: the number of an absolute one, ``{relative: <f>}`` of a relative
one. A settable without one carries ``motion_band`` when the channel it reads
back declares motion in its ``simulation`` seed: the envelope
:func:`osprey_connectors.simulation.envelope.motion_envelope` derives from that
seed, which only a simulated lane settles within. A settable with neither
settles within the profile's floor.

A device carries ``unit`` when the channel it reports under its name states a
non-empty ``unit``: a paired settable's readback, an unpaired settable's
setpoint, a readable's channel. A paired settable whose setpoint states one
also carries ``setpoint_unit``. A device whose channel states no unit carries
no unit key.

A readable on a waveform channel of one dimension carries ``shape: [<n>]``,
the channel's ``shape``, and each run records it as an array of that length.
A readback of more than one dimension is no device: writing the view warns,
once per build, naming each one.

Writing the view warns, once per build, of each scenario that moves a
toleranced readback beyond its tolerance, and, per lane that does not serve
the simulator, of the setpoints that declare no ``tolerance`` and so settle
within the lane's floor; a render whose inputs turn the tolerance warnings off
writes the same file and prints none.
A device's name is its address. Entries follow the facility file's channel
order. A render carries the view when it runs a Bluesky lane, under every
control system: the compose generator stages it for a control system that
talks to the network and leaves it unstaged for the simulator served in
process, which speaks no Channel Access.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey.facility.views import ViewInputs
from osprey_facility import FACILITY_FILE
from osprey_facility.records import FacilityRecords

__all__ = [
    "BLUESKY_DEVICES_FILE",
    "BLUESKY_DEVICES_SCHEMA",
    "bluesky_configured",
    "bluesky_document",
    "write_bluesky_view",
]

BLUESKY_DEVICES_FILE = "bluesky_devices.yml"
BLUESKY_DEVICES_SCHEMA = "osprey.facility.bluesky_devices/1"


def _source(facility_file: Path) -> Any:
    from osprey.channel_roster import RosterSource, RosterSourceKind

    return RosterSource(
        kind=RosterSourceKind.FACILITY, path=facility_file, spelled=facility_file.name
    )


def _records(records: FacilityRecords, facility_file: Path) -> list[Any]:
    from osprey.channel_roster import channel_record

    source = _source(facility_file)
    return [channel_record(channel, source, records) for channel in records.of_kind("channel")]


def _tolerances(records: FacilityRecords) -> dict[str, Mapping[str, float]]:
    """Each setpoint's settle ``tolerance`` record, by address; a further address of a wiring record takes its record's."""
    tolerances: dict[str, Mapping[str, float]] = {}
    for channel in records.of_kind("channel"):
        held = records.channel(channel["id"]) or channel
        if held.get("tolerance"):
            tolerances[channel["id"]] = held["tolerance"]
    return tolerances


def _motion_bands(records: FacilityRecords) -> dict[str, float]:
    """Each channel's motion envelope, by address."""
    from osprey_connectors.simulation.envelope import motion_envelope

    return {
        channel["id"]: motion_envelope(channel.get("simulation"))
        for channel in records.of_kind("channel")
    }


def _units(records: FacilityRecords) -> dict[str, str]:
    """Each channel's declared ``unit``, by address; a further address of a wiring record takes its record's."""
    units: dict[str, str] = {}
    for channel in records.of_kind("channel"):
        unit = (records.channel(channel["id"]) or channel).get("unit")
        if isinstance(unit, str) and unit:
            units[channel["id"]] = unit
    return units


def _completions(records: list[Any], facility_file: Path) -> dict[str, Any]:
    """Each command's completion signal, by address, as the roster states it."""
    from osprey.channel_roster import RosterResult

    roster = RosterResult(records=tuple(records), source=_source(facility_file))
    completions = {}
    for record in roster.write_records:
        completion = roster.completion(record.address)
        if completion is not None:
            completions[record.address] = completion
    return completions


def bluesky_document(records: FacilityRecords) -> dict[str, list[dict[str, Any]]]:
    """The device document of one facility file.

    Args:
        records: The facility file's records.

    Returns:
        ``{settables, triggerables, readables}`` as the worker's device file
        holds them.
    """
    from osprey.services.bluesky_bridge.substrate_devices import devices_document

    roster = _records(records, Path(FACILITY_FILE))
    return devices_document(
        roster,
        _tolerances(records),
        _motion_bands(records),
        _units(records),
        _completions(roster, Path(FACILITY_FILE)),
    )


def _lanes(inputs: ViewInputs) -> list[tuple[str, bool, float]]:
    """Each rendered Bluesky lane as ``(service key, serves the simulator, floor)``.

    Lane one serves the simulator when the rendered ``control_system`` selects
    it; the VA lane always does, the live and stand-in lanes never. A lane's
    floor is its rendered ``settle_tolerance``, else the worker's default.
    """
    from osprey.bluesky_bridge_connection import LANE_KEYS, LANE_ONE, SECOND_LANE_KEYS
    from osprey.services.bluesky_bridge.devices.connector import DEFAULT_SETTLE_TOLERANCE
    from osprey_connectors.types import is_simulated, resolve_control_system_type

    services = inputs.rendered_config.get("services")
    lanes: list[tuple[str, bool, float]] = []
    for key in LANE_KEYS:
        block = services.get(key) if isinstance(services, Mapping) else None
        if not isinstance(block, Mapping):
            continue
        if key == LANE_ONE:
            system = inputs.rendered_config.get("control_system")
            simulated = is_simulated(resolve_control_system_type(system))
        else:
            simulated = key == SECOND_LANE_KEYS["va"]
        floor = float(block.get("settle_tolerance", DEFAULT_SETTLE_TOLERANCE))
        lanes.append((key, simulated, floor))
    return lanes


def _warn_once(inputs: ViewInputs, warning: str) -> None:
    """Print one build warning, once per build."""
    from osprey.cli import output

    if inputs.reported is not None:
        if warning in inputs.reported:
            return
        inputs.reported.add(warning)
    summary, _, detail = warning.partition("\n")
    output.warn(summary, detail or None)


def _beyond_one_dimension(records: list[Any]) -> str | None:
    """The warning naming each readback of more than one dimension, or ``None``."""
    addresses = [
        record.address
        for record in records
        if record.direction == "read" and record.shape is not None and len(record.shape) > 1
    ]
    if not addresses:
        return None
    summary = f"{len(addresses)} readback channel(s) of more than one dimension are no plan device"
    return "\n".join((summary, *addresses))


def bluesky_configured(inputs: ViewInputs) -> bool:
    """Whether the render runs a Bluesky lane.

    Args:
        inputs: The render's view inputs.

    Returns:
        True when any Bluesky lane key is a block under the rendered
        ``services``.
    """
    from osprey.bluesky_bridge_connection import LANE_KEYS

    services = inputs.rendered_config.get("services")
    if not isinstance(services, Mapping):
        return False
    return any(isinstance(services.get(lane), Mapping) for lane in LANE_KEYS)


def write_bluesky_view(root: Path, inputs: ViewInputs) -> list[Path]:
    """Write the Bluesky devices view into ``root``.

    Args:
        root: The render's ``data/`` directory.
        inputs: The render's view inputs.

    Returns:
        The file written.
    """
    from osprey.services.bluesky_bridge.substrate_devices import write_devices_file
    from osprey_facility.validate import tolerance_warnings

    root.mkdir(parents=True, exist_ok=True)
    facility_file = root.parent / FACILITY_FILE
    target = root / BLUESKY_DEVICES_FILE
    records = _records(inputs.records, facility_file)
    write_devices_file(
        target,
        records,
        source=_source(facility_file),
        schema=BLUESKY_DEVICES_SCHEMA,
        tolerances=_tolerances(inputs.records),
        motion_bands=_motion_bands(inputs.records),
        units=_units(inputs.records),
        completions=_completions(records, facility_file),
    )
    for warning in tolerance_warnings(inputs.records, _lanes(inputs)):
        _warn_once(inputs, warning)
    left_out = _beyond_one_dimension(records)
    if left_out is not None:
        _warn_once(inputs, left_out)
    return [target]
