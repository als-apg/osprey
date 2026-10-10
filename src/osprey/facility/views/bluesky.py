"""The Bluesky devices view: the facility's channels as the worker's device file.

Written to ``<render>/data/bluesky_devices.yml``::

    schema: osprey.facility.bluesky_devices/1
    # generated-file header
    settables:
      - {name: <setpoint>, setpoint: <setpoint>, readback: <pair>, settle_tolerance: <x>}
      - {name: <setpoint>, setpoint: <setpoint>, readback: <pair>, motion_band: <band>}
    readables:
      - {name: <readback>, pv: <readback>}

One settable per setpoint channel, whose ``readback`` is the channel's ``pair``
when the pair is another channel and is absent otherwise; one readable per
readback channel, paired or not; a channel whose role is ``none`` is no device.
A settable carries ``settle_tolerance`` when its setpoint declares a
``tolerance``: the number of an absolute one, ``{relative: <f>}`` of a relative
one. A settable without one carries ``motion_band`` when the channel it reads
back declares motion in its ``simulation`` seed: the envelope
:func:`osprey_connectors.simulation.envelope.motion_envelope` derives from that
seed, which only a simulated lane settles within. A settable with neither
settles within the profile's floor.

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

from osprey.facility import FACILITY_FILE
from osprey.facility.views import ViewInputs

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


def _records(doc: Mapping[str, Any], facility_file: Path) -> list[Any]:
    from osprey.channel_roster import channel_record

    source = _source(facility_file)
    return [channel_record(channel, source) for channel in doc.get("channels", [])]


def _tolerances(doc: Mapping[str, Any]) -> dict[str, Mapping[str, float]]:
    """Each setpoint's declared ``tolerance`` record, by address."""
    return {
        channel["id"]: channel["tolerance"]
        for channel in doc.get("channels", [])
        if channel.get("tolerance")
    }


def _motion_bands(doc: Mapping[str, Any]) -> dict[str, float]:
    """Each channel's motion envelope, by address."""
    from osprey_connectors.simulation.envelope import motion_envelope

    return {
        channel["id"]: motion_envelope(channel.get("simulation"))
        for channel in doc.get("channels", [])
    }


def bluesky_document(doc: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """The device document of one facility file.

    Args:
        doc: The facility file.

    Returns:
        ``{settables, readables}`` as the worker's device file holds them.
    """
    from osprey.services.bluesky_bridge.substrate_devices import devices_document

    return devices_document(
        _records(doc, Path(FACILITY_FILE)), _tolerances(doc), _motion_bands(doc)
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
    """Print one tolerance warning, once per build."""
    from osprey.cli import output

    if inputs.reported is not None:
        if warning in inputs.reported:
            return
        inputs.reported.add(warning)
    summary, _, detail = warning.partition("\n")
    output.warn(summary, detail or None)


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
    from osprey.facility.validate import tolerance_warnings
    from osprey.services.bluesky_bridge.substrate_devices import write_devices_file

    root.mkdir(parents=True, exist_ok=True)
    facility_file = root.parent / FACILITY_FILE
    target = root / BLUESKY_DEVICES_FILE
    write_devices_file(
        target,
        _records(inputs.doc, facility_file),
        source=_source(facility_file),
        schema=BLUESKY_DEVICES_SCHEMA,
        tolerances=_tolerances(inputs.doc),
        motion_bands=_motion_bands(inputs.doc),
    )
    if inputs.tolerance_warnings:
        for warning in tolerance_warnings(inputs.doc, _lanes(inputs)):
            _warn_once(inputs, warning)
    return [target]
