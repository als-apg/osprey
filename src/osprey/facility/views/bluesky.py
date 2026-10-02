"""The Bluesky devices view: the facility's channels as the worker's device file.

Written to ``<render>/data/bluesky_devices.yml``::

    schema: osprey.facility.bluesky_devices/1
    # generated-file header
    settables:
      - {name: <setpoint>, setpoint: <setpoint>, readback: <pair>}
    readables:
      - {name: <readback>, pv: <readback>}

One settable per setpoint channel, whose ``readback`` is the channel's ``pair``
when the pair is another channel and is absent otherwise; one readable per
readback channel, paired or not; a channel whose role is ``none`` is no device.
A device's name is its address. Entries follow the facility file's channel
order. A render carries the view when it runs a Bluesky lane, under every
control system: the compose generator stages it for a connector that drives
channels and leaves it unstaged under the mock.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
    from osprey.channel_roster.sources import _record

    source = _source(facility_file)
    return [_record(channel, source) for channel in doc.get("channels", [])]


def bluesky_document(doc: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    """The device document of one facility file.

    Args:
        doc: The facility file.

    Returns:
        ``{settables, readables}`` as the worker's device file holds them.
    """
    from osprey.facility.render import FACILITY_FILE
    from osprey.services.bluesky_bridge.substrate_devices import devices_document

    return devices_document(_records(doc, Path(FACILITY_FILE)))


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
    from osprey.facility.render import FACILITY_FILE
    from osprey.services.bluesky_bridge.substrate_devices import write_devices_file

    root.mkdir(parents=True, exist_ok=True)
    facility_file = root.parent / FACILITY_FILE
    target = root / BLUESKY_DEVICES_FILE
    write_devices_file(
        target,
        _records(inputs.doc, facility_file),
        source=_source(facility_file),
        schema=BLUESKY_DEVICES_SCHEMA,
    )
    return [target]
