"""Build a served tree for a test: facility layers, then the build, then the views.

A test that drives a connector hands it the addresses it reads and writes as a
facility tree, the way a deployment does. :func:`served_tree` writes those
addresses as channel records under ``<root>/data/facility``, builds the
facility file and renders it into ``<root>/build``, and returns that render's
simulator view, the directory a connector serves from. :func:`in_process_config`
names that view in the in-process simulator's ``connect()`` settings.

Every setpoint is writable: the render runs the simulated target with
``limits_checking.mode: optional``, so a setpoint without a limits record takes
no limits. A test that checks limits does so through its own validator, never
through the tree.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from osprey_connectors.simulation.view import SimulatorView

__all__ = [
    "IN_PROCESS_RENDER_CONFIG",
    "LEAK_VISIBILITY_FACTOR",
    "READBACK_FLOOR",
    "SIMULATOR_VIEW",
    "assert_unmoved",
    "assert_write_visible",
    "in_process_config",
    "served_active",
    "served_tree",
    "unmoved_band",
]

#: The ``connect()`` setting that names the simulator view the in-process simulator serves.
SIMULATOR_VIEW = "simulator_view"

#: The rendered configuration the tree is rendered under: the simulator served
#: in process, with every setpoint writable.
IN_PROCESS_RENDER_CONFIG: Mapping[str, Any] = {
    "control_system": {
        "type": "virtual_accelerator",
        "connector": {"virtual_accelerator": {"serving": "in_process"}},
        "limits_checking": {"enabled": True, "mode": "optional"},
    }
}

#: Absolute slack on a readback check beyond the declared motion: float
#: rounding of the served value, far below any write a test makes.
READBACK_FLOOR = 1e-9

#: How far above a readback check's band a demand sits from the held value, so
#: a leaked write cannot hide inside the readback's declared motion.
LEAK_VISIBILITY_FACTOR = 100.0

#: The views every render of this process has named as not written; each is
#: named once.
_REPORTED: set[str] = set()


def served_tree(
    root: Path,
    setpoints: Iterable[str] | Mapping[str, str] = (),
    readings: Iterable[str] = (),
    *,
    channels: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Build and render a tree serving the given addresses.

    Args:
        root: An empty directory; the tree is written to ``root/data/facility``
            and rendered into ``root/build``.
        setpoints: The addresses a test writes, each a writable setpoint. A
            mapping pairs each setpoint with its readback address, which joins
            the tree as a readback.
        readings: The addresses a test only reads, each a readback.
        channels: Further channel-record fields by address, in the facility
            schema's own spelling (``value_type``, ``options``, ``unit``,
            ``description``, ``simulation``), merged into that address's
            record.

    Returns:
        The render's simulator view, ``root/build/data/simulator``.

    Raises:
        AssertionError: An address of ``channels`` is not among the setpoints,
            their pairs or the readings; the view does not serve every address
            given; or a setpoint given is not writable in it.
    """
    from osprey.facility.build import build_facility
    from osprey.facility.render import render_facility_outputs

    pairs = dict(setpoints) if isinstance(setpoints, Mapping) else dict.fromkeys(setpoints)
    readbacks = {*readings, *(pair for pair in pairs.values() if pair is not None)} - set(pairs)

    records: list[dict[str, Any]] = []
    for address in sorted(pairs):
        record: dict[str, Any] = {"id": address, "role": "setpoint"}
        if pairs[address] is not None:
            record["pair"] = pairs[address]
        records.append(record)
    records.extend({"id": address} for address in sorted(readbacks))
    further = dict(channels or {})
    unknown = sorted(set(further) - set(pairs) - readbacks)
    assert not unknown, f"channels names addresses the tree does not hold: {unknown}"
    for record in records:
        record.update(further.get(record["id"], {}))

    facility_dir = root / "data" / "facility"
    (facility_dir / "records").mkdir(parents=True)
    (facility_dir / "records" / "channels.yaml").write_text(
        yaml.safe_dump(records, sort_keys=False), encoding="utf-8"
    )

    render_dir = root / "build"
    render_dir.mkdir()
    document = build_facility(facility_dir, project_name="served")
    render_facility_outputs(
        render_dir, document, IN_PROCESS_RENDER_CONFIG, facility_dir, omitted_reported=_REPORTED
    )

    view = render_dir / "data" / "simulator"
    _check_served(view, set(pairs), readbacks)
    return view


def in_process_config(view: Path, **settings: Any) -> dict[str, Any]:
    """The in-process simulator's connector block serving ``view``.

    Also its ``connect()`` settings: the block states ``serving: in_process``,
    which the factory reads and ``connect()`` ignores.

    Args:
        view: A simulator view, as :func:`served_tree` returns it.
        settings: The connector's other settings.

    Returns:
        ``settings`` with the venue and the view named under
        :data:`SIMULATOR_VIEW`.
    """
    return {"serving": "in_process", SIMULATOR_VIEW: str(view), **settings}


def served_active(view: SimulatorView) -> list[str]:
    """The scenario set the simulator serving ``view`` runs, ``nominal`` first.

    Read from the state directory the in-process simulator itself resolves for
    the view, so the set is the one the served readings are drawn under. A
    render with no state file runs ``nominal`` alone.
    """
    from osprey_connectors.control_system._va_in_process_connector import simulation_state_dir
    from osprey_connectors.simulation.state import read_active_state, resolve_active_scenarios

    return resolve_active_scenarios(read_active_state(simulation_state_dir(view.path))[0])


def unmoved_band(view: SimulatorView, address: str, active: Sequence[str] = ()) -> float:
    """The band two reads of a quiet ``address`` keep within under ``active``.

    Each read carries an independent draw of the motion the served tree
    declares for the channel (``SimulatorView.motion_envelope``), so two reads
    differ by at most twice it; :data:`READBACK_FLOOR` covers float rounding. A
    channel that declares no motion gets the floor alone.
    """
    return 2 * view.motion_envelope(address, active=active) + READBACK_FLOOR


def assert_write_visible(demand: float, held: float, band: float) -> None:
    """``demand`` sits far enough from ``held`` that a landed write shows past ``band``.

    Raises:
        AssertionError: ``demand`` is within :data:`LEAK_VISIBILITY_FACTOR`
            bands of ``held``.
    """
    assert abs(demand - held) >= LEAK_VISIBILITY_FACTOR * band, (
        f"the demand {demand} sits {abs(demand - held)} from the held {held}, too close "
        f"to the band {band} for a read-back check to see a leaked write"
    )


def assert_unmoved(address: str, before: float, after: float, band: float) -> None:
    """``address`` read ``after`` within ``band`` of ``before``.

    Raises:
        AssertionError: The two reads differ by more than ``band``.
    """
    assert abs(after - before) <= band, (
        f"{address} read {before} before and {after} after, beyond the band {band}"
    )


def _check_served(view: Path, setpoints: set[str], readbacks: set[str]) -> None:
    import json

    addresses = json.loads((view / "addresses.json").read_text(encoding="utf-8"))
    channels = json.loads((view / "variables.json").read_text(encoding="utf-8"))["channels"]
    missing = (setpoints | readbacks) - set(addresses["channels"])
    assert not missing, f"the view does not serve {sorted(missing)}"
    locked = sorted(
        channel["address"]
        for channel in channels
        if channel["address"] in setpoints and not channel["writable"]
    )
    assert not locked, f"the view does not write {locked}"
