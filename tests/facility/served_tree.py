"""Build a served tree for a test: facility layers, then the build, then the views.

A test that drives a connector hands it the addresses it reads and writes as a
facility tree, the way a deployment does. :func:`served_tree` writes those
addresses as channel records under ``<root>/data/facility``, builds the
facility file and renders it into ``<root>/build``, and returns that render's
simulator view, the directory a connector serves from. :func:`mock_config`
names that view in a mock connector's ``connect()`` settings.

Every setpoint is writable: the render runs the simulated target with
``limits_checking.mode: optional``, so a setpoint without a limits record takes
no limits. A test that checks limits does so through its own validator, never
through the tree.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

__all__ = ["MOCK_RENDER_CONFIG", "SIMULATOR_VIEW", "mock_config", "served_tree"]

#: The ``connect()`` setting that names the simulator view a mock serves.
SIMULATOR_VIEW = "simulator_view"

#: The rendered configuration the tree is rendered under: the simulated target,
#: with every setpoint writable.
MOCK_RENDER_CONFIG: Mapping[str, Any] = {
    "control_system": {
        "type": "mock",
        "limits_checking": {"enabled": True, "mode": "optional"},
    }
}

#: The views every render of this process has named as not written; each is
#: named once.
_REPORTED: set[str] = set()


def served_tree(
    root: Path,
    setpoints: Iterable[str] | Mapping[str, str] = (),
    readings: Iterable[str] = (),
) -> Path:
    """Build and render a tree serving the given addresses.

    Args:
        root: An empty directory; the tree is written to ``root/data/facility``
            and rendered into ``root/build``.
        setpoints: The addresses a test writes, each a writable setpoint. A
            mapping pairs each setpoint with its readback address, which joins
            the tree as a readback.
        readings: The addresses a test only reads, each a readback.

    Returns:
        The render's simulator view, ``root/build/data/simulator``.

    Raises:
        AssertionError: The view does not serve every address given, or a
            setpoint given is not writable in it.
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

    facility_dir = root / "data" / "facility"
    (facility_dir / "records").mkdir(parents=True)
    (facility_dir / "records" / "channels.yaml").write_text(
        yaml.safe_dump(records, sort_keys=False), encoding="utf-8"
    )

    render_dir = root / "build"
    render_dir.mkdir()
    document = build_facility(facility_dir, project_name="served")
    render_facility_outputs(
        render_dir, document, MOCK_RENDER_CONFIG, facility_dir, omitted_reported=_REPORTED
    )

    view = render_dir / "data" / "simulator"
    _check_served(view, set(pairs), readbacks)
    return view


def mock_config(view: Path, **settings: Any) -> dict[str, Any]:
    """A mock connector's ``connect()`` settings serving ``view``.

    Args:
        view: A simulator view, as :func:`served_tree` returns it.
        settings: The connector's other settings.

    Returns:
        ``settings`` with the view named under :data:`SIMULATOR_VIEW`.
    """
    return {SIMULATOR_VIEW: str(view), **settings}


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
