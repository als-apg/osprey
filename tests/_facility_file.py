"""Write the facility file a build leaves at the root of a render.

The channel roster reads ``<render root>/facility.json``. ``osprey build``
writes that file; a test that stands up a render by hand and then asks for
channels has to write one too, or the roster answers with the absence that
says the file is not built.

The file is built from a ``data/facility`` tree by the same in-memory build
the real one runs, and serialised by the same writer, so a test's file has the
shape a build's has -- defaults filled, records sorted.

Import-light on purpose: nothing of osprey is imported at module import.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

#: The shipped demo's facility file, built once per process.
_DEMO_BYTES: bytes | None = None


def channel_tree(
    setpoints: Mapping[str, str | None] | Iterable[str] = (),
    readbacks: Iterable[str] = (),
    unpaired: Iterable[str] = (),
) -> dict[str, Any]:
    """A tree holding channel records and nothing else.

    Args:
        setpoints: Each setpoint address to the readback address it is paired
            with, or to ``None`` for a setpoint that is its own pair. A bare
            iterable of addresses is all-``None``.
        readbacks: The readback addresses. A setpoint's pair is added to them.
        unpaired: Addresses whose role is ``none``.

    Returns:
        The tree, for :func:`write_facility_file`.
    """
    pairs = dict(setpoints) if isinstance(setpoints, Mapping) else dict.fromkeys(setpoints)
    records: list[dict[str, Any]] = []
    for address, pair in pairs.items():
        record: dict[str, Any] = {"id": address, "role": "setpoint"}
        if pair is not None:
            record["pair"] = pair
        records.append(record)
    paired = [pair for pair in pairs.values() if pair is not None]
    records += [{"id": address} for address in dict.fromkeys([*readbacks, *paired])]
    records += [{"id": address, "role": "none"} for address in unpaired]
    return {"records/channels.yaml": records}


def write_facility_file(
    render: Path | str, tree: Mapping[str, Any] | None = None, *, name: str = "demo"
) -> Path:
    """Build *tree* and write its facility file at the root of *render*.

    Args:
        render: The render root, the directory holding ``config.yml``.
        tree: A ``data/facility`` tree in the shape
            ``tests.facility._synthetic_trees`` writes; ``None`` is a project
            with no ``data/facility`` at all.
        name: The project name.

    Returns:
        The path of the facility file that now exists on disk.
    """
    from osprey.facility.build import build_facility
    from osprey.facility.render import FACILITY_FILE, facility_bytes
    from tests.facility._synthetic_trees import write_tree

    with tempfile.TemporaryDirectory() as scratch:
        facility_dir = Path(scratch) / "facility"
        if tree is not None:
            write_tree(facility_dir, tree)
        document = build_facility(facility_dir, project_name=name)
    target = Path(render) / FACILITY_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(facility_bytes(document))
    return target


def write_demo_facility_file(render: Path | str) -> Path:
    """Write the facility file of the tree the control-assistant preset ships.

    2912 channels: 396 setpoints, each paired with a readback, and 2516
    readbacks.

    Args:
        render: The render root, the directory holding ``config.yml``.

    Returns:
        The path of the facility file that now exists on disk.
    """
    global _DEMO_BYTES
    from osprey.facility.render import FACILITY_FILE

    if _DEMO_BYTES is None:
        from importlib.resources import as_file, files

        from osprey.facility.build import build_facility
        from osprey.facility.render import facility_bytes

        resource = files("osprey.templates").joinpath(
            "apps", "control_assistant", "data", "facility"
        )
        with as_file(resource) as facility_dir:
            _DEMO_BYTES = facility_bytes(build_facility(facility_dir, project_name="demo"))
    target = Path(render) / FACILITY_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_DEMO_BYTES)
    return target
