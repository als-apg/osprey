"""Write the facility file a build leaves at the root of a render.

The channel roster reads ``<render root>/facility.json``. ``osprey build``
writes that file; a test that stands up a render by hand and then asks for
channels has to write one too, or the roster answers with the absence that
says the file is not built.

The file is built from a ``facility`` tree by the same in-memory build
the real one runs, and serialised by the same writer, so a test's file has the
shape a build's has -- defaults filled, records sorted.

Import-light on purpose: nothing of osprey is imported at module import.
"""

from __future__ import annotations

import copy
import functools
import json
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tests._builds import FRAMEWORK_CHECKS

#: The shipped demo's facility file, built once per process.
_DEMO_BYTES: bytes | None = None

#: The header of the facility document the readers read, spelled here so this
#: module imports nothing of osprey; ``tests/facility/test_reading.py`` holds it
#: in ``osprey_facility.reading.READS``.
FACILITY_V4 = "osprey.facility.facility/4"

#: The name of the facility file at the root of a render.
_FACILITY_FILE = "facility.json"


def write_facility_document(
    root: Path | str, document: object, *, schema: str | None = FACILITY_V4
) -> Path:
    """Write any payload as the facility file at the root of *root*.

    The raw form: no build runs and nothing is checked, so a test can write a
    document a reader must refuse -- a foreign header, a header-less document,
    a non-object. A test that writes a header outside
    ``osprey_facility.reading.READS`` carries the ``foreign_header`` marker.

    Args:
        root: The directory that holds ``facility.json``, normally a render root.
        document: The payload, serialised as JSON as it is. A mapping gets a
            ``schema`` key set to *schema* (on a copy); any other payload is
            written untouched.
        schema: The header a mapping is written with; ``None`` writes the
            mapping exactly as given, header or none.

    Returns:
        The path of the facility file that now exists on disk.
    """
    payload = document
    if isinstance(document, Mapping) and schema is not None:
        payload = {**document, "schema": schema}
    target = Path(root) / _FACILITY_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def write_facility_v4(root: Path | str, document: Mapping[str, Any] | None = None) -> Path:
    """Write a ``/4`` facility document at the root of *root*.

    The form for a test whose document a reader reads: the header is always
    :data:`FACILITY_V4`. No build runs, so the tables are the test's own.

    Args:
        root: The directory that holds ``facility.json``, normally a render root.
        document: The document's tables; ``None`` is a document of the header
            alone.

    Returns:
        The path of the facility file that now exists on disk.

    Raises:
        ValueError: *document* names another header; a test that means one
            writes it with :func:`write_facility_document`.
    """
    document = dict(document or {})
    if document.get("schema", FACILITY_V4) != FACILITY_V4:
        raise ValueError(
            f"write_facility_v4 writes {FACILITY_V4}, not {document['schema']!r}; "
            "use write_facility_document for another header"
        )
    return write_facility_document(root, document, schema=FACILITY_V4)


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
        tree: A ``facility`` tree in the shape
            ``tests.facility._synthetic_trees`` writes; ``None`` is a project
            with no ``facility`` at all.
        name: The project name.

    Returns:
        The path of the facility file that now exists on disk.
    """
    from osprey_facility.build import build_facility
    from osprey_facility.render import FACILITY_FILE, facility_bytes
    from tests.facility._synthetic_trees import write_tree

    with tempfile.TemporaryDirectory() as scratch:
        facility_dir = Path(scratch) / "facility"
        if tree is not None:
            write_tree(facility_dir, tree)
        document = build_facility(facility_dir, project_name=name, extra_checks=FRAMEWORK_CHECKS)
    target = Path(render) / FACILITY_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(facility_bytes(document))
    return target


@functools.cache
def _demo_document() -> dict[str, Any]:
    from osprey_facility.build import build_facility
    from tests._preset_data import packaged_facility_dir

    return build_facility(
        packaged_facility_dir("demo"), project_name="demo", extra_checks=FRAMEWORK_CHECKS
    )


def demo_facility_document() -> dict[str, Any]:
    """The facility document of the tree the demo preset ships.

    Built once per process; each call returns its own copy, which the caller
    may change. The tree ships ``identity.yaml``, so the project name a build
    passes reaches nothing in the document: every project that builds this
    tree gets this document.

    Returns:
        A copy of the document ``build_facility`` returns for the tree.
    """
    return copy.deepcopy(_demo_document())


def write_demo_facility_file(render: Path | str) -> Path:
    """Write the facility file of the tree the demo preset ships.

    4765 channels: 677 setpoints, all but the 29 gauge trip resets paired with a
    readback, and 4088 readbacks.

    Args:
        render: The render root, the directory holding ``config.yml``.

    Returns:
        The path of the facility file that now exists on disk.
    """
    global _DEMO_BYTES
    from osprey_facility.render import FACILITY_FILE

    if _DEMO_BYTES is None:
        from osprey_facility.render import facility_bytes

        _DEMO_BYTES = facility_bytes(_demo_document())
    target = Path(render) / FACILITY_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_DEMO_BYTES)
    return target


def write_facility_views(render: Path | str, bundle: str = "demo") -> list[Path]:
    """Write the facility outputs a build writes into *render*, from *bundle*'s tree.

    The facility file and every view the render's ``config.yml`` asks for, by
    the build's own writer, so the render's agent facts name the models its
    config serves. A bundle that packages no ``facility`` tree writes
    nothing.

    Args:
        render: The render root, the directory holding ``config.yml``.
        bundle: The app bundle whose packaged facility tree is built.

    Returns:
        The files written, sorted.
    """
    import yaml

    from osprey.facility.views.outputs import render_facility_outputs
    from osprey_facility.build import build_facility
    from tests._preset_data import packaged_facility_dir

    render = Path(render)
    config = yaml.safe_load((render / "config.yml").read_text(encoding="utf-8")) or {}
    facility_dir = packaged_facility_dir(bundle)
    if not facility_dir.is_dir():
        return []
    if facility_dir == packaged_facility_dir("demo"):
        document = demo_facility_document()
    else:
        document = build_facility(
            facility_dir,
            project_name=str(config.get("project_name", render.name)),
            extra_checks=FRAMEWORK_CHECKS,
        )
    return render_facility_outputs(render, document, config, facility_dir)
