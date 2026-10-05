"""Served trees to compare: the demo the control assistant ships, and emitted ones.

Two kinds of tree serve a model in this repository: the demo tree
(``templates/apps/control_assistant/data``), and one per Middle Layer export
the repo commits a 2.0 ``va.json`` beside, which the emit lane writes. Each is
read back here through the loaders the served process reads it with, so a
module comparing trees compares what a deployment would serve.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.services.virtual_accelerator.bindings import BindingsDocument, load_bindings
from osprey.services.virtual_accelerator.manifest.paths import PACKAGE_PATHS, ManifestPaths

__all__ = ["FIXTURES", "Tree", "_demo_tree", "emit_export_tree"]

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "mml"


@dataclass(frozen=True)
class Tree:
    """One deployment tree's served model, and the export it came from.

    Attributes:
        name: The tree's name.
        paths: Where the tree's files sit, so a loader reads them the way the
            served process does.
        document: The bindings the tree serves: which addresses are driven, of
            which family, to which nominal.
        addresses: Every address the tree's channel database carries.
        export: The Middle Layer export the tree was emitted from, or ``None``
            for the demo tree, whose files are committed rather than emitted.
    """

    name: str
    paths: ManifestPaths
    document: BindingsDocument
    addresses: frozenset[str]
    export: dict[str, Any] | None


def _demo_tree() -> Tree:
    """The tree the control assistant ships, whose files are committed."""
    from osprey.services.virtual_accelerator.manifest import build_manifest

    manifest = build_manifest()
    return Tree(
        name="demo",
        paths=PACKAGE_PATHS,
        document=load_bindings(PACKAGE_PATHS.va_bindings),
        addresses=frozenset(channel["address"] for channel in manifest["channels"]),
        export=None,
    )


def emit_export_tree(name: str, directory: Path, stem: str, root: Path) -> Tree:
    """Run the virtual-accelerator emit lane over one committed 2.0 export.

    The lane is run in the order ``osprey mml emit`` runs it -- the deck first,
    because the bindings stamp its digest, then the bindings and the
    starting-state seed -- and writes into ``root`` the layout
    :class:`ManifestPaths` resolves, so every document is read back by the
    loader the served process reads it with.

    Args:
        name: The fixture directory's name, used as the tree's name.
        directory: The fixture directory holding the export.
        stem: The export's file stem, e.g. ``quokka.sr``.
        root: An empty directory to emit the tree into.

    Returns:
        The emitted tree.
    """
    from osprey.services.mml.emit.context import build_context
    from osprey.services.mml.emit.va import emit_bindings, emit_lattice, emit_machine
    from osprey.services.mml.family import FamilyView
    from osprey.services.mml.loaders.mat import load_lattice
    from osprey.services.mml.normalize import normalize_family
    from osprey.services.mml.va.elements import address_elements
    from osprey.services.mml.va.verdicts import propose

    system = stem.split(".")[1].upper()
    ao = json.loads((directory / f"{stem}.ao.json").read_text())
    va = json.loads((directory / f"{stem}.va.json").read_text())
    ring = load_lattice(directory / f"{stem}.lattice.mat")
    export = {
        raw: body for raw, body in ao.items() if not raw.startswith("_") and isinstance(body, dict)
    }
    views = {raw: FamilyView(system, raw, normalize_family(body)) for raw, body in export.items()}
    verdicts = propose(va, ring, views)

    ao_path = root / "ao.json"
    ao_path.write_text(json.dumps({system: ao}))
    mapping_path = root / "mapping.yaml"
    mapping_path.write_text(f"facility:\n  token: {name}\n")
    ctx = build_context(ao_path, mapping_path, {})

    paths = ManifestPaths(data_root=root / "data")
    paths.machine_json.parent.mkdir(parents=True, exist_ok=True)
    keyed = {(system, raw): verdict for raw, verdict in verdicts.items()}
    judged_va = {(system, raw): block for raw, block in va["families"].items()}
    # What each family's devices drive. The bindings anchor their slice
    # factors on it and the seeds start a supply at the mean of the same
    # devices, so both lanes are handed the one mapping.
    element_bindings = dict(address_elements(va, ring, verdicts).bindings)
    emit_lattice(ring, paths.lattice_json, ctx, write=False)
    bindings_text, _findings = emit_bindings(
        keyed,
        list(views.values()),
        element_bindings,
        judged_va,
        ctx,
        system=system,
        energy_gev=va["lattice"]["energy_gev"],
    )
    paths.va_bindings.write_text(bindings_text)
    machine_text, _seeds = emit_machine(
        keyed,
        list(views.values()),
        judged_va,
        _export_mapping(name, system, views),
        ctx,
        element_bindings,
    )
    paths.machine_json.write_text(machine_text)

    return Tree(
        name=name,
        paths=paths,
        document=load_bindings(paths.va_bindings),
        addresses=frozenset(_export_addresses(views.values())),
        export=export,
    )


def _export_mapping(token: str, system: str, views: dict):
    """The prose side of a mapping, for the emitters that read a channel's words.

    The seeds are read off the export, not off the mapping, so what a reviewer
    wrote about a family does not change them -- a mapping naming every family
    with no prose at all is enough to run the lane.
    """
    from osprey.services.mml.mapping.schema import (
        Facility,
        Family,
        Field,
        Mapping,
        System,
        VirtualAccelerator,
    )

    return Mapping(
        facility=Facility(token=token, title=token, description=None, provenance="human"),
        systems={system: System(raw=system, name=system, description=None, provenance="human")},
        section_order=(system,),
        families={
            raw: Family(
                raw=raw,
                rename=None,
                branch=None,
                class_="BPM",
                aliases=(),
                description=None,
                provenance="human",
                channels=1,
                fields={name: Field(description=None, provenance="human") for name in view.fields},
            )
            for raw, view in views.items()
        },
        directions={},
        judgments={},
        virtual_accelerator=VirtualAccelerator(system=system, families={}),
    )


def _export_addresses(views) -> set[str]:
    """Every address the export names, which is what its channel database carries."""
    addresses: set[str] = set()
    for view in views:
        for field in view.fields.values():
            for key in field.keys:
                for slot in field.slots(key):
                    if isinstance(slot, str) and slot.strip():
                        addresses.add(slot.strip())
    return addresses
