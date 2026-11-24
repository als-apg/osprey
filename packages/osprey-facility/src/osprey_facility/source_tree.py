"""Where each source of a facility tree lives: the tree's one layout declaration.

A facility tree is the directory ``osprey build`` reads facility sources from.
Each slot below is one place in it: a single file (``limits.yaml``), a
directory (``records/``, ``knowledge/``), or a keyed directory that holds one
``<key>.yaml`` per key (``measurement/<model>.yaml``). A slot may sit inside
another (the record files inside ``records/``); its path is then relative to
that parent, so moving the parent moves it.

Every layer under ``imported/<layer>/`` is flat: a slot a layer also carries
has its own fixed name there (``models.yaml``, ``decks``), which never follows
the slot's authored path.

The registry is the only module state. Every accessor reads it when called,
so a consumer that computes its tables from these functions at its own import
follows whatever registry is bound at that moment. Standard library only.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath

__all__ = [
    "FILE",
    "display_path",
    "keyed_dir",
    "keyed_file",
    "layer_dir",
    "layer_entry",
    "layer_root",
    "path",
    "slot",
    "slots",
]

#: A slot that is one file.
FILE = "file"
#: A slot that is a directory.
_DIRECTORY = "directory"
#: A slot that is a directory of one ``<key>.yaml`` per key.
_KEYED_DIRECTORY = "keyed directory"

_KINDS = frozenset({FILE, _DIRECTORY, _KEYED_DIRECTORY})

#: The suffix of each file of a keyed directory.
_KEYED_SUFFIX = ".yaml"

#: Where the facility tree sits in a deployment, as messages name it.
_DISPLAY_PREFIX = "data/facility"

#: The slot every imported layer's directory sits in.
_LAYER_ROOT = "imported"


@dataclass(frozen=True)
class _FacilitySlot:
    """One place in a facility tree.

    Attributes:
        name: The slot's stable identity.
        path: Where it is authored, POSIX, relative to the facility root, or to
            its parent when it has one.
        kind: :data:`FILE`, :data:`_DIRECTORY` or :data:`_KEYED_DIRECTORY`.
        parent: The slot it sits inside, if any.
        layer_name: Its flat name inside ``imported/<layer>/``, for a slot a
            layer also carries.
        record_kind: The record kind a record file holds.
        content: True for what a deployment writes about its own use of the
            machine, false for a fact of the machine.
    """

    name: str
    path: str
    kind: str
    parent: str | None = None
    layer_name: str | None = None
    record_kind: str | None = None
    content: bool = False


_FACILITY_SLOTS: tuple[_FacilitySlot, ...] = (
    _FacilitySlot("identity", "identity.yaml", FILE),
    _FacilitySlot("classes", "classes.yaml", FILE),
    _FacilitySlot("signals", "signals.yaml", FILE),
    _FacilitySlot("models", "models.yaml", FILE, layer_name="models.yaml"),
    _FacilitySlot("limits", "limits.yaml", FILE),
    _FacilitySlot("seeds", "seeds.yaml", FILE),
    _FacilitySlot("fixes", "fixes.yaml", FILE),
    _FacilitySlot("documents", "documents.yaml", FILE),
    _FacilitySlot("policy", "policy.yaml", FILE),
    _FacilitySlot("records", "records", _DIRECTORY),
    _FacilitySlot(
        "places", "places.yaml", FILE, "records", layer_name="places.yaml", record_kind="place"
    ),
    _FacilitySlot(
        "devices", "devices.yaml", FILE, "records", layer_name="devices.yaml", record_kind="device"
    ),
    _FacilitySlot(
        "channels",
        "channels.yaml",
        FILE,
        "records",
        layer_name="channels.yaml",
        record_kind="channel",
    ),
    _FacilitySlot(
        "groups", "groups.yaml", FILE, "records", layer_name="groups.yaml", record_kind="group"
    ),
    _FacilitySlot(
        "relations",
        "relations.yaml",
        FILE,
        "records",
        layer_name="relations.yaml",
        record_kind="relation",
    ),
    _FacilitySlot(_LAYER_ROOT, "imported", _DIRECTORY),
    _FacilitySlot("scenarios", "scenarios", _KEYED_DIRECTORY),
    _FacilitySlot("measurement", "measurement", _KEYED_DIRECTORY),
    _FacilitySlot("decks", "decks", _DIRECTORY, layer_name="decks"),
    _FacilitySlot("knowledge", "knowledge", _DIRECTORY, content=True),
    _FacilitySlot("ariel", "ariel", _DIRECTORY, content=True),
    _FacilitySlot("ariel_vocabulary", "vocabulary.yml", FILE, "ariel", content=True),
    _FacilitySlot(
        "benchmark_queries", "benchmarks/cross_paradigm/queries", _DIRECTORY, content=True
    ),
    _FacilitySlot(
        "in_context_queries", "in_context_queries.json", FILE, "benchmark_queries", content=True
    ),
    _FacilitySlot("tree_queries", "tree_queries.json", FILE, "benchmark_queries", content=True),
)


def slots() -> tuple[_FacilitySlot, ...]:
    """Every slot, in declaration order."""
    return _FACILITY_SLOTS


def slot(name: str) -> _FacilitySlot:
    """The slot named *name*.

    Raises:
        KeyError: No slot has that name.
    """
    return _find(_FACILITY_SLOTS, name)


def path(name: str) -> str:
    """Where slot *name* is authored, relative to the facility root."""
    return _resolve(_FACILITY_SLOTS, name)


def keyed_file(name: str, key: str) -> str:
    """The file of *key* in keyed directory *name*: ``measurement/<key>.yaml``.

    The key ``*`` gives the glob of every file of the directory.
    """
    return f"{_path_of(name, _KEYED_DIRECTORY)}/{key}{_KEYED_SUFFIX}"


def keyed_dir(name: str, key: str) -> str:
    """The directory of *key* beside its file in keyed directory *name*: ``scenarios/<key>``."""
    return f"{_path_of(name, _KEYED_DIRECTORY)}/{key}"


def _inside(name: str, *parts: str) -> str:
    """A path inside directory slot *name*: ``_inside("knowledge", "index.md")``."""
    if not parts:
        raise ValueError(f"{name!r} names nothing inside the slot")
    return "/".join((_path_of(name, _DIRECTORY, _KEYED_DIRECTORY), *parts))


def layer_root() -> str:
    """The directory every imported layer's directory sits in: ``imported``."""
    return path(_LAYER_ROOT)


def layer_dir(layer: str, *parts: str) -> str:
    """Imported layer *layer*'s directory, or a path under it: ``imported/<layer>``."""
    return _inside(_LAYER_ROOT, layer, *parts)


def layer_entry(layer: str, name: str) -> str:
    """Slot *name*'s flat entry inside imported layer *layer*: ``imported/mml/decks``.

    Raises:
        ValueError: A layer never carries slot *name*.
    """
    entry = slot(name).layer_name
    if entry is None:
        raise ValueError(f"a layer carries no {name!r} slot")
    return f"{layer_dir(layer)}/{entry}"


def display_path(relative: str) -> str:
    """A path relative to the facility root as messages name it: ``data/facility/<relative>``."""
    return f"{_DISPLAY_PREFIX}/{relative}"


# --- the registry ---------------------------------------------------------------------


def _find(registry: tuple[_FacilitySlot, ...], name: str) -> _FacilitySlot:
    for row in registry:
        if row.name == name:
            return row
    raise KeyError(f"no facility slot named {name!r}")


def _resolve(registry: tuple[_FacilitySlot, ...], name: str) -> str:
    row = _find(registry, name)
    if row.parent is None:
        return row.path
    return f"{_resolve(registry, row.parent)}/{row.path}"


def _path_of(name: str, *kinds: str) -> str:
    row = slot(name)
    if row.kind not in kinds:
        raise ValueError(f"facility slot {name!r} is a {row.kind}, not a {' or '.join(kinds)}")
    return path(name)


def _with_paths(paths: Mapping[str, str]) -> tuple[_FacilitySlot, ...]:
    """The bound registry with the named slots' paths replaced.

    Raises:
        KeyError: A name is no slot.
        ValueError: The result is not a valid registry.
    """
    unknown = set(paths) - {row.name for row in _FACILITY_SLOTS}
    if unknown:
        raise KeyError(f"no facility slot named {sorted(unknown)}")
    registry = tuple(
        dataclasses.replace(row, path=paths[row.name]) if row.name in paths else row
        for row in _FACILITY_SLOTS
    )
    _check(registry)
    return registry


def _check(registry: tuple[_FacilitySlot, ...]) -> None:
    """Raise ValueError unless *registry* is a valid facility layout."""
    seen: dict[str, _FacilitySlot] = {}
    for row in registry:
        if row.name in seen:
            raise ValueError(f"facility slot {row.name!r} is declared twice")
        if row.kind not in _KINDS:
            raise ValueError(f"facility slot {row.name!r} has unknown kind {row.kind!r}")
        pure = PurePosixPath(row.path)
        if (
            not row.path
            or "\\" in row.path
            or pure.is_absolute()
            or pure.as_posix() != row.path
            or any(part in (".", "..") for part in row.path.split("/"))
        ):
            raise ValueError(f"facility slot {row.name!r} path {row.path!r} is not relative POSIX")
        if row.parent is not None:
            parent = seen.get(row.parent)
            if parent is None:
                raise ValueError(f"facility slot {row.name!r} sits in undeclared {row.parent!r}")
            if parent.kind == FILE:
                raise ValueError(f"facility slot {row.name!r} sits in file slot {row.parent!r}")
        if row.record_kind is not None and (row.kind != FILE or row.layer_name is None):
            raise ValueError(f"record slot {row.name!r} is not a file a layer carries")
        seen[row.name] = row
    resolved = {row.name: _resolve(registry, row.name) for row in registry}
    for row in registry:
        if row.kind == FILE:
            nested = sorted(
                other
                for other, where in resolved.items()
                if where.startswith(f"{resolved[row.name]}/")
            )
            if nested:
                raise ValueError(f"facility slots {nested} sit inside file slot {row.name!r}")
    for what, values in (
        ("path", list(resolved.values())),
        ("layer name", [row.layer_name for row in registry if row.layer_name is not None]),
    ):
        doubled = sorted({value for value in values if values.count(value) > 1})
        if doubled:
            raise ValueError(f"facility slots share a {what}: {doubled}")


_check(_FACILITY_SLOTS)
