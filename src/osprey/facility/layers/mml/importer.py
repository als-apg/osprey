"""The mml layer's importer: MML exports written as record sources under ``imported/mml/``.

:func:`import_mml` reads one or more exports, loads the layer's mapping (or
writes a draft and stops, :func:`~osprey.facility.layers.mml.mapping.load_or_draft`)
and writes the layer's record files. It writes sources, never a view: the
build merges them with every other layer.

What is written, relative to ``data/facility/``:

* ``imported/mml/devices.yaml``: one device per device of a family that
  carries a channel, its id and the slots it stands for resolved by the
  identity model (:mod:`~osprey.facility.layers.mml.identity`): the export's
  ``CommonNames`` entry at the slot's position, slots naming one device being
  one record; typed by the mapping's family ``class`` (the nearest class
  every such family's class descends from); its ``names`` carry the export's
  ``CommonNames`` slot and its ``attributes`` the export's ``DeviceList`` row
  and ``ElementList`` slot, each only when the family states one per device.
* ``imported/mml/channels.yaml``: one channel per address, ``on`` the one
  device that binds it, or, bound by several, naming each in ``endpoint_of``
  and ``on`` none; its ``role`` follows the field's direction
  (:func:`~osprey.facility.layers.mml.mapping.field_roles`) and a setpoint
  that reads back through its family's ``Monitor`` names that device's
  ``Monitor`` address as its ``pair``.
* ``imported/mml/groups.yaml``: one group per family, id the family's mapped
  token; same-named families of several exports are one group whose members
  are the union of theirs.
* ``imported/mml/models.yaml``: one ``pyat`` model per imported system, named
  by the mapping. A transport line (``state.is_transport`` of the export's
  ``<stem>.model.json``, else ``MachineType: Transport`` in its AD) runs
  ``solve: single_pass`` from the ``twiss_in`` of the first element of its
  deck that carries ``TwissData``; a transport line without one stops the
  import (``mapping-undecided``). A model whose export saved a deck names it
  as its ``deck`` and carries the ``wiring`` the wiring pass derives
  (:mod:`~osprey.facility.layers.mml.wiring`), one record per wired address.
* ``imported/mml/decks/<model>.json``: the deck each such model is served,
  its wired elements renamed so every wiring record's element is in it
  exactly once (:mod:`~osprey.facility.layers.mml.decks`).
* ``imported/mml/<model>.response.json``: the export's ``<stem>.response.json``
  copied byte for byte.

Families are read through the reviewer's judgment answers, so every list is
the grain the mapping settled on. Every record file is sorted by id, so an
unchanged import rewrites the same bytes. The layer directory holds what the
latest import carried: a response export or a deck of a model this import
does not carry is removed.

Nothing is written until every record and every model's wiring is derived, so
a stop leaves the layer directory as it was.

The export loaders pull numpy and scipy and the deck reader pulls pyAT, so
each is imported inside the function that needs it.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from osprey.facility.layers.mml.decks import DECKS_DIR, write_deck
from osprey.facility.layers.mml.identity import common_class, device_ids, endpoints
from osprey.facility.layers.mml.mapping import (
    FieldAnswer,
    ImportStop,
    Mapping,
    OwnerMap,
    field_roles,
    load_or_draft,
)

if TYPE_CHECKING:  # the export services stay out of the import graph
    from osprey.services.mml.family import FamilyView, FieldView
    from osprey.services.mml.mapping.schema import Mapping as ExportAnswers

__all__ = [
    "ENGINE",
    "LAYER_DIR",
    "MODEL_SUFFIX",
    "TRANSPORT",
    "Exports",
    "import_mml",
    "read_exports",
    "write_records",
]

#: Where the layer's sources live, relative to ``data/facility/``.
LAYER_DIR = "imported/mml"

#: The engine every imported model runs on: an MML deck is a pyAT lattice.
ENGINE = "pyat"

#: File-name suffix of the Middle Layer model's own answers beside an export.
MODEL_SUFFIX = ".model.json"

#: The AD ``MachineType`` of a transport line.
TRANSPORT = "Transport"

#: File-name suffix of a model's copied response export under the layer.
_RESPONSE_SUFFIX = ".response.json"

#: The ``twiss_in`` keys pyAT reads, each from its ``TwissData`` spelling.
_TWISS_KEYS: tuple[tuple[str, str], ...] = (
    ("beta", "beta"),
    ("alpha", "alpha"),
    ("dispersion", "Dispersion"),
    ("closed_orbit", "ClosedOrbit"),
)

_SETPOINT = "setpoint"


@dataclass
class Exports:
    """What an import read from its exports, keyed by raw system token.

    Attributes:
        ao: The merged export, ``{system: {family: body}}`` plus bookkeeping.
        ad: Each system's accelerator data.
        va: Each system's sampled model facts (``<stem>.va.json``).
        responses: Each system's response export (``<stem>.response.json``).
        states: Each system's ``state`` block of ``<stem>.model.json``.
        decks: Each system's saved deck (``<stem>.lattice.mat``).
    """

    ao: dict[str, Any]
    ad: dict[str, Any] = field(default_factory=dict)
    va: dict[str, Any] = field(default_factory=dict)
    responses: dict[str, Path] = field(default_factory=dict)
    states: dict[str, dict[str, Any]] = field(default_factory=dict)
    decks: dict[str, Path] = field(default_factory=dict)

    @property
    def systems(self) -> list[str]:
        """The imported systems, in import order."""
        from osprey.services.mml.systems import IMPORT_ORDER_KEY

        return list(self.ao[IMPORT_ORDER_KEY])


def read_exports(paths: Sequence[Path]) -> Exports:
    """Read MML exports and the siblings filed beside each.

    Each path is an export's AO (``<stem>.ao.json``, any family-keyed JSON, or
    a ``.mat`` export). The siblings of an export carrying exactly one system
    are found by its stem: ``.va.json``, ``.response.json``, ``.model.json``
    and ``.lattice.mat``.

    Args:
        paths: The exports, in the order their systems are imported.

    Returns:
        The merged export and every sibling found.

    Raises:
        click.ClickException: An input cannot be read or names no system.
    """
    import click

    from osprey.services.mml.loaders.json_any import (
        AO_SUFFIX,
        LATTICE_SUFFIX,
        RESPONSE_SUFFIX,
        VA_SUFFIX,
        load_json,
        load_sibling,
        paired_sibling,
    )
    from osprey.services.mml.systems import input_systems, merge_inputs, resolve_system

    pairs = []
    for path in paths:
        if path.suffix.lower() == ".mat":
            from osprey.services.mml.loaders.mat import load_mat

            loaded = load_mat(path)
            if loaded.lattice is not None:
                raise click.UsageError(
                    f"Cannot import {path}: it is a deck; name the export's AO file."
                )
        else:
            loaded = load_json(path)
        pairs.append((loaded, resolve_system(loaded, None)))
    ao, ad = merge_inputs(pairs)
    exports = Exports(ao=ao, ad=ad)

    for loaded, token in pairs:
        carried = input_systems(loaded, token)
        if len(carried) != 1 or not loaded.source.name.endswith(AO_SUFFIX):
            continue
        (system,) = carried
        va = paired_sibling(loaded.source, VA_SUFFIX)
        if va is not None:
            exports.va[system] = load_sibling(va)
        response = paired_sibling(loaded.source, RESPONSE_SUFFIX)
        if response is not None:
            exports.responses[system] = response
        model = paired_sibling(loaded.source, MODEL_SUFFIX)
        if model is not None:
            state = load_sibling(model).get("state")
            if isinstance(state, dict):
                exports.states[system] = state
        deck = paired_sibling(loaded.source, LATTICE_SUFFIX)
        if deck is not None:
            exports.decks[system] = deck
    return exports


def import_mml(paths: Sequence[Path], facility_dir: Path) -> list[Path]:
    """Import MML exports as the mml layer's record sources.

    Args:
        paths: The exports' AO files.
        facility_dir: The ``data/facility`` directory.

    Returns:
        Every file written, in write order.

    Raises:
        ImportStop: ``mapping-draft`` when the mapping was absent and a draft
            was written; ``mapping-undecided`` while a slot it needs is
            undecided; ``export-invalid`` or ``reference-missing`` from the
            wiring pass.
        MappingError: The mapping has the wrong structure.
    """
    exports = read_exports(paths)
    mapping = load_or_draft(facility_dir, exports.ao, exports.ad or None, exports.va or None)
    return write_records(exports, mapping, facility_dir)


# -- records ------------------------------------------------------------------


def write_records(exports: Exports, mapping: Mapping, facility_dir: Path) -> list[Path]:
    """Write the layer's record files and decks and copy each response export.

    Args:
        exports: What :func:`read_exports` read.
        mapping: The layer's mapping, every slot decided.
        facility_dir: The ``data/facility`` directory.

    Returns:
        Every file written, in write order.

    Raises:
        ImportStop: ``mapping-undecided`` for a system or family the mapping
            does not name, or a transport line without initial twiss;
            ``export-invalid`` or ``reference-missing`` from the wiring pass.
        MappingError: The mapping answers a cavity voltage for a deck that
            holds its cavity.
    """
    from osprey.services.mml.judgments import judged_family_views

    roles = field_roles(mapping)
    answers = _export_answers(mapping)
    views: list[FamilyView] = []
    systems: dict[str, str] = {}
    models: list[dict[str, Any]] = []
    judged: dict[str, dict[str, FamilyView]] = {}
    for system in exports.systems:
        systems[system] = _model_name(mapping, system)
        judged[system] = {}
        for view in judged_family_views(system, exports.ao[system], answers):
            judged[system][view.raw_name] = view
            if view.channel_count == 0:
                continue
            family = mapping.families.get(view.raw_name)
            if family is None:
                raise ImportStop(
                    "mapping-undecided", [f"families.{view.raw_name}: the export carries it"]
                )
            if family.channels == 0:
                continue
            views.append(view)
        models.append(_model(exports, system, systems[system]))

    ids = device_ids(views, systems)
    owners = endpoints(views, ids)
    branches = {name: branch.parent for name, branch in mapping.branches.items()}
    devices: dict[str, dict[str, Any]] = {}
    channels: dict[str, dict[str, Any]] = {}
    groups: dict[str, dict[str, Any]] = {}
    for view, slot_ids in zip(views, ids, strict=True):
        family = mapping.families[view.raw_name]
        for device_id, device in zip(slot_ids, _devices(view, family.class_), strict=True):
            _add_device(devices, device_id, device, branches)
        _channels(view, slot_ids, owners, mapping, roles, channels)
        _group(groups, mapping.mapped(view.raw_name), mapping, view.raw_name, slot_ids)

    family_ids: dict[str, dict[str, list[str]]] = {system: {} for system in exports.systems}
    for view, slots in zip(views, ids, strict=True):
        family_ids[view.system][view.raw_name] = slots
    served = _wire(exports, mapping, models, judged, family_ids, owners, channels, answers)

    layer = facility_dir / LAYER_DIR
    layer.mkdir(parents=True, exist_ok=True)
    written = [
        _dump(layer / "devices.yaml", _sorted(devices.values(), "id")),
        _dump(layer / "channels.yaml", _sorted(channels.values(), "id")),
        _dump(layer / "groups.yaml", _sorted(groups.values(), "id")),
        _dump(layer / "models.yaml", _sorted(models, "name")),
    ]
    decks = [write_deck(deck, facility_dir, name) for name, deck in served]
    written.extend(decks)
    for stale in sorted((facility_dir / DECKS_DIR).glob("*.json")):
        if stale not in decks:
            stale.unlink()
    written.extend(_copy_responses(exports, mapping, layer))
    return written


def _model_name(mapping: Mapping, system: str) -> str:
    model = mapping.models.get(system)
    if model is None:
        raise ImportStop("mapping-undecided", [f"models.{system}: the export carries it"])
    return model.name


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _devices(view: FamilyView, klass: str | None) -> Iterable[dict[str, Any]]:
    """The fields of each device of one family, in device order."""
    names = view.aligned("CommonNames")
    elements = view.aligned("ElementList")
    rows = view.device_rows
    for index in range(view.n_devices):
        device: dict[str, Any] = {}
        if klass is not None:
            device["class"] = klass
        name = _text(names[index]) if names is not None else None
        if name is not None:
            device["names"] = [name]
        attributes: dict[str, Any] = {}
        if rows is not None:
            row = [_integer(part) for part in rows[index]]
            if None not in row:
                attributes["DeviceList"] = row
        element = _integer(elements[index]) if elements is not None else None
        if element is not None:
            attributes["ElementList"] = element
        if attributes:
            device["attributes"] = attributes
        yield device


def _add_device(
    devices: dict[str, dict[str, Any]],
    device_id: str,
    device: dict[str, Any],
    branches: dict[str, str],
) -> None:
    """Record one slot's device, or fold it into the device its id already names.

    The first slot's names and attributes stand; the class is the nearest one
    every slot's family class descends from, absent when they share none.
    """
    found = devices.get(device_id)
    if found is None:
        devices[device_id] = {"id": device_id, **device}
        return
    klass = common_class(found.get("class"), device.get("class"), branches)
    merged: dict[str, Any] = {"id": device_id}
    if klass is not None:
        merged["class"] = klass
    for key in ("names", "attributes"):
        value = found.get(key, device.get(key))
        if value is not None:
            merged[key] = value
    devices[device_id] = merged


def _field_scalar(fld: FieldView, key: str, index: int, n_devices: int) -> str | None:
    """The scalar ``fld.body[key]`` gives one device: its slot of a per-device list, or itself."""
    value = fld.body.get(key)
    if isinstance(value, (list, tuple)):
        return _text(value[index]) if len(value) == n_devices else None
    return _text(value)


def _channels(
    view: FamilyView,
    ids: list[str],
    owners: dict[str, list[str]],
    mapping: Mapping,
    roles: dict[str, Any],
    channels: dict[str, dict[str, Any]],
) -> None:
    """Add one channel per address of one family that no earlier family wrote.

    An address one device binds is ``on`` it; an address several devices bind
    names each in ``endpoint_of`` and belongs to no device.
    """
    family = mapping.families[view.raw_name]
    for fld in view.fields.values():
        role = roles.get(f"{view.raw_name}.{fld.name}")
        described = family.fields.get(fld.name)
        description = described.description if described is not None else None
        paired = view.fields.get(role.pair) if role is not None and role.pair else None
        for key in fld.keys:
            pairs = paired.slots(key) if paired is not None and key in paired.keys else []
            for index, slot in enumerate(fld.slots(key)[: view.n_devices]):
                address = _text(slot)
                if address is None or address in channels:
                    continue
                channel: dict[str, Any] = {"id": address}
                bound = owners.get(address, [ids[index]])
                if len(bound) > 1:
                    channel["endpoint_of"] = sorted(bound)
                else:
                    channel["on"] = {"device": bound[0]}
                if role is not None:
                    channel["role"] = role.role
                    pair = _text(pairs[index]) if index < len(pairs) else None
                    if role.role == _SETPOINT and pair is not None and pair != address:
                        channel["pair"] = pair
                unit = _field_scalar(fld, "HWUnits", index, view.n_devices)
                if unit is not None:
                    channel["unit"] = unit
                if description is not None:
                    channel["description"] = description
                channels[address] = channel


def _group(
    groups: dict[str, dict[str, Any]], token: str, mapping: Mapping, raw: str, ids: list[str]
) -> None:
    """Record one family as a group, or add its devices to the group of its name."""
    group = groups.get(token)
    if group is None:
        family = mapping.families[raw]
        group = {"id": token}
        if family.description is not None:
            group["description"] = family.description
        if family.aliases:
            group["names"] = list(family.aliases)
        group["members"] = []
        signals = {
            name: fld.description
            for name, fld in sorted(family.fields.items())
            if fld.description is not None
        }
        if signals:
            group["signals"] = signals
        groups[token] = group
    group["members"] = sorted({*group["members"], *ids})


# -- models -------------------------------------------------------------------


def _is_transport(exports: Exports, system: str) -> bool:
    """Whether a system is a transport line: its model state says so, else its AD does."""
    stated = exports.states.get(system, {}).get("is_transport")
    if stated is not None:
        return bool(stated)
    ad = exports.ad.get(system)
    return isinstance(ad, dict) and _text(ad.get("MachineType")) == TRANSPORT


def _model(exports: Exports, system: str, name: str) -> dict[str, Any]:
    model: dict[str, Any] = {"name": name, "engine": ENGINE}
    if _is_transport(exports, system):
        deck = exports.decks.get(system)
        twiss = _twiss_in(deck) if deck is not None else None
        if twiss is None:
            raise ImportStop("mapping-undecided", [f"{name}: transport line without initial twiss"])
        model["settings"] = {ENGINE: {"solve": "single_pass", "twiss_in": twiss}}
    return model


def _wire(
    exports: Exports,
    mapping: Mapping,
    models: list[dict[str, Any]],
    judged: dict[str, dict[str, FamilyView]],
    family_ids: dict[str, dict[str, list[str]]],
    owners: dict[str, list[str]],
    channels: dict[str, dict[str, Any]],
    answers: ExportAnswers,
) -> list[tuple[str, Any]]:
    """Wire every imported model and name its deck and wiring on its entry.

    Each entry of ``models`` that has a deck gains ``deck``, the path its
    served deck is written to relative to ``data/facility/``, and ``wiring``,
    in the order ``name, engine, deck, settings, wiring``; its ``settings``
    are left as they are. Nothing is written here.

    Returns:
        Each wired model's name and the deck it is served, in import order.

    Raises:
        ImportStop: ``export-invalid``, ``reference-missing`` or
            ``mapping-undecided``, as the wiring pass states them.
    """
    from osprey.facility.layers.mml.wiring import wire_model

    served: list[tuple[str, Any]] = []
    for entry, system in zip(models, exports.systems, strict=True):
        model = mapping.models[system]
        wired = wire_model(
            model,
            exports.decks.get(system),
            exports.va.get(system),
            exports.ad.get(system),
            judged[system],
            family_ids[system],
            owners,
            channels,
            answers,
        )
        if wired is None:
            continue
        settings = entry.pop("settings", None)
        entry["deck"] = f"{DECKS_DIR}/{model.name}.json"
        if settings is not None:
            entry["settings"] = settings
        entry["wiring"] = wired.records
        served.append((model.name, wired.deck))
    return served


def _twiss_field(data: Any, name: str) -> Any:
    if isinstance(data, dict):
        return data.get(name)
    names = getattr(getattr(data, "dtype", None), "names", None)
    if names is not None and name in names:
        return data[name]
    return getattr(data, name, None)


def _twiss_in(deck: Path) -> dict[str, list[float]] | None:
    """pyAT's ``twiss_in`` from the first deck element carrying ``TwissData``."""
    import numpy as np

    from osprey.services.mml.loaders.mat import load_lattice

    for element in load_lattice(deck):
        data = getattr(element, "TwissData", None)
        if data is None:
            continue
        twiss: dict[str, list[float]] = {}
        for key, stated in _TWISS_KEYS:
            value = _twiss_field(data, stated)
            if value is not None:
                twiss[key] = [float(item) for item in np.ravel(np.asarray(value, dtype=float))]
        return twiss
    return None


def _copy_responses(exports: Exports, mapping: Mapping, layer: Path) -> list[Path]:
    """Copy each response export under its model's name and drop the rest."""
    copied: list[Path] = []
    for system in exports.systems:
        source = exports.responses.get(system)
        if source is None:
            continue
        target = layer / f"{_model_name(mapping, system)}{_RESPONSE_SUFFIX}"
        shutil.copyfile(source, target)
        copied.append(target)
    for stale in sorted(layer.glob(f"*{_RESPONSE_SUFFIX}")):
        if stale not in copied:
            stale.unlink()
    return copied


# -- judgments ----------------------------------------------------------------


@dataclass(frozen=True)
class _Answers:
    """The reviewer's judgment answers in the shape the export services read."""

    judgments: dict[str, Any]


def _export_answers(mapping: Mapping) -> ExportAnswers:
    """The mapping's judgment answers, spelled in the export services' own types.

    The services apply answers by type, so each ``{field: <name>}`` row answer
    and each owner map is re-spelled in their classes; the literal answers
    carry over as they are.
    """
    from osprey.services.mml.mapping import schema

    def row(answer: Any) -> Any:
        return schema.FieldAnswer(answer.name) if isinstance(answer, FieldAnswer) else answer

    judgments = {
        raw: schema.FamilyJudgments(
            rows_beyond={
                name: {signal: row(answer) for signal, answer in answers.items()}
                for name, answers in found.rows_beyond.items()
            },
            unbound_devices=dict(found.unbound_devices),
            shared_pvs=(
                schema.OwnerMap(dict(found.shared_pvs.owners))
                if isinstance(found.shared_pvs, OwnerMap)
                else found.shared_pvs
            ),
            shared_pvs_present=found.shared_pvs_present,
        )
        for raw, found in mapping.judgments.items()
    }
    return cast("ExportAnswers", _Answers(judgments))


# -- writing ------------------------------------------------------------------


def _sorted(rows: Iterable[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: str(row[key]))


def _dump(path: Path, rows: list[dict[str, Any]]) -> Path:
    import yaml

    text = yaml.safe_dump(
        rows, sort_keys=False, default_flow_style=False, allow_unicode=True, width=100
    )
    path.write_text(text, encoding="utf-8")
    return path
