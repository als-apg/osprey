"""The mml layer's importer: MML exports written as record sources under ``imported/mml/``.

:func:`import_mml` reads one or more exports, holds each export's deck to the
lattice fingerprint the export states (:func:`check_decks`; a deck of another
lattice stops the import, an energy that differs alone is said), loads the
layer's mapping (or writes a draft and stops, :func:`~osprey.facility.layers.mml.mapping.load_or_draft`)
checks it against the exports
(:func:`~osprey.facility.layers.mml.mapping.check_mapping`; a mapping with a
problem stops the import before anything is written) and writes the layer's
record files. It writes sources, never a view: the
build merges them with every other layer.

What is written, relative to ``data/facility/``:

* ``imported/mml/devices.yaml``: one device per device of a family that
  carries a channel, its id and the slots it stands for resolved by the
  identity model (:mod:`~osprey.facility.layers.mml.identity`): the export's
  ``CommonNames`` entry at the slot's position, or what the mapping's
  ``devices`` answer says where the export names none, slots naming one
  device being one record; typed by the mapping's family ``class`` (the nearest class
  every such family's class descends from); its ``label`` is the export's
  ``CommonNames`` slot, and it carries no ``attributes``.
* ``imported/mml/channels.yaml``: one channel per address, ``on`` the one
  device that binds it, or, bound by several, naming each in ``endpoint_of``
  and ``on`` none; its ``role`` follows the field's direction
  (:func:`~osprey.facility.layers.mml.mapping.field_roles`): an address any
  ``write`` field names is a setpoint, whichever field named it first. A
  setpoint that reads back through its family's ``Monitor`` names that
  device's ``Monitor`` address as its ``pair``, unless that address reads back
  several setpoints of the field, when it pairs none. A field the mapping
  gives a ``signal`` role writes it on each of its channels. A channel's
  description is ``<device>: <field sentence>``, the device named by its
  label, else its id, and a shared endpoint by each of its devices.
* ``imported/mml/groups.yaml``: one group per physical family, id the family's
  mapped token; same-named families of several exports are one group whose
  members are the union of theirs. Families whose devices are the same set are
  one group (plane- and field-split twins), named by their common stem; its
  ``description`` is the family's sentence.
* the model's ``tune`` addresses, from the mapping's ``tune`` block: each a
  readback channel, the record a family field wrote where one did; a waveform
  block's address is ``value_type: waveform`` with ``shape`` its number of
  planes.
* ``imported/mml/rows.json``: the device each export ``DeviceList`` row
  became, per model and family (:mod:`~osprey.facility.layers.mml.rows`),
  which the response check reads.
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

The wiring pass's lines are printed once the records are written: one per
family wired without a stated hardware nominal. After the records,
:func:`import_mml` seeds the authored files that do not
exist yet (``limits.yaml``, ``seeds.yaml``, ``measurement/<model>.yaml``,
``classes.yaml`` and ``identity.yaml``; :mod:`~osprey.facility.layers.mml.seed`)
and prints what the seeding reports.

The export loaders pull numpy and scipy and the deck reader pulls pyAT, so
each is imported inside the function that needs it.
"""

from __future__ import annotations

import shutil
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import click

from osprey.facility.layers.mml.decks import DECKS_DIR, write_deck
from osprey.facility.layers.mml.identity import (
    common_class,
    device_ids,
    endpoints,
    split_identities,
)
from osprey.facility.layers.mml.mapping import (
    LAYER_DIR,
    MAPPING_FILE,
    READBACK_ROLE,
    SETPOINT_ROLE,
    ImportStop,
    Mapping,
    Problem,
    TuneBlock,
    _text,
    check_mapping,
    exported_number,
    field_roles,
    load_or_draft,
)
from osprey.facility.layers.mml.response_check import RESPONSE_SUFFIX
from osprey.facility.layers.mml.rows import write_rows

if TYPE_CHECKING:  # the export services stay out of the import graph
    from osprey.facility.layers.mml.family import FamilyView, FieldView

__all__ = [
    "ENGINE",
    "LAYER_DIR",
    "MODEL_SUFFIX",
    "TRANSPORT",
    "Exports",
    "MappingProblems",
    "check_decks",
    "import_mml",
    "read_exports",
    "write_records",
]

#: The engine every imported model runs on: an MML deck is a pyAT lattice.
ENGINE = "pyat"

#: File-name suffix of the Middle Layer model's own answers beside an export.
MODEL_SUFFIX = ".model.json"

#: The AD ``MachineType`` of a transport line.
TRANSPORT = "Transport"

#: The ``twiss_in`` keys pyAT reads, each from its ``TwissData`` spelling.
_TWISS_KEYS: tuple[tuple[str, str], ...] = (
    ("beta", "beta"),
    ("alpha", "alpha"),
    ("dispersion", "Dispersion"),
    ("closed_orbit", "ClosedOrbit"),
)

#: The largest exported ``Tolerance`` that states none: a machine-epsilon placeholder.
_NO_TOLERANCE = 1e-12

#: The keys of a channel record, in the order they are written.
_CHANNEL_KEYS = (
    "id",
    "endpoint_of",
    "on",
    "role",
    "pair",
    "tolerance",
    "signal",
    "value_type",
    "shape",
    "unit",
    "description",
)


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
        from osprey.facility.layers.mml.systems import IMPORT_ORDER_KEY

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

    from osprey.facility.layers.mml.loaders.json_any import (
        AO_SUFFIX,
        LATTICE_SUFFIX,
        RESPONSE_SUFFIX,
        VA_SUFFIX,
        load_json,
        load_sibling,
        paired_sibling,
    )
    from osprey.facility.layers.mml.systems import input_systems, merge_inputs, resolve_system

    pairs = []
    for path in paths:
        if path.suffix.lower() == ".mat":
            from osprey.facility.layers.mml.loaders.mat import load_mat

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


#: What every line about a deck that disagrees with its export ends in.
_PAIRING_REMEDY = "import the lattice the export was sampled from, or export again over this one"


def check_decks(exports: Exports) -> list[str]:
    """Hold each export's deck to the lattice fingerprint the export states.

    A deck and the export sampled over it travel as separate files, and the
    fingerprint in the export's ``lattice`` block is all that pairs them. Each
    paired deck is read and its fingerprint recomputed. An export with no
    sampled facts states none, and its deck is not judged.

    Args:
        exports: What :func:`read_exports` read.

    Returns:
        One line per thing to say about a deck the import still files: a deck
        whose export states no usable fingerprint, and a deck whose energy
        alone differs from the one its export states.

    Raises:
        ImportStop: ``export-invalid``, one line per deck that is another
            lattice than its export states, naming the first fact the two
            disagree on.
    """
    from osprey.facility.layers.mml.fingerprint import check_fingerprint, lattice_fingerprint
    from osprey.facility.layers.mml.loaders.mat import load_lattice

    refused: list[str] = []
    lines: list[str] = []
    for system in exports.systems:
        deck = exports.decks.get(system)
        facts = exports.va.get(system)
        stated = facts.get("lattice") if isinstance(facts, dict) else None
        if deck is None or stated is None:
            continue
        if not isinstance(stated, dict) or "refused" in stated:
            reason = stated.get("refused") if isinstance(stated, dict) else stated
            lines.append(
                f"import mml: deck unchecked: {system}: the export states no lattice "
                f"fingerprint ({reason}); the deck {deck.name} is filed unchecked"
            )
            continue
        said = [
            (
                mismatch.refuses,
                f"{system}: the deck {deck.name} holds {mismatch.field} {mismatch.actual!r} "
                f"and the export states {mismatch.expected!r}; {_PAIRING_REMEDY}",
            )
            for mismatch in check_fingerprint(stated, lattice_fingerprint(load_lattice(deck)))
        ]
        refusing = [line for refuses, line in said if refuses]
        if refusing:
            refused.append(refusing[0])
        else:
            lines.extend(f"import mml: deck energy: {line}" for _, line in said)
    if refused:
        raise ImportStop("export-invalid", refused)
    return lines


class MappingProblems(click.ClickException):
    """The mapping fails its check: one line per problem, then how many there are.

    Args:
        path: The mapping file.
        problems: What the check refused, in its order.
    """

    exit_code = 1

    def __init__(self, path: Path, problems: Sequence[Problem]) -> None:
        self.path = path
        self.problems = tuple(problems)
        count = len(self.problems)
        noun = "problem" if count == 1 else "problems"
        lines = [str(problem) for problem in self.problems]
        lines.append(f"{count} {noun} in {path}; fix each and check again.")
        super().__init__("\n".join(lines))

    def show(self, file: IO[Any] | None = None) -> None:
        """Write the lines alone, with no ``Error: `` prefix.

        Args:
            file: The stream to write to; stderr when omitted.
        """
        if file is None:
            click.echo(self.format_message(), err=True, color=self.show_color)
        else:
            click.echo(self.format_message(), file=file, color=self.show_color)


def import_mml(paths: Sequence[Path], facility_dir: Path) -> list[Path]:
    """Import MML exports as the mml layer's record sources and seed the authored files.

    Args:
        paths: The exports' AO files.
        facility_dir: The ``data/facility`` directory.

    Returns:
        Every file written, in write order: the layer's records, then each
        authored file seeded because it did not exist.

    Raises:
        ImportStop: ``export-invalid`` for a deck that is another lattice than
            its export states; ``mapping-draft`` when the mapping was absent
            and a draft was written; ``mapping-undecided`` while a slot it
            needs is undecided; ``mapping-invalid`` for a ``devices`` answer
            the export cannot carry, or where a wired channel would name a
            device another family binds it as, before anything is written;
            ``export-invalid`` or ``reference-missing`` from the wiring pass.
        MappingProblems: The mapping fails its check; nothing is written.
        MappingError: The mapping has the wrong structure.
    """
    from osprey.facility.layers.mml.seed import seed_once

    exports = read_exports(paths)
    for line in check_decks(exports):
        click.echo(line)
    mapping = load_or_draft(facility_dir, exports.ao, exports.ad or None, exports.va or None)
    problems = check_mapping(mapping, exports.ao)
    if problems:
        raise MappingProblems(facility_dir / MAPPING_FILE, problems)
    judged, views = _carried(exports, mapping)
    written, untoleranced = _write_records(exports, mapping, facility_dir, judged, views)
    if untoleranced:
        click.echo(f"{untoleranced} setpoint devices export no usable `Setpoint.Tolerance`")
    seeded = seed_once(exports, mapping, facility_dir, views)
    for line in seeded.lines:
        click.echo(line)
    return [*written, *seeded.written]


# -- records ------------------------------------------------------------------


def write_records(exports: Exports, mapping: Mapping, facility_dir: Path) -> list[Path]:
    """Write the layer's record files and decks and copy each response export.

    Args:
        exports: What :func:`read_exports` read.
        mapping: The layer's mapping, every slot decided and its check
            against these exports clean.
        facility_dir: The ``data/facility`` directory.

    Returns:
        Every file written, in write order.

    Raises:
        ImportStop: ``mapping-undecided`` for a family whose devices the
            mapping leaves unidentified or a transport line without initial
            twiss; ``mapping-invalid`` for a ``devices`` answer the export
            cannot carry, or where a wired channel would name a device
            another family binds it as, before anything is written;
            ``export-invalid`` or ``reference-missing`` from the wiring pass.
        MappingProblems: The mapping leaves out a system or a family these
            exports carry; nothing is written.
        MappingError: The mapping answers a cavity voltage for a deck that
            holds its cavity.
    """
    unnamed = _unnamed(exports, mapping)
    if unnamed:
        raise MappingProblems(facility_dir / MAPPING_FILE, unnamed)
    judged, views = _carried(exports, mapping)
    return _write_records(exports, mapping, facility_dir, judged, views)[0]


def _unnamed(exports: Exports, mapping: Mapping) -> list[Problem]:
    """The exported systems and families the mapping leaves out, as its check words them."""
    from osprey.facility.layers.mml.family import family_views

    problems = [
        Problem("models", f"leaves out the exported system {system}")
        for system in exports.systems
        if system not in mapping.models
    ]
    families = {
        view.raw_name
        for system in exports.systems
        for view in family_views(system, exports.ao[system])
    }
    problems.extend(
        Problem("families", f"leaves out the exported family {family}")
        for family in sorted(families - set(mapping.families))
    )
    return problems


def _write_records(
    exports: Exports,
    mapping: Mapping,
    facility_dir: Path,
    judged: dict[str, dict[str, FamilyView]],
    views: list[FamilyView],
) -> tuple[list[Path], int]:
    """Write the layer's files from the families as :func:`_carried` judged them.

    Returns:
        Every file written, and how many setpoints carry no ``tolerance``
        because their export states no usable ``Tolerance``.
    """
    roles = field_roles(mapping)
    systems = {system: _model_name(mapping, system) for system in exports.systems}
    models = [_model(exports, system, systems[system]) for system in exports.systems]

    answered = {
        raw: family.devices
        for raw, family in mapping.families.items()
        if family.devices is not None
    }
    ids = device_ids(views, systems, answered)
    owners = endpoints(views, ids)
    branches = {name: branch.parent for name, branch in mapping.branches.items()}
    devices: dict[str, dict[str, Any]] = {}
    channels: dict[str, dict[str, Any]] = {}
    groups: dict[str, dict[str, Any]] = {}
    untoleranced: set[str] = set()
    export_rows: list[dict[str, Any]] = []
    for view, slot_ids in zip(views, ids, strict=True):
        family = mapping.families[view.raw_name]
        for device_id, device in zip(slot_ids, _devices(view, family.class_), strict=True):
            _add_device(devices, device_id, device, branches)
    for view, slot_ids in zip(views, ids, strict=True):
        for device_id, row in zip(slot_ids, _export_rows(view), strict=True):
            if row is not None:
                export_rows.append(
                    {
                        "model": systems[view.system],
                        "family": view.raw_name,
                        "device_list": row,
                        "device": device_id,
                    }
                )
        _channels(view, slot_ids, owners, mapping, roles, channels, untoleranced, devices)
        _group(groups, mapping.mapped(view.raw_name), mapping, view.raw_name, slot_ids)
    groups = _physical_groups(groups)

    for system in exports.systems:
        _tune_channels(mapping.models[system].tune, channels)

    family_ids: dict[str, dict[str, list[str]]] = {system: {} for system in exports.systems}
    for view, slots in zip(views, ids, strict=True):
        family_ids[view.system][view.raw_name] = slots
    served, lines, wired = _wire(exports, mapping, models, judged, family_ids, owners, channels)
    split = split_identities(views, ids, wired)
    if split:
        raise ImportStop("mapping-invalid", split)

    layer = facility_dir / LAYER_DIR
    layer.mkdir(parents=True, exist_ok=True)
    written = [
        _dump(layer / "devices.yaml", _sorted(devices.values(), "id")),
        _dump(layer / "channels.yaml", _sorted(channels.values(), "id")),
        _dump(layer / "groups.yaml", _sorted(groups.values(), "id")),
        _dump(layer / "models.yaml", _sorted(models, "name")),
        write_rows(layer, export_rows),
    ]
    decks = [write_deck(deck, facility_dir, name) for name, deck in served]
    written.extend(decks)
    for stale in sorted((facility_dir / DECKS_DIR).glob("*.json")):
        if stale not in decks:
            stale.unlink()
    written.extend(_copy_responses(exports, mapping, layer))
    for line in lines:
        click.echo(line)
    untoleranced -= {address for address, channel in channels.items() if "tolerance" in channel}
    return written, len(untoleranced)


def _carried(
    exports: Exports, mapping: Mapping
) -> tuple[dict[str, dict[str, FamilyView]], list[FamilyView]]:
    """Every family as the reviewer judged it, and those that carry channel records.

    Returns:
        ``{system: {raw family: view}}`` of every judged family, and the views
        of the families the mapping gives channels, in import order.
    """
    from osprey.facility.layers.mml.judgments import judged_family_views

    judged: dict[str, dict[str, FamilyView]] = {}
    views: list[FamilyView] = []
    for system in exports.systems:
        judged[system] = {}
        for view in judged_family_views(system, exports.ao[system], mapping):
            judged[system][view.raw_name] = view
            if view.channel_count == 0:
                continue
            if mapping.families[view.raw_name].channels == 0:
                continue
            views.append(view)
    return judged, views


def _model_name(mapping: Mapping, system: str) -> str:
    return mapping.models[system].name


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _devices(view: FamilyView, klass: str | None) -> Iterable[dict[str, Any]]:
    """The fields of each device of one family, in device order.

    A device's ``label`` is its ``CommonNames`` entry.
    """
    names = view.aligned("CommonNames")
    for index in range(view.n_devices):
        device: dict[str, Any] = {}
        if klass is not None:
            device["class"] = klass
        name = _text(names[index]) if names is not None else None
        if name is not None:
            device["label"] = name
        yield device


def _export_rows(view: FamilyView) -> list[list[int] | None]:
    """Each device slot's export ``DeviceList`` row, ``None`` where it is not all integers."""
    rows = view.device_rows
    out: list[list[int] | None] = []
    for index in range(view.n_devices):
        if rows is None:
            out.append(None)
            continue
        row = [_integer(part) for part in rows[index]]
        out.append(None if None in row else [part for part in row if part is not None])
    return out


def _add_device(
    devices: dict[str, dict[str, Any]],
    device_id: str,
    device: dict[str, Any],
    branches: dict[str, str],
) -> None:
    """Record one slot's device, or fold it into the device its id already names.

    The first slot's label stands; the class is the nearest one every slot's
    family class descends from, absent when they share none.
    """
    found = devices.get(device_id)
    if found is None:
        devices[device_id] = {"id": device_id, **device}
        return
    klass = common_class(found.get("class"), device.get("class"), branches)
    merged: dict[str, Any] = {"id": device_id}
    if klass is not None:
        merged["class"] = klass
    for key in ("label",):
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


def _field_number(fld: FieldView, key: str, index: int, n_devices: int) -> float | None:
    """The finite number ``fld.body[key]`` gives one device: its slot of a per-device list, or itself."""
    value = fld.body.get(key)
    if isinstance(value, (list, tuple)):
        value = value[index] if len(value) == n_devices else None
    return exported_number(value)


def _tolerance(
    fld: FieldView, index: int, n_devices: int, unit: str | None
) -> dict[str, float] | None:
    """A setpoint's ``tolerance`` from its write field's ``Tolerance``, or ``None``.

    Only a finite tolerance above 1e-12 in a stated unit is one: an export's
    ``Inf`` or a machine-epsilon placeholder says the field has none.
    """
    value = _field_number(fld, "Tolerance", index, n_devices)
    if unit is None or value is None or value <= _NO_TOLERANCE:
        return None
    return {"absolute": value}


def _channels(
    view: FamilyView,
    ids: list[str],
    owners: dict[str, list[str]],
    mapping: Mapping,
    roles: dict[str, Any],
    channels: dict[str, dict[str, Any]],
    untoleranced: set[str],
    devices: dict[str, dict[str, Any]],
) -> None:
    """Add one channel per address of one family that no earlier family wrote.

    An address one device binds is ``on`` it; an address several devices bind
    names each in ``endpoint_of`` and belongs to no device. An address a
    ``write`` field names is a setpoint: when an earlier field wrote its
    channel as anything else, the channel takes the setpoint role, its pair
    and the write field's unit, tolerance and description, and keeps the rest.
    A setpoint takes its ``tolerance`` from the write field's ``Tolerance``
    in the field's unit; one whose export states no usable tolerance is added
    to ``untoleranced``. A channel is described as ``<owner>: <field
    sentence>`` (:func:`_described`).
    """
    family = mapping.families[view.raw_name]
    for fld in view.fields.values():
        role = roles.get(f"{view.raw_name}.{fld.name}")
        described = family.fields.get(fld.name)
        description = described.description if described is not None else None
        signal = described.signal if described is not None else None
        paired = view.fields.get(role.pair) if role is not None and role.pair else None
        shared = _shared_pairs(fld, paired, view.n_devices)
        for key in fld.keys:
            pairs = paired.slots(key) if paired is not None and key in paired.keys else []
            for index, slot in enumerate(fld.slots(key)[: view.n_devices]):
                address = _text(slot)
                if address is None:
                    continue
                writes = role is not None and role.role == SETPOINT_ROLE
                pair = _text(pairs[index]) if index < len(pairs) else None
                if pair in shared:
                    pair = None
                found = channels.get(address)
                if found is not None:
                    if writes and found.get("role") != SETPOINT_ROLE:
                        found["role"] = SETPOINT_ROLE
                        if pair is not None and pair != address:
                            found["pair"] = pair
                        if signal is not None:
                            found["signal"] = signal
                        unit = _field_scalar(fld, "HWUnits", index, view.n_devices)
                        if unit is not None:
                            found["unit"] = unit
                        tolerance = _tolerance(fld, index, view.n_devices, unit)
                        if tolerance is not None:
                            found["tolerance"] = tolerance
                        else:
                            untoleranced.add(address)
                        if description is not None:
                            found["description"] = _described(found, description, devices)
                        channels[address] = {
                            key: found[key] for key in _CHANNEL_KEYS if key in found
                        }
                    continue
                channel: dict[str, Any] = {"id": address}
                bound = owners.get(address, [ids[index]])
                if len(bound) > 1:
                    channel["endpoint_of"] = sorted(bound)
                else:
                    channel["on"] = {"device": bound[0]}
                if role is not None:
                    channel["role"] = role.role
                    if writes and pair is not None and pair != address:
                        channel["pair"] = pair
                if signal is not None:
                    channel["signal"] = signal
                unit = _field_scalar(fld, "HWUnits", index, view.n_devices)
                if unit is not None:
                    channel["unit"] = unit
                if writes:
                    tolerance = _tolerance(fld, index, view.n_devices, unit)
                    if tolerance is not None:
                        channel["tolerance"] = tolerance
                    else:
                        untoleranced.add(address)
                if description is not None:
                    channel["description"] = _described(channel, description, devices)
                channels[address] = {key: channel[key] for key in _CHANNEL_KEYS if key in channel}


def _described(channel: dict[str, Any], sentence: str, devices: dict[str, dict[str, Any]]) -> str:
    """``<owner>: <sentence>``, the owner the ``on`` device or each ``endpoint_of`` device.

    A device is named by its ``label``, else its id.
    """
    on = channel.get("on") or {}
    owned = [on["device"]] if on.get("device") else list(channel.get("endpoint_of") or [])
    owner = ", ".join(str(devices.get(d, {}).get("label") or d) for d in owned)
    return f"{owner}: {sentence}" if owner else sentence


def _shared_pairs(fld: FieldView, paired: FieldView | None, devices: int) -> set[str]:
    """The addresses of ``paired`` that read back more than one address of ``fld``."""
    if paired is None:
        return set()
    read: dict[str, set[str]] = {}
    for key in fld.keys:
        if key not in paired.keys:
            continue
        for slot, back in zip(fld.slots(key)[:devices], paired.slots(key), strict=False):
            address, pair = _text(slot), _text(back)
            if address is not None and pair is not None and pair != address:
                read.setdefault(pair, set()).add(address)
    return {pair for pair, addresses in read.items() if len(addresses) > 1}


def _tune_channels(tune: TuneBlock | None, channels: dict[str, dict[str, Any]]) -> None:
    """Make each tune address a readback channel, a waveform block's a waveform of its planes.

    A family field that wrote the address keeps its record, typed here; an
    address no field wrote is written as a readback.
    """
    if tune is None:
        return
    for address in dict.fromkeys(tune.planes.values()):
        channel = channels.setdefault(address, {"id": address, "role": READBACK_ROLE})
        if tune.address is not None:
            channel["value_type"] = "waveform"
            channel["shape"] = [len(tune.planes)]
        channels[address] = {key: channel[key] for key in _CHANNEL_KEYS if key in channel}


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
        groups[token] = group
    group["members"] = sorted({*group["members"], *ids})


def _stem(tokens: Sequence[str]) -> str:
    """The tokens' longest common prefix, less its trailing non-alphanumerics."""
    prefix = tokens[0]
    for token in tokens[1:]:
        while not token.startswith(prefix):
            prefix = prefix[:-1]
    while prefix and not prefix[-1].isalnum():
        prefix = prefix[:-1]
    return prefix


def _physical_groups(groups: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Fold the groups whose members are the same set into one group per physical family.

    A fold is named by its tokens' common stem when that has two characters
    or more, is no other group's id and is no other fold's stem, else by its
    first token in sorted order. Its ``names`` are the union of the twins' names and tokens, sorted;
    its ``description`` the twins' distinct descriptions in token order,
    joined by one space.
    """
    folds: dict[tuple[str, ...], list[str]] = {}
    for token in sorted(groups):
        folds.setdefault(tuple(groups[token]["members"]), []).append(token)
    stems = Counter(_stem(tokens) for tokens in folds.values() if len(tokens) > 1)
    out: dict[str, dict[str, Any]] = {}
    for members, tokens in folds.items():
        if len(tokens) == 1:
            out[tokens[0]] = groups[tokens[0]]
            continue
        stem = _stem(tokens)
        outside = set(groups) - set(tokens)
        unique = len(stem) >= 2 and stem not in outside and stems[stem] == 1
        group_id = stem if unique else tokens[0]
        group: dict[str, Any] = {"id": group_id}
        descriptions = [groups[t]["description"] for t in tokens if groups[t].get("description")]
        if descriptions:
            group["description"] = " ".join(dict.fromkeys(descriptions))
        names = {*tokens, *(name for t in tokens for name in groups[t].get("names", []))}
        group["names"] = sorted(names)
        group["members"] = list(members)
        out[group_id] = group
    return out


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
) -> tuple[list[tuple[str, Any]], list[str], dict[str, str]]:
    """Wire every imported model and name its deck and wiring on its entry.

    Each entry of ``models`` that has a deck gains ``deck``, the path its
    served deck is written to relative to ``data/facility/``, and ``wiring``,
    in the order ``name, engine, deck, settings, wiring``; its ``settings``
    are left as they are. Nothing is written here.

    Returns:
        Each wired model's name and the deck it is served, in import order;
        the lines the wiring pass prints; and the raw family whose record
        names an element on each address, over every model.

    Raises:
        ImportStop: ``export-invalid``, ``reference-missing`` or
            ``mapping-undecided``, as the wiring pass states them.
    """
    from osprey.facility.layers.mml.wiring import wire_model

    served: list[tuple[str, Any]] = []
    lines: list[str] = []
    wired_by: dict[str, str] = {}
    for entry, system in zip(models, exports.systems, strict=True):
        model = mapping.models[system]
        wired = wire_model(
            model,
            exports.decks.get(system),
            exports.ao.get(system),
            exports.va.get(system),
            exports.ad.get(system),
            judged[system],
            family_ids[system],
            owners,
            channels,
            mapping,
        )
        if wired is None:
            continue
        settings = entry.pop("settings", None)
        entry["deck"] = f"{DECKS_DIR}/{model.name}.json"
        if settings is not None:
            entry["settings"] = settings
        entry["wiring"] = wired.records
        served.append((model.name, wired.deck))
        lines.extend(wired.lines)
        wired_by.update(wired.families)
    return served, lines, wired_by


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

    from osprey.facility.layers.mml.loaders.mat import load_lattice

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
        target = layer / f"{_model_name(mapping, system)}{RESPONSE_SUFFIX}"
        shutil.copyfile(source, target)
        copied.append(target)
    for stale in sorted(layer.glob(f"*{RESPONSE_SUFFIX}")):
        if stale not in copied:
            stale.unlink()
    return copied


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
