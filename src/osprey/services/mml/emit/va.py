"""The virtual-accelerator artifacts ``osprey mml emit`` writes.

The lane turns one system's export -- its lattice, its calibrations and its
nominals -- into the files a served virtual accelerator boots from. This module
holds the emitters; the CLI splices them in and writes what they return through
``write_if_changed``, so every emitter is a function of its inputs alone.
:func:`emit_lattice` can write its file itself, for a caller that wants the
deck on disk the moment it is rendered, and returns the text like the others so
a caller that must know every refusal first writes it in its own turn.

Two of those files are the machine's starting state:

* ``data/simulation/machine.json`` -- one entry per address the export states a
  nominal for. The served model boots its records from these values, and the
  mock connector's simulation engine synthesizes the rest of the tree from
  them, so a nominal reaches both paradigms through one file. A family the
  model does not drive still seeds its channels, marked
  ``nominal_seed_only`` so a reader can tell a value the model maintains from
  one that only starts a channel off somewhere plausible.
* ``data/machine_state_channels.json`` -- the machine-state view's channel
  list: the monitor-only families with a single device, which is what a
  machine-wide reading looks like in the Middle Layer grain (beam current,
  tunes, lifetime), as opposed to a per-device monitor family like the beam
  position monitors, whose thousand addresses are not a machine-state view.

A third says what the model does when one of those channels is written:

* ``data/simulation/va_bindings.json`` -- one binding per coupled address: the
  element it drives and the attribute a write lands in, the calibration it
  converts through, where and how its readback is served, and the device's
  nominal -- and, for a monitor, the calibration the facility states for that
  device's reading, carried for a later readout-error model and applied by
  nothing. :func:`emit_bindings` renders it through the served schema's own
  ``dump_bindings``, so this lane can only write a document the service would
  load, and it names the digest of the lattice the same run wrote.

The starting-state documents carry :data:`PROVENANCE_KEY` as their FIRST key,
holding ``ctx.provenance_string``, and so does the bindings document. That stamp is the whole of how the emit pre-flight
tells a file this lane wrote from one a person hand-authored: an unstamped
file at either path is refused with the ``rm`` line that names it, rather than
overwritten. Both readers already tolerate it -- ``load_machine_json_channels``
reads only ``channels``, and the machine-state loader keys off each entry's
``label`` member, so a top-level string is not mistaken for a channel.

Rules the two emitters share:

* Families are read through the caller's judged views, so a reviewer's answers
  are already applied to every device list a value is aligned against. The
  export's per-device rows are checked against the judged device count and a
  disagreement is refused by name: rows that have shifted against their family
  would seed the right values at the wrong addresses, silently.
* A nominal returned in physics units is refused rather than seeded -- the
  channels are hardware -- as is a non-finite one, and so is a second value
  landing on an address another device already seeded differently. All three
  are returned as :class:`NominalSeed` rows carrying the refusal, for the
  report to list; the first value on an address stands.
* Addresses are sorted and every value is written verbatim, so a re-emit of an
  unchanged export produces byte-identical text.

Pure: stdlib plus the family view, the mapping schema, the emit context, the
element table the addressing pass built, and the bindings document's own schema
and curve evaluators -- which are what decide whether a readback collapses to
the value that was written, on the same arithmetic the served write path uses.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.services.mml.canonical import write_if_changed
from osprey.services.mml.emit.context import LATTICE_ARTIFACT, EmitContext
from osprey.services.mml.family import FamilyView, FieldView
from osprey.services.mml.mapping.schema import Mapping, VAFamily
from osprey.services.mml.va.elements import ElementBinding
from osprey.services.virtual_accelerator.bindings import (
    ENERGY_SCALINGS,
    PROVENANCE_KEY,
    READOUT_KEYS,
    Binding,
    BindingsDocument,
    Calibration,
    Linear,
    Readout,
    Slice,
    Table,
    dump_bindings,
)
from osprey.services.virtual_accelerator.lattice.calibration import to_hardware, to_physics

__all__ = [
    "IDENTITY_TOLERANCE",
    "PHYSICS_UNITS",
    "PROVENANCE_KEY",
    "SEED_ONLY_KEY",
    "ChannelBand",
    "NominalSeed",
    "emit_bindings",
    "emit_channel_limits",
    "emit_lattice",
    "emit_machine",
    "emit_state_channels",
]

#: The channel key marking a value the model does not maintain.
SEED_ONLY_KEY = "nominal_seed_only"

#: The units word an export uses for a nominal it could not read in hardware
#: units; such a nominal seeds nothing.
PHYSICS_UNITS = "physics"

#: What the machine-state document says about itself, above its entries.
_STATE_COMMENT = (
    "The machine-state view's channels: every monitor-only single-device family "
    "of the Middle Layer export. Written by `osprey mml emit`; edit the export "
    "or the mapping rather than this file."
)

#: Top-level key of the machine document's prose.
_COMMENT_KEY = "_comment"


@dataclass(frozen=True)
class NominalSeed:
    """One nominal the machine emitter read, seeded or refused.

    Attributes:
        system: The raw system token the family sits under.
        family: The family's raw name.
        field: The field the nominal was read through (``Setpoint`` for a
            family that has one, ``Monitor`` for a monitor-only family).
        address: The channel the value seeds, or ``""`` when the refusal
            covers the whole nominal rather than one device.
        device: The 1-based device position the value belongs to.
        value: The seeded value, or ``None`` when it was refused.
        at_type: The AT field the export read the nominal from, for the report
            to name beside a family the model does not drive.
        synthetic: Whether the Middle Layer fabricated the value rather than
            reading it from the deck.
        seed_only: Whether the model leaves the value where it starts.
        refused: Why nothing was seeded, or ``None`` when it was.
    """

    system: str
    family: str
    field: str
    address: str
    device: int
    value: float | None
    at_type: str
    synthetic: bool
    seed_only: bool
    refused: str | None = None


def emit_machine(
    verdicts: dict[tuple[str, str], VAFamily],
    views: Iterable[FamilyView],
    judged_va: dict[tuple[str, str], dict],
    mapping: Mapping,
    ctx: EmitContext,
) -> tuple[str, tuple[NominalSeed, ...]]:
    """Build ``machine.json``: one seed per nominal the export states.

    Args:
        verdicts: What the virtual accelerator does with each family, keyed by
            ``(raw system, raw family)``. A family with no entry, or one whose
            verdict is anything but ``couple``, is seeded ``nominal_seed_only``:
            the model drives nothing there, so its value only starts the
            channel off.
        views: The judged family views to seed from, in any order.
        judged_va: Each family's ``va.json`` block in the judged device order,
            keyed by ``(raw system, raw family)``; a family with no block
            states no nominal and is skipped. Judged with the family's device
            count, so a reviewer's answer moves the block's rows with the
            family's.
        mapping: The parsed mapping, read for the prose each channel is
            described with.
        ctx: The provenance of this emit run.

    Returns:
        The document text, and one :class:`NominalSeed` per nominal read --
        the seeded ones and the refused ones alike, in family and device order,
        for ``VA-REPORT.md`` to list.

    Raises:
        ValueError: A family's exported rows do not line up with its judged
            device count. Seeding past that would put values on the wrong
            addresses with nothing to see.
    """
    channels: dict[str, dict[str, Any]] = {}
    seeds: list[NominalSeed] = []
    for view in views:
        block = judged_va.get((view.system, view.raw_name))
        if not isinstance(block, dict):
            continue
        nominals = block.get("nominals")
        if not isinstance(nominals, dict):
            continue
        _require_devices(view, block)
        verdict = verdicts.get((view.system, view.raw_name))
        seed_only = verdict is None or verdict.verdict != "couple"
        for field_name in sorted(nominals):
            nominal = nominals[field_name]
            field_view = view.fields.get(field_name)
            if not isinstance(nominal, dict) or field_view is None:
                continue
            seeds.extend(
                _field_seeds(view, field_view, field_name, nominal, seed_only, mapping, channels)
            )

    document: dict[str, Any] = {PROVENANCE_KEY: ctx.provenance_string}
    document["name"] = _machine_name(mapping)
    document["description"] = (
        "Starting values of the Middle Layer export, one per channel it states "
        "a nominal for. Written by `osprey mml emit`; edit the export or the "
        "mapping rather than this file."
    )
    document["channels"] = {address: channels[address] for address in sorted(channels)}
    return _text(document), tuple(seeds)


def emit_state_channels(views: Iterable[FamilyView], mapping: Mapping, ctx: EmitContext) -> str:
    """Build ``machine_state_channels.json``: the machine-state view's channels.

    A family is listed when it has exactly one device and none of its fields is
    written -- the Middle Layer grain of a machine-wide reading. Direction comes
    from the mapping, which the emit lane has already refused to run without, so
    a family whose fields are all undecided is left out: nothing states that it
    is a reading rather than a knob.

    Args:
        views: The judged family views to draw from, in any order.
        mapping: The parsed mapping, read for each family's prose, its mapped
            token and the direction of its fields.
        ctx: The provenance of this emit run.

    Returns:
        The document text: the provenance stamp, one line of prose, then one
        ``{label, group}`` entry per address, sorted by address.
    """
    entries: dict[str, dict[str, str]] = {}
    for view in views:
        if view.n_devices != 1 or not _is_monitor_only(view, mapping):
            continue
        group = _group(view, mapping)
        for field_name in sorted(view.fields):
            field_view = view.fields[field_name]
            label = _state_label(view, field_name, mapping)
            for key in field_view.keys:
                for slot in field_view.slots(key):
                    address = _address(slot)
                    if address is not None:
                        entries.setdefault(address, {"label": label, "group": group})

    document: dict[str, Any] = {PROVENANCE_KEY: ctx.provenance_string}
    document[_COMMENT_KEY] = _STATE_COMMENT
    for address in sorted(entries):
        document[address] = entries[address]
    return _text(document)


def _field_seeds(
    view: FamilyView,
    field_view: FieldView,
    field_name: str,
    nominal: dict,
    seed_only: bool,
    mapping: Mapping,
    channels: dict[str, dict[str, Any]],
) -> list[NominalSeed]:
    """Seed one field's channels from its nominal, returning what it stated."""
    at_type = _word(nominal.get("at_type"))
    synthetic = bool(nominal.get("synthetic"))
    units_word = _word(nominal.get("units"))
    values = _values(view, field_name, nominal.get("values"))

    def row(address: str, device: int, value: float | None, refused: str | None) -> NominalSeed:
        return NominalSeed(
            system=view.system,
            family=view.raw_name,
            field=field_name,
            address=address,
            device=device,
            value=value,
            at_type=at_type,
            synthetic=synthetic,
            seed_only=seed_only,
            refused=refused,
        )

    if units_word.lower() == PHYSICS_UNITS:
        return [row("", 0, None, f"nominal read in {units_word} units, not hardware")]

    hw_units = _word(field_view.body.get("HWUnits"))
    description = _channel_description(view, field_name, mapping)
    seeds: list[NominalSeed] = []
    for index, raw_value in enumerate(values):
        addresses = _addresses(field_view, index)
        value = _number(raw_value)
        if not addresses:
            continue
        if value is None:
            seeds.append(row(addresses[0], index + 1, None, f"non-finite nominal {raw_value!r}"))
            continue
        for address in addresses:
            seeded = channels.get(address)
            if seeded is not None and seeded["value"] != value:
                seeds.append(
                    row(
                        address,
                        index + 1,
                        None,
                        f"address already seeded with {seeded['value']!r}; "
                        "two devices state one channel",
                    )
                )
                continue
            entry: dict[str, Any] = {"value": value}
            if hw_units:
                entry["units"] = hw_units
            if description:
                entry["description"] = description
            if seed_only:
                entry[SEED_ONLY_KEY] = True
            channels.setdefault(address, entry)
            seeds.append(row(address, index + 1, value, None))
    return seeds


def _values(view: FamilyView, field_name: str, values: Any) -> list[Any]:
    """Return one nominal value per device, a scalar broadcast to all of them.

    Raises:
        ValueError: The list states a different number of devices than the
            judged family has.
    """
    if not isinstance(values, (list, tuple)):
        return [values] * view.n_devices
    if len(values) != view.n_devices:
        raise ValueError(
            f"{view.system}.{view.raw_name}: the export states {len(values)} "
            f"{field_name} nominal(s) for a family judged to have "
            f"{view.n_devices} device(s); the rows no longer line up with the "
            "devices they were sampled for"
        )
    return list(values)


def _require_devices(view: FamilyView, block: dict) -> None:
    """Refuse a ``va.json`` block whose device rows are not the family's.

    Raises:
        ValueError: The block's ``device_list`` states a different device
            count than the judged family view.
    """
    devices = _block_devices(block.get("device_list"))
    if devices and devices != view.n_devices:
        raise ValueError(
            f"{view.system}.{view.raw_name}: its virtual-accelerator block "
            f"states {devices} device(s) and the judged family has "
            f"{view.n_devices}; judge the block with devices=<the family's "
            "device count> so both documents read one device order"
        )


def _block_devices(device_list: Any) -> int:
    """Return how many devices a ``device_list`` states, 0 when it states none.

    The Nx2 rule is ``FamilyView.device_rows``': a flat pair of numbers is one
    device, rows of two are one device each.
    """
    if not isinstance(device_list, (list, tuple)) or not device_list:
        return 0
    if all(isinstance(row, (list, tuple)) and len(row) == 2 for row in device_list):
        return len(device_list)
    if len(device_list) == 2 and all(_number(item) is not None for item in device_list):
        return 1
    return len(device_list)


def _addresses(field_view: FieldView, index: int) -> list[str]:
    """Return the addresses one device sits at, across the field's channel keys."""
    found: list[str] = []
    for key in field_view.keys:
        slots = field_view.slots(key)
        address = _address(slots[index]) if index < len(slots) else None
        if address is not None and address not in found:
            found.append(address)
    return found


def _address(slot: Any) -> str | None:
    """Return a channel slot as an address, or ``None`` when it names none."""
    return slot.strip() if isinstance(slot, str) and slot.strip() else None


def _number(value: Any) -> float | None:
    """Return a finite number as a float, or ``None``.

    Normalisation spells a non-finite number as a string (``"NaN"``, ``"Inf"``),
    so a string is never a value here, only a refusal.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _word(value: Any) -> str:
    """Return a non-blank string stripped, else ``""``."""
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _is_monitor_only(view: FamilyView, mapping: Mapping) -> bool:
    """Whether the mapping states this family is read and never written."""
    directions = [
        mapping.directions[key].direction
        for key in (f"{view.raw_name}.{field}" for field in view.fields)
        if key in mapping.directions
    ]
    return "read" in directions and "write" not in directions


def _group(view: FamilyView, mapping: Mapping) -> str:
    """Return the machine-state group a family's channels are shown under."""
    system = mapping.systems.get(view.system)
    return (system.name if system is not None else view.system).lower()


def _state_label(view: FamilyView, field_name: str, mapping: Mapping) -> str:
    """Return the label the machine-state view shows one field's channel under."""
    family = mapping.families.get(view.raw_name)
    prose = _word(None if family is None else family.description)
    name = view.raw_name if family is None else mapping.mapped(view.raw_name)
    label = prose or name
    return f"{label} ({field_name})" if len(view.fields) > 1 else label


def _channel_description(view: FamilyView, field_name: str, mapping: Mapping) -> str:
    """Return the prose a seeded channel is described with, or ``""``."""
    family = mapping.families.get(view.raw_name)
    if family is None:
        return ""
    field = family.fields.get(field_name)
    return _word(None if field is None else field.description) or _word(family.description)


def _machine_name(mapping: Mapping) -> str:
    """Return what the machine is called, from the facility block."""
    return _word(mapping.facility.title) or _word(mapping.facility.token) or "machine"


def _text(document: dict) -> str:
    """Render one document, in the two-space JSON both readers are given."""
    return json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


# --- the saved lattice -------------------------------------------------------
#
# The third file the lane writes is the deck itself. It is not built here: pyAT
# renders its own document and this only takes one key out of it, so a deck the
# served model loads is byte-for-byte what pyAT would have saved.

#: The key pyAT stamps its own version into. It is dropped from the emitted
#: deck -- see :func:`emit_lattice`.
AT_VERSION_KEY = "at_version"


def emit_lattice(ring: Any, out_path: Path, ctx: EmitContext, *, write: bool = True) -> str:
    """Render the served deck as pyAT JSON, recording its digest in ``ctx``.

    The export ships its lattice as a ``.mat``, which is never byte-stable --
    scipy stamps the wall clock into the header -- so a re-export of an
    unchanged deck would rewrite the file and invalidate every digest taken
    over it. The emitted copy is therefore JSON, written through
    ``write_if_changed``: identical inputs leave the file untouched, down to
    its modification time. The ``.mat`` the export was read from is not
    touched at all, and neither is ``data/lattice/``, which the Lattice
    Dashboard keeps reading.

    One key is dropped on the way through: pyAT writes its own version into
    ``at_version``, which would rewrite the whole file on every upgrade of a
    library that changed nothing about this deck. The version is recorded on
    ``ctx`` instead, beside the digest the bindings document stamps as its
    ``lattice_sha256`` -- both under :data:`~...emit.context.LATTICE_ARTIFACT`,
    whatever this file is named.

    One element is dropped too, and not by this function: pyAT saves a
    ``RingParam`` marker as the lattice's properties rather than as an element,
    so the deck read back from this file is the deck that went in minus those
    markers, its energy and periodicity intact. The element the Middle Layer
    calls position *n* is therefore not this file's *n*th element -- which is
    why the bindings document binds by element NAME, and why those names are
    made unique before the deck is saved.

    Args:
        ring: The deck to save, renamed by the addressing pass -- an
            ``at.Lattice``, since the element names written here are the names
            the bindings document binds by.
        out_path: Where to write it, ``data/simulation/lattice.json`` in a
            served tree. Missing parent directories are created.
        ctx: The provenance of this emit run; the rendered text's sha256 and
            the pyAT version are recorded on it.
        write: Whether to put the text on disk. The digest is taken over the
            text, never over the file, so a caller that must know every refusal
            before it touches the tree renders with ``write=False`` and writes
            the returned text itself.

    Returns:
        The rendered text, so a caller can write or report on it without
        re-reading it.

    Raises:
        ValueError: The deck holds a value JSON cannot state -- a non-finite
            number in an element attribute. Nothing is written.
        OSError: The file could not be written; ``out_path`` is left as it was.
    """
    import at

    document = json.loads(_saved_lattice(ring, at))
    at_version = document.pop(AT_VERSION_KEY, "")
    text = _lattice_text(document, out_path)
    if write:
        write_if_changed(out_path, text)
    ctx.record_artifact(LATTICE_ARTIFACT, text, writer=str(at_version))
    return text


def _saved_lattice(ring: Any, at: Any) -> str:
    """Return pyAT's own JSON rendering of a ring, without going through disk.

    ``at.save_json`` prints to stdout when it is given no file name, which is
    the only way to reach its encoder -- the one thing that knows how each
    element class spells itself -- without writing a file first.
    """
    rendered = io.StringIO()
    with contextlib.redirect_stdout(rendered):
        at.save_json(ring)
    return rendered.getvalue()


def _lattice_text(document: dict, out_path: Path) -> str:
    """Render the stripped deck, naming the file when a value cannot be stated.

    Raises:
        ValueError: An element attribute holds a non-finite number, which no
            reader of this file could parse back.
    """
    try:
        return _text(document)
    except ValueError as exc:
        raise ValueError(
            f"{out_path}: the deck holds a value JSON cannot state ({exc}); "
            "the lattice the export saved is not one a served model could read back"
        ) from exc


# -- va_bindings.json ---------------------------------------------------------


#: How close the round trip has to come back to the hardware value it started
#: from -- the calibration applied, then the facility's own inverse -- for the
#: readback to be served as the written value itself rather than through that
#: inverse. The served write path collapses on the same number.
IDENTITY_TOLERANCE = 1e-9

#: The kinds carrying a physics strength the ring rescales with its rigidity.
#: Every other kind is written ``none`` whatever the export's word was: the
#: export states the scaling of a conversion, and a cavity or an energy knob
#: has no strength for it to apply to.
_RIGID_KINDS: frozenset[str] = frozenset({"strength", "kick"})

#: The kinds whose value is shared out over a split device's pieces rather than
#: written whole to each of them, which is what the slice weights say.
_SHARED_KINDS: frozenset[str] = frozenset({"kick"})


def emit_bindings(
    verdicts: dict[tuple[str, str], VAFamily],
    views: Iterable[FamilyView],
    element_bindings: dict[str, tuple[ElementBinding, ...]],
    judged_va: dict[tuple[str, str], dict],
    ctx: EmitContext,
    *,
    system: str,
    energy_gev: float,
) -> str:
    """Build ``va_bindings.json``: what each coupled address does to the deck.

    One binding per device of every coupled family: the element it drives and
    the attribute a write lands in, the calibration that converts the write to
    physics, where and how the readback is served, and the device's nominal.
    The energy knob is the exception -- it binds no element, so it is written
    once, on the first device row's address, with the table that maps that
    setpoint to beam energy.

    How a readback is served is decided per device, not per family:
    ``same_as_setpoint`` when one address carries both, ``identity`` when the
    exported inverse returns the written value to within
    :data:`IDENTITY_TOLERANCE` over the calibration's own points, and
    ``inverse`` otherwise.

    A monitor binding also carries the facility's own calibration of that
    device's reading, where the export states one, and nothing else does: it
    is a reading that a readout-error model perturbs, and a driven family's
    exported gains belong to its setpoint, which the calibration states
    already. Nothing on the served path reads those numbers today, and
    carrying them moves no reading -- their gain and offset are inside the
    conversions beside them, so applying them here would count them twice.

    A calibration is never inverted to stand in for a
    missing inverse -- the two directions are sampled data in their own right
    -- so a family serving a readback it has no inverse for is refused.

    Args:
        verdicts: What the virtual accelerator does with each family, keyed by
            ``(raw system, raw family)`` as :func:`emit_machine` reads them. A
            family that does not couple binds nothing.
        views: The judged family views, in any order; the addresses and the
            device count are read off them, so a reviewer's answers are
            already applied to both.
        element_bindings: What each family's devices drive, keyed by raw
            family: :attr:`~osprey.services.mml.va.elements.Addressing.bindings`
            of the same system, whose element names are the emitted deck's.
        judged_va: Each family's ``va.json`` block in the judged device order,
            keyed by ``(raw system, raw family)``.
        ctx: The provenance of this emit run, carrying the digest of the
            lattice this run wrote, which the bindings name.
        system: The raw system token the bindings describe.
        energy_gev: The beam energy the deck was built at, as the export's
            lattice block states it.

    Returns:
        The document text, exactly as :func:`load_bindings` reads it back.

    Raises:
        ValueError: The lattice has not been written yet, a family's rows no
            longer line up with its judged devices, a coupled family states
            none of the facts a binding is made of, or the assembled document
            breaks a schema rule -- two families writing one element field and
            two devices claiming one address among them.
    """
    digest = ctx.lattice_sha256
    if not digest:
        raise ValueError(
            "the bindings name the digest of the lattice they were derived "
            f"against and this run has written none: emit {LATTICE_ARTIFACT} "
            "before the bindings"
        )
    grain = {view.raw_name: view for view in views if view.system == system}
    bindings: list[Binding] = []
    for family in sorted(name for held, name in verdicts if held == system):
        verdict = verdicts[(system, family)]
        view = grain.get(family)
        block = judged_va.get((system, family))
        if verdict.verdict != "couple" or verdict.kind is None or view is None:
            continue
        if not isinstance(block, dict):
            continue
        _require_devices(view, block)
        bindings.extend(
            _family_bindings(family, verdict, view, block, element_bindings.get(family, ()))
        )

    document = BindingsDocument(
        system=system,
        energy_gev=float(energy_gev),
        lattice_sha256=digest,
        bindings=tuple(bindings),
        provenance=ctx.provenance_string,
    )
    return dump_bindings(document)


def _family_bindings(
    family: str,
    verdict: VAFamily,
    view: FamilyView,
    block: dict,
    entries: tuple[ElementBinding, ...],
) -> list[Binding]:
    """Bind one coupled family: one entry per device that drives an element."""
    kind = str(verdict.kind)
    if kind == "energy":
        return [_energy_binding(family, verdict, view, block)]

    written = "Monitor" if kind == "monitor" else "Setpoint"
    field_view = view.fields.get(written)
    if field_view is None:
        raise ValueError(
            f"{view.system}.{family}: couples as {kind} and states no {written} "
            "channel for a binding to be keyed by"
        )
    source = verdict.nominal_source or written
    devices = view.n_devices
    elements = _element_binding_by_device(view, entries)
    rows: list[Binding] = []
    for device in range(devices):
        element = elements.get(device)
        address = _first_address(field_view, device)
        if element is None or address is None:
            continue
        where = f"{view.system}.{family} device {device + 1}"
        calibration = _curve_for_device(block.get(written), "calibration", device, devices, where)
        if calibration is None:
            raise ValueError(
                f"{where}: couples as {kind} and its {written} block states no "
                "calibration; nothing says what the written value is worth in physics"
            )
        inverse = _curve_for_device(block.get("Monitor"), "monitor_inverse", device, devices, where)
        nominal = _nominal_for(block, source, device, devices, where)
        if nominal is None and kind != "monitor":
            raise ValueError(
                f"{where}: couples as {kind} and states no hardware nominal for "
                f"{address}; a driven device starts the model somewhere"
            )
        served = _first_address(view.fields.get("Monitor"), device)
        rule, readback_address, applied = _readback_rule(
            kind, address, served, calibration, inverse, nominal, where
        )
        # Only a reading is calibrated on its way out of the control system,
        # so only a monitor's readout is carried; a driven family's exported
        # gains calibrate its setpoint, which the calibration already states.
        reading = block.get("Monitor") if kind == "monitor" else None
        readout = _readout_for_device(reading, device, devices, where)
        rows.append(
            Binding(
                kind=kind,
                family=family,
                setpoint_address=address,
                readback_address=readback_address,
                readback=rule,
                element=element.element,
                attribute=element.attribute,
                index=element.index,
                slices=_binding_slices(kind, element),
                owner=element.owner,
                calibration=calibration,
                monitor_inverse=applied,
                nominal=nominal,
                energy_scaling=_energy_scaling(kind, block),
                energy_table=None,
                readout=readout,
            )
        )
    return rows


def _energy_binding(family: str, verdict: VAFamily, view: FamilyView, block: dict) -> Binding:
    """Bind the energy knob: one address, no element, and the ramp's table.

    The knob is a property of the ring, so one device drives it however many
    the family lists -- the row the export sampled ``bend2gev`` at, which is
    its first. The rows with addresses of their own are left unbound and
    named in ``PROFILE.md``, not silently bound to the same table.
    """
    table = _sampled_curve(_energy_points(block, "grid"), _energy_points(block, "values"))
    if table is None:
        raise ValueError(
            f"{view.system}.{family}: couples as the energy knob and states no "
            "energy_table; that table is the whole of how a setpoint becomes an energy"
        )
    device = _energy_device(view, block)
    where = f"{view.system}.{family} device {device + 1}"
    address = _first_address(view.fields.get("Setpoint"), device)
    if address is None:
        raise ValueError(f"{where}: couples as the energy knob and names no Setpoint channel")
    nominal = _nominal_for(
        block, verdict.nominal_source or "Setpoint", device, view.n_devices, where
    )
    if nominal is None:
        raise ValueError(f"{where}: couples as the energy knob and states no hardware nominal")
    served = _first_address(view.fields.get("Monitor"), device)
    same = served is None or served == address
    return Binding(
        kind="energy",
        family=family,
        setpoint_address=address,
        readback_address=None if same else served,
        readback="same_as_setpoint" if same else "identity",
        element=None,
        attribute=None,
        index=None,
        slices=(),
        owner=None,
        calibration=None,
        monitor_inverse=None,
        nominal=nominal,
        energy_scaling="none",
        energy_table=table,
    )


def _readback_rule(
    kind: str,
    address: str,
    served: str | None,
    calibration: Calibration,
    inverse: Calibration | None,
    nominal: float | None,
    where: str,
) -> tuple[str, str | None, Calibration | None]:
    """Decide how one device's readback is served, and through which curve."""
    if kind == "monitor":
        if inverse is None:
            raise ValueError(
                f"{where}: reads a physics quantity and states no monitor_inverse; "
                "the facility's own inverse is the only way back to hardware units"
            )
        return "inverse", None, inverse
    if served is None or served == address:
        return "same_as_setpoint", None, None
    if inverse is None:
        raise ValueError(
            f"{where}: serves its readback on {served} and states no monitor_inverse; "
            "a calibration is never inverted to stand in for one"
        )
    if _serves_identity(calibration, inverse, nominal):
        return "identity", served, None
    return "inverse", served, inverse


def _serves_identity(
    calibration: Calibration, inverse: Calibration | None, nominal: float | None
) -> bool:
    """Whether the inverse returns the hardware value the calibration was given.

    The round trip is the served write path's own -- the calibration, then the
    inverse -- read at the points the calibration was sampled over, so the two
    agree here exactly where the served readback would.
    """
    if inverse is None:
        return False
    points = _round_trip_grid(calibration, nominal)
    returned = to_hardware(inverse, to_physics(calibration, points))
    return all(
        abs(float(value) - point) <= IDENTITY_TOLERANCE
        for value, point in zip(returned, points, strict=False)
    )


def _round_trip_grid(calibration: Calibration, nominal: float | None) -> tuple[float, ...]:
    """The hardware points the round trip is read at.

    A sampled calibration is read at its own points. A straight line is read
    at two: the origin and the device's nominal, which is enough to separate a
    slope that does not come back from an offset that does not.
    """
    if isinstance(calibration, Table):
        return calibration.grid
    return (0.0, float(nominal) if nominal else 1.0)


def _binding_slices(kind: str, element: ElementBinding) -> tuple[Slice, ...]:
    """The elements one device writes, each with the share it carries there.

    A split device's pieces are weighted only where the value is shared out
    over them: a kick is divided by its piece count, as the Middle Layer's own
    read of a split corrector is, while a strength or a monitor reading
    describes each piece whole.
    """
    pieces = len(element.slices)
    weight = 1.0 / pieces if kind in _SHARED_KINDS and pieces > 1 else 1.0
    return tuple(Slice(element=piece.element, weight=weight) for piece in element.slices)


def _energy_scaling(kind: str, block: dict) -> str:
    """Whether a binding's physics value moves with the beam rigidity."""
    if kind not in _RIGID_KINDS:
        return "none"
    setpoint = block.get("Setpoint")
    word = _word(setpoint.get("energy_scaling")) if isinstance(setpoint, dict) else ""
    return word if word in ENERGY_SCALINGS else "none"


def _element_binding_by_device(
    view: FamilyView, entries: tuple[ElementBinding, ...]
) -> dict[int, ElementBinding]:
    """Pair each element row with the device position of the judged family.

    The rows are matched by the device they name rather than by their order,
    so a family whose middle device states no element keeps every other row
    against the device it was addressed for.
    """
    positions: dict[tuple[int, ...], int] = {}
    for index, row in enumerate(view.device_rows or []):
        key = _device_key(row)
        if key is not None:
            positions.setdefault(key, index)
    found: dict[int, ElementBinding] = {}
    for entry in entries:
        index = positions.get(tuple(entry.device))
        if index is not None:
            found.setdefault(index, entry)
    return found


def _energy_device(view: FamilyView, block: dict) -> int:
    """The device position the energy table was sampled at, 0 when unstated."""
    table = block.get("energy_table")
    wanted = _device_key(table.get("device_row")) if isinstance(table, dict) else None
    for index, row in enumerate(view.device_rows or []):
        if wanted is not None and _device_key(row) == wanted:
            return index
    return 0


def _device_key(row: Any) -> tuple[int, ...] | None:
    """One device row as the whole numbers that name it, or ``None``."""
    if not isinstance(row, (list, tuple)):
        return None
    numbers = [_number(cell) for cell in row]
    return tuple(int(number) for number in numbers if number is not None) or None


def _first_address(field_view: FieldView | None, device: int) -> str | None:
    """The address one device answers on, or ``None`` where its slot is blank."""
    if field_view is None:
        return None
    addresses = _addresses(field_view, device)
    return addresses[0] if addresses else None


def _curve_for_device(
    field_block: Any, key: str, device: int, devices: int, where: str
) -> Calibration | None:
    """One device's conversion, read out of the field block that states it."""
    spec = field_block.get(key) if isinstance(field_block, dict) else None
    if not isinstance(spec, dict):
        return None
    kind = _word(spec.get("kind"))
    if kind == "linear":
        gain = _number(_per_device_entry(spec.get("gain"), device, devices, f"{where} {key} gain"))
        offset = _number(
            _per_device_entry(spec.get("offset"), device, devices, f"{where} {key} offset")
        )
        return None if gain is None or offset is None else Linear(gain=gain, offset=offset)
    if kind == "table":
        return _sampled_curve(
            _sampled_row(spec.get("grid"), device, devices, f"{where} {key} grid"),
            _sampled_row(spec.get("values"), device, devices, f"{where} {key} values"),
        )
    return None


def _readout_for_device(field_block: Any, device: int, devices: int, where: str) -> Readout | None:
    """One device's readout calibration, as its own field block states it.

    Each number is optional twice over: a key the facility never stated is
    absent from the block, and a key stated for the family as a whole may
    still be a non-finite entry for this device -- the export's spelling for
    a device the facility's own tables do not cover. Both mean the same thing
    here, and both are carried as "not stated" rather than as a number that
    would read as no correction at all.

    Returns:
        The readout, or ``None`` where this device is left uncalibrated.
    """
    spec = field_block.get("readout") if isinstance(field_block, dict) else None
    if not isinstance(spec, dict):
        return None
    stated = {
        key: _number(_per_device_entry(spec[key], device, devices, f"{where} readout {key}"))
        for key in READOUT_KEYS
        if key in spec
    }
    found = {key: value for key, value in stated.items() if value is not None}
    return Readout(**found) if found else None


def _sampled_curve(grid: Any, values: Any) -> Table | None:
    """A sampled conversion, cut down to the points the facility could state.

    A table is exported over the whole hardware range with the points outside
    its finite span spelled as a non-finite word, so the curve a binding
    carries is the pairs that are numbers at both ends.
    """
    if not isinstance(grid, (list, tuple)) or not isinstance(values, (list, tuple)):
        return None
    pairs = [
        (point, value)
        for point, value in (
            (_number(one), _number(other)) for one, other in zip(grid, values, strict=False)
        )
        if point is not None and value is not None
    ]
    if len(pairs) < 2:
        return None
    return Table(grid=tuple(point for point, _ in pairs), values=tuple(value for _, value in pairs))


def _energy_points(block: dict, key: str) -> Any:
    """One axis of the energy table, which describes a ramp, not a device."""
    table = block.get("energy_table")
    return table.get(key) if isinstance(table, dict) else None


def _nominal_for(block: dict, field: str, device: int, devices: int, where: str) -> float | None:
    """One device's nominal hardware value, or ``None`` where it states none."""
    nominals = block.get("nominals")
    nominal = nominals.get(field) if isinstance(nominals, dict) else None
    if not isinstance(nominal, dict):
        return None
    if _word(nominal.get("units")).lower() == PHYSICS_UNITS:
        return None
    return _number(
        _per_device_entry(nominal.get("values"), device, devices, f"{where} {field} nominal")
    )


def _per_device_entry(value: Any, device: int, devices: int, what: str) -> Any:
    """One device's entry of a per-device sequence, a scalar broadcast to all.

    Raises:
        ValueError: The sequence states a different number of devices than the
            judged family has, so its rows no longer name the devices they
            were sampled for.
    """
    if not isinstance(value, (list, tuple)):
        return value
    if len(value) != devices:
        raise ValueError(
            f"{what}: the export states {len(value)} row(s) for a family judged "
            f"to have {devices} device(s); the rows no longer line up with the "
            "devices they were sampled for"
        )
    return value[device]


def _sampled_row(value: Any, device: int, devices: int, what: str) -> Any:
    """One device's row of sampled points, flat where the family has one device."""
    if (
        isinstance(value, (list, tuple))
        and devices == 1
        and not any(isinstance(item, (list, tuple)) for item in value)
    ):
        return list(value)
    return _per_device_entry(value, device, devices, what)


# --- the shared channel-limits database --------------------------------------
#
# ``data/channel_limits.json`` is not this lane's file. It is the write-safety
# database every write is validated against, and a facility that already runs
# one has authored bands this lane knows nothing about. So the lane writes it
# the way a guest writes in someone else's book: on a tree it creates it states
# the whole channel set, and on a tree that already carries the file it touches
# nothing but ``min_value``/``max_value`` on the addresses its own bindings
# name. Every entry it wrote carries :data:`PROVENANCE_KEY`, which is the whole
# of how a later run tells its own band from a band a person typed: a stamped
# entry is re-derived, an unstamped one is copied across unchanged -- key order
# and all -- so every foreign entry stays byte-identical, and a coupled address
# whose band a person has since edited is reported for the pre-flight to refuse
# rather than overwritten.
#
# The stamp is per entry and never at the top of the document, unlike the two
# starting-state files: those the lane owns whole, this one it shares, and a
# key at the top of a facility's own file is a key the lane does not own. The
# pre-flight reads the stamp per address for the same reason -- the question it
# asks is about one band, not about the file.
#
# ``LimitsValidator._load_limits_database`` fails the WHOLE file on one entry
# carrying a key it does not know, which is why the stamp is spelled with a
# leading underscore: every ``_``-prefixed key is documentation there. So the
# file this lane writes loads through the validator and through ``catalog.py``
# alike.

#: The two entry keys this lane owns, and nothing else.
_MIN_KEY = "min_value"
_MAX_KEY = "max_value"

#: Whether a write to the address is allowed at all. Written once, when the
#: lane creates the entry; never touched again, because a facility that turned
#: an address off meant it.
_WRITABLE_KEY = "writable"

#: The one functional non-address key of the document, kept verbatim.
_DEFAULTS_KEY = "defaults"

#: The field key a Middle Layer family states its operating band under, and the
#: field a coupled setpoint is banded from.
_RANGE_KEY = "Range"
_LIMITS_FIELD = "Setpoint"


@dataclass(frozen=True)
class ChannelBand:
    """One coupled setpoint's band, as this run derives it.

    Attributes:
        address: The channel the band applies to.
        family: The family the address belongs to.
        min_value: The bottom of the band, or ``None`` when the export states
            none -- no ``Range``, or an infinite bound, which is no bound.
        max_value: The top of the band, or ``None``, for the same reasons.
        nominal: The device's nominal hardware value, for a report to show
            beside a band it widened.
        widened: Whether the nominal sat outside the exported ``Range`` and
            widened the band to include it. The served model refuses to boot on
            a nominal outside its band, so the band gives way -- and says so.
        refused: Why the band was not written, or ``None`` when it was. Set
            only for an address the file already bands differently without
            this lane's stamp: a person's edit, which the pre-flight refuses.
    """

    address: str
    family: str
    min_value: float | None
    max_value: float | None
    nominal: float | None
    widened: bool
    refused: str | None = None


def emit_channel_limits(
    existing: dict | None,
    bindings: Iterable[Binding],
    channel_addresses: Iterable[str],
    ctx: EmitContext,
    *,
    views: Iterable[FamilyView],
    system: str,
) -> tuple[str, tuple[ChannelBand, ...]]:
    """Build ``channel_limits.json``: what a write to each address may do.

    On a tree that carries no such file, the lane states the whole machine:
    one entry per address the channel database carries, plus every address the
    bindings name. A coupled setpoint is writable and banded from its family's
    Setpoint ``Range``; every other entry -- a readback, a monitor, a channel
    the virtual accelerator does not drive -- is ``writable: false``.

    On a tree that already carries one, the lane touches only the addresses its
    own bindings name, and only their bands. ``defaults``, every per-entry
    ``confirm``, and every entry this lane did not stamp are copied across
    exactly as they were, so a facility's own file survives the emit. The
    document's keys are sorted, so a re-emit of an unchanged export produces
    byte-identical text.

    Args:
        existing: The file already on the tree as plain JSON types, or ``None``
            when there is none. Never modified.
        bindings: The bindings this run emitted --
            :attr:`~osprey.services.virtual_accelerator.bindings.BindingsDocument.bindings`
            of the document :func:`emit_bindings` just wrote.
        channel_addresses: Every address the channel database carries. Read on
            a tree the lane creates and ignored on a merge, where an address
            the lane does not own is an address it does not state.
        ctx: The provenance of this emit run; its string is the stamp.
        views: The judged family views, in any order -- the operating bands
            and the device each address sits at are read off them, so a
            reviewer's dropped device is already gone from both.
        system: The raw system token the bindings describe, as
            :func:`emit_bindings` was given.

    Returns:
        The document text, and one :class:`ChannelBand` per coupled setpoint:
        the band written, with ``widened`` set where the nominal pushed it out
        and ``refused`` set where a person's own band stands in its place. The
        refused rows are what the emit pre-flight stops on.

    Raises:
        ValueError: A family's per-device ``Range`` states a different number
            of devices than the judged family has, so its rows no longer name
            the devices they were measured for.
    """
    grain = {view.raw_name: view for view in views if view.system == system}
    banded: dict[str, tuple[float | None, float | None, bool, str, float | None]] = {}
    read_only: list[str] = []
    for binding in bindings:
        if binding.is_writable:
            if binding.setpoint_address not in banded:
                low, high, widened = _setpoint_band(binding, grain.get(binding.family))
                banded[binding.setpoint_address] = (
                    low,
                    high,
                    widened,
                    binding.family,
                    _number(binding.nominal),
                )
        else:
            read_only.append(binding.setpoint_address)
        if binding.readback_address:
            read_only.append(binding.readback_address)
    owned_read_only = [
        address for address in dict.fromkeys(read_only) if address and address not in banded
    ]

    if existing is None:
        document = _created_limits(banded, owned_read_only, channel_addresses, ctx)
        refusals: dict[str, str] = {}
    else:
        document, refusals = _merged_limits(existing, banded, owned_read_only, ctx)

    rows = [
        ChannelBand(
            address=address,
            family=family,
            min_value=low,
            max_value=high,
            nominal=nominal,
            widened=widened and address not in refusals,
            refused=refusals.get(address),
        )
        for address, (low, high, widened, family, nominal) in sorted(banded.items())
    ]
    return _text(_ordered_limits(document)), tuple(rows)


def _created_limits(
    banded: dict[str, tuple[float | None, float | None, bool, str, float | None]],
    read_only: Iterable[str],
    channel_addresses: Iterable[str],
    ctx: EmitContext,
) -> dict[str, Any]:
    """The whole machine, on a tree that carries no limits file yet."""
    document: dict[str, Any] = {}
    for address in channel_addresses:
        if address and address not in banded:
            document[address] = _read_only_entry(ctx)
    for address in read_only:
        document[address] = _read_only_entry(ctx)
    for address, (low, high, _widened, _family, _nominal) in banded.items():
        document[address] = _band_entry(ctx, low, high)
    return document


def _merged_limits(
    existing: dict,
    banded: dict[str, tuple[float | None, float | None, bool, str, float | None]],
    read_only: Iterable[str],
    ctx: EmitContext,
) -> tuple[dict[str, Any], dict[str, str]]:
    """The facility's own file, with this lane's bands folded into it."""
    document = dict(existing)
    refusals: dict[str, str] = {}
    for address, (low, high, _widened, _family, _nominal) in banded.items():
        entry = document.get(address)
        if not isinstance(entry, dict):
            document[address] = _band_entry(ctx, low, high)
            continue
        if PROVENANCE_KEY in entry:
            document[address] = _restamped(entry, ctx, low, high)
            continue
        held = (_number(entry.get(_MIN_KEY)), _number(entry.get(_MAX_KEY)))
        if (low, high) != held and (low is not None or high is not None):
            refusals[address] = (
                f"{address}: the file bands it {_band_words(*held)} and carries no "
                f"{PROVENANCE_KEY} stamp, while this export bands it "
                f"{_band_words(low, high)}; the band was written by hand and is "
                "left alone"
            )
    for address in read_only:
        entry = document.get(address)
        if entry is None:
            document[address] = _read_only_entry(ctx)
        elif isinstance(entry, dict) and PROVENANCE_KEY in entry:
            document[address] = _restamped(entry, ctx, None, None)
    return document, refusals


def _setpoint_band(
    binding: Binding, view: FamilyView | None
) -> tuple[float | None, float | None, bool]:
    """One coupled setpoint's band: its family's ``Range``, widened to its nominal."""
    low, high = _range_pair(view, binding.setpoint_address)
    nominal = _number(binding.nominal)
    widened = False
    if nominal is not None:
        if low is not None and nominal < low:
            low, widened = nominal, True
        if high is not None and nominal > high:
            high, widened = nominal, True
    return low, high, widened


def _range_pair(view: FamilyView | None, address: str) -> tuple[float | None, float | None]:
    """The ``Range`` pair one address sits under, as two finite bounds.

    A flat pair is the whole family's band and reaches every device; a
    per-device table states one row each, and the row is the device's own. An
    infinite bound is no bound at all and is left unwritten, which is what the
    write-safety database means by an absent ``min_value``/``max_value``.
    """
    if view is None:
        return None, None
    field_view = view.fields.get(_LIMITS_FIELD)
    if field_view is None:
        return None, None
    declared = field_view.body.get(_RANGE_KEY)
    if not isinstance(declared, (list, tuple)) or not declared:
        return None, None
    pair: Any = declared
    if any(isinstance(row, (list, tuple)) for row in declared):
        index = _device_index(field_view, address)
        if index is None:
            return None, None
        pair = _per_device_entry(
            list(declared),
            index,
            view.n_devices,
            f"{view.system}.{view.raw_name} {_LIMITS_FIELD} {_RANGE_KEY}",
        )
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None, None
    low, high = _number(pair[0]), _number(pair[1])
    if low is not None and high is not None and low > high:
        low, high = high, low
    return low, high


def _device_index(field_view: FieldView, address: str) -> int | None:
    """The 0-based device position an address sits at, across the field's keys."""
    for key in field_view.keys:
        for index, slot in enumerate(field_view.slots(key)):
            if _address(slot) == address:
                return index
    return None


def _band_entry(ctx: EmitContext, low: float | None, high: float | None) -> dict[str, Any]:
    """A fresh entry for a coupled setpoint: stamped, writable, banded."""
    entry: dict[str, Any] = {PROVENANCE_KEY: ctx.provenance_string, _WRITABLE_KEY: True}
    if low is not None:
        entry[_MIN_KEY] = low
    if high is not None:
        entry[_MAX_KEY] = high
    return entry


def _read_only_entry(ctx: EmitContext) -> dict[str, Any]:
    """A fresh entry for an address nothing writes."""
    return {PROVENANCE_KEY: ctx.provenance_string, _WRITABLE_KEY: False}


def _restamped(
    entry: dict, ctx: EmitContext, low: float | None, high: float | None
) -> dict[str, Any]:
    """An entry this lane wrote before, re-derived: the band and the stamp, nothing else."""
    out = dict(entry)
    out[PROVENANCE_KEY] = ctx.provenance_string
    for key, value in ((_MIN_KEY, low), (_MAX_KEY, high)):
        if value is None:
            out.pop(key, None)
        else:
            out[key] = value
    return out


def _band_words(low: float | None, high: float | None) -> str:
    """One band, as a refusal names it."""
    return f"[{'none' if low is None else low}, {'none' if high is None else high}]"


def _ordered_limits(document: dict[str, Any]) -> dict[str, Any]:
    """The document in one deterministic order: metadata, ``defaults``, addresses."""
    ordered: dict[str, Any] = {
        key: document[key] for key in sorted(document) if key.startswith("_")
    }
    if _DEFAULTS_KEY in document:
        ordered[_DEFAULTS_KEY] = document[_DEFAULTS_KEY]
    for key in sorted(document):
        if not key.startswith("_") and key != _DEFAULTS_KEY:
            ordered[key] = document[key]
    return ordered
