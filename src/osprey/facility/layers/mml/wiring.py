"""The mml layer's wiring pass: what each imported model's channels do to its deck.

The mapping states only family words: the field a family is wired through
(``element_field``), the engine block in pyAT's words and the shape of its
calibration. Everything per address comes from the export: the element each
device drives from the deck the export saved, addressed and renamed by
:mod:`~osprey.facility.layers.mml.decks`, and the numbers of each conversion
from the model facts sampled beside it (``<stem>.va.json``).

:func:`wire_model` takes one model of the mapping and returns the deck the
model is served and its wiring records, in the shape an authored
``models.yaml`` states them::

    address: <the channel>
    element: <deck element>            # or slices: [{element, weight?, device?}]; neither for energy
    engine: {attribute, index} | {axis} | {attribute: energy}
    calibration:
      curve: {linear: {gain, offset}} | {table: {grid, values}}
      inverse: <curve>                 # where the export states one
      energy_scaling: none | brho

One record is written per supply: the address a device of the family answers
on through its wired field. Each device of a supply that reads back on
another address of its family's ``Monitor`` field wires that address too, with
the same element or slices, engine block and calibration, so every readback
of a wired supply follows it. Records are sorted by address and share no
object, so the file they are written to repeats each value in full.

Rules of the derivation:

* **One supply, one record.** A supply the export names against several
  devices feeds them in series, so they are one record with a slice each. The
  curve is the first device's; every other device carries the factor that puts
  it at its own strength where the supply starts, the mean of the hardware
  nominals its devices state. A device split over several deck elements is a
  slice per piece; a kick is shared out over the pieces, a strength or a
  reading describes each piece whole. A slice names its device only where the
  address is a shared endpoint of several.
* **A device with no element is not wired.** It is on the supply and not in
  the deck, so it enters no slice and no mean.
* **The tunes are read off the solve.** A model's ``tune`` block wires its
  address to the engine's tunes, ``{attribute: tune}`` naming no element: a
  waveform block one record reading every plane, a scalar block one record
  per plane, ``index`` 0, 1 or 2 for ``x``, ``y`` or ``s``.
* **The energy knob names no element.** Its engine block,
  ``{attribute: energy}``, drives a property of the whole deck, so its record
  is ``{address, engine, calibration}`` and the dipoles its devices are placed
  at are named for the reader, never bound.
* **Curves are the export's.** ``curve`` is the calibration of the wired field
  and ``inverse`` the ``monitor_inverse`` of the family's ``Monitor`` field;
  neither is derived from the other. A sampled curve keeps the points that are
  numbers at both ends, and where its grid turns back on itself it keeps the
  stretch holding the supply's operating point, because a table is read by
  interpolating on a grid that runs one way.
* **Energy scaling** is the word the export states beside the wired field's
  calibration, carried for a strength or a kick and ``none`` for everything
  else.

Stops, each an :class:`~osprey.facility.layers.mml.mapping.ImportStop`:

* ``export-invalid``: the export and the mapping or the deck disagree -- the
  mapping wires families and the export saved no deck; a wired family the
  export places no device of; whatever the deck pass refuses; a driven device
  that states no hardware nominal where its sampled conversion turns back or
  its supply feeds several devices in series; a device stating no
  calibration, or one of another shape than the mapping names; rows that no
  longer line up with the family's devices; an address or an element field two
  records claim.
* ``reference-missing``: a wired address no channel record carries.

pyAT is imported inside the functions that need it.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Container, Sequence
from collections.abc import Mapping as Map
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.facility.layers.mml import decks
from osprey.facility.layers.mml.mapping import (
    MONITOR_FIELD,
    TUNE_PLANES,
    EngineBlock,
    ImportStop,
    MappingError,
    Model,
    TuneBlock,
    WiringFamily,
    exported_number,
)
from osprey.simulation.engines.calibration import Calibration, Linear, Table, evaluate

if TYPE_CHECKING:  # the export services stay out of the import graph
    from osprey.services.mml.family import FamilyView, FieldView
    from osprey.services.mml.mapping.schema import Mapping as ExportAnswers

__all__ = [
    "ENERGY_SCALINGS",
    "EXPORT_INVALID",
    "REFERENCE_MISSING",
    "Wired",
    "wire_model",
]

#: The stop word for an export that disagrees with its mapping or its deck.
EXPORT_INVALID = "export-invalid"

#: The stop word for a wired address no channel record carries.
REFERENCE_MISSING = "reference-missing"

#: The words a calibration's ``energy_scaling`` may hold, the default first.
ENERGY_SCALINGS: tuple[str, ...] = ("none", "brho")

#: Electron-volts per GeV: an export states energies in GeV, a deck in eV.
_EV_PER_GEV = 1.0e9

#: What a nominal's ``units`` reads when it is not a hardware value.
_PHYSICS_UNITS = "physics"

_MONITOR = "monitor"
_KICK = "kick"
_RF = "rf"
_STRENGTH = "strength"
_ENERGY = "energy"

#: The kinds carrying a physics strength the beam rigidity rescales.
_RIGID_KINDS: frozenset[str] = frozenset({_STRENGTH, _KICK})

#: The kinds whose value is shared out over a split device's pieces.
_SHARED_KINDS: frozenset[str] = frozenset({_KICK})

#: The kinds the model drives, each device starting from a hardware nominal.
_DRIVEN_KINDS: frozenset[str] = frozenset({_STRENGTH, _KICK, _RF})

#: The mapping's word for each curve shape.
_SHAPES: dict[type, str] = {Linear: "linear", Table: "table"}


@dataclass(frozen=True)
class Wired:
    """One model's served deck and its wiring.

    Attributes:
        deck: The deck the model is served, as
            :func:`~osprey.facility.layers.mml.decks.served_deck` returns it;
            every element a record names carries that name exactly once.
        records: The wiring records, sorted by address.
        lines: What the import prints about the wiring, one line each.
    """

    deck: Any
    records: list[dict[str, Any]]
    lines: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Slice:
    element: str
    weight: float
    device: int


def wire_model(
    model: Model,
    deck_path: Path | None,
    va_block: Map[str, Any] | None,
    ad_block: Map[str, Any] | None,
    views: Map[str, FamilyView],
    device_ids: Map[str, Sequence[str]],
    endpoints: Map[str, Sequence[str]],
    channels: Container[str],
    answers: ExportAnswers,
) -> Wired | None:
    """Address one model's deck and derive its wiring records.

    Args:
        model: One model of the layer's mapping, every slot decided.
        deck_path: The deck the export saved for the model (``<stem>.lattice.mat``),
            or ``None`` where it saved none.
        va_block: The model's sampled facts (``<stem>.va.json``), or ``None``.
        ad_block: The model's accelerator data, or ``None``.
        views: The model's families read through the reviewer's judgment
            answers, keyed by raw family token.
        device_ids: The device id of each slot of a family that carries
            channel records, keyed by raw family token.
        endpoints: The devices binding each address, as the identity model
            resolves them.
        channels: The addresses the import writes a channel record for.
        answers: The judgment answers, in the shape the export services read.

    Returns:
        The served deck and the wiring records, or ``None`` for a model that
        wires nothing and whose export saved no deck.

    Raises:
        ImportStop: ``export-invalid`` or ``reference-missing``, as the module
            states them; ``mapping-undecided`` when a cavity is to be built and
            the mapping answers no voltage for it.
        MappingError: The mapping answers a voltage for a deck that holds its
            cavity.
    """
    from osprey.services.mml.loaders.mat import load_lattice

    name = model.name
    if deck_path is None:
        if model.wiring or model.tune is not None:
            raise ImportStop(
                EXPORT_INVALID, [f"{name}: the mapping wires families and the export saved no deck"]
            )
        return None
    facts: Map[str, Any] = va_block if isinstance(va_block, Map) else {}
    try:
        addressing = decks.address_elements(model, load_lattice(deck_path), facts, ad_block)
        unplaced = [
            f"{name}: family {family} is wired through {wiring.element_field} "
            "and the export places none of its devices"
            for family, wiring in model.wiring.items()
            if family not in addressing.bindings
        ]
        if unplaced:
            raise ImportStop(EXPORT_INVALID, unplaced)
        lines: list[str] = []
        records = _records(model, addressing, facts, views, device_ids, endpoints, answers, lines)
        if model.tune is not None:
            records = _with_tune(records, model.tune)
        served = decks.served_deck(addressing)
    except MappingError:
        raise
    except ValueError as exc:
        raise ImportStop(EXPORT_INVALID, [f"{name}: {exc}"]) from exc
    missing = [record["address"] for record in records if record["address"] not in channels]
    if missing:
        raise ImportStop(
            REFERENCE_MISSING,
            [
                f"wiring {name}/{address}: no channel record carries the address"
                for address in missing
            ],
        )
    return Wired(deck=served, records=records, lines=lines)


def _records(
    model: Model,
    addressing: decks.Addressing,
    facts: Map[str, Any],
    views: Map[str, FamilyView],
    device_ids: Map[str, Sequence[str]],
    endpoints: Map[str, Sequence[str]],
    answers: ExportAnswers,
    lines: list[str],
) -> list[dict[str, Any]]:
    """Every wiring record of one model, sorted by address.

    Each family wired without a hardware nominal adds its line to ``lines``.

    Raises:
        ValueError: Two records claim one address or one element field, or
            whatever :func:`_family_records` refuses.
    """
    from osprey.services.mml.judgments import judged_va_block

    wired: dict[str, tuple[str, dict[str, Any]]] = {}
    written: dict[tuple[str, str], str] = {}
    for family, entries in addressing.bindings.items():
        wiring = model.wiring[family]
        view = views.get(family)
        if view is None:
            raise ValueError(
                f"family {family} is wired through {wiring.element_field} "
                "and the export carries no channel of that field"
            )
        block = judged_va_block(
            model.raw, family, {model.raw: facts}, answers, devices=view.n_devices
        )
        _require_devices(family, view, block)
        records, unstated, flipped = _family_records(
            family,
            wiring,
            view,
            block,
            entries,
            device_ids.get(family),
            endpoints,
            addressing.deck,
        )
        if unstated:
            lines.append(
                f"import mml: nominal not stated: {model.name}: family {family}; "
                "start value from the deck"
            )
        lines.extend(
            f"import mml: polarity: {model.name}: family {family} device {device + 1}; "
            "the deck holds the other sign"
            for device in flipped
        )
        for address, readbacks, body in records:
            for piece in _elements(body):
                key = (piece, _field_name(body["engine"]))
                if key in written:
                    raise ValueError(
                        f"element {piece} field {key[1]} is written by "
                        f"family {written[key]} and family {family}"
                    )
                written[key] = family
            for claimed in (address, *readbacks):
                if claimed in wired:
                    raise ValueError(
                        f"address {claimed} is wired by family {wired[claimed][0]} "
                        f"and family {family}"
                    )
                wired[claimed] = (family, {"address": claimed, **copy.deepcopy(body)})
    return [wired[address][1] for address in sorted(wired)]


#: The engine attribute a record reads the tunes on.
_TUNE = "tune"


def _with_tune(records: list[dict[str, Any]], tune: TuneBlock) -> list[dict[str, Any]]:
    """The records with the tune block's, sorted by address.

    Raises:
        ValueError: A tune address is wired by a family too.
    """
    if tune.address is not None:
        added = [{"address": tune.address, "engine": {"attribute": _TUNE}}]
    else:
        added = [
            {"address": address, "engine": {"attribute": _TUNE, "index": TUNE_PLANES.index(plane)}}
            for plane, address in tune.planes.items()
        ]
    wired = {str(record["address"]) for record in records}
    for record in added:
        if record["address"] in wired:
            raise ValueError(
                f"address {record['address']} is wired by a family and by the tune block"
            )
    return sorted([*records, *added], key=lambda record: str(record["address"]))


def _elements(body: Map[str, Any]) -> list[str]:
    """Every deck element one record names."""
    if "element" in body:
        return [str(body["element"])]
    return [str(piece["element"]) for piece in body.get("slices", [])]


def _field_name(engine: Map[str, Any]) -> str:
    """An element field, as a refusal writes it."""
    if "attribute" not in engine:
        return str(engine.get("axis"))
    if engine.get("index") is None:
        return str(engine["attribute"])
    return f"{engine['attribute']}[{engine['index']}]"


def _family_records(
    family: str,
    wiring: WiringFamily,
    view: FamilyView,
    block: Map[str, Any],
    entries: Sequence[decks.ElementBinding],
    ids: Sequence[str] | None,
    endpoints: Map[str, Sequence[str]],
    deck: Any,
) -> tuple[list[tuple[str, list[str], dict[str, Any]]], bool, list[int]]:
    """One family's records: per supply its address, its readbacks and the record body.

    A supply's readbacks are the ``Monitor`` addresses its devices read back
    on, each device's own, other than the supply's address.

    A driven device that states no hardware nominal is wired all the same
    where nothing hangs on the nominal: the start value comes from the deck.
    Two things do hang on it, and refuse such a device: a sampled conversion
    that turns back, whose stretch the operating point picks, and a supply
    feeding several devices in series, whose shares the nominals set.

    Returns:
        The records, whether a device was wired without a nominal, and the
        devices whose polarity the deck holds the other way, in device order.

    Raises:
        ValueError: The family carries no channel of its wired field, a driven
            device states no hardware nominal where a turning conversion or a
            series needs it, a device serves a readback on an address of its
            own with no ``monitor_inverse`` stated, or whatever
            :func:`_supply_record` refuses.
    """
    engine = wiring.engine
    written = wiring.element_field
    field_view = view.fields.get(written) if written is not None else None
    if engine is None or written is None or field_view is None:
        raise ValueError(
            f"family {family} is wired through {written} "
            "and the export carries no channel of that field"
        )
    kind = _kind(engine)
    devices = view.n_devices
    gap = _nominal_gap(block, written, devices) if kind in _DRIVEN_KINDS else None

    elements = _element_by_device(view, entries)
    supplies: dict[str, list[int]] = {}
    for device in range(devices):
        address = _first_address(field_view, device)
        if address is not None and (kind == _ENERGY or device in elements):
            supplies.setdefault(address, []).append(device)

    records: list[tuple[str, list[str], dict[str, Any]]] = []
    unstated = False
    flipped: list[int] = []
    monitor = view.fields.get(MONITOR_FIELD)
    for address, members in supplies.items():
        if gap is not None and any(
            _nominal_for(block, written, member, devices, f"family {family}") is None
            for member in members
        ):
            if len(members) > 1 or _turns_back(block, written, members[0], devices, family):
                raise ValueError(f"family {family} is wired through {written} and the export {gap}")
            unstated = True
        body = _supply_record(family, kind, engine, wiring, view, block, elements, members, deck)
        flipped.extend(
            sorted({piece.device for piece in body.get("slices", ()) if piece.weight < 0.0})
        )
        if kind != _ENERGY:
            shared = len(endpoints.get(address, ())) > 1
            body = _stated(body, ids if shared else None)
        readbacks: list[str] = []
        for member in members if kind != _MONITOR else ():
            served = _first_address(monitor, member)
            if served is None or served == address or served in readbacks:
                continue
            if "inverse" not in body["calibration"]:
                raise ValueError(
                    f"family {family} device {member + 1}: serves its readback "
                    f"on {served} and states no monitor_inverse"
                )
            readbacks.append(served)
        records.append((address, readbacks, body))
    return records, unstated, sorted(flipped)


def _turns_back(block: Map[str, Any], written: str, device: int, devices: int, family: str) -> bool:
    """Whether a device's sampled conversion, either way, turns back on its grid."""
    where = f"family {family} device {device + 1}"
    for field_block, key in (
        (block.get(written), "calibration"),
        (block.get(MONITOR_FIELD), "monitor_inverse"),
    ):
        curve = _curve_for_device(field_block, key, device, devices, where)
        if isinstance(curve, Table) and _stretches(curve.grid) != [(0, len(curve.grid) - 1)]:
            return True
    return False


def _supply_record(
    family: str,
    kind: str,
    engine: EngineBlock,
    wiring: WiringFamily,
    view: FamilyView,
    block: Map[str, Any],
    elements: Map[int, decks.ElementBinding],
    members: list[int],
    deck: Any,
) -> dict[str, Any]:
    """One supply's record body, whether it feeds one device or several in series.

    The supply starts at the mean of the hardware nominals its devices state
    and converts through the first device's curve. With one device the mean is
    its own nominal and its slice weighs what a split shares out.

    Returns:
        ``{slices, engine, calibration}`` with ``slices`` as :class:`_Slice`
        rows, for :func:`_stated` to spell; ``{engine, calibration}`` for the
        energy knob, which drives a property of the whole deck and names no
        element.

    Raises:
        ValueError: The first device states no calibration or one of another
            shape than the mapping names, a sampled curve reads one way
            nowhere, or a device of a series sits at no strength.
    """
    written = str(wiring.element_field)
    devices = view.n_devices
    reference = members[0]
    where = f"family {family} device {reference + 1}"
    nominals = [
        _nominal_for(block, written, device, devices, f"family {family} device {device + 1}")
        for device in members
    ]
    stated = [value for value in nominals if value is not None]
    hardware = math.fsum(stated) / len(stated) if stated else 0.0
    if kind == _ENERGY:
        return _energy_record(family, engine, wiring, block, hardware)

    sampled = _curve_for_device(block.get(written), "calibration", reference, devices, where)
    if sampled is None:
        raise ValueError(f"{where}: its {written} block states no calibration")
    shape = _SHAPES[type(sampled)]
    if wiring.calibration is not None and shape != wiring.calibration:
        raise ValueError(
            f"{where}: the mapping names a {wiring.calibration} calibration "
            f"and the export states a {shape} one"
        )
    curve = _one_way(sampled, where, "calibration", hardware=hardware)
    inverse = _curve_for_device(
        block.get(MONITOR_FIELD), "monitor_inverse", reference, devices, where
    )
    if inverse is not None:
        inverse = _one_way(inverse, where, "monitor_inverse", hardware=hardware, on_values=True)

    calibration: dict[str, Any] = {"curve": _curve_record(curve)}
    if inverse is not None:
        calibration["inverse"] = _curve_record(inverse)
    calibration["energy_scaling"] = _energy_scaling(kind, block.get(written))
    return {
        "slices": _slices(
            family, kind, written, view, block, elements, members, evaluate(curve, hardware), deck
        ),
        "engine": _engine_record(engine),
        "calibration": calibration,
    }


def _energy_record(
    family: str, engine: EngineBlock, wiring: WiringFamily, block: Map[str, Any], hardware: float
) -> dict[str, Any]:
    """The energy knob's record body: its supply's current to the deck energy.

    The curve is the export's energy table, its energies stated in GeV and
    written in eV, the unit the deck states its energy in, so the deck's own
    energy reads back through the inverse to the current the supply sits at.
    The inverse is the same table read the other way: a table that runs one
    way on both axes converts back exactly.

    Raises:
        ValueError: The export states no energy table with two finite points,
            the mapping names a linear calibration, or the table reads one way
            nowhere.
    """
    where = f"family {family}"
    table = block.get("energy_table")
    grid = table.get("grid") if isinstance(table, Map) else None
    values = table.get("values") if isinstance(table, Map) else None
    sampled = _sampled_curve(
        grid,
        [
            _EV_PER_GEV * number if (number := exported_number(value)) is not None else value
            for value in values
        ]
        if isinstance(values, (list, tuple))
        else values,
    )
    if sampled is None:
        raise ValueError(f"{where}: drives the energy and the export states no energy table")
    if wiring.calibration is not None and wiring.calibration != _SHAPES[Table]:
        raise ValueError(
            f"{where}: the mapping names a {wiring.calibration} calibration "
            "and the export states the energy as a table"
        )
    curve = _one_way(sampled, where, "energy table", hardware=hardware)
    inverse = _one_way(
        Table(grid=curve.values, values=curve.grid),  # type: ignore[union-attr]
        where,
        "energy table",
        hardware=hardware,
        on_values=True,
    )
    return {
        "engine": _engine_record(engine),
        "calibration": {
            "curve": _curve_record(curve),
            "inverse": _curve_record(inverse),
            "energy_scaling": ENERGY_SCALINGS[0],
        },
    }


def _stated(body: dict[str, Any], ids: Sequence[str] | None) -> dict[str, Any]:
    """Spell a record body the way a models file states it.

    One slice of weight 1 on the channel's own device is an ``element``; every
    other record states ``slices``, a weight of 1 and the channel's own device
    left out.

    Args:
        body: What :func:`_supply_record` returned.
        ids: The family's device id per slot when the address is a shared
            endpoint, so each slice names its device; ``None`` otherwise.
    """
    slices: list[_Slice] = body["slices"]
    rest = {key: value for key, value in body.items() if key != "slices"}
    if ids is None and len(slices) == 1 and slices[0].weight == 1.0:
        return {"element": slices[0].element, **rest}
    rows: list[dict[str, Any]] = []
    for piece in slices:
        row: dict[str, Any] = {"element": piece.element}
        if piece.weight != 1.0:
            row["weight"] = piece.weight
        if ids is not None:
            row["device"] = ids[piece.device]
        rows.append(row)
    return {"slices": rows, **rest}


def _kind(engine: EngineBlock) -> str:
    """What a wired family is to the model, read off its engine block."""
    if engine.axis is not None:
        return _MONITOR
    if engine.attribute == decks.KICK:
        return _KICK
    if engine.attribute == decks.FREQUENCY:
        return _RF
    if engine.attribute == decks.ENERGY:
        return _ENERGY
    return _STRENGTH


def _engine_record(engine: EngineBlock) -> dict[str, Any]:
    """An engine block as a record states it: only the words the mapping names."""
    record: dict[str, Any] = {}
    if engine.attribute is not None:
        record["attribute"] = engine.attribute
    if engine.index is not None:
        record["index"] = engine.index
    if engine.axis is not None:
        record["axis"] = engine.axis
    return record


def _curve_record(curve: Calibration) -> dict[str, Any]:
    """A curve in the facility file's shape."""
    if isinstance(curve, Linear):
        return {"linear": {"gain": curve.gain, "offset": curve.offset}}
    return {"table": {"grid": list(curve.grid), "values": list(curve.values)}}


def _energy_scaling(kind: str, field_block: Any) -> str:
    """Whether a record's physics value moves with the beam rigidity."""
    if kind not in _RIGID_KINDS or not isinstance(field_block, Map):
        return ENERGY_SCALINGS[0]
    word = _word(field_block.get("energy_scaling"))
    return word if word in ENERGY_SCALINGS else ENERGY_SCALINGS[0]


# -- slices -------------------------------------------------------------------


def _slices(
    family: str,
    kind: str,
    written: str,
    view: FamilyView,
    block: Map[str, Any],
    elements: Map[int, decks.ElementBinding],
    members: list[int],
    start_physics: float,
    deck: Any,
) -> list[_Slice]:
    """Every element one supply writes, each with the share it carries there.

    Two shares and a sign multiply into one weight. The split share divides a
    value over the pieces one device is modelled as. The series factor is what
    one device of a series holds against the supply: the physics its own curve
    puts it at over the physics the supply's curve answers at the starting
    value. The sign is the device's polarity: where the physics the export's
    curve puts the device at from its stated nominal and the strength the deck
    holds on its first piece have opposite signs, the deck is wound the other
    way, and the slice carries -1 so the device starts at its stated nominal.

    Raises:
        ValueError: A device of a series sits at no strength while the supply
            sits at some; no fixed share puts a device at zero and still moves
            it with the supply.
    """
    devices = view.n_devices
    rows: list[_Slice] = []
    for device in members:
        pieces = elements[device].slices
        share = 1.0 / len(pieces) if kind in _SHARED_KINDS and len(pieces) > 1 else 1.0
        factor = 1.0
        where = f"family {family} device {device + 1}"
        nominal = _nominal_for(block, written, device, devices, where)
        physics = start_physics if nominal is not None else 0.0
        if len(members) > 1:
            own = _curve_for_device(block.get(written), "calibration", device, devices, where)
            physics = evaluate(own, nominal) if own is not None and nominal is not None else 0.0
            factor = _series_factor(where, physics, start_physics)
        held = _deck_strength(deck, pieces[0].position, elements[device].engine)
        sign = -1.0 if held is not None and physics * held < 0.0 else 1.0
        rows.extend(_Slice(piece.element, share * factor * sign, device) for piece in pieces)
    return rows


def _deck_strength(deck: Any, position: int, engine: EngineBlock) -> float | None:
    """The finite value the deck holds in the field an engine block writes, or ``None``."""
    if engine.attribute is None:
        return None
    value: Any = getattr(deck[position], engine.attribute, None)
    try:
        number = float(value if engine.index is None else value[engine.index])
    except (IndexError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _series_factor(where: str, strength: float, start_physics: float) -> float:
    """What one device of a series holds against the supply at the starting value.

    A supply whose own curve answers nothing there has no ratio to divide by,
    and its devices all sit at nothing too, so each moves one for one with it.
    """
    if not math.isfinite(start_physics) or start_physics == 0.0:
        return 1.0
    factor = strength / start_physics
    if not math.isfinite(factor) or factor == 0.0:
        raise ValueError(
            f"{where}: its supply feeds it in series, the export puts it at {strength:.6g} "
            f"and the supply itself at {start_physics:.6g}; a device held at nothing by a "
            "supply that is not cannot be a fixed share of it"
        )
    return factor


def _element_by_device(
    view: FamilyView, entries: Sequence[decks.ElementBinding]
) -> dict[int, decks.ElementBinding]:
    """Pair each element row with the device position of the judged family.

    Rows are matched by the device they name, not by their order, so a family
    whose middle device states no element keeps every other row against the
    device it was addressed for.
    """
    positions: dict[tuple[int, ...], int] = {}
    for index, row in enumerate(view.device_rows or []):
        key = _device_key(row)
        if key is not None:
            positions.setdefault(key, index)
    found: dict[int, decks.ElementBinding] = {}
    for entry in entries:
        position = positions.get(tuple(entry.device))
        if position is not None:
            found.setdefault(position, entry)
    return found


def _device_key(row: Any) -> tuple[int, ...] | None:
    """One device row as the whole numbers that name it, or ``None``."""
    if not isinstance(row, (list, tuple)):
        return None
    numbers = [exported_number(cell) for cell in row]
    return tuple(int(number) for number in numbers if number is not None) or None


def _first_address(field_view: FieldView | None, device: int) -> str | None:
    """The address one device answers on, or ``None`` where its slot is blank."""
    if field_view is None:
        return None
    for key in field_view.keys:
        slots = field_view.slots(key)
        address = _word(slots[device]) if device < len(slots) else ""
        if address:
            return address
    return None


# -- the export's rows --------------------------------------------------------


def _require_devices(family: str, view: FamilyView, block: Map[str, Any]) -> None:
    """Refuse sampled facts whose device rows are not the family's.

    Raises:
        ValueError: The block's ``device_list`` states another device count
            than the judged family has.
    """
    from osprey.services.mml.family import device_rows

    rows = device_rows(block.get("device_list"))
    if rows is not None and len(rows) != view.n_devices:
        raise ValueError(
            f"family {family}: its sampled facts state {len(rows)} devices "
            f"and the family has {view.n_devices}"
        )


def _nominal_gap(block: Map[str, Any], field: str, devices: int) -> str | None:
    """Say why a driven family cannot start, in a clause that reads after "the export"."""
    nominals = block.get("nominals")
    nominal = nominals.get(field) if isinstance(nominals, Map) else None
    if not isinstance(nominal, Map):
        return f"states no hardware nominal: {field} states none"
    if _word(nominal.get("units")).lower() == _PHYSICS_UNITS:
        return "states no hardware nominal: its nominal is in physics units"
    values = nominal.get("values")
    rows = list(values) if isinstance(values, (list, tuple)) else [values] * max(devices, 1)
    missing = sum(1 for value in rows if exported_number(value) is None)
    if not missing:
        return None
    return f"states no hardware nominal for {missing} of its {len(rows)} devices"


def _nominal_for(
    block: Map[str, Any], field: str, device: int, devices: int, where: str
) -> float | None:
    """One device's nominal hardware value, or ``None`` where it states none."""
    nominals = block.get("nominals")
    nominal = nominals.get(field) if isinstance(nominals, Map) else None
    if not isinstance(nominal, Map):
        return None
    if _word(nominal.get("units")).lower() == _PHYSICS_UNITS:
        return None
    return exported_number(
        _per_device_entry(nominal.get("values"), device, devices, f"{where} {field} nominal")
    )


def _per_device_entry(value: Any, device: int, devices: int, what: str) -> Any:
    """One device's entry of a per-device sequence, a scalar broadcast to all.

    Raises:
        ValueError: The sequence states another number of devices than the
            judged family has, so its rows no longer name the devices they
            were sampled for.
    """
    if not isinstance(value, (list, tuple)):
        return value
    if len(value) != devices:
        raise ValueError(
            f"{what}: the export states {len(value)} rows for a family of {devices} devices"
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


def _word(value: Any) -> str:
    """A non-blank string stripped, else ``""``."""
    return value.strip() if isinstance(value, str) and value.strip() else ""


# -- curves -------------------------------------------------------------------


def _curve_for_device(
    field_block: Any, key: str, device: int, devices: int, where: str
) -> Calibration | None:
    """One device's conversion, read out of the field block that states it."""
    spec = field_block.get(key) if isinstance(field_block, Map) else None
    if not isinstance(spec, Map):
        return None
    kind = _word(spec.get("kind"))
    if kind == "linear":
        gain = exported_number(
            _per_device_entry(spec.get("gain"), device, devices, f"{where} {key} gain")
        )
        offset = exported_number(
            _per_device_entry(spec.get("offset"), device, devices, f"{where} {key} offset")
        )
        return None if gain is None or offset is None else Linear(gain=gain, offset=offset)
    if kind == "table":
        return _sampled_curve(
            _sampled_row(spec.get("grid"), device, devices, f"{where} {key} grid"),
            _sampled_row(spec.get("values"), device, devices, f"{where} {key} values"),
        )
    return None


def _sampled_curve(grid: Any, values: Any) -> Table | None:
    """A sampled conversion, cut down to the points the export could state.

    A table is exported over the whole hardware range with the points outside
    its finite span spelled as text, so the curve a record carries is the pairs
    that are numbers at both ends.
    """
    if not isinstance(grid, (list, tuple)) or not isinstance(values, (list, tuple)):
        return None
    pairs = [
        (point, value)
        for point, value in (
            (exported_number(one), exported_number(other))
            for one, other in zip(grid, values, strict=False)
        )
        if point is not None and value is not None
    ]
    if len(pairs) < 2:
        return None
    return Table(grid=tuple(point for point, _ in pairs), values=tuple(value for _, value in pairs))


def _one_way(
    curve: Calibration, where: str, what: str, *, hardware: float, on_values: bool = False
) -> Calibration:
    """Keep the stretch of a sampled conversion the device itself sits on.

    A table is read by interpolating on its grid and continuing along its end
    segments, so the grid has to run strictly one way. A sampled conversion
    that turns back still answers one value over each stretch between its
    turning points, and the stretch that matters is the one the device runs
    in.

    Which stretch that is, is settled by a sampled point and not by a span:
    where a curve turns back it travels the same values twice, so both
    branches span the operating point. The sampled point closest to the
    device's hardware value belongs to one branch, and the longest stretch
    holding that point is kept; where the point belongs to none, the nearest
    stretch is.

    Args:
        curve: The conversion as the export states it; a straight line is
            returned untouched.
        where: The family and device, for a refusal to name.
        what: The conversion's name in the export, for the same.
        hardware: The supply's starting value, in the hardware units one of
            the two axes is stated in.
        on_values: Whether that axis is the values rather than the grid, which
            is what a conversion back to hardware states.

    Returns:
        The conversion to write.

    Raises:
        ValueError: No two neighbouring grid points of the table differ, so no
            stretch of it reads one way.
    """
    if not isinstance(curve, Table):
        return curve
    stretches = _stretches(curve.grid)
    if not stretches:
        raise ValueError(
            f"{where}: its {what} repeats one sampled point across the whole grid, "
            "so no stretch of it converts one way"
        )
    sits = _nearest(curve.values if on_values else curve.grid, hardware)
    start, end = min(stretches, key=lambda run: (_apart(run, sits), run[0] - run[1]))
    if (start, end) == (0, len(curve.grid) - 1):
        return curve
    return Table(grid=curve.grid[start : end + 1], values=curve.values[start : end + 1])


def _stretches(grid: tuple[float, ...]) -> list[tuple[int, int]]:
    """Every maximal run of a grid that rises or falls throughout, as index pairs.

    Two runs meeting at a turning point share it. A pair of neighbours that
    repeat a value belongs to neither: a curve that stands still there
    converts nothing.
    """
    runs: list[tuple[int, int]] = []
    start = 0
    rising: bool | None = None
    for position in range(1, len(grid)):
        step = grid[position] - grid[position - 1]
        if step == 0.0:
            if position - 1 > start:
                runs.append((start, position - 1))
            start, rising = position, None
            continue
        if rising is None:
            rising = step > 0.0
            continue
        if (step > 0.0) != rising:
            runs.append((start, position - 1))
            start, rising = position - 1, step > 0.0
    if len(grid) - 1 > start:
        runs.append((start, len(grid) - 1))
    return runs


def _nearest(axis: tuple[float, ...], point: float) -> int:
    """The position of the sampled point closest to ``point``; the earlier of two equally close."""
    return min(range(len(axis)), key=lambda position: abs(axis[position] - point))


def _apart(run: tuple[int, int], position: int) -> int:
    """How many sampled points lie between one stretch and a position; zero inside it."""
    start, end = run
    return max(start - position, position - end, 0)
