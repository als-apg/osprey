"""The computed slots of every wiring record.

A wiring record carries eight slots the build fills and no author states:
``direction``, ``unit``, ``default``, ``value_range``, ``role``, ``plane``,
``refresh`` and ``marks``. ``default`` waits for a deck; ``role``, ``plane``,
``refresh`` and ``marks`` wait for an engine, and are what the model's engine
``describe()`` says the record is, in the facility schema's words;
``direction``, ``unit`` and ``value_range`` are filled for every model. This module is their one writer: it loads each physics model's engine
once, and the simulator view reads the description from the record. It runs
after S5, so every id a record names resolves and the record rules hold, and
before S6, whose checks compare against the ``default`` it writes.

A wired element the deck does not hold exactly once stops with one line naming
the wiring record, whichever slice or role it sits on: ``wiring-conflict`` when
the deck repeats it, ``engine-invalid`` when the deck lacks it
(``element_stop``). An engine signals such a stop by raising a
``FacilityBuildError`` that carries ``element`` and ``count`` (0 absent, above 1
repeated). Every engine stop on a wiring record names the file that states the
record as its source.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

from osprey.facility import DEFAULT_ROLE, TEXTURE
from osprey.facility.errors import FacilityBuildError
from osprey.facility.provenance import add_defaults
from osprey.facility.validate import Validated, need, stating_files

__all__ = ["element_stop", "fill_wiring_slots"]

_MODELS_FILE = "models.yaml"

#: The keys of an engine's description every described wiring record carries.
_DESCRIPTION_KEYS = ("role", "plane", "refresh")
#: The key of an engine's description listing its marks; written only when it lists one.
_MARKS_KEY = "marks"


def fill_wiring_slots(validated: Validated) -> list[FacilityBuildError]:
    """Fill the computed slots of each model's wiring records.

    The slots are written in place on ``validated.document``:

    * ``default``: the engine's start value for the address, from the deck
      at ``facility_dir / model.deck``;
    * ``direction``: ``write`` for a setpoint channel, else ``read``;
    * ``unit``: the channel's unit, when the channel states one;
    * ``value_range``: ``[min_value, max_value]`` of the channel's limits
      record, when that record states both bounds;
    * ``role``, ``plane``, ``refresh`` and ``marks``: what the model's engine
      ``describe()`` says the record is; ``plane`` is null for a record that
      steers or reads no plane, and ``marks`` is absent for a record the
      engine marks with none.

    ``default`` waits for a deck; the description waits for an engine, which every model but a ``texture`` one names. Each filled slot
    is recorded in the record's ``provenance.defaults``. The engine is reached
    through the ``osprey.simulation.engines`` entry-point group, read once per
    build. A record the engine has no start value for, or cannot describe, is
    left without what it lacks and stops; the model's other records are still
    filled. A description that contradicts the facility file stops the model.

    Args:
        validated: What stages S1 to S5 produced.

    Returns:
        Every stop, sorted by record id; empty when every slot was filled.

    Raises:
        RuntimeError: The stage ran before S2 produced the document.
    """
    from osprey.simulation.engines import ENTRY_POINT_GROUP

    document: dict[str, Any] = need(validated.document)
    channels: dict[str, dict[str, Any]] = {c["id"]: c for c in document.get("channels", [])}
    limits: dict[str, dict[str, Any]] = {
        r["address"]: r for r in (document.get("limits") or {}).get("records", [])
    }
    setpoint_of = {
        channel.get("pair", channel["id"]): channel["id"]
        for channel in channels.values()
        if _role(channel) == "setpoint"
    }
    pairs = frozenset(pair for pair, setpoint in setpoint_of.items() if pair != setpoint)
    engines = metadata.entry_points(group=ENTRY_POINT_GROUP)
    errors: list[FacilityBuildError] = []
    for model in document.get("models", []):
        if model.get("engine") == TEXTURE:
            for record in model.get("wiring", []):
                if record["address"] in channels:
                    address = record["address"]
                    _fill_record(record, channels[address], limits.get(address))
            continue
        errors.extend(
            _fill_model(
                model, validated.facility_dir, engines, channels, limits, setpoint_of, pairs
            )
        )
    return sorted(errors, key=lambda error: error.record_id)


def _fill_model(
    model: dict[str, Any],
    facility_dir: Path,
    engines: metadata.EntryPoints,
    channels: Mapping[str, Mapping[str, Any]],
    limits: Mapping[str, Mapping[str, Any]],
    setpoint_of: Mapping[str, str],
    pairs: frozenset[str],
) -> list[FacilityBuildError]:
    from osprey.simulation.engines import ENTRY_POINT_GROUP

    name = model["engine"]
    if name not in engines.names:
        return [
            FacilityBuildError(
                "engine-missing",
                model["name"],
                stating_files(model, None, fallback=_MODELS_FILE),
                "install the engine's package, or name an engine the environment registers",
                record_kind="model",
                detail=f"engine {name} is not registered under {ENTRY_POINT_GROUP}",
            )
        ]
    engine = engines[name].load()
    describe = getattr(engine, "describe", None)
    if describe is None:
        return [
            FacilityBuildError(
                "engine-invalid",
                model["name"],
                stating_files(model, None, fallback=_MODELS_FILE),
                "add describe() to the engine plug-in",
                record_kind="model",
                detail=f"engine {name} states no describe(), so its wiring records carry no role",
            )
        ]
    records = [r for r in model.get("wiring", []) if r["address"] in channels]
    if "deck" not in model:
        errors: list[FacilityBuildError] = []
        for record in records:
            _fill_record(record, channels[record["address"]], limits.get(record["address"]))
            errors.extend(_describe_record(describe, model, record, pairs))
        return errors
    readbacks = {
        r["address"]: setpoint_of.get(r["address"])
        for r in records
        if _role(channels[r["address"]]) != "setpoint"
    }
    try:
        values = engine.start_values(
            facility_dir / model["deck"],
            model.get("wiring", []),
            model.get("settings"),
            readbacks=readbacks,
            value_types={
                r["address"]: str(channels[r["address"]].get("value_type", "float"))
                for r in records
            },
            shapes={
                r["address"]: list(channels[r["address"]]["shape"])
                for r in records
                if "shape" in channels[r["address"]]
            },
        )
    except FacilityBuildError as stop:
        record = next((r for r in model.get("wiring", []) if r["id"] == stop.record_id), None)
        if record is None:
            return [stop]
        translated = element_stop(stop, record, model["name"])
        if translated is not None:
            return [translated]
        return [
            FacilityBuildError(
                stop.kind,
                stop.record_id,
                stating_files(record, None, fallback=_MODELS_FILE),
                stop.remedy,
                record_kind=stop.record_kind,
                detail=stop.detail,
            )
        ]
    errors = []
    for record in records:
        address = record["address"]
        if address not in values:
            errors.append(
                FacilityBuildError(
                    "engine-invalid",
                    record["id"],
                    stating_files(record, None, fallback=_MODELS_FILE),
                    "name an element or slices for the record, or leave the channel unwired",
                    record_kind="wiring",
                    detail=f"{address}: no element to read a start value from",
                )
            )
            continue
        _fill_record(record, channels[address], limits.get(address), values[address])
        errors.extend(_describe_record(describe, model, record, pairs))
    return errors


def _describe_record(
    describe: Callable[[Any], Mapping[str, Any]],
    model: Mapping[str, Any],
    record: dict[str, Any],
    pairs: frozenset[str],
) -> list[FacilityBuildError]:
    """Write what the engine says a record is; the stop when it cannot say or says wrong."""
    try:
        description = describe(record)
    except ValueError as error:
        return [
            FacilityBuildError(
                "engine-invalid",
                str(record["id"]),
                stating_files(record, None, fallback=_MODELS_FILE),
                "name an element, slices or an engine block the engine describes",
                record_kind="wiring",
                detail=str(error),
            )
        ]
    try:
        _check_description(model, record, description, pairs)
    except FacilityBuildError as stop:
        return [stop]
    slots = {key: description[key] for key in _DESCRIPTION_KEYS}
    marks = _marks(description)
    if marks:
        slots[_MARKS_KEY] = sorted(marks)
    _write_slots(record, slots)
    return []


def _marks(description: Mapping[str, Any]) -> list[Any]:
    """The marks a description lists, whatever shape the engine gave them in."""
    marks = description.get(_MARKS_KEY) or []
    return list(marks) if isinstance(marks, list | tuple | set | frozenset) else [marks]


def _check_description(
    model: Mapping[str, Any],
    record: Mapping[str, Any],
    description: Mapping[str, Any],
    pairs: frozenset[str],
) -> None:
    """Stop when an engine's description contradicts the schema or the facility file.

    Every word a description says is one the facility schema states for its slot,
    and only a ``write`` record carries the ``command`` mark.
    """
    from osprey.facility.schema import core

    address = str(record["address"])
    words: dict[str, frozenset[Any]] = {
        "role": frozenset(member.value for member in core.WiringRoleEnum),
        "plane": frozenset(member.value for member in core.PlaneEnum) | {None},
        "refresh": frozenset(member.value for member in core.RefreshEnum),
        _MARKS_KEY: frozenset(member.value for member in core.WiringMarkEnum),
    }
    said = [(key, description.get(key)) for key in _DESCRIPTION_KEYS]
    said += [(_MARKS_KEY, mark) for mark in _marks(description)]
    for key, word in said:
        if not isinstance(word, str | None) or word not in words[key]:
            stated = ", ".join(sorted(str(w) for w in words[key] if w is not None))
            raise FacilityBuildError(
                "engine-invalid",
                str(model["name"]),
                stating_files(model, None, fallback=_MODELS_FILE),
                f"make the engine's describe() give {address} a {key} of {stated}",
                record_kind="model",
                detail=(
                    f"the engine describes {address} with {key} {word!r}, "
                    "which the facility schema does not state"
                ),
            )
    command = core.WiringMarkEnum.command.value
    if command in _marks(description) and record.get("direction") != "write":
        raise FacilityBuildError(
            "engine-invalid",
            str(model["name"]),
            stating_files(model, None, fallback=_MODELS_FILE),
            f"make the engine's describe() mark write records alone {command}",
            record_kind="model",
            detail=f"the engine marks {address} {command}; its direction is read",
        )
    if record.get("direction") == "write":
        expected, why = "setpoint", "its direction is write"
    elif address in pairs:
        expected, why = "readback", "a setpoint pairs with it"
    elif record.get("element") is None and not record.get("slices"):
        expected, why = "output", "it names no element"
    else:
        return
    if description.get("role") != expected:
        raise FacilityBuildError(
            "engine-invalid",
            str(model["name"]),
            stating_files(model, None, fallback=_MODELS_FILE),
            f"make the engine's describe() call {address} a {expected}",
            record_kind="model",
            detail=(
                f"the engine describes {address} as a {description.get('role')}; "
                f"it is a {expected}: {why}"
            ),
        )


def element_stop(
    stop: FacilityBuildError, record: Mapping[str, Any], model: str
) -> FacilityBuildError | None:
    """The build's line for an engine stop on a wired element, if it is one.

    Args:
        stop: What the engine raised while reading the record's deck.
        record: The wiring record that names the element.
        model: The model whose deck was read.

    Returns:
        ``wiring-conflict`` for an element the deck repeats, ``engine-invalid``
        for one it lacks, each naming the wiring record and the model; ``None``
        when the stop carries no element count.
    """
    count = getattr(stop, "count", None)
    if getattr(stop, "element", None) is None or not isinstance(count, int):
        return None
    return FacilityBuildError(
        "wiring-conflict" if count > 1 else "engine-invalid",
        str(record["id"]),
        stating_files(record, None, fallback=_MODELS_FILE),
        stop.remedy,
        record_kind="wiring",
        detail=f"{stop.detail} of model {model}",
    )


def _fill_record(
    record: dict[str, Any],
    channel: Mapping[str, Any],
    limit: Mapping[str, Any] | None,
    default: float | list[float] | None = None,
) -> None:
    """Write a record's computed slots; ``default`` only when a deck gave one."""
    slots: dict[str, Any] = {"direction": "write" if _role(channel) == "setpoint" else "read"}
    if default is not None:
        slots["default"] = default
    if "unit" in channel:
        slots["unit"] = channel["unit"]
    if limit is not None and "min_value" in limit and "max_value" in limit:
        slots["value_range"] = [float(limit["min_value"]), float(limit["max_value"])]
    _write_slots(record, slots)


def _write_slots(record: dict[str, Any], slots: Mapping[str, Any]) -> None:
    """Write computed slots onto a record and record them as filled."""
    present = sorted(set(slots) & set(record))
    if present:
        raise RuntimeError(f"wiring {record['id']} already carries computed slots {present}")
    record.update(slots)
    record["provenance"] = add_defaults(record.get("provenance", {}), slots)


def _role(channel: Mapping[str, Any]) -> str:
    return str(channel.get("role", DEFAULT_ROLE))
