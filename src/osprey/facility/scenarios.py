"""The faults a scenario writes into a model, and the logbook it narrates.

A scenario faults a model in two spellings, both under ``faults.<model>``:

* by channel address, ``{<address>: {<fault field>: <value>}}``;
* by the engine's own fault name, ``{<engine variable>: <value>}``.

The engine plug-in names what a model built from a wiring declares through
``fault_variables(wiring)``: fault name -> a slot carrying ``address``,
``field``, ``value_range`` and ``options``. A plug-in is reached through the
``osprey.simulation.engines`` entry-point group, never by import. Every value
is coerced as a float and must lie inside its slot's ``value_range`` or equal
one of its ``options``.

A scenario's ``logbook`` block reads as :class:`ScenarioLogEntry` records
through :func:`scenario_logbook`; each entry's ``when`` is ``{days_ago,
time}``, resolved against the activation anchor when the entry is seeded.
An entry's ``attachments`` list names its pictures, each item one of
``{path: <picture file>}`` or ``{plot: <plot spec .json>}``, relative to the
scenario's own directory ``scenarios/<name>/``.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Iterator, Mapping
from datetime import time
from importlib import metadata
from pathlib import Path
from typing import Any

from osprey.facility.errors import FacilityBuildError
from osprey_connectors.relative_time import RelativeTimestamp
from osprey_connectors.simulation.machine import ScenarioLogEntry, _parse_log_attachments

__all__ = [
    "FaultRoster",
    "ScenarioLogEntry",
    "check_scenario_attachments",
    "check_scenario_engines",
    "fault_roster",
    "map_fault_errors",
    "scenario_logbook",
]


def scenario_logbook(
    scenario: Mapping[str, Any], directory: Path | None = None
) -> tuple[ScenarioLogEntry, ...]:
    """The entries a scenario's ``logbook`` block narrates, in block order.

    Args:
        scenario: One scenario record: its ``name`` and, when stated, its
            ``logbook`` list.
        directory: The scenario's own directory, which its entries'
            ``attachments`` resolve against; needed only when an entry
            attaches a picture.

    Returns:
        The entries; empty when the scenario states no logbook. A shipped
        picture is its absolute path inside ``directory``, a plot spec its
        parsed spec.

    Raises:
        ValueError: An entry is malformed, or attaches a picture with no
            ``directory`` given; the message names the scenario, the entry and
            the key.
    """
    name = str(scenario.get("name"))
    raw = scenario.get("logbook") or []
    if not isinstance(raw, list):
        raise ValueError(f"Scenario {name!r} logbook: must be a list of entries")
    return tuple(_log_entry(name, item, directory) for item in raw)


def _log_entry(scenario: str, raw: Any, directory: Path | None) -> ScenarioLogEntry:
    prefix = f"Scenario {scenario!r} logbook"
    if not isinstance(raw, Mapping):
        raise ValueError(f"{prefix}: each entry must be a mapping")
    entry_id = raw.get("entry_id")
    if not isinstance(entry_id, str) or not entry_id:
        raise ValueError(f"{prefix}: 'entry_id' must be a non-empty string, got {entry_id!r}")
    prefix = f"{prefix} entry {entry_id!r}"

    def text(key: str) -> str:
        value = raw.get(key)
        if not isinstance(value, str):
            raise ValueError(f"{prefix}: {key!r} must be a string, got {value!r}")
        return value

    def strings(key: str) -> tuple[str, ...]:
        value = raw.get(key, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{prefix}: {key!r} must be a list of strings, got {value!r}")
        return tuple(value)

    loto_tag = raw.get("loto_tag")
    if loto_tag is not None and not isinstance(loto_tag, str):
        raise ValueError(f"{prefix}: 'loto_tag' must be a string or null, got {loto_tag!r}")
    extra = raw.get("extra", {})
    if not isinstance(extra, Mapping):
        raise ValueError(f"{prefix}: 'extra' must be a mapping, got {extra!r}")
    attachments = raw.get("attachments", [])
    if attachments and directory is None:
        raise ValueError(f"{prefix}: 'attachments' resolve against the scenario's directory")
    return ScenarioLogEntry(
        entry_id=entry_id,
        when=_relative_timestamp(prefix, raw.get("when")),
        author=text("author"),
        title=text("title"),
        text=text("text"),
        tags=strings("tags"),
        categories=strings("categories"),
        loto_tag=loto_tag,
        extra=dict(extra),
        attachments=(
            _parse_log_attachments(prefix, attachments, directory) if directory is not None else ()
        ),
    )


def _relative_timestamp(prefix: str, raw: Any) -> RelativeTimestamp:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{prefix}: 'when' must be a mapping with 'days_ago' and 'time'")
    days_ago = raw.get("days_ago")
    if isinstance(days_ago, bool) or not isinstance(days_ago, int) or days_ago < 0:
        raise ValueError(f"{prefix}: 'days_ago' must be a non-negative integer, got {days_ago!r}")
    raw_time = raw.get("time")
    if not isinstance(raw_time, str):
        raise ValueError(f"{prefix}: 'when.time' must be an 'HH:MM:SS' string, got {raw_time!r}")
    try:
        parsed = time.fromisoformat(raw_time)
    except ValueError:
        raise ValueError(
            f"{prefix}: 'when.time' must be a valid 'HH:MM:SS' time of day, got {raw_time!r}"
        ) from None
    if parsed.tzinfo is not None:
        raise ValueError(
            f"{prefix}: 'when.time' is local time and must not carry a timezone offset, "
            f"got {raw_time!r}"
        )
    return RelativeTimestamp(days_ago=days_ago, time=parsed)


class FaultRoster:
    """The faults one model's engine declares, by fault name and by address.

    Attributes:
        engine: The engine's name.
        slots: Fault name -> slot (``address``, ``field``, ``value_range``,
            ``options``, ``element``).
    """

    def __init__(self, engine: str, slots: Mapping[str, Any]) -> None:
        self.engine = engine
        self.slots = dict(slots)

    def fields(self, address: str) -> dict[str, Any]:
        """The fault fields ``address`` carries, each with its slot."""
        return {slot.field: slot for slot in self.slots.values() if slot.address == address}

    def element(self, address: str) -> str | None:
        """The element ``address`` is wired to, as its fault slots name it."""
        for slot in self.slots.values():
            if slot.address == address and getattr(slot, "element", None) is not None:
                return str(slot.element)
        return None

    def sibling(self, address: str, field: str) -> str | None:
        """The address on the same element that carries ``field``, if another does."""
        element = self.element(address)
        if element is None:
            return None
        for slot in self.slots.values():
            if (
                slot.field == field
                and slot.address != address
                and getattr(slot, "element", None) == element
            ):
                return str(slot.address)
        return None


def fault_roster(
    model: Mapping[str, Any], channels: Mapping[str, Mapping[str, Any]]
) -> FaultRoster | None:
    """The faults ``model``'s engine declares for its wiring.

    A wiring record without a ``direction`` takes ``write`` when its channel
    is a setpoint, else ``read``.

    Args:
        model: The model record.
        channels: Every channel record, by address.

    Returns:
        The roster; ``None`` when no plug-in is registered under the model's
        engine name. A plug-in without ``fault_variables`` declares no faults.
    """
    from osprey.simulation.engines import ENTRY_POINT_GROUP

    name = str(model.get("engine"))
    engines = metadata.entry_points(group=ENTRY_POINT_GROUP)
    if name not in engines.names:
        return None
    plugin = engines[name].load()
    wiring = [
        {**record, "direction": record.get("direction", _direction(channels, record))}
        for record in model.get("wiring") or []
        if isinstance(record, dict)
    ]
    declare = getattr(plugin, "fault_variables", None)
    slots = declare(wiring) if declare is not None else {}
    return FaultRoster(name, slots)


def _direction(channels: Mapping[str, Mapping[str, Any]], record: Mapping[str, Any]) -> str:
    channel = channels.get(str(record.get("address")), {})
    return "write" if channel.get("role") == "setpoint" else "read"


def _error(kind: str, name: str, files: list[str], detail: str, remedy: str) -> FacilityBuildError:
    return FacilityBuildError(kind, name, files, remedy, record_kind="scenario", detail=detail)


def _value_errors(
    name: str, files: list[str], slot: str, value: Any, fault: Any
) -> Iterator[FacilityBuildError]:
    """Coerce one fault value as a float and hold it to its slot's range or options."""
    from osprey_connectors.simulation.values import coerce

    try:
        number = coerce(value, "float", None, None)
    except ValueError as exc:
        yield _error(
            "value-invalid", name, files, f"`{slot}`: {exc}", f"write a float value for `{slot}`"
        )
        return
    options = getattr(fault, "options", None)
    if options is not None:
        if number in options:
            return
        listing = ", ".join(f"{o:g}" for o in options)
        yield _error(
            "seed-invalid",
            name,
            files,
            f"`{slot}` {number:g} is not one of {listing}",
            f"write one of {listing}",
        )
        return
    bounds = fault.value_range
    if bounds is None:
        return
    low, high = bounds
    if low <= number <= high:
        return
    yield _error(
        "seed-invalid",
        name,
        files,
        f"`{slot}` {number:g} lies {'below' if number < low else 'above'} its range "
        f"[{low:g}, {high:g}]",
        f"write a value inside [{low:g}, {high:g}]",
    )


def map_fault_errors(
    name: str,
    files: list[str],
    model: str,
    address: str,
    value: Mapping[str, Any],
    roster: FaultRoster,
) -> list[FacilityBuildError]:
    """Check a ``{<fault field>: <value>}`` map a scenario writes on one address.

    Args:
        name: The scenario's name.
        files: The scenario's file.
        model: The model the map sits under.
        address: The faulted channel's address.
        value: The map.
        roster: The model's fault roster.

    Returns:
        One stop per field that is not a fault field of ``address``, whose
        value is not a float, or whose value lies outside its range or is not
        one of its options.
    """
    errors: list[FacilityBuildError] = []
    fields = roster.fields(address)
    for field_name in sorted(value, key=str):
        field = str(field_name)
        slot = f"faults.{model}.{address}.{field}"
        fault = fields.get(field)
        if fault is None:
            errors.append(_unknown_field(name, files, slot, address, field, fields, roster))
            continue
        errors.extend(_value_errors(name, files, slot, value[field_name], fault))
    return errors


def _unknown_field(
    name: str,
    files: list[str],
    slot: str,
    address: str,
    field: str,
    fields: Mapping[str, Any],
    roster: FaultRoster,
) -> FacilityBuildError:
    detail = f"{slot} is not a fault field of {address}"
    sibling = roster.sibling(address, field)
    if sibling is not None:
        remedy = f"move `{field}` to {sibling}, which carries it for {roster.element(address)}"
    elif fields:
        remedy = f"use one of {', '.join(sorted(fields))}"
    else:
        remedy = f"write a value for {address} itself; it carries no fault fields"
    return _error("value-invalid", name, files, detail, remedy)


def check_scenario_attachments(
    document: Mapping[str, Any], facility_dir: Path
) -> list[FacilityBuildError]:
    """Check every file a scenario's logbook entries attach.

    Each item of an entry's ``attachments`` is one of ``{path: <picture>}`` or
    ``{plot: <plot spec .json>}``, relative to ``scenarios/<name>/``. A path
    that leaves that directory, an item of another shape, a picture whose
    suffix or bytes are not an accepted image, or a plot spec that is not a
    valid ``.json`` spec is ``value-invalid``; a named file that does not
    exist is ``reference-missing``.

    Args:
        document: The combined document.
        facility_dir: The ``data/facility`` directory the scenarios came from.

    Returns:
        Every stop, in scenario, entry and item order.
    """
    errors: list[FacilityBuildError] = []
    for scenario in document.get("scenarios") or []:
        if not isinstance(scenario, dict) or not isinstance(scenario.get("logbook"), list):
            continue
        name = str(scenario.get("name"))
        files = [f"scenarios/{name}.yaml"]
        root = (facility_dir / "scenarios" / name).resolve()
        for entry in scenario["logbook"]:
            if not isinstance(entry, dict):
                continue
            slot = f"logbook.{entry.get('entry_id')}.attachments"
            items = entry.get("attachments", [])
            if not isinstance(items, list):
                errors.append(
                    _error(
                        "value-invalid",
                        name,
                        files,
                        f"`{slot}` must be a list, got {items!r}",
                        f"write `{slot}` as a list of `path` or `plot` items",
                    )
                )
                continue
            for item in items:
                errors.extend(_attachment_errors(name, files, slot, root, item))
    return errors


def _attachment_errors(
    name: str, files: list[str], slot: str, root: Path, item: Any
) -> Iterator[FacilityBuildError]:
    """The stops of one attachment item of a scenario's logbook entry."""
    from osprey_connectors.simulation.machine import (
        _IMAGE_SIGNATURES,
        _SIGNATURE_BYTES,
        _matches_signature,
        parse_plot_spec,
    )

    where = f"scenarios/{name}/"
    if not isinstance(item, Mapping) or len(item) != 1 or next(iter(item)) not in ("path", "plot"):
        yield _error(
            "value-invalid",
            name,
            files,
            f"`{slot}` item {item!r} is not one of `path` or `plot`",
            f"write each item of `{slot}` as `path: <picture>` or `plot: <plot spec .json>`",
        )
        return
    ((key, rel),) = item.items()
    if not isinstance(rel, str) or not rel:
        yield _error(
            "value-invalid",
            name,
            files,
            f"`{slot}` {key} must be a non-empty path, got {rel!r}",
            f"name a file inside {where}",
        )
        return
    path = (root / rel).resolve()
    if Path(rel).is_absolute() or not path.is_relative_to(root):
        yield _error(
            "value-invalid",
            name,
            files,
            f"`{slot}` {key} {rel} leaves {where}",
            f"name a file inside {where}",
        )
        return
    if key == "plot" and path.suffix.lower() != ".json":
        yield _error(
            "value-invalid",
            name,
            files,
            f"`{slot}` plot {rel} is not a .json plot spec",
            "name a .json plot spec, or attach the picture as `path`",
        )
        return
    signatures = _IMAGE_SIGNATURES.get(path.suffix.lower()) if key == "path" else None
    if key == "path" and signatures is None:
        accepted = ", ".join(sorted(_IMAGE_SIGNATURES))
        yield _error(
            "value-invalid",
            name,
            files,
            f"`{slot}` path {rel} is not a picture (accepted: {accepted})",
            f"attach a picture with one of the suffixes {accepted}",
        )
        return
    if not path.is_file():
        yield _error(
            "reference-missing",
            name,
            files,
            f"`{slot}` names {where}{rel}, which does not exist",
            f"add {where}{rel} or correct `{slot}`",
        )
        return
    if signatures is not None:
        with path.open("rb") as handle:
            head = handle.read(_SIGNATURE_BYTES)
        if not any(_matches_signature(head, signature) for signature in signatures):
            yield _error(
                "value-invalid",
                name,
                files,
                f"`{slot}` path {rel} does not hold {path.suffix.lower()} image data",
                f"replace {where}{rel} with a {path.suffix.lower()} picture",
            )
        return
    try:
        parse_plot_spec(json.loads(path.read_text(encoding="utf-8")), f"plot spec {rel}")
    except (ValueError, UnicodeDecodeError) as exc:
        yield _error(
            "value-invalid",
            name,
            files,
            f"`{slot}` plot {rel} is not a valid plot spec: {exc}",
            f"correct {where}{rel}",
        )


def check_scenario_engines(document: Mapping[str, Any]) -> list[FacilityBuildError]:
    """Check every scenario fault keyed by an engine variable rather than a channel.

    A key under ``faults.<model>`` that names no channel must be one of the
    fault names the model's engine declares (``engine-invalid`` otherwise);
    its value is coerced as a float and held to that fault's range.

    Args:
        document: The combined document, its wiring slots filled.

    Returns:
        Every stop, in scenario, model and key order.
    """
    channels = {str(c["id"]): c for c in document.get("channels") or [] if isinstance(c, dict)}
    models = {str(m["name"]): m for m in document.get("models") or [] if isinstance(m, dict)}
    rosters: dict[str, FaultRoster | None] = {}
    errors: list[FacilityBuildError] = []
    for scenario in document.get("scenarios") or []:
        if not isinstance(scenario, dict) or not isinstance(scenario.get("faults"), dict):
            continue
        name = str(scenario.get("name"))
        files = [f"scenarios/{name}.yaml"]
        for model_name, targets in sorted(scenario["faults"].items(), key=lambda kv: str(kv[0])):
            model = models.get(str(model_name))
            if model is None or not isinstance(targets, dict):
                continue
            keys = sorted((k for k in targets if str(k) not in channels), key=str)
            if not keys:
                continue
            if str(model_name) not in rosters:
                rosters[str(model_name)] = fault_roster(model, channels)
            roster = rosters[str(model_name)]
            for key in keys:
                errors.extend(
                    _named_fault(name, files, str(model_name), str(key), targets[key], roster)
                )
    return errors


def _named_fault(
    name: str,
    files: list[str],
    model: str,
    key: str,
    value: Any,
    roster: FaultRoster | None,
) -> Iterator[FacilityBuildError]:
    slots = roster.slots if roster is not None else {}
    slot = f"faults.{model}.{key}"
    fault = slots.get(key)
    if fault is None:
        engine = roster.engine if roster is not None else "engine"
        article = "an" if engine[:1] in "aeiou" else "a"
        remedy = f"name a channel address or {article} {engine} variable (<address>/<field>)"
        closest = difflib.get_close_matches(key, sorted(slots), n=3)
        if closest:
            remedy += f"; closest: {', '.join(closest)}"
        yield _error(
            "engine-invalid",
            name,
            files,
            f"{slot} names no channel and no {engine} variable",
            remedy,
        )
        return
    yield from _value_errors(name, files, slot, value, fault)
