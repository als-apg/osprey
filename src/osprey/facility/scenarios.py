"""The faults a scenario writes into a model, checked against the model's engine.

A scenario faults a model in two spellings, both under ``faults.<model>``:

* by channel address, ``{<address>: {<fault field>: <value>}}``;
* by the engine's own fault name, ``{<engine variable>: <value>}``.

The engine plug-in names what a model built from a wiring declares through
``fault_variables(wiring)``: fault name -> a slot carrying ``address``,
``field``, ``value_range`` and ``options``. A plug-in is reached through the
``osprey.simulation.engines`` entry-point group, never by import. Every value
is coerced as a float and must lie inside its slot's ``value_range`` or equal
one of its ``options``.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterator, Mapping
from importlib import metadata
from typing import Any

from osprey.facility.errors import FacilityBuildError

__all__ = ["FaultRoster", "check_scenario_engines", "fault_roster", "map_fault_errors"]


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
