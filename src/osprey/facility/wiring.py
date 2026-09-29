"""The computed slots of every wiring record.

A wiring record of a deck-bearing model carries four slots the build fills and
no author states: ``direction``, ``unit``, ``default`` and ``value_range``. This
module is their one writer. It runs after S5, so every id a record names
resolves and the record rules hold, and before S6, whose checks compare against
the ``default`` it writes.

A wired element the deck does not hold exactly once stops with one line naming
the wiring record, whichever slice or role it sits on: ``wiring-conflict`` when
the deck repeats it, ``engine-invalid`` when the deck lacks it
(``element_stop``). An engine signals such a stop by raising a
``FacilityBuildError`` that carries ``element`` and ``count`` (0 absent, above 1
repeated).
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

from osprey.facility.errors import FacilityBuildError
from osprey.facility.provenance import add_defaults
from osprey.facility.validate import Validated

__all__ = ["element_stop", "fill_wiring_slots"]

#: The entry-point group every simulation engine registers under.
_ENGINE_GROUP = "osprey.simulation.engines"

_MODELS_FILE = "models.yaml"


def fill_wiring_slots(validated: Validated) -> list[FacilityBuildError]:
    """Fill the computed slots of each deck-bearing model's wiring records.

    The slots are written in place on ``validated.document``:

    * ``default``: the engine's start value for the address, from the deck
      at ``facility_dir / model.deck``;
    * ``direction``: ``write`` for a setpoint channel, else ``read``;
    * ``unit``: the channel's unit, when the channel states one;
    * ``value_range``: ``[min_value, max_value]`` of the channel's limits
      record, when that record states both bounds.

    Each filled slot is recorded in the record's ``provenance.defaults``. A
    model without a deck is left untouched. The engine is reached through the
    ``osprey.simulation.engines`` entry-point group and loaded only here. A
    record the engine has no start value for is left unfilled and stops; the
    model's other records are still filled.

    Args:
        validated: What stages S1 to S5 produced.

    Returns:
        Every stop, sorted by record id; empty when every slot was filled.

    Raises:
        RuntimeError: The stage ran before S2 produced the document.
    """
    document = validated.document
    if document is None:
        raise RuntimeError("a stage ran before the stage that produces its input")
    channels: dict[str, dict[str, Any]] = {c["id"]: c for c in document.get("channels", [])}
    limits: dict[str, dict[str, Any]] = {
        r["address"]: r for r in (document.get("limits") or {}).get("records", [])
    }
    setpoint_of = {
        channel.get("pair", channel["id"]): channel["id"]
        for channel in channels.values()
        if _role(channel) == "setpoint"
    }
    engines = metadata.entry_points(group=_ENGINE_GROUP)
    errors: list[FacilityBuildError] = []
    for model in document.get("models", []):
        if "deck" not in model:
            continue
        errors.extend(
            _fill_model(model, validated.facility_dir, engines, channels, limits, setpoint_of)
        )
    return sorted(errors, key=lambda error: error.record_id)


def _fill_model(
    model: dict[str, Any],
    facility_dir: Path,
    engines: metadata.EntryPoints,
    channels: Mapping[str, Mapping[str, Any]],
    limits: Mapping[str, Mapping[str, Any]],
    setpoint_of: Mapping[str, str],
) -> list[FacilityBuildError]:
    name = model["engine"]
    if name not in engines.names:
        return [
            FacilityBuildError(
                "engine-missing",
                model["name"],
                _sources(model),
                "install the engine's package, or name an engine the environment registers",
                record_kind="model",
                detail=f"engine {name} is not registered under {_ENGINE_GROUP}",
            )
        ]
    engine = engines[name].load()
    records = [r for r in model.get("wiring", []) if r["address"] in channels]
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
        )
    except FacilityBuildError as stop:
        record = next((r for r in model.get("wiring", []) if r["id"] == stop.record_id), None)
        translated = element_stop(stop, record, model["name"]) if record is not None else None
        return [translated or stop]
    errors: list[FacilityBuildError] = []
    for record in records:
        address = record["address"]
        if address not in values:
            errors.append(
                FacilityBuildError(
                    "engine-invalid",
                    record["id"],
                    _sources(record),
                    "name an element or slices for the record, or leave the channel unwired",
                    record_kind="wiring",
                    detail=f"{address}: no element to read a start value from",
                )
            )
            continue
        _fill_record(record, values[address], channels[address], limits.get(address))
    return errors


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
        _sources(record),
        stop.remedy,
        record_kind="wiring",
        detail=f"{stop.detail} of model {model}",
    )


def _fill_record(
    record: dict[str, Any],
    default: float,
    channel: Mapping[str, Any],
    limit: Mapping[str, Any] | None,
) -> None:
    slots: dict[str, Any] = {
        "direction": "write" if _role(channel) == "setpoint" else "read",
        "default": default,
    }
    if "unit" in channel:
        slots["unit"] = channel["unit"]
    if limit is not None and "min_value" in limit and "max_value" in limit:
        slots["value_range"] = [float(limit["min_value"]), float(limit["max_value"])]
    present = sorted(set(slots) & set(record))
    if present:
        raise RuntimeError(f"wiring {record['id']} already carries computed slots {present}")
    record.update(slots)
    record["provenance"] = add_defaults(record.get("provenance", {}), slots)


def _role(channel: Mapping[str, Any]) -> str:
    return str(channel.get("role", "readback"))


def _sources(record: Mapping[str, Any]) -> list[str]:
    provenance = record.get("provenance") or {}
    files = sorted({str(source["file"]) for source in provenance.get("sources", [])})
    return files or [_MODELS_FILE]
