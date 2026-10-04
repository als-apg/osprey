"""The simulator view: the models a render serves, their addresses and decks.

Written into ``<render>/data/simulator/``::

    served_models.json   {schema: osprey.facility.served_models/1, models: [...]}
    addresses.json       {schema: osprey.facility.addresses/1, channels: [...], status: [...]}
    decks/<model>.json   a byte copy of each deck-bearing model's deck
    variables.json       {schema: osprey.facility.simulator/1, code, models: [...], channels: [...]}
    seeds.json           {schema: osprey.facility.seeds/1, seeds: {<address>: <seed record>}}
    scenarios.json       {schema: osprey.facility.scenarios/1, scenarios: [...]}

``models`` lists the served physics models sorted by name, then ``texture``;
readers building physics children or selectors skip engine ``texture``.
``channels`` is every channel address of the facility file, sorted; ``status``
is ``<code>:SIM:<model>:STATUS`` for each served physics model and appears in
no other view. A deck is copied for every model that names one, served or not,
so a render's deck set does not depend on ``simulation.models``.

``variables.json`` lists every model record of the facility file, sorted by
name, as ``{name, engine, served, settings, deck, wiring}``: ``deck`` is the
copy's path under ``data/simulator/`` or null, ``wiring`` is
``simulator_wiring`` of the model (empty for a model without wiring). Its
``channels`` are every channel, sorted by address, as ``{address, role, pair,
value_type, options?, shape?, unit, description, writable, value_range,
owner}``: ``role`` defaults to ``readback`` and ``value_type`` to ``float``
as in the facility file; ``pair`` is a setpoint's readback (itself when it names none) and
null on any other role; ``owner`` is the model wiring the address, else
``texture``. ``writable`` and ``value_range`` come from the channel's limits
record (``value_range`` = ``[min_value, max_value]`` when it states both, else
null); a setpoint without a record is writable only when the simulated
target's limits mode is ``optional``, and every other channel without a record
is not writable.

``seeds.json`` maps each channel carrying a seed record to that record.
``scenarios.json`` lists every scenario, sorted by name, with each block it
states carried verbatim; ``faults`` maps each faulted model to ``{writes}``,
plus ``inactive: model not served`` when the render does not serve it.

``simulator_wiring`` gives one model's wiring entries: each wired address with
its element (or slices), engine block and calibration, plus the channel facts
the build filled in (direction, unit, default, value_range) and, for a
waveform channel, its ``value_type`` and ``shape``.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey.facility import TEXTURE
from osprey.facility.build import FacilityDocument
from osprey.facility.views import ViewInputs, view_bytes

__all__ = [
    "ADDRESSES_FILE",
    "ADDRESSES_SCHEMA",
    "DECKS_DIR",
    "INACTIVE_UNSERVED",
    "SCENARIOS_FILE",
    "SCENARIOS_SCHEMA",
    "SEEDS_FILE",
    "SEEDS_SCHEMA",
    "SERVED_MODELS_FILE",
    "SERVED_MODELS_SCHEMA",
    "VARIABLES_FILE",
    "VARIABLES_SCHEMA",
    "simulator_wiring",
    "status_address",
    "write_simulator_view",
]

SERVED_MODELS_FILE = "served_models.json"
SERVED_MODELS_SCHEMA = "osprey.facility.served_models/1"
ADDRESSES_FILE = "addresses.json"
ADDRESSES_SCHEMA = "osprey.facility.addresses/1"
DECKS_DIR = "decks"
VARIABLES_FILE = "variables.json"
VARIABLES_SCHEMA = "osprey.facility.simulator/1"
SEEDS_FILE = "seeds.json"
SEEDS_SCHEMA = "osprey.facility.seeds/1"
SCENARIOS_FILE = "scenarios.json"
SCENARIOS_SCHEMA = "osprey.facility.scenarios/1"

#: The mark a scenario's faults carry for a model the render does not serve.
INACTIVE_UNSERVED = "model not served"

#: The channel keys ``variables.json`` carries only when the record states them.
_CHANNEL_OPTIONAL_KEYS = ("options", "shape", "precision")

#: The scenario blocks ``scenarios.json`` carries verbatim when stated.
_SCENARIO_BLOCKS = (
    "description",
    "overrides",
    "archiver",
    "logbook",
    "drivers",
    "couple",
    "noise",
)

#: The keys a wiring entry may carry, in emission order.
_WIRING_ENTRY_KEYS = (
    "id",
    "address",
    "element",
    "slices",
    "engine",
    "calibration",
    "direction",
    "unit",
    "default",
    "value_range",
)


def simulator_wiring(facility: FacilityDocument, model: str) -> list[dict[str, Any]]:
    """The wiring entries of one model, in the facility file's record order.

    Each entry holds the keys of ``_WIRING_ENTRY_KEYS`` the wiring record
    carries, and, where the wired channel is a waveform, the channel's
    ``value_type`` and ``shape``; a key the record lacks is absent from its
    entry.

    Args:
        facility: The in-memory facility file.
        model: The model's name.

    Returns:
        One entry per wiring record of the model; empty for a model without
        wiring. The entries share no objects with ``facility``.

    Raises:
        KeyError: ``facility`` has no model named ``model``.
    """
    channels = {str(channel["id"]): channel for channel in facility.get("channels", [])}
    for entry in facility.get("models", []):
        if entry["name"] == model:
            return [
                {
                    **{
                        key: copy.deepcopy(record[key])
                        for key in _WIRING_ENTRY_KEYS
                        if key in record
                    },
                    **_waveform(channels.get(str(record["address"]), {})),
                }
                for record in entry.get("wiring", [])
            ]
    raise KeyError(f"the facility file has no model {model!r}")


def _waveform(channel: Mapping[str, Any]) -> dict[str, Any]:
    """A waveform channel's ``value_type`` and ``shape``; nothing for any other channel."""
    if channel.get("value_type") != "waveform":
        return {}
    return {"value_type": "waveform", "shape": list(channel.get("shape") or [])}


def status_address(code: str, model: str) -> str:
    """The status address the simulator serves for one physics model.

    Args:
        code: The facility's identity ``code``.
        model: The model's name.

    Returns:
        ``<code>:SIM:<model>:STATUS``.
    """
    return f"{code}:SIM:{model}:STATUS"


def _unlisted_setpoints_writable(rendered_config: Mapping[str, Any]) -> bool:
    """Whether a setpoint without a limits record is writable on the simulated target."""
    from osprey_connectors.types import (
        LIMITS_MODE_OPTIONAL,
        resolve_control_system_type,
        type_limits_posture,
    )

    section = rendered_config.get("control_system") or {}
    posture = type_limits_posture(section, resolve_control_system_type(section))
    return posture.mode == LIMITS_MODE_OPTIONAL


def _variables_document(inputs: ViewInputs) -> dict[str, Any]:
    from osprey.facility.views.limits import limits_document

    doc = inputs.doc
    models = sorted(doc.get("models", []), key=lambda model: str(model["name"]))
    owners: dict[str, str] = {}
    for model in models:
        for record in model.get("wiring", []):
            owners.setdefault(str(record["address"]), str(model["name"]))
    limits = limits_document(doc)
    unlisted_writable = _unlisted_setpoints_writable(inputs.rendered_config)

    channels: list[dict[str, Any]] = []
    for record in sorted(doc.get("channels", []), key=lambda channel: str(channel["id"])):
        address = str(record["id"])
        role = record.get("role", "readback")
        entry = limits.get(address)
        if entry is not None:
            writable = bool(entry["writable"])
            bounds = (entry.get("min_value"), entry.get("max_value"))
            value_range = list(bounds) if None not in bounds else None
        else:
            writable = role == "setpoint" and unlisted_writable
            value_range = None
        channel: dict[str, Any] = {
            "address": address,
            "role": role,
            "pair": (record.get("pair") or address) if role == "setpoint" else None,
            "value_type": record.get("value_type", "float"),
            "unit": record.get("unit"),
            "description": record.get("description"),
            "writable": writable,
            "value_range": value_range,
            "owner": owners.get(address, TEXTURE),
        }
        for key in _CHANNEL_OPTIONAL_KEYS:
            if record.get(key) is not None:
                channel[key] = copy.deepcopy(record[key])
        channels.append(channel)

    return {
        "schema": VARIABLES_SCHEMA,
        "code": str(doc["identity"]["code"]),
        "models": [
            {
                "name": model["name"],
                "engine": model.get("engine"),
                "served": model["name"] in inputs.served,
                "settings": copy.deepcopy(model.get("settings") or {}),
                "deck": f"{DECKS_DIR}/{model['name']}.json" if model.get("deck") else None,
                "wiring": simulator_wiring(doc, str(model["name"])),
            }
            for model in models
        ],
        "channels": channels,
    }


def _seeds_document(doc: FacilityDocument) -> dict[str, Any]:
    seeds = {
        str(channel["id"]): copy.deepcopy(channel["simulation"])
        for channel in doc.get("channels", [])
        if channel.get("simulation") is not None
    }
    return {"schema": SEEDS_SCHEMA, "seeds": seeds}


def _scenarios_document(inputs: ViewInputs) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    for scenario in sorted(inputs.doc.get("scenarios", []), key=lambda s: str(s["name"])):
        entry: dict[str, Any] = {"name": scenario["name"]}
        for block in _SCENARIO_BLOCKS:
            if scenario.get(block) is not None:
                entry[block] = copy.deepcopy(scenario[block])
        faults = scenario.get("faults")
        if faults is not None:
            entry["faults"] = {}
            for model, writes in faults.items():
                fault: dict[str, Any] = {"writes": copy.deepcopy(writes)}
                if model not in inputs.served:
                    fault["inactive"] = INACTIVE_UNSERVED
                entry["faults"][model] = fault
        scenarios.append(entry)
    return {"schema": SCENARIOS_SCHEMA, "scenarios": scenarios}


def write_simulator_view(root: Path, inputs: ViewInputs) -> list[Path]:
    """Write the simulator view into ``root``.

    Args:
        root: The view's directory, ``<render>/data/simulator``.
        inputs: The render's view inputs.

    Returns:
        The files written, sorted.
    """
    doc = inputs.doc
    code = str(doc["identity"]["code"])
    physics = [name for name in inputs.served if name != TEXTURE]
    documents = {
        SERVED_MODELS_FILE: {"schema": SERVED_MODELS_SCHEMA, "models": list(inputs.served)},
        ADDRESSES_FILE: {
            "schema": ADDRESSES_SCHEMA,
            "channels": sorted(str(channel["id"]) for channel in doc.get("channels", [])),
            "status": [status_address(code, name) for name in physics],
        },
        VARIABLES_FILE: _variables_document(inputs),
        SEEDS_FILE: _seeds_document(doc),
        SCENARIOS_FILE: _scenarios_document(inputs),
    }

    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, document in documents.items():
        target = root / name
        target.write_bytes(view_bytes(document))
        written.append(target)

    decks = root / DECKS_DIR
    for model in doc.get("models", []):
        deck = model.get("deck")
        if deck is None:
            continue
        decks.mkdir(exist_ok=True)
        target = decks / f"{model['name']}.json"
        target.write_bytes((inputs.facility_dir / deck).read_bytes())
        written.append(target)
    return sorted(written)
