"""The simulator view: the models a render serves, their addresses and decks.

Written into ``<render>/data/simulator/``::

    served_models.json   {schema: osprey.facility.served_models/1, models: [...]}
    addresses.json       {schema: osprey.facility.addresses/1, channels: [...], status: [...],
                          server_health: [...]}
    decks/<model><suffix>   a byte copy of each deck-bearing model's deck, under its own suffix
    variables.json       {schema: osprey.facility.simulator/4, code, models: [...], channels: [...]}
    seeds.json           {schema: osprey.facility.seeds/2, seeds: {<address>: <seed record>}}
    follows.json         {schema: osprey.facility.follows/1, totals: {...}, pairs: [...]}
    scenarios.json       {schema: osprey.facility.scenarios/3, scenarios: [...]}
    scenarios/<name>/    a byte copy of each file a scenario's logbook entries attach

``models`` lists the served physics models sorted by name, then ``texture``;
readers building physics children or selectors skip engine ``texture``.
``channels`` is every channel address of the facility file, sorted; ``status``
is ``<code>:SIM:<model>:STATUS`` for each served physics model and appears in
no other view. A deck is copied for every model that names one, served or not,
so a render's deck set does not depend on ``simulation.models``.

``variables.json`` lists every model record of the facility file, sorted by
name, as ``{name, engine, served, settings, deck, wiring, inputs?}``: ``deck`` is the
copy's path under ``data/simulator/`` or null, ``wiring`` is
``simulator_wiring`` of the model (empty for a model without wiring) and
``inputs`` is the model's ``inputs`` verbatim, in declared order, present only
when the model declares one, so a model without inputs is written as under
``/3``. Its
``channels`` are every channel, sorted by address, as ``{address, role, pair,
value_type, options?, shape?, unit, description, writable, value_range,
owner, on}``: ``role`` defaults to ``readback`` and ``value_type`` to ``float``
as in the facility file; ``pair`` is a setpoint's readback (itself when it names none) and
null on any other role; ``owner`` is the model wiring the address, as a record's
``address`` or under its ``also``, else ``texture``; ``on`` is the node the facility record puts the channel on,
``{device: id}`` or ``{place: id}``, or null. ``writable`` and ``value_range`` come from the channel's limits
record (``value_range`` = ``[min_value, max_value]`` when it states both, else
null); a setpoint without a record is writable only when the simulated
target's limits mode is ``optional``, and every other channel without a record
is not writable.

``seeds.json`` maps each channel carrying a seed record to that record, less
``follows``, which a setpoint's ``pair`` in ``variables.json`` already carries;
a seed of ``follows`` alone is not listed.
``follows.json`` says for each pair, a setpoint whose ``pair`` names another
channel, whether the simulation follows it: ``pairs`` lists ``{setpoint,
readback, follow, why, copy?, labels?}`` sorted by readback, then setpoint,
where ``follow`` is ``default``, ``seed`` (the readback's seed ``follows`` names
the setpoint) or ``not followed``; a followed pair states ``copy``, ``value`` or
``index``, and one copied by index lists ``labels``, ``[setpoint state, readback
label]`` per index; ``totals`` counts each word. The runtime does not read it; a
setpoint's ``pair`` in ``variables.json`` is the readback it follows, so the two
agree.
``scenarios.json`` lists every scenario, sorted by name, with each block it
states carried verbatim, but that a key of ``overrides``, ``couple``, ``noise``,
``channel_faults`` or a model's fault writes, and an entry of ``still``, that
names a served model's wiring record under ``also`` is written as the record's ``address``;
``faults`` maps each faulted model to ``{writes}``,
plus ``inactive: model not served`` when the render does not serve it;
``stopped_servers`` keeps only the IOCs this render serves, and each named IOC of
a model the render does not serve moves to ``inactive_servers``, mapped to
``model not served``, and stops nothing; such a name is refused only when its
scenario is one of the render's ``simulation.default_scenarios``;
``still`` is the sorted list of float addresses its ``all``, addresses and
selections resolve to; a scenario named ``still`` that stills every float
reading is listed when no file defines one. Each
file a logbook entry's ``attachments`` names, relative to the facility's
``scenarios/<name>/``, is copied to the same path under the view's
``scenarios/<name>/``, so the entries read back against the view's copy.

``simulator_wiring`` gives one model's wiring entries: each wired address with
the further addresses it states under ``also``, its element (or slices), engine
block and calibration, plus the slots the build
filled from the channel, its limits record and the model's engine (direction,
unit, default, value_range, engine_role, plane, refresh, marks) and, for a waveform
channel, its ``value_type`` and ``shape``.
"""

from __future__ import annotations

import copy
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from osprey.facility.views import ViewInputs, view_bytes
from osprey_connectors.simulation.servers import served_servers
from osprey_connectors.simulation.state import STILL_SCENARIO
from osprey_connectors.simulation.values import DEFAULT_VALUE_TYPE, is_waveform
from osprey_connectors.simulation.view import (
    ADDRESSES_FILE,
    ADDRESSES_SCHEMA,
    DECKS_DIR,
    SCENARIOS_DIR,
    SCENARIOS_FILE,
    SCENARIOS_SCHEMA,
    SEEDS_FILE,
    SEEDS_SCHEMA,
    SERVED_MODELS_FILE,
    SERVED_MODELS_SCHEMA,
    SETPOINT,
    VARIABLES_FILE,
    VARIABLES_SCHEMA,
    SimulatorView,
)
from osprey_facility import DEFAULT_ROLE, STILL_ALL, TEXTURE, wiring_addresses
from osprey_facility.records import FacilityRecords
from osprey_facility.scenarios import scenario_logbook
from osprey_facility.simulated import resolved_still, server_health_addresses, status_address

__all__ = [
    "ADDRESSES_FILE",
    "FOLLOWS_FILE",
    "FOLLOWS_SCHEMA",
    "ADDRESSES_SCHEMA",
    "DECKS_DIR",
    "INACTIVE_UNSERVED",
    "SCENARIOS_DIR",
    "SCENARIOS_FILE",
    "SCENARIOS_SCHEMA",
    "SEEDS_FILE",
    "SEEDS_SCHEMA",
    "SERVED_MODELS_FILE",
    "SERVED_MODELS_SCHEMA",
    "VARIABLES_FILE",
    "VARIABLES_SCHEMA",
    "served_deck_path",
    "simulator_wiring",
    "unlisted_setpoints_writable",
    "withdraw_scenarios",
    "write_simulator_view",
]


#: The file saying which pairs the simulation follows, and why; no runtime reads it.
FOLLOWS_FILE = "follows.json"
FOLLOWS_SCHEMA = "osprey.facility.follows/1"

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
    "sessions",
    "drivers",
    "couple",
    "noise",
    "channel_faults",
    "stopped_servers",
)

#: The scenario blocks keyed by channel address.
_ADDRESS_KEYED_BLOCKS = ("overrides", "couple", "noise", "channel_faults")

#: The scenario ``scenarios.json`` lists when no file defines ``still``.
_BUILT_IN_STILL = {
    "name": STILL_SCENARIO,
    "description": "Every reading serves without drift, couplings or noise.",
    "still": STILL_ALL,
}

#: The keys a wiring entry may carry, in emission order: the wiring record's
#: authored keys, then the slots the build computed.
_WIRING_ENTRY_KEYS = (
    "id",
    "address",
    "also",
    "element",
    "slices",
    "engine",
    "calibration",
    "direction",
    "unit",
    "default",
    "value_range",
    "engine_role",
    "plane",
    "refresh",
    "marks",
)

#: The channels and models last wired and each model's wiring of them, keyed on
#: the model: every render of one build hands in the same document, and a render
#: whose served set or configuration differs writes this view again, so each model
#: is wired once per build. Keyed on the two lists the wiring reads, which every
#: read model of one document shares. One slot: the view reads one build's document
#: at a time.
_wired: list[tuple[Any, Any, dict[str, list[dict[str, Any]]]]] = []


def simulator_wiring(records: FacilityRecords, model: str) -> list[dict[str, Any]]:
    """The wiring entries of one model, in the facility file's record order.

    Each entry holds the keys of ``_WIRING_ENTRY_KEYS`` the wiring record
    carries; where the wired channel is a waveform, the entry adds the
    channel's ``value_type`` and ``shape``. A key the record lacks is absent
    from its entry.

    Args:
        records: The facility file's records.
        model: The model's name.

    Returns:
        One entry per wiring record of the model; empty for a model without
        wiring. The entries share no objects with the facility file.

    Raises:
        KeyError: The facility file has no model named ``model``.
    """
    entry = records.require("model", model)
    channels, models = records.document.get("channels"), records.document.get("models")
    slot = next((s for s in _wired if s[0] is channels and s[1] is models), None)
    if slot is None:
        slot = (channels, models, {})
        _wired[:] = [slot]
    wiring = slot[2].get(model)
    if wiring is None:
        wiring = slot[2][model] = _wire(records, entry)
    return copy.deepcopy(wiring)


def _wire(records: FacilityRecords, entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compute :func:`simulator_wiring` of ``entry`` without the per-build memo."""
    return [
        {
            **{key: copy.deepcopy(record[key]) for key in _WIRING_ENTRY_KEYS if key in record},
            **_waveform(records.record("channel", str(record["address"])) or {}),
        }
        for record in entry.get("wiring", [])
    ]


def served_deck_path(model: Mapping[str, Any]) -> str | None:
    """The file the view serves a model's deck as, relative to the view.

    Args:
        model: A facility-file model record.

    Returns:
        ``decks/<model><suffix>``, the suffix of the deck the model names;
        ``None`` for a model with no deck.
    """
    deck = model.get("deck")
    if not deck:
        return None
    return f"{DECKS_DIR}/{model['name']}{Path(deck).suffix}"


def _waveform(channel: Mapping[str, Any]) -> dict[str, Any]:
    """A waveform channel's ``value_type`` and ``shape``; nothing for any other channel."""
    if not is_waveform(channel.get("value_type")):
        return {}
    return {"value_type": "waveform", "shape": list(channel.get("shape") or [])}


def unlisted_setpoints_writable(rendered_config: Mapping[str, Any]) -> bool:
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

    doc = inputs.records.document
    models = sorted(doc.get("models", []), key=lambda model: str(model["name"]))
    owners: dict[str, str] = {}
    for model in models:
        for record in model.get("wiring", []):
            for address in wiring_addresses(record):
                owners.setdefault(address, str(model["name"]))
    limits = limits_document(inputs.records)
    unlisted_writable = unlisted_setpoints_writable(inputs.rendered_config)

    from osprey_facility.pairing import simulated_pairs

    followed = {
        setpoint: readback
        for readback, setpoint in simulated_pairs(
            {str(c["id"]): c for c in doc.get("channels", [])}
        ).items()
    }
    channels: list[dict[str, Any]] = []
    for record in sorted(doc.get("channels", []), key=lambda channel: str(channel["id"])):
        address = str(record["id"])
        role = record.get("role", DEFAULT_ROLE)
        entry = limits.get(address)
        if entry is not None:
            writable = bool(entry["writable"])
            bounds = (entry.get("min_value"), entry.get("max_value"))
            value_range = list(bounds) if None not in bounds else None
        else:
            writable = role == SETPOINT and unlisted_writable
            value_range = None
        channel: dict[str, Any] = {
            "address": address,
            "role": role,
            "pair": followed.get(address, address) if role == SETPOINT else None,
            "value_type": record.get("value_type", DEFAULT_VALUE_TYPE),
            "unit": record.get("unit"),
            "description": record.get("description"),
            "writable": writable,
            "value_range": value_range,
            "owner": owners.get(address, TEXTURE),
            "on": copy.deepcopy(record.get("on")),
        }
        for key in _CHANNEL_OPTIONAL_KEYS:
            if record.get(key) is not None:
                channel[key] = copy.deepcopy(record[key])
        channels.append(channel)

    return {
        "schema": VARIABLES_SCHEMA,
        "code": str(doc["identity"]["code"]),
        "models": [_model_entry(inputs, model) for model in models],
        "channels": channels,
    }


def _model_entry(inputs: ViewInputs, model: Mapping[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": model["name"],
        "engine": model.get("engine"),
        "served": model["name"] in inputs.served,
        "settings": copy.deepcopy(model.get("settings") or {}),
        "deck": served_deck_path(model),
        "wiring": simulator_wiring(inputs.records, str(model["name"])),
    }
    if model.get("inputs"):
        entry["inputs"] = copy.deepcopy(list(model["inputs"]))
    return entry


def _seeds_document(doc: Mapping[str, Any]) -> dict[str, Any]:
    # `follows` is decided at build time and carried by a setpoint's `pair` in
    # variables.json; the runtime reads seeds for values and motion only.
    seeds = {}
    for channel in doc.get("channels", []):
        seed = channel.get("simulation")
        if seed is None:
            continue
        kept = {key: copy.deepcopy(value) for key, value in seed.items() if key != "follows"}
        if kept:
            seeds[str(channel["id"])] = kept
    return {"schema": SEEDS_SCHEMA, "seeds": seeds}


def _follows_document(doc: Mapping[str, Any]) -> dict[str, Any]:
    from osprey_facility.pairing import follow_totals, pair_follows

    follows = pair_follows({str(c["id"]): c for c in doc.get("channels", [])})
    return {
        "schema": FOLLOWS_SCHEMA,
        "totals": follow_totals(follows),
        "pairs": [
            {key: value for key, value in follow._asdict().items() if value is not None}
            for follow in follows
        ],
    }


def _split_stopped_servers(
    inputs: ViewInputs, scenario: Mapping[str, Any]
) -> tuple[list[str], dict[str, str]]:
    """A scenario's ``stopped_servers``: the names this render stops, and the inactive rest.

    The validator refuses a name that is no simulated IOC of the facility;
    only the render knows which models it serves. A name of an unserved model
    stops nothing, so it follows the rule for a fault on an unserved model: it
    is marked ``model not served`` and left out of the stop. It is refused only
    when the scenario is one of the render's default scenarios, which would
    otherwise come up stopping nothing.

    Raises:
        FacilityBuildError: ``value-invalid``: a default scenario names an IOC
            this render does not serve.
    """
    from osprey.simulation.apply import resolve_default_scenarios
    from osprey_facility.errors import FacilityBuildError

    name = str(scenario["name"])
    physics = [
        str(model["name"])
        for model in inputs.records.document.get("models", [])
        if model["name"] in inputs.served
        and model["name"] != TEXTURE
        and model.get("engine") != TEXTURE
    ]
    known = served_servers(physics)
    stopped = [str(ioc) for ioc in scenario["stopped_servers"] if ioc in known]
    unserved = [str(ioc) for ioc in scenario["stopped_servers"] if ioc not in known]
    if unserved and name in resolve_default_scenarios(inputs.rendered_config):
        raise FacilityBuildError(
            "value-invalid",
            name,
            [f"{SCENARIOS_DIR}/{name}.yaml"],
            f"serve the model in this render, remove `{unserved[0]}` from `stopped_servers`, "
            f"or drop `{name}` from `simulation.default_scenarios`; "
            f"this render serves {', '.join(f'`{ioc}`' for ioc in sorted(known))}",
            record_kind="scenario",
            detail=f"default scenario `{name}`'s `stopped_servers` names `{unserved[0]}`, "
            "which this render does not serve",
        )
    return stopped, dict.fromkeys(unserved, INACTIVE_UNSERVED)


def _record_addresses(document: Mapping[str, Any], served: Iterable[str]) -> dict[str, str]:
    """Each address a served model's wiring record names under ``also``, mapped to its ``address``.

    A model the render does not serve leaves every address of its wiring to
    the texture model, each a channel of its own.
    """
    served = set(served)
    return {
        further: address
        for model in document.get("models", [])
        if model.get("name") in served
        for record in model.get("wiring", [])
        for address, *also in [wiring_addresses(record)]
        for further in also
    }


def _by_record_address(block: Any, record_of: Mapping[str, str]) -> Any:
    """A copy of a scenario block keyed by address, each key the address of its wiring record.

    A key naming a record under ``also`` becomes the record's ``address``; an
    entry keyed by that ``address`` itself stands over it.
    """
    if not isinstance(block, Mapping) or not any(str(key) in record_of for key in block):
        return copy.deepcopy(block)
    keyed = {record_of[str(key)]: value for key, value in block.items() if str(key) in record_of}
    keyed.update({key: value for key, value in block.items() if str(key) not in record_of})
    return copy.deepcopy(keyed)


def _scenarios_document(inputs: ViewInputs) -> dict[str, Any]:
    record_of = _record_addresses(inputs.records.document, inputs.served)
    scenarios: list[dict[str, Any]] = []
    for scenario in sorted(
        inputs.records.document.get("scenarios", []), key=lambda s: str(s["name"])
    ):
        entry: dict[str, Any] = {"name": scenario["name"]}
        for block in _SCENARIO_BLOCKS:
            if scenario.get(block) is None:
                continue
            if block in _ADDRESS_KEYED_BLOCKS:
                entry[block] = _by_record_address(scenario[block], record_of)
            else:
                entry[block] = copy.deepcopy(scenario[block])
        if scenario.get("stopped_servers"):
            entry["stopped_servers"], inactive = _split_stopped_servers(inputs, scenario)
            if inactive:
                entry["inactive_servers"] = inactive
        if scenario.get("still") is not None:
            still = resolved_still(inputs.records, scenario)
            entry["still"] = sorted({record_of.get(address, address) for address in still})
        faults = scenario.get("faults")
        if faults is not None:
            entry["faults"] = {}
            for model, writes in faults.items():
                fault: dict[str, Any] = {"writes": _by_record_address(writes, record_of)}
                if model not in inputs.served:
                    fault["inactive"] = INACTIVE_UNSERVED
                entry["faults"][model] = fault
        scenarios.append(entry)
    if all(str(entry["name"]) != STILL_SCENARIO for entry in scenarios):
        scenarios.append(
            {**_BUILT_IN_STILL, "still": resolved_still(inputs.records, _BUILT_IN_STILL)}
        )
        scenarios.sort(key=lambda entry: str(entry["name"]))
    return {"schema": SCENARIOS_SCHEMA, "scenarios": scenarios}


def _copy_scenario_files(root: Path, inputs: ViewInputs) -> list[Path]:
    """Copy each file a scenario's logbook entries attach into the view.

    Every scenario's entries are read first
    (:func:`~osprey_facility.scenarios.scenario_logbook`), so an attachment
    that leaves its scenario directory, names a missing file or holds a
    malformed picture is refused before anything is copied. A file two entries
    attach is copied, and listed, once.
    """
    copies: dict[Path, Path] = {}
    for scenario in inputs.records.document.get("scenarios", []):
        name = str(scenario["name"])
        source = inputs.facility_dir / SCENARIOS_DIR / name
        scenario_logbook(scenario, source)
        for entry in scenario.get("logbook") or []:
            for item in entry.get("attachments") or []:
                for rel in item.values():
                    copies.setdefault(root / SCENARIOS_DIR / name / rel, source / rel)
    for target, origin in copies.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(origin.read_bytes())
    return list(copies)


def write_simulator_view(root: Path, inputs: ViewInputs) -> list[Path]:
    """Write the simulator view into ``root``.

    Args:
        root: The view's directory, ``<render>/data/simulator``.
        inputs: The render's view inputs.

    Returns:
        The files written, in the order they were written.
    """
    doc = inputs.records.document
    code = str(doc["identity"]["code"])
    physics = [name for name in inputs.served if name != TEXTURE]
    documents = {
        SERVED_MODELS_FILE: {"schema": SERVED_MODELS_SCHEMA, "models": list(inputs.served)},
        ADDRESSES_FILE: {
            "schema": ADDRESSES_SCHEMA,
            "channels": sorted(str(channel["id"]) for channel in doc.get("channels", [])),
            "status": [status_address(code, name) for name in physics],
            "server_health": [
                address for name in inputs.served for address in server_health_addresses(code, name)
            ],
        },
        VARIABLES_FILE: _variables_document(inputs),
        SEEDS_FILE: _seeds_document(doc),
        FOLLOWS_FILE: _follows_document(doc),
        SCENARIOS_FILE: _scenarios_document(inputs),
    }

    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, document in documents.items():
        target = root / name
        target.write_bytes(view_bytes(document))
        written.append(target)

    written.extend(_copy_scenario_files(root, inputs))

    for model in doc.get("models", []):
        served = served_deck_path(model)
        if served is None:
            continue
        (root / DECKS_DIR).mkdir(exist_ok=True)
        target = root / served
        target.write_bytes((inputs.facility_dir / model["deck"]).read_bytes())
        written.append(target)
    return written


def withdraw_scenarios(root: Path, names: Iterable[str]) -> None:
    """Withdraw scenarios from the simulator view written into ``root``.

    ``scenarios.json`` lists the named scenarios no more and their
    ``scenarios/<name>/`` copies are gone; every other scenario and every
    other view file is left as written.

    Args:
        root: The view's directory, ``<render>/data/simulator``.
        names: The scenarios to withdraw.

    Raises:
        KeyError: the view lists no scenario of one of ``names``; nothing is
            withdrawn.
    """
    withdrawn = set(names)
    listed = SimulatorView.open(root).scenarios()
    missing = withdrawn - {str(scenario["name"]) for scenario in listed}
    if missing:
        raise KeyError(f"the simulator view lists no scenario {sorted(missing)}")
    kept = [_thawed(scenario) for scenario in listed if str(scenario["name"]) not in withdrawn]
    (root / SCENARIOS_FILE).write_bytes(view_bytes({"schema": SCENARIOS_SCHEMA, "scenarios": kept}))
    for name in withdrawn:
        shutil.rmtree(root / SCENARIOS_DIR / name, ignore_errors=True)


def _thawed(value: Any) -> Any:
    """A JSON value of a read-only view document, as plain dicts and lists."""
    if isinstance(value, Mapping):
        return {key: _thawed(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thawed(item) for item in value]
    return value
