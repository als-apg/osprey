"""Stage a render's simulator view, its channel records and scenarios, as the build does."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from osprey.facility.sources import read_yaml
from osprey.facility.views import view_bytes
from osprey_connectors.simulation.records import (
    AddressesDocument,
    ScenariosDocument,
    SeedsDocument,
    ServedModelsDocument,
    VariablesDocument,
    ViewDocument,
)
from osprey_connectors.simulation.view import SCENARIOS_DIR

#: The documents of a texture-only view with no channels, by document.
_MINIMAL_VIEW: Mapping[type[ViewDocument], Mapping[str, Any]] = {
    ServedModelsDocument: {"models": ["texture"]},
    AddressesDocument: {
        "channels": [],
        "status": [],
        "ioc_health": ["T:SIM:texture:HEARTBEAT", "T:SIM:texture:UPTIME", "T:SIM:texture:ENGINEER"],
    },
    VariablesDocument: {
        "code": "T",
        "models": [
            {
                "name": "texture",
                "engine": "texture",
                "served": True,
                "settings": {},
                "deck": None,
                "wiring": [],
            }
        ],
        "channels": [],
    },
    SeedsDocument: {"seeds": {}},
}


def write_scenarios_view(
    render: Path,
    scenarios: Mapping[str, Mapping[str, Any]],
    files: Path | None = None,
) -> Path:
    """Write ``<render>/data/simulator/scenarios.json`` listing ``scenarios``.

    Each other view document not already present is written as a minimal
    texture-only view, so the directory opens as a simulator view.

    Args:
        render: The render's root.
        scenarios: Scenario name -> its blocks (``description``, ``archiver``,
            ``logbook``, ...), carried verbatim but for a facility file's
            ``faults``, ``{model: writes}``, which the build writes as
            ``{model: {writes}}``.
        files: A facility ``scenarios/`` directory whose ``<name>/``
            subdirectories, holding the files the entries attach, are copied
            beside the view as the build copies them.

    Returns:
        The file written.
    """
    document = {
        "schema": ScenariosDocument.SCHEMA,
        "scenarios": [_view_scenario(name, scenarios[name]) for name in sorted(scenarios)],
    }
    path = render / "data" / "simulator" / ScenariosDocument.FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(view_bytes(document))
    for kind, minimal in _MINIMAL_VIEW.items():
        if not (path.parent / kind.FILE).is_file():
            (path.parent / kind.FILE).write_bytes(view_bytes({"schema": kind.SCHEMA, **minimal}))
    if files is not None:
        for name in scenarios:
            if (files / name).is_dir():
                shutil.copytree(files / name, path.parent / SCENARIOS_DIR / name)
    return path


def _view_scenario(name: str, blocks: Mapping[str, object]) -> dict[str, object]:
    """A scenario's blocks as ``scenarios.json`` states them."""
    scenario = {"name": name, **dict(blocks)}
    faults = scenario.get("faults")
    if faults is not None:
        scenario["faults"] = {
            model: writes if "writes" in writes else {"writes": writes}
            for model, writes in faults.items()
        }
    return scenario


def facility_scenarios(directory: Path) -> dict[str, dict[str, Any]]:
    """The scenarios of a facility's ``scenarios/`` directory, by file stem."""
    return {
        path.stem: read_yaml(path.read_text(encoding="utf-8")) or {}
        for path in sorted(directory.glob("*.yaml"))
    }


def view_channel(address: str, owner: str = "texture", **fields: Any) -> dict[str, Any]:
    """One channel record, every field filled.

    A readback by default, owned by ``owner``; a setpoint pairs with itself
    unless ``pair`` says otherwise. Any other field in ``fields`` replaces its
    default or is added to the record.
    """
    role = fields.pop("role", "readback")
    return {
        "address": address,
        "role": role,
        "pair": address if role == "setpoint" else None,
        "value_type": "float",
        "unit": None,
        "description": None,
        "writable": False,
        "value_range": None,
        "owner": owner,
        "on": None,
        **fields,
    }


#: The fields of a :func:`write_texture_view` channel that go to ``seeds.json``.
_SEED_FIELDS = ("nominal", "noise", "drift", "clamp")


def write_texture_view(
    render: Path,
    channels: Mapping[str, Mapping[str, Any]],
    scenarios: Mapping[str, Mapping[str, Any]] | None = None,
    models: Sequence[Mapping[str, Any]] = (),
) -> Path:
    """Write a whole simulator view, every document carrying its schema.

    Args:
        render: The render's root.
        channels: Address -> its ``value_type`` (``float`` when absent),
            ``options``, ``role`` (``readback`` when absent) and ``owner``
            (``texture`` when absent), plus the seed fields ``nominal``,
            ``noise``, ``drift`` and ``clamp``.
        scenarios: Scenario name -> its blocks, as :func:`write_scenarios_view`
            takes them.
        models: Served physics models, each a ``variables.json`` model record
            (``name``, ``engine``, ``settings``, ``deck`` and ``wiring``, every
            wiring record carrying its ``engine_role``, ``plane`` and ``refresh``);
            each is listed as served with its status and IOC health addresses.

    Returns:
        The view directory, ``<render>/data/simulator``.
    """
    records = []
    seeds: dict[str, dict[str, Any]] = {}
    for address in sorted(channels):
        spec = dict(channels[address])
        role = spec.get("role", "readback")
        records.append(
            view_channel(
                address,
                spec.get("owner", "texture"),
                role=role,
                value_type=spec.get("value_type", "float"),
                writable=role == "setpoint",
                **({"options": spec["options"]} if "options" in spec else {}),
            )
        )
        seed = {field: spec[field] for field in _SEED_FIELDS if field in spec}
        if seed:
            seeds[address] = seed
    physics = [{"served": True, **dict(model)} for model in models]
    names = sorted(str(model["name"]) for model in physics)
    view = render / "data" / "simulator"
    documents = {
        ServedModelsDocument: {"models": [*names, "texture"]},
        AddressesDocument: {
            "channels": sorted(channels),
            "status": [f"T:SIM:{name}:STATUS" for name in names],
            "ioc_health": [
                f"T:SIM:{name}:{signal}"
                for name in (*names, "texture")
                for signal in ("HEARTBEAT", "UPTIME", "ENGINEER")
            ],
        },
        VariablesDocument: {"code": "T", "models": physics, "channels": records},
        SeedsDocument: {"seeds": seeds},
    }
    view.mkdir(parents=True, exist_ok=True)
    for kind, document in documents.items():
        (view / kind.FILE).write_bytes(view_bytes({"schema": kind.SCHEMA, **document}))
    write_scenarios_view(render, {"nominal": {}, **(scenarios or {})})
    return view
